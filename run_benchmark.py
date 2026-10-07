#!/usr/bin/env python3
"""Benchmark harness: drives llama-server and appends one row per configuration to results.csv.

For every model x context length x batch size it starts a fresh llama-server, sends
`batch size` simultaneous streaming requests, and records:

  ttft_ms          time from sending a request to its first generated token (prefill latency)
  inter_token_ms   average gap between generated tokens for one request
  tokens_per_sec   decode rate of one request (1000 / inter_token_ms)
  peak_mem_gb      peak GPU memory in use while the server was up (GiB)
  avg_watts        mean GPU power during the timed requests (NVIDIA telemetry, NOT wall power)

Each value is the mean over the simultaneous requests, then the median over repetitions.
The model is loaded and warmed up before timing starts, so ttft_ms never includes load time.

Same code on every machine; only the environment changes:
  RESULTS_DIR      where results.csv and server logs go   (default: ./results, /app/results in Docker)
  MODELS_DIR       where GGUF files live / are downloaded (default: /models if present, else ./models)
  BUILD_INFO_PATH  build description written into runtime_version (default: /opt/BUILD_INFO)

Examples:
  python run_benchmark.py --dry-run
  python run_benchmark.py --only Qwen3-8B --contexts 1024 --batch-sizes 1,2,4
  python run_benchmark.py --contexts 1024,4096,16384 --batch-sizes 1,2,4,8,16 --skip-existing
"""

import argparse
import asyncio
import csv
import json
import logging
import os
import re
import shlex
import shutil
import socket
import statistics
import subprocess
import sys
import threading
import time
import uuid
from datetime import datetime, timezone
from pathlib import Path

import aiohttp
import jsonschema
import psutil
import requests

try:
    import pynvml

    NVMLError = pynvml.NVMLError
except ImportError:  # no NVIDIA tooling on this machine (laptop dry run)
    pynvml = None

    class NVMLError(Exception):
        pass


REPO_DIR = Path(__file__).resolve().parent
SCHEMA = json.loads((REPO_DIR / "results_schema.json").read_text())
COLUMNS = SCHEMA["required"]  # same order as the results.csv header

RESULTS_DIR = Path(os.environ.get("RESULTS_DIR", REPO_DIR / "results"))
MODELS_DIR = Path(os.environ.get("MODELS_DIR", "/models" if Path("/models").is_dir() else REPO_DIR / "models"))
BUILD_INFO_PATH = Path(os.environ.get("BUILD_INFO_PATH", "/opt/BUILD_INFO"))

# Columns this harness cannot always measure. They are left blank in the CSV and the
# notes column says so, instead of writing a made-up number.
MAY_BE_UNMEASURED = ("avg_watts", "ambient_c")

# Spare room per request slot beyond prompt + output, so a request can never overflow its slot.
CTX_MARGIN = 16
WARMUP_PROMPT_TOKENS = 32
WARMUP_OUTPUT_TOKENS = 8

FILLER = (
    "The benchmark prompt is plain filler text whose only job is to occupy context. "
    "Throughput depends on how many tokens are processed, not on what they say. "
)

log = logging.getLogger("bench")


class SkipConfig(Exception):
    """This configuration cannot be measured on this machine; move on to the next one."""


# --------------------------------------------------------------------------- telemetry


class TelemetrySampler:
    """Samples memory and power in a background thread while a server is running."""

    def __init__(self, interval_s=0.1):
        self.interval_s = interval_s
        self.handles = []
        self.gpu_name = None
        if pynvml is not None:
            try:
                pynvml.nvmlInit()
                self.handles = [pynvml.nvmlDeviceGetHandleByIndex(i) for i in range(pynvml.nvmlDeviceGetCount())]
                names = [pynvml.nvmlDeviceGetName(h) for h in self.handles]
                names = [n.decode() if isinstance(n, bytes) else n for n in names]
                self.gpu_name = " + ".join(names) if names else None
            except NVMLError as e:
                log.warning("NVML unavailable (%s); falling back to process memory, no power reading", e)
                self.handles = []

    def gpu_mem_bytes(self):
        """GPU memory in use right now, or None if this machine does not report it."""
        try:
            return sum(pynvml.nvmlDeviceGetMemoryInfo(h).used for h in self.handles) if self.handles else None
        except NVMLError:
            return None

    def start(self, pid):
        self._proc = psutil.Process(pid)
        self.mem_source = "gpu" if self.handles else "process"
        self.peak_mem_bytes = 0
        self._watts = []
        self._measuring = threading.Event()
        self._stop = threading.Event()
        self._thread = threading.Thread(target=self._loop, daemon=True)
        self._thread.start()

    def stop(self):
        self._stop.set()
        self._thread.join()

    def measuring(self, on):
        """Power is averaged only over the timed requests, not over model load or warmup."""
        self._measuring.set() if on else self._measuring.clear()

    @property
    def avg_watts(self):
        return statistics.fmean(self._watts) if self._watts else None

    def _loop(self):
        while not self._stop.is_set():
            self._sample()
            self._stop.wait(self.interval_s)

    def _sample(self):
        mem = None
        if self.mem_source == "gpu":
            try:
                mem = sum(pynvml.nvmlDeviceGetMemoryInfo(h).used for h in self.handles)
            except NVMLError:
                # Unified-memory machines (e.g. GB10) may not report per-GPU memory.
                self.mem_source = "process"
        if mem is None:
            try:
                mem = self._proc.memory_info().rss
            except psutil.Error:
                mem = 0
        self.peak_mem_bytes = max(self.peak_mem_bytes, mem)
        if self.handles and self._measuring.is_set():
            try:
                self._watts.append(sum(pynvml.nvmlDeviceGetPowerUsage(h) for h in self.handles) / 1000.0)
            except NVMLError:
                pass  # this GPU does not report power


# --------------------------------------------------------------------------- llama-server


class LlamaServer:
    """Runs one llama-server process for one configuration."""

    def __init__(self, server_bin, model_path, n_ctx, parallel, gpu_layers, extra_args, log_path, startup_timeout_s):
        self.port = _free_port()
        self.url = f"http://127.0.0.1:{self.port}"
        self.cmd = [
            server_bin, "-m", str(model_path), "-c", str(n_ctx), "-np", str(parallel),
            "-ngl", str(gpu_layers), "--host", "127.0.0.1", "--port", str(self.port), *extra_args,
        ]
        self.log_path = log_path
        self.startup_timeout_s = startup_timeout_s
        self.proc = None

    def __enter__(self):
        self.log_path.parent.mkdir(parents=True, exist_ok=True)
        self._log_file = open(self.log_path, "w")
        self.proc = subprocess.Popen(self.cmd, stdout=self._log_file, stderr=subprocess.STDOUT)
        deadline = time.monotonic() + self.startup_timeout_s
        while time.monotonic() < deadline:
            if self.proc.poll() is not None:
                self._log_file.close()
                # Most often the model plus context does not fit in GPU memory.
                raise SkipConfig(f"llama-server exited during startup (code {self.proc.returncode}); see {self.log_path}")
            try:
                if requests.get(f"{self.url}/health", timeout=2).status_code == 200:
                    return self
            except requests.RequestException:
                pass
            time.sleep(0.5)
        self.__exit__(None, None, None)
        raise SkipConfig(f"llama-server not ready after {self.startup_timeout_s}s; see {self.log_path}")

    def __exit__(self, *exc):
        if self.proc.poll() is None:
            self.proc.terminate()
            try:
                self.proc.wait(timeout=30)
            except subprocess.TimeoutExpired:
                self.proc.kill()
                self.proc.wait()
        self._log_file.close()

    def offloaded_layers(self):
        """(layers on GPU, layers in model) from the server's startup log, or None if it did not say."""
        match = re.search(r"offloaded (\d+)/(\d+) layers to GPU", self.log_path.read_text(errors="replace"))
        return (int(match.group(1)), int(match.group(2))) if match else None

    def slot_ctx(self):
        """Context available to one request slot, as the server itself reports it."""
        try:
            return requests.get(f"{self.url}/props", timeout=10).json()["default_generation_settings"]["n_ctx"]
        except (requests.RequestException, KeyError, ValueError):
            return None

    def n_ctx_train(self):
        """Longest context the model was trained for."""
        try:
            return requests.get(f"{self.url}/v1/models", timeout=10).json()["data"][0]["meta"]["n_ctx_train"]
        except (requests.RequestException, KeyError, IndexError, ValueError, TypeError):
            return None

    def tokenize(self, text, add_special):
        r = requests.post(f"{self.url}/tokenize", json={"content": text, "add_special": add_special}, timeout=300)
        r.raise_for_status()
        return r.json()["tokens"]


def _free_port():
    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]


def runtime_version(server_bin):
    """llama-server's own version lines plus the build description, e.g. the CUDA arch it was compiled for."""
    out = subprocess.run([server_bin, "--version"], capture_output=True, text=True)
    lines = [line.strip() for line in (out.stdout + out.stderr).splitlines() if line.strip()]
    wanted = [line for line in lines if line.startswith(("version:", "built with"))] or lines
    if BUILD_INFO_PATH.exists():
        wanted.append(BUILD_INFO_PATH.read_text().strip())
    return "; ".join(wanted)


# --------------------------------------------------------------------------- requests


def filler_tokens(server, count):
    """`count` tokens of filler text, tokenized by the model's own tokenizer."""
    repeats = max(1, count // 8)
    while True:
        tokens = server.tokenize(FILLER * repeats, add_special=False)
        if len(tokens) >= count:
            return tokens[:count]
        repeats *= 2


def build_prompts(server, body, length, batch):
    """One prompt of exactly `length` tokens per request. Each starts differently so that
    no request can reuse another's cached prefix."""
    prompts = []
    for i in range(batch):
        header = server.tokenize(f"[{uuid.uuid4().hex} request {i}] ", add_special=True)
        prompts.append((header + body)[:length])
    return prompts


async def one_request(session, url, prompt_tokens, n_predict):
    payload = {
        "prompt": prompt_tokens,
        "n_predict": n_predict,
        "stream": True,
        "cache_prompt": False,
        "ignore_eos": True,  # always generate exactly n_predict tokens
        "temperature": 0,
        "seed": 0,
    }
    t_first = t_last = None
    chunks = 0
    final = None
    buf = b""
    t_send = time.perf_counter()
    async with session.post(f"{url}/completion", json=payload) as resp:
        resp.raise_for_status()
        async for data in resp.content.iter_any():
            now = time.perf_counter()
            buf += data
            while b"\n" in buf:
                line, buf = buf.split(b"\n", 1)
                if not line.startswith(b"data: "):
                    continue
                event = json.loads(line[6:])
                if event.get("stop"):
                    final = event
                    continue
                if t_first is None:
                    t_first = now
                t_last = now
                chunks += 1
    if final is None or t_first is None:
        raise RuntimeError("stream ended without generating any tokens")
    n_predicted = final.get("tokens_predicted", chunks)
    if n_predicted < 2:
        raise RuntimeError(f"only {n_predicted} token(s) generated; cannot compute inter-token latency")
    inter_token_ms = (t_last - t_first) * 1000 / (n_predicted - 1)
    return {
        "ttft_ms": (t_first - t_send) * 1000,
        "inter_token_ms": inter_token_ms,
        "tokens_per_sec": 1000 / inter_token_ms,
        "n_predicted": n_predicted,
        "prompt_tokens": final.get("tokens_evaluated"),
        "prefill_tps": (final.get("timings") or {}).get("prompt_per_second"),
        "t_last": t_last,
    }


async def run_requests(url, prompts, n_predict, timeout_s):
    """Send all prompts at the same moment. Returns per-request results and the aggregate
    end-to-end rate: all generated tokens / wall time from send to the last token."""
    timeout = aiohttp.ClientTimeout(total=timeout_s)
    async with aiohttp.ClientSession(timeout=timeout) as session:
        t0 = time.perf_counter()
        results = await asyncio.gather(*[one_request(session, url, p, n_predict) for p in prompts])
    wall = max(r["t_last"] for r in results) - t0
    return results, sum(r["n_predicted"] for r in results) / wall


# --------------------------------------------------------------------------- one configuration


def measure_config(args, spec, model_path, context, batch, sampler, n_ctx_train_seen):
    """Returns the measured values for one model x context x batch size, or raises SkipConfig."""
    version = model_version(spec)
    needed = context + args.output_tokens
    _check_train_ctx(args, n_ctx_train_seen.get(version), needed)

    gpu_mem_before = sampler.gpu_mem_bytes()
    log_path = RESULTS_DIR / "logs" / f"{spec['model']}_ctx{context}_b{batch}.log"
    server = LlamaServer(
        args.server_bin, model_path, (needed + CTX_MARGIN) * batch, batch,
        args.gpu_layers, args.server_args, log_path, args.startup_timeout,
    )
    with server:
        sampler.start(server.proc.pid)
        try:
            n_ctx_train_seen[version] = server.n_ctx_train()
            _check_train_ctx(args, n_ctx_train_seen[version], needed)
            # A model that only partly fits may be split between GPU and CPU without any
            # error, which would give plausible-looking but meaningless numbers. Prefer the
            # server's own layer count; at its default log level it does not print one, so
            # fall back to checking that the GPU now holds about as much as the model file.
            offloaded = server.offloaded_layers()
            gpu_mem_after = sampler.gpu_mem_bytes()
            if offloaded is not None:
                if offloaded[0] < min(args.gpu_layers, offloaded[1]):
                    raise SkipConfig(f"only {offloaded[0]} of {offloaded[1]} layers went to the GPU")
            elif gpu_mem_before is not None and gpu_mem_after is not None:
                on_gpu, model_bytes = gpu_mem_after - gpu_mem_before, model_path.stat().st_size
                if args.gpu_layers >= 999 and on_gpu < 0.9 * model_bytes:
                    raise SkipConfig(f"only {on_gpu / 2**30:.2f} GiB of the {model_bytes / 2**30:.2f} GiB model is on the GPU")
            else:
                log.warning("  could not confirm GPU offload from %s", log_path)
            slot_ctx = server.slot_ctx()
            if slot_ctx is not None and slot_ctx < needed:
                raise SkipConfig(f"server gives each slot {slot_ctx} tokens of context, need {needed}")

            body = filler_tokens(server, max(context, WARMUP_PROMPT_TOKENS))

            # Warmup: first request after load is slow, and it tells us whether the server
            # adds tokens of its own (e.g. a BOS) so the timed prompts can be sized exactly.
            warm = build_prompts(server, body, WARMUP_PROMPT_TOKENS, 1)
            warm_results, _ = asyncio.run(run_requests(server.url, warm, WARMUP_OUTPUT_TOKENS, args.request_timeout))
            evaluated = warm_results[0]["prompt_tokens"]
            added_by_server = (evaluated - WARMUP_PROMPT_TOKENS) if evaluated is not None else 0

            reps = []
            for _ in range(args.reps):
                prompts = build_prompts(server, body, context - added_by_server, batch)
                sampler.measuring(True)
                try:
                    results, aggregate_tps = asyncio.run(
                        run_requests(server.url, prompts, args.output_tokens, args.request_timeout)
                    )
                finally:
                    sampler.measuring(False)
                for r in results:
                    if r["prompt_tokens"] is not None and r["prompt_tokens"] != context:
                        raise SkipConfig(f"server evaluated {r['prompt_tokens']} prompt tokens, expected {context}")
                    if r["n_predicted"] != args.output_tokens:
                        raise SkipConfig(f"server generated {r['n_predicted']} tokens, expected {args.output_tokens}")
                prefill = [r["prefill_tps"] for r in results if r["prefill_tps"]]
                reps.append({
                    "ttft_ms": statistics.fmean(r["ttft_ms"] for r in results),
                    "inter_token_ms": statistics.fmean(r["inter_token_ms"] for r in results),
                    "tokens_per_sec": statistics.fmean(r["tokens_per_sec"] for r in results),
                    "aggregate_e2e_tps": aggregate_tps,
                    "prefill_tps": statistics.fmean(prefill) if prefill else None,
                })
        finally:
            sampler.stop()

    prefill = [rep["prefill_tps"] for rep in reps if rep["prefill_tps"] is not None]
    return {
        "ttft_ms": statistics.median(rep["ttft_ms"] for rep in reps),
        "inter_token_ms": statistics.median(rep["inter_token_ms"] for rep in reps),
        "tokens_per_sec": statistics.median(rep["tokens_per_sec"] for rep in reps),
        "aggregate_e2e_tps": statistics.median(rep["aggregate_e2e_tps"] for rep in reps),
        "prefill_tps": statistics.median(prefill) if prefill else None,
        "peak_mem_gb": sampler.peak_mem_bytes / 2**30,
        "avg_watts": sampler.avg_watts,
    }


def _check_train_ctx(args, n_ctx_train, needed):
    if n_ctx_train is not None and needed > n_ctx_train and not args.allow_beyond_train_ctx:
        raise SkipConfig(f"needs {needed} tokens of context but the model was trained for {n_ctx_train}")


# --------------------------------------------------------------------------- rows and CSV


def model_version(spec):
    return f"{spec['hf_repo']}:{spec['quantization']}"


def build_row(args, spec, context, batch, measured, sampler, runtime_ver):
    gpu = sampler.gpu_name or "none"  # existing_configs() reads this back out of the notes
    notes = [
        args.env_label,
        f"GPU={gpu}",
        "llama-server",
        "context_len=actual input prompt tokens",
        f"concurrency={batch}",
        f"output_target={args.output_tokens} tokens/request",
        f"reps={args.reps} (median)",
        f"aggregate_e2e_tps={measured['aggregate_e2e_tps']:.3f}",
    ]
    if measured["prefill_tps"] is not None:
        notes.append(f"prefill_tps={measured['prefill_tps']:.1f}")
    if sampler.mem_source == "gpu":
        notes.append("peak_mem_gb=GPU memory in use (GiB)")
    else:
        notes.append("peak_mem_gb=llama-server process RSS (GiB)")
    if measured["avg_watts"] is not None:
        notes.append("avg_watts=NVIDIA device telemetry, NOT wall power")
    else:
        notes.append("avg_watts not measured")
    if args.ambient_c is None:
        notes.append("ambient_c not measured")
    if args.notes:
        notes.append(args.notes)

    return {
        "timestamp": datetime.now(timezone.utc).isoformat().replace("+00:00", "Z"),
        "model": spec["model"],
        "version": model_version(spec),
        "params": spec["params"],
        "quantization": spec["quantization"],
        "runtime": "llama.cpp",
        "runtime_version": runtime_ver,
        "context_len": context,
        "batch_size": batch,
        "ttft_ms": round(measured["ttft_ms"], 3),
        "inter_token_ms": round(measured["inter_token_ms"], 3),
        "tokens_per_sec": round(measured["tokens_per_sec"], 3),
        "peak_mem_gb": round(measured["peak_mem_gb"], 3),
        "avg_watts": None if measured["avg_watts"] is None else round(measured["avg_watts"], 3),
        "ambient_c": args.ambient_c,
        "notes": "; ".join(notes),
    }


def validate_row(row):
    """Check the row against results_schema.json. A column listed in MAY_BE_UNMEASURED may
    be empty; every other column must satisfy the schema exactly."""
    schema = json.loads(json.dumps(SCHEMA))
    for column in MAY_BE_UNMEASURED:
        schema["properties"][column]["type"] = ["number", "null"]
    jsonschema.validate(row, schema)


def append_row(path, row):
    path.parent.mkdir(parents=True, exist_ok=True)
    new_file = not path.exists() or path.stat().st_size == 0
    with open(path, "a", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=COLUMNS)
        if new_file:
            writer.writeheader()
        writer.writerow(row)  # None is written as an empty cell


def existing_configs(path):
    """(version, context_len, batch_size, GPU) of llama.cpp rows already in the CSV. The GPU
    is part of the key so a run on one machine never hides the same run on another."""
    if not path.exists() or path.stat().st_size == 0:
        return set()
    with open(path, newline="") as f:
        reader = csv.DictReader(f)
        if reader.fieldnames != COLUMNS:
            sys.exit(f"{path} has a different header than results_schema.json; refusing to append to it")
        done = set()
        for r in reader:
            gpu = re.search(r"GPU=([^;]*)", r["notes"])
            if r["runtime"] == "llama.cpp" and gpu:
                done.add((r["version"], int(r["context_len"]), int(r["batch_size"]), gpu.group(1).strip()))
        return done


# --------------------------------------------------------------------------- models


def ensure_model(spec):
    """Path to the model's GGUF file, downloading it into MODELS_DIR if it is not there yet."""
    local = MODELS_DIR / spec["hf_file"]
    if local.exists():
        return local
    from huggingface_hub import hf_hub_download

    log.info("downloading %s/%s into %s", spec["hf_repo"], spec["hf_file"], MODELS_DIR)
    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    return Path(hf_hub_download(repo_id=spec["hf_repo"], filename=spec["hf_file"], local_dir=MODELS_DIR))


# --------------------------------------------------------------------------- main


def int_list(text):
    return [int(x) for x in text.split(",") if x.strip()]


def parse_args():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--models", type=Path, default=REPO_DIR / "models.json", help="JSON list of models to benchmark")
    p.add_argument("--only", help="comma-separated model names from the models file; default is all of them")
    p.add_argument("--contexts", type=int_list, default=[1024, 4096, 16384], help="prompt lengths in tokens")
    p.add_argument("--batch-sizes", type=int_list, default=[1, 2, 4, 8, 16], help="numbers of simultaneous requests")
    p.add_argument("--output-tokens", type=int, default=128, help="tokens generated per request")
    p.add_argument("--reps", type=int, default=3, help="timed repetitions per configuration; the median is reported")
    p.add_argument("--output", type=Path, default=RESULTS_DIR / "results.csv", help="CSV file to append rows to")
    p.add_argument("--skip-existing", action="store_true", help="skip configurations already present in the output CSV")
    p.add_argument("--ambient-c", type=float, help="measured room temperature; left blank if not given")
    p.add_argument("--env-label", default="AWS dry run", help="first entry of the notes column, e.g. where this ran")
    p.add_argument("--notes", help="extra text appended to the notes column")
    p.add_argument("--server-bin", default="llama-server", help="llama-server executable")
    p.add_argument("--server-args", type=shlex.split, default=[], help="extra llama-server arguments, as one quoted string")
    p.add_argument("--gpu-layers", type=int, default=999, help="layers to put on the GPU (-ngl)")
    p.add_argument("--startup-timeout", type=int, default=900, help="seconds to wait for a model to load")
    p.add_argument("--request-timeout", type=int, default=3600, help="seconds to wait for one round of requests")
    p.add_argument("--allow-beyond-train-ctx", action="store_true",
                   help="also run contexts longer than the model was trained for")
    p.add_argument("--dry-run", action="store_true", help="print the configurations that would run, then exit")
    return p.parse_args()


def main():
    args = parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s", datefmt="%H:%M:%S")

    specs = json.loads(args.models.read_text())
    if args.only:
        wanted = {name.strip() for name in args.only.split(",")}
        unknown = wanted - {s["model"] for s in specs}
        if unknown:
            sys.exit(f"not in {args.models}: {', '.join(sorted(unknown))}")
        specs = [s for s in specs if s["model"] in wanted]

    sampler = TelemetrySampler()
    gpu = sampler.gpu_name or "none"
    done = existing_configs(args.output) if args.skip_existing else set()
    plan = [
        (spec, context, batch)
        for spec in specs for context in args.contexts for batch in args.batch_sizes
        if (model_version(spec), context, batch, gpu) not in done
    ]
    log.info("%d configuration(s) to run, results -> %s", len(plan), args.output)
    if args.dry_run:
        for spec, context, batch in plan:
            print(f"{spec['model']:<16} context={context:<7} batch={batch}")
        return

    server_bin = shutil.which(args.server_bin)
    if server_bin is None:
        sys.exit(f"llama-server not found: {args.server_bin}")
    args.server_bin = server_bin
    runtime_ver = runtime_version(server_bin)
    log.info("runtime: %s | GPU: %s", runtime_ver, gpu)

    written, skipped = 0, []
    n_ctx_train_seen = {}
    model_paths = {}
    for spec, context, batch in plan:
        label = f"{spec['model']} context={context} batch={batch}"
        try:
            if spec["model"] not in model_paths:
                model_paths[spec["model"]] = ensure_model(spec)
            log.info("running %s", label)
            measured = measure_config(args, spec, model_paths[spec["model"]], context, batch, sampler, n_ctx_train_seen)
            row = build_row(args, spec, context, batch, measured, sampler, runtime_ver)
            validate_row(row)
            append_row(args.output, row)
            written += 1
            log.info("  ttft=%.0f ms  decode=%.1f tok/s  aggregate=%.1f tok/s  peak_mem=%.2f GiB",
                     row["ttft_ms"], row["tokens_per_sec"], measured["aggregate_e2e_tps"], row["peak_mem_gb"])
        except SkipConfig as e:
            skipped.append((label, str(e)))
            log.warning("  skipped: %s", e)
        except Exception as e:  # one broken configuration must not end a long sweep
            skipped.append((label, f"{type(e).__name__}: {e}"))
            log.exception("  failed: %s", label)

    log.info("done: %d row(s) written, %d configuration(s) skipped", written, len(skipped))
    for label, reason in skipped:
        log.info("  skipped %s: %s", label, reason)
    if plan and written == 0:
        sys.exit(1)


if __name__ == "__main__":
    main()
