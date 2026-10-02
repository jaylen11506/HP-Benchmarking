#!/usr/bin/env python3
"""Runs run_benchmark.py against tests/fake_llama_server.py and checks the rows it writes.

No GPU, model, or llama.cpp build needed:  python tests/smoke_test.py
"""

import csv
import json
import os
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
FAKE_SERVER = REPO / "tests" / "fake_llama_server.py"
COLUMNS = json.loads((REPO / "results_schema.json").read_text())["required"]


def run(tmp, *cli, env=None, output="results.csv"):
    """Run the harness in a scratch directory; returns (exit code, rows in the output CSV)."""
    full_env = {
        **os.environ,
        "RESULTS_DIR": str(tmp / "results"),
        "MODELS_DIR": str(tmp / "models"),
        "BUILD_INFO_PATH": str(tmp / "BUILD_INFO"),
        **(env or {}),
    }
    out = tmp / "results" / output
    proc = subprocess.run(
        [sys.executable, str(REPO / "run_benchmark.py"), "--models", str(tmp / "models.json"),
         "--server-bin", str(FAKE_SERVER), "--output", str(out), "--output-tokens", "16", "--reps", "2", *cli],
        env=full_env, capture_output=True, text=True,
    )
    rows = []
    if out.exists():
        with open(out, newline="") as f:
            reader = csv.DictReader(f)
            assert reader.fieldnames == COLUMNS, reader.fieldnames
            rows = list(reader)
    return proc, rows


def main():
    FAKE_SERVER.chmod(0o755)
    tmp = Path(tempfile.mkdtemp(prefix="bench-smoke-"))
    (tmp / "models").mkdir()
    (tmp / "models" / "fake.gguf").write_bytes(b"not a real model")
    (tmp / "BUILD_INFO").write_text("llama.cpp=fake GGML_CUDA=OFF CUDA_ARCH=0\n")
    (tmp / "models.json").write_text(json.dumps([
        {"model": "Fake-1B", "params": "1B", "quantization": "Q4_K_M", "hf_repo": "fake/Fake-1B-GGUF", "hf_file": "fake.gguf"}
    ]))

    # 1. A normal sweep writes one row per configuration with the fake server's known timings.
    proc, rows = run(tmp, "--contexts", "256,1024", "--batch-sizes", "1,4")
    assert proc.returncode == 0, proc.stderr
    assert len(rows) == 4, proc.stderr
    assert [(r["context_len"], r["batch_size"]) for r in rows] == [("256", "1"), ("256", "4"), ("1024", "1"), ("1024", "4")]
    for r in rows:
        assert r["model"] == "Fake-1B" and r["version"] == "fake/Fake-1B-GGUF:Q4_K_M" and r["runtime"] == "llama.cpp"
        assert r["runtime_version"] == (
            "version: 0.0.0-fake (build 1, commit 0000000); built with fake compiler for test; "
            "llama.cpp=fake GGML_CUDA=OFF CUDA_ARCH=0"
        ), r["runtime_version"]
        assert 50 <= float(r["ttft_ms"]) <= 250, r["ttft_ms"]  # fake prefill is 50 ms
        assert 10 <= float(r["inter_token_ms"]) <= 25, r["inter_token_ms"]  # fake token gap is 10 ms
        assert abs(float(r["tokens_per_sec"]) - 1000 / float(r["inter_token_ms"])) < 5
        assert float(r["peak_mem_gb"]) > 0
        assert r["ambient_c"] == "" and "ambient_c not measured" in r["notes"]
        if r["avg_watts"] == "":
            assert "avg_watts not measured" in r["notes"]
        assert f"concurrency={r['batch_size']}" in r["notes"] and "aggregate_e2e_tps=" in r["notes"]
    # Four simultaneous requests should deliver clearly more total tokens/s than one.
    agg = {(r["context_len"], r["batch_size"]): float(r["notes"].split("aggregate_e2e_tps=")[1].split(";")[0]) for r in rows}
    assert agg[("256", "4")] > 2 * agg[("256", "1")], agg

    # 2. Re-running with --skip-existing adds nothing.
    proc, rows = run(tmp, "--contexts", "256,1024", "--batch-sizes", "1,4", "--skip-existing")
    assert proc.returncode == 0 and len(rows) == 4, proc.stderr

    # 3. A context longer than the model was trained for is skipped, not written.
    proc, rows = run(tmp, "--contexts", "16384", "--batch-sizes", "1", output="too_long.csv")
    assert proc.returncode == 1 and rows == [], proc.stderr
    assert "trained for 8192" in proc.stderr, proc.stderr

    # 4. A server that prepends a token of its own still gets prompts of exactly context_len.
    proc, rows = run(tmp, "--contexts", "256", "--batch-sizes", "2", env={"FAKE_ADDED_TOKENS": "1"}, output="bos.csv")
    assert proc.returncode == 0 and len(rows) == 1, proc.stderr

    # 5. A configuration that does not fit in memory is skipped and the sweep carries on.
    proc, rows = run(tmp, "--contexts", "1024", "--batch-sizes", "1,4",
                     env={"FAKE_MAX_TOTAL_CTX": "2000"}, output="oom.csv")
    assert proc.returncode == 0, proc.stderr
    assert [r["batch_size"] for r in rows] == ["1"], proc.stderr
    assert "exited during startup" in proc.stderr

    # 6. A model only partly on the GPU is skipped instead of producing misleading numbers.
    proc, rows = run(tmp, "--contexts", "256", "--batch-sizes", "1", env={"FAKE_OFFLOADED": "20/33"}, output="partial.csv")
    assert proc.returncode == 1 and rows == [], proc.stderr
    assert "only 20 of 33 layers went to the GPU" in proc.stderr, proc.stderr

    # 7. --skip-existing is per machine: rows measured on another GPU do not hide this run.
    other = tmp / "results" / "other_gpu.csv"
    other.write_text((tmp / "results" / "results.csv").read_text().replace("GPU=none", "GPU=Tesla T4"))
    proc, rows = run(tmp, "--contexts", "256", "--batch-sizes", "1", "--skip-existing", output="other_gpu.csv")
    assert proc.returncode == 0 and len(rows) == 5, proc.stderr

    # 8. Appending to the team's existing results.csv keeps every earlier row intact.
    (tmp / "results").mkdir(exist_ok=True)
    shutil.copy(REPO / "results.csv", tmp / "results" / "team.csv")
    before = (tmp / "results" / "team.csv").read_text()
    proc, rows = run(tmp, "--contexts", "256", "--batch-sizes", "1", "--ambient-c", "21.5", output="team.csv")
    assert proc.returncode == 0, proc.stderr
    after = (tmp / "results" / "team.csv").read_text()
    assert after.startswith(before) and len(rows) == before.count("\n"), (len(rows), before.count("\n"))
    assert rows[-1]["ambient_c"] == "21.5" and "ambient_c not measured" not in rows[-1]["notes"]

    shutil.rmtree(tmp)
    print("smoke test passed")


if __name__ == "__main__":
    main()
