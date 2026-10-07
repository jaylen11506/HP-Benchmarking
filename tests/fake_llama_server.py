#!/usr/bin/env python3
"""Stand-in for llama-server so run_benchmark.py can be tested without a GPU or a model.

Accepts the same flags the harness passes and serves the same endpoints with fixed,
known timings: FAKE_PREFILL_S before the first token, FAKE_TOKEN_S between tokens.
One "token" is one whitespace-separated word; BOS is token 1.
"""

import argparse
import asyncio
import json
import os
import sys

from aiohttp import web

PREFILL_S = float(os.environ.get("FAKE_PREFILL_S", "0.05"))
TOKEN_S = float(os.environ.get("FAKE_TOKEN_S", "0.01"))
N_CTX_TRAIN = int(os.environ.get("FAKE_N_CTX_TRAIN", "8192"))
# Pretend the server prepends this many tokens of its own to every prompt.
ADDED_TOKENS = int(os.environ.get("FAKE_ADDED_TOKENS", "0"))
# Refuse to start when the total context is above this, like running out of GPU memory.
MAX_TOTAL_CTX = int(os.environ.get("FAKE_MAX_TOTAL_CTX", "0"))
# "layers on GPU/layers in model", as llama-server prints it while loading.
OFFLOADED = os.environ.get("FAKE_OFFLOADED", "33/33")


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("-m")
    p.add_argument("-c", type=int, default=4096)
    p.add_argument("-np", type=int, default=1)
    p.add_argument("-ngl", type=int, default=0)
    p.add_argument("--host", default="127.0.0.1")
    p.add_argument("--port", type=int, default=8080)
    p.add_argument("--version", action="store_true")
    return p.parse_args()


def make_app(args):
    async def health(_):
        return web.json_response({"status": "ok"})

    async def props(_):
        return web.json_response({"default_generation_settings": {"n_ctx": args.c // args.np}})

    async def models(_):
        return web.json_response({"data": [{"id": "fake", "meta": {"n_ctx_train": N_CTX_TRAIN}}]})

    async def tokenize(request):
        body = await request.json()
        tokens = [2 + (hash(word) % 1000) for word in body["content"].split()]
        if body.get("add_special"):
            tokens = [1] + tokens
        return web.json_response({"tokens": tokens})

    async def completion(request):
        body = await request.json()
        n_predict = body["n_predict"]
        evaluated = len(body["prompt"]) + ADDED_TOKENS
        resp = web.StreamResponse(headers={"Content-Type": "text/event-stream"})
        await resp.prepare(request)
        await asyncio.sleep(PREFILL_S)
        for i in range(n_predict):
            if i:
                await asyncio.sleep(TOKEN_S)
            await resp.write(b"data: " + json.dumps({"content": "x", "stop": False}).encode() + b"\n\n")
        final = {
            "content": "",
            "stop": True,
            "tokens_predicted": n_predict,
            "tokens_evaluated": evaluated,
            "timings": {"prompt_n": evaluated, "prompt_per_second": evaluated / PREFILL_S},
        }
        await resp.write(b"data: " + json.dumps(final).encode() + b"\n\n")
        await resp.write_eof()
        return resp

    app = web.Application(client_max_size=1024**3)
    app.add_routes([
        web.get("/health", health),
        web.get("/props", props),
        web.get("/v1/models", models),
        web.post("/tokenize", tokenize),
        web.post("/completion", completion),
    ])
    return app


def main():
    args = parse_args()
    if args.version:
        print("ggml_init: some unrelated startup line", file=sys.stderr)
        print("version: 0.0.0-fake (build 1, commit 0000000)", file=sys.stderr)
        print("built with fake compiler for test", file=sys.stderr)
        return
    if MAX_TOTAL_CTX and args.c > MAX_TOTAL_CTX:
        sys.exit("fake: out of memory")
    print(f"load_tensors: offloaded {OFFLOADED} layers to GPU", file=sys.stderr, flush=True)
    web.run_app(make_app(args), host=args.host, port=args.port, print=None)


if __name__ == "__main__":
    main()
