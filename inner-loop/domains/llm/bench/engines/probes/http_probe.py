#!/usr/bin/env python3
"""Probe an OpenAI-compatible HTTP inference server."""

from __future__ import annotations

import argparse
import json
import sys
import time

import httpx


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--base-url", required=True)
    p.add_argument("--model", default=None)
    p.add_argument("--max-tokens", type=int, default=16)
    args = p.parse_args()

    base = args.base_url.rstrip("/")
    t0 = time.perf_counter()
    try:
        with httpx.Client(base_url=base, timeout=120.0) as client:
            models = client.get("/v1/models")
            models.raise_for_status()
            model = args.model
            if not model:
                body = models.json()
                data = body.get("data") or []
                if not data:
                    print(json.dumps({"ok": False, "error": "no models in /v1/models"}))
                    return 1
                model = data[0]["id"]
            resp = client.post(
                "/v1/chat/completions",
                json={
                    "model": model,
                    "messages": [{"role": "user", "content": "Say hi in three words."}],
                    "max_tokens": args.max_tokens,
                    "temperature": 0.0,
                    "stream": False,
                },
            )
            resp.raise_for_status()
            text = resp.json()["choices"][0]["message"]["content"]
    except Exception as ex:
        print(json.dumps({"ok": False, "error": str(ex)[:400], "latency_s": round(time.perf_counter() - t0, 3)}))
        return 1

    print(
        json.dumps(
            {
                "ok": True,
                "model": model,
                "sample": text[:200],
                "latency_s": round(time.perf_counter() - t0, 3),
            }
        )
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
