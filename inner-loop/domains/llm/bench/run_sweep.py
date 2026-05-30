#!/usr/bin/env python3
"""Run workload sweeps against a live OpenAI-compatible inference server."""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

BENCH_ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(BENCH_ROOT))

from clients.openai_bench import run_workload_sync  # noqa: E402
from lib import BENCH_ROOT, gpu_name_slug, load_yaml, write_json  # noqa: E402


def build_messages(workload: dict, prompts: dict) -> list[dict]:
    msgs: list[dict] = []
    if "system_key" in workload:
        msgs.append({"role": "system", "content": prompts["prompts"][workload["system_key"]].strip()})
    prompt = prompts["prompts"][workload["prompt_key"]].strip()
    msgs.append({"role": "user", "content": prompt})
    return msgs


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--base-url", required=True, help="e.g. http://127.0.0.1:8000")
    p.add_argument("--engine", default="vllm")
    p.add_argument("--model", default=None)
    p.add_argument("--workload", action="append", help="Workload id from workloads.yaml")
    p.add_argument("--concurrency", type=int, action="append")
    p.add_argument("--n-requests", type=int, default=None)
    p.add_argument("--warmup", type=int, default=None)
    p.add_argument("-o", "--output", type=Path)
    args = p.parse_args()

    cfg = load_yaml("config.yaml")
    wl_cfg = load_yaml("workloads.yaml")
    defaults = cfg["defaults"]
    model = args.model or cfg["model"]["id"]
    engine_spec = cfg["engines"].get(args.engine, {})
    track = engine_spec.get("track", "?")

    workloads = wl_cfg["workloads"]
    if args.workload:
        workloads = {k: v for k, v in workloads.items() if k in args.workload}

    runs = []
    t0 = time.time()
    for wl_id, wl in workloads.items():
        conc_list = args.concurrency or wl.get("concurrency", [1])
        messages = build_messages(wl, wl_cfg)
        for conc in conc_list:
            print(f"Running {wl_id} @ concurrency={conc}...", flush=True)
            metrics = run_workload_sync(
                args.base_url,
                model=model,
                messages=messages,
                max_tokens=wl["max_tokens"],
                concurrency=conc,
                n_requests=args.n_requests or defaults["measure_requests"],
                warmup=args.warmup if args.warmup is not None else defaults["warmup_requests"],
                temperature=defaults["temperature"],
                top_p=defaults["top_p"],
            )
            runs.append(
                {
                    "engine": args.engine,
                    "track": track,
                    "workload": wl_id,
                    "concurrency": conc,
                    "model": model,
                    "metrics": metrics,
                }
            )
            print(f"  tok/s={metrics.get('output_tok_s')} ttft_p50={metrics.get('ttft_ms', {}).get('p50')}", flush=True)

    slug = gpu_name_slug()
    out = args.output or (BENCH_ROOT / "results" / f"baselines_{slug}_{args.engine}.json")
    write_json(
        out,
        {
            "gpu_slug": slug,
            "model": model,
            "engine": args.engine,
            "track": track,
            "base_url": args.base_url,
            "elapsed_s": round(time.time() - t0, 2),
            "runs": runs,
        },
    )
    print(f"\nWrote {out}")


if __name__ == "__main__":
    main()
