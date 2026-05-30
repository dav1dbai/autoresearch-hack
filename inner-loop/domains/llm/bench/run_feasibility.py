#!/usr/bin/env python3
"""Phase 0: check which engines can run Qwen3.6-27B (or documented substitutes)."""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path

BENCH_ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(BENCH_ROOT))

from lib import BENCH_ROOT, gpu_name_slug, load_yaml, run_cmd, write_json  # noqa: E402


def probe_engine(name: str, spec: dict, cfg: dict, *, base_url: str | None, smoke_mirage: bool) -> dict:
    track = spec.get("track", "?")
    kind = spec.get("kind", "http")
    row: dict = {"engine": name, "track": track, "kind": kind, "status": "unknown", "details": {}}

    if kind == "http":
        if not base_url:
            row["status"] = "skipped"
            row["details"]["reason"] = "pass --base-url or start server and re-run with --engine"
            return row
        port = spec.get("port", cfg["defaults"]["port"])
        url = base_url if "://" in base_url else f"http://127.0.0.1:{port}"
        model = cfg["model"]["id"]
        rc, out = run_cmd(
            [
                sys.executable,
                str(BENCH_ROOT / "engines/probes/http_probe.py"),
                "--base-url",
                url,
                "--model",
                model,
            ],
            timeout=180,
        )
        try:
            body = json.loads(out.splitlines()[-1])
        except Exception:
            body = {"ok": False, "error": out[-400:]}
        row["details"] = body
        row["status"] = "pass" if rc == 0 and body.get("ok") else "fail"
        return row

    if name == "mirage":
        mirage_home = os.environ.get("MIRAGE_HOME", "/opt/mirage")
        cmd = [
            sys.executable,
            str(BENCH_ROOT / "engines/probes/mirage_probe.py"),
            "--mirage-home",
            mirage_home,
            "--demo-model",
            spec.get("demo_model", "Qwen/Qwen3-8B"),
            "--target-model",
            spec.get("target_model", cfg["model"]["bf16_id"]),
        ]
        if smoke_mirage:
            cmd.append("--smoke")
        rc, out = run_cmd(cmd, timeout=2000 if smoke_mirage else 60)
        try:
            body = json.loads(out.splitlines()[-1])
        except Exception:
            body = {"ok": False, "error": out[-400:]}
        row["details"] = body
        row["details"]["notes"] = spec.get("notes", "")
        if body.get("import_ok") and not smoke_mirage:
            row["status"] = "partial"
            row["details"]["target_qwen36"] = "blocked_until_port"
        elif body.get("ok"):
            row["status"] = "partial"
            row["details"]["target_qwen36"] = "blocked_until_port"
        else:
            row["status"] = "fail"
        return row

    if kind == "megakernel" and spec.get("probe", "").endswith("import_probe.py"):
        import_name = spec.get("import_name", name)
        rc, out = run_cmd(
            [
                sys.executable,
                str(BENCH_ROOT / "engines/probes/import_probe.py"),
                "--import-name",
                import_name,
            ],
            timeout=30,
        )
        try:
            body = json.loads(out.splitlines()[-1])
        except Exception:
            body = {"ok": False, "error": out[-400:]}
        row["details"] = body
        row["details"]["notes"] = spec.get("notes", "")
        row["status"] = "partial" if body.get("ok") else "fail"
        row["details"]["target_qwen36"] = "blocked_until_port"
        return row

    row["status"] = "skipped"
    row["details"]["reason"] = "no probe configured"
    return row


def collect_env() -> dict:
    env = {"cuda_visible": os.environ.get("CUDA_VISIBLE_DEVICES")}
    try:
        import torch

        if torch.cuda.is_available():
            env["gpu"] = torch.cuda.get_device_name(0)
            env["torch"] = torch.__version__
    except Exception as ex:
        env["gpu_error"] = str(ex)
    return env


def main() -> None:
    p = argparse.ArgumentParser(description="Phase 0 feasibility gate for LLM engines")
    p.add_argument("--engine", action="append", help="Limit to one or more engines from config.yaml")
    p.add_argument("--base-url", help="OpenAI-compatible base URL for HTTP engines")
    p.add_argument("--smoke-mirage", action="store_true", help="Run Mirage MPK Qwen3-8B megakernel smoke")
    p.add_argument("-o", "--output", type=Path, help="Output JSON path")
    args = p.parse_args()

    cfg = load_yaml("config.yaml")
    engines: dict = cfg["engines"]
    if args.engine:
        engines = {k: v for k, v in engines.items() if k in args.engine}

    t0 = time.time()
    results = []
    for name, spec in engines.items():
        print(f"Probing {name}...", flush=True)
        row = probe_engine(name, spec, cfg, base_url=args.base_url, smoke_mirage=args.smoke_mirage)
        results.append(row)
        print(f"  -> {row['status']}", flush=True)

    artifact = {
        "model": cfg["model"],
        "gpu_slug": gpu_name_slug(),
        "environment": collect_env(),
        "elapsed_s": round(time.time() - t0, 2),
        "engines": results,
    }
    out = args.output or (BENCH_ROOT / "results" / f"feasibility_{gpu_name_slug()}.json")
    write_json(out, artifact)
    print(f"\nWrote {out}")
    failed = [r for r in results if r["status"] == "fail"]
    if failed:
        sys.exit(1)


if __name__ == "__main__":
    main()
