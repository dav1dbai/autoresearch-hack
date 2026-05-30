#!/usr/bin/env python3
"""Probe Mirage MPK install and Qwen3 megakernel smoke (demo model)."""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
from pathlib import Path


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--mirage-home", default=os.environ.get("MIRAGE_HOME", "/opt/mirage"))
    p.add_argument("--demo-model", default="Qwen/Qwen3-8B")
    p.add_argument("--target-model", default="Qwen/Qwen3.6-27B")
    p.add_argument("--smoke", action="store_true", help="Run MPK demo with --max-new-tokens 4")
    args = p.parse_args()

    mirage_home = Path(args.mirage_home)
    demo = mirage_home / "demo" / "qwen3" / "demo.py"
    out: dict = {
        "ok": False,
        "mirage_home": str(mirage_home),
        "demo_exists": demo.is_file(),
        "target_model": args.target_model,
        "demo_model": args.demo_model,
        "target_supported": False,
        "notes": "MPK ships Qwen3-8B demo; Qwen3.6-27B GDN requires a new graph port.",
    }

    try:
        import mirage as mi  # noqa: F401

        out["import_ok"] = True
        out["mirage_version"] = getattr(mi, "__version__", "unknown")
    except Exception as ex:
        out["import_ok"] = False
        out["error"] = f"import mirage: {ex}"
        print(json.dumps(out))
        return 1

    if not demo.is_file():
        out["error"] = f"missing demo: {demo}"
        print(json.dumps(out))
        return 1

    if args.smoke:
        cmd = [
            "python",
            str(demo),
            "--use-mirage",
            "--model",
            args.demo_model,
            "--max-new-tokens",
            "4",
            "--max-seq-length",
            "512",
            "--prompt",
            "Hello",
        ]
        proc = subprocess.run(cmd, cwd=mirage_home / "demo" / "qwen3", capture_output=True, text=True, timeout=1800)
        out["smoke_rc"] = proc.returncode
        out["smoke_tail"] = (proc.stdout + proc.stderr)[-800:]
        out["ok"] = proc.returncode == 0
    else:
        out["ok"] = True

    print(json.dumps(out))
    return 0 if out["ok"] else 1


if __name__ == "__main__":
    sys.exit(main())
