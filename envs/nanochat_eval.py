"""Harness-controlled nanochat evaluator (D-06) — not agent stdout."""
from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path

_AR_PREPARE = Path(__file__).resolve().parent.parent / "ar" / "prepare.py"


def evaluate_workdir(workdir: Path) -> float:
    """Parse val_bpb from run.log via harness — ignores bare stdout injection."""
    log_path = workdir / "run.log"
    if not log_path.exists():
        raise FileNotFoundError(f"missing {log_path}")
    log_text = log_path.read_text()
    m = re.search(r"^val_bpb:\s*([\d.]+)", log_text, re.MULTILINE)
    if not m:
        raise ValueError(f"val_bpb not found in {log_path}")
    return float(m.group(1))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--workdir", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    val_bpb = evaluate_workdir(args.workdir)
    args.out.write_text(json.dumps({"val_bpb": val_bpb}), encoding="utf-8")


if __name__ == "__main__":
    main()
