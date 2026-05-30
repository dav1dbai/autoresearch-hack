"""Normalize a KernelBench eval into a [0,1] reward.

Wraps the existing deterministic eval (correctness gate via allclose, then
median CUDA-event timing) and maps it onto a bounded reward so heterogeneous
tasks can be averaged. Incorrect kernels get exactly 0.0 — correctness is a
hard gate, never partially rewarded.

Usage:
    python score_kernel.py <task_path> <submission_path> [--cap 3.0]
"""

from __future__ import annotations

import argparse
import json

from eval_submission import eval_submission


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("task")
    parser.add_argument("submission")
    parser.add_argument("--cap", type=float, default=3.0, help="speedup at which reward saturates to 1.0")
    parser.add_argument("--atol", type=float, default=1e-3)
    parser.add_argument("--rtol", type=float, default=1e-3)
    args = parser.parse_args()

    try:
        result = eval_submission(args.task, args.submission, atol=args.atol, rtol=args.rtol)
    except Exception as exc:  # a kernel that fails to import/compile scores 0
        print(json.dumps({"reward": 0.0, "correct": False, "error": str(exc)[-2000:]}))
        return

    if not result.get("correct"):
        result["reward"] = 0.0
    else:
        speedup = float(result.get("speedup", 0.0))
        result["reward"] = round(min(max(speedup, 0.0), args.cap) / args.cap, 4)
    print(json.dumps(result))


if __name__ == "__main__":
    main()
