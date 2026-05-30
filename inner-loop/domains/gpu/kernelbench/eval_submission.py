"""Evaluate a KernelBench submission against a reference task.

Usage:
    uv run python eval_submission.py <task_path> <submission_path> [--atol 1e-3] [--rtol 1e-3]

The submission must define a `ModelNew` class that is a drop-in replacement
for the reference `Model`. Scores: correctness (allclose over randomized inputs)
and speedup (median wall-clock via CUDA events).
"""
import argparse
import importlib.util
import json
import sys
import torch


def load_module(path: str, name: str = "mod"):
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def eval_submission(
    task_path: str,
    submission_path: str,
    num_correct: int = 5,
    num_perf: int = 20,
    warmup: int = 5,
    atol: float = 1e-3,
    rtol: float = 1e-3,
):
    device = torch.device("cuda")
    task = load_module(task_path, "task")
    sub = load_module(submission_path, "submission")

    ref_model = task.Model(*task.get_init_inputs()).to(device).eval()
    new_model = sub.ModelNew(*task.get_init_inputs()).to(device).eval()

    # --- Correctness: run submission FIRST, then reference (Kevin-32B bug) ---
    print(f"Correctness ({num_correct} trials, atol={atol}, rtol={rtol}):")
    for i in range(num_correct):
        inputs = [x.to(device) if isinstance(x, torch.Tensor) else x for x in task.get_inputs()]
        with torch.no_grad():
            out_new = new_model(*inputs)
            out_ref = ref_model(*inputs)
        ok = torch.allclose(out_new, out_ref, atol=atol, rtol=rtol)
        if not ok:
            max_diff = (out_new - out_ref).abs().max().item()
            print(f"  Trial {i}: FAIL (max_diff={max_diff:.6e})")
            return {"correct": False, "fail_trial": i, "max_diff": max_diff}
        print(f"  Trial {i}: PASS")

    # --- Performance ---
    inputs = [x.to(device) if isinstance(x, torch.Tensor) else x for x in task.get_inputs()]

    def time_model(model, n, w):
        for _ in range(w):
            with torch.no_grad():
                model(*inputs)
        torch.cuda.synchronize()
        times = []
        for _ in range(n):
            torch.cuda.synchronize()
            s = torch.cuda.Event(enable_timing=True)
            e = torch.cuda.Event(enable_timing=True)
            s.record()
            with torch.no_grad():
                model(*inputs)
            e.record()
            torch.cuda.synchronize()
            times.append(s.elapsed_time(e))
        times.sort()
        return times[len(times) // 2]

    ref_ms = time_model(ref_model, num_perf, warmup)
    new_ms = time_model(new_model, num_perf, warmup)
    speedup = ref_ms / new_ms if new_ms > 0 else 0.0

    print(f"\nPerformance ({num_perf} trials):")
    print(f"  Reference: {ref_ms:.3f} ms")
    print(f"  Submission: {new_ms:.3f} ms")
    print(f"  Speedup: {speedup:.2f}x")

    return {
        "correct": True,
        "ref_ms": round(ref_ms, 4),
        "new_ms": round(new_ms, 4),
        "speedup": round(speedup, 4),
    }


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("task", help="Path to KernelBench task .py file")
    parser.add_argument("submission", help="Path to submission .py file defining ModelNew")
    parser.add_argument("--atol", type=float, default=1e-3)
    parser.add_argument("--rtol", type=float, default=1e-3)
    parser.add_argument("--trials", type=int, default=20)
    args = parser.parse_args()

    result = eval_submission(args.task, args.submission, atol=args.atol, rtol=args.rtol, num_perf=args.trials)
    print(f"\n{json.dumps(result, indent=2)}")
