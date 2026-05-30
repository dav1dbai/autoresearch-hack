"""Autoresearch inner loop for KernelBench: iteratively generate and evaluate
Triton kernel submissions for a given task.

Karpathy-style keep/discard loop:
  1. Pick a task
  2. Generate a candidate Triton kernel (via Anthropic API)
  3. Evaluate correctness + speedup
  4. If correct and faster, keep as new best
  5. Repeat

Usage:
    uv run python autoresearch_loop.py <task_path> [--iters 10] [--budget-s 1200]

Requires ANTHROPIC_API_KEY env var.
"""
from __future__ import annotations

import argparse
import importlib.util
import json
import os
import re
import sys
import time
from pathlib import Path

import anthropic
import torch


def load_module(path: str, name: str = "mod"):
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def eval_candidate(task_path: str, candidate_path: str, atol=1e-3, rtol=1e-3,
                   num_correct=5, num_perf=10, warmup=3) -> dict:
    device = torch.device("cuda")
    task = load_module(task_path, "task")
    cand = load_module(candidate_path, "candidate")

    ref_model = task.Model(*task.get_init_inputs()).to(device).eval()
    try:
        new_model = cand.ModelNew(*task.get_init_inputs()).to(device).eval()
    except Exception as e:
        return {"correct": False, "error": f"init: {e}"}

    for i in range(num_correct):
        inputs = [x.to(device) if isinstance(x, torch.Tensor) else x for x in task.get_inputs()]
        try:
            with torch.no_grad():
                out_new = new_model(*inputs)
                out_ref = ref_model(*inputs)
        except Exception as e:
            return {"correct": False, "error": f"fwd trial {i}: {e}"}
        if not torch.allclose(out_new, out_ref, atol=atol, rtol=rtol):
            diff = (out_new - out_ref).abs().max().item()
            return {"correct": False, "error": f"trial {i} max_diff={diff:.6e}"}

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

    del ref_model, new_model
    torch.cuda.empty_cache()

    return {
        "correct": True,
        "ref_ms": round(ref_ms, 4),
        "new_ms": round(new_ms, 4),
        "speedup": round(ref_ms / new_ms, 4) if new_ms > 0 else 0.0,
    }


GENERATE_PROMPT = """You are optimizing a GPU kernel. Below is a PyTorch reference module.

Write a drop-in replacement `ModelNew` class that produces numerically equivalent
outputs and runs as fast as possible. Use Triton kernels where beneficial.

Reference code:
```python
{reference_code}
```

{history_section}

Output ONLY a complete Python file defining `ModelNew`. Include all necessary imports.
Do not include any explanation or markdown fences — just the raw Python code."""


def generate_candidate(task_path: str, history: list[dict], work_dir: Path,
                       iteration: int, client: anthropic.Anthropic,
                       model: str = "claude-sonnet-4-20250514") -> Path:
    ref_code = Path(task_path).read_text()

    history_lines = []
    for h in history[-5:]:
        if h["correct"]:
            history_lines.append(f"- iter {h['iter']}: speedup={h['speedup']:.2f}x (kept={h['kept']})")
        else:
            history_lines.append(f"- iter {h['iter']}: FAILED ({h.get('error', '?')[:80]})")

    history_section = ""
    if history_lines:
        history_section = "Previous attempts:\n" + "\n".join(history_lines)
        best = [h for h in history if h.get("kept")]
        if best:
            best_path = best[-1].get("path")
            if best_path and Path(best_path).exists():
                history_section += f"\n\nCurrent best submission ({best[-1]['speedup']:.2f}x):\n```python\n{Path(best_path).read_text()}\n```"

    prompt = GENERATE_PROMPT.format(reference_code=ref_code, history_section=history_section)

    response = client.messages.create(
        model=model,
        max_tokens=8192,
        messages=[{"role": "user", "content": prompt}],
    )

    code = response.content[0].text.strip()
    # Strip markdown fences if present
    if code.startswith("```"):
        lines = code.split("\n")
        code = "\n".join(lines[1:-1] if lines[-1].strip() == "```" else lines[1:])
    # Also try to extract from ```python ... ``` blocks
    m = re.search(r"```python\n(.*?)```", code, re.DOTALL)
    if m:
        code = m.group(1).strip()

    out_path = work_dir / f"candidate_{iteration}.py"
    out_path.write_text(code)
    return out_path


def autoresearch_loop(task_path: str, max_iters: int = 10, budget_s: int = 1200,
                      model: str = "claude-sonnet-4-20250514"):
    client = anthropic.Anthropic()  # uses ANTHROPIC_API_KEY env var

    work_dir = Path("runs") / Path(task_path).stem / time.strftime("%Y%m%d_%H%M%S")
    work_dir.mkdir(parents=True, exist_ok=True)

    log_path = work_dir / "results.jsonl"
    history: list[dict] = []
    best_speedup = 1.0
    best_path = None
    t0 = time.time()

    print(f"Task: {task_path}")
    print(f"Work dir: {work_dir}")
    print(f"Model: {model}")
    print(f"Budget: {max_iters} iters / {budget_s}s wall clock")
    print()

    for i in range(max_iters):
        if time.time() - t0 > budget_s:
            print(f"Budget exhausted after {i} iterations")
            break

        print(f"--- Iteration {i} ---")

        try:
            cand_path = generate_candidate(task_path, history, work_dir, i, client, model)
            print(f"  Generated: {cand_path}")
        except Exception as e:
            entry = {"iter": i, "correct": False, "error": f"gen: {e}", "kept": False}
            history.append(entry)
            print(f"  Generation failed: {e}")
            continue

        result = eval_candidate(task_path, str(cand_path))
        kept = result.get("correct", False) and result.get("speedup", 0) > best_speedup

        entry = {"iter": i, **result, "kept": kept, "path": str(cand_path)}
        history.append(entry)

        if kept:
            best_speedup = result["speedup"]
            best_path = str(cand_path)
            print(f"  NEW BEST: {best_speedup:.2f}x speedup")
        elif result.get("correct"):
            print(f"  Correct but {result['speedup']:.2f}x (best: {best_speedup:.2f}x)")
        else:
            print(f"  Failed: {result.get('error', '?')[:80]}")

        with open(log_path, "a") as f:
            f.write(json.dumps(entry) + "\n")

    elapsed = time.time() - t0
    print(f"\nDone: {len(history)} iters in {elapsed:.0f}s")
    print(f"Best speedup: {best_speedup:.2f}x")
    if best_path:
        print(f"Best submission: {best_path}")

    summary = {
        "task": task_path,
        "best_speedup": best_speedup,
        "best_path": best_path,
        "total_iters": len(history),
        "correct_iters": sum(1 for h in history if h.get("correct")),
        "elapsed_s": round(elapsed, 1),
    }
    (work_dir / "summary.json").write_text(json.dumps(summary, indent=2))
    return summary


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("task", help="Path to KernelBench task")
    parser.add_argument("--iters", type=int, default=10)
    parser.add_argument("--budget-s", type=int, default=1200)
    parser.add_argument("--model", default="claude-sonnet-4-20250514")
    args = parser.parse_args()

    result = autoresearch_loop(args.task, args.iters, args.budget_s, args.model)
    print(json.dumps(result, indent=2))
