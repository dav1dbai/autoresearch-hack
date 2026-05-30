"""Run all KernelBench baselines on the current GPU and save results."""
import importlib.util
import json
import os
import sys
import time

import torch


def load_module(path, name="task"):
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def bench_task(task_path, idx, num_correct=3, num_perf=10, warmup=3):
    device = torch.device("cuda")
    mod = load_module(task_path, f"task_{idx}")
    model = mod.Model(*mod.get_init_inputs()).to(device).eval()

    # Allocate inputs once — get_inputs() is the bottleneck (huge tensors)
    inputs = [x.to(device) if isinstance(x, torch.Tensor) else x for x in mod.get_inputs()]

    for i in range(num_correct):
        with torch.no_grad():
            out = model(*inputs)
        if torch.isnan(out).any():
            return {"error": f"NaN in trial {i}"}

    for _ in range(warmup):
        with torch.no_grad():
            model(*inputs)
    torch.cuda.synchronize()

    times = []
    for _ in range(num_perf):
        s = torch.cuda.Event(enable_timing=True)
        e = torch.cuda.Event(enable_timing=True)
        torch.cuda.synchronize()
        s.record()
        with torch.no_grad():
            model(*inputs)
        e.record()
        torch.cuda.synchronize()
        times.append(s.elapsed_time(e))

    times.sort()
    return {
        "median_ms": round(times[len(times) // 2], 4),
        "mean_ms": round(sum(times) / len(times), 4),
        "min_ms": round(times[0], 4),
        "max_ms": round(times[-1], 4),
    }


if __name__ == "__main__":
    level = sys.argv[1] if len(sys.argv) > 1 else "level1"
    task_dir = f"/root/KernelBench/KernelBench/{level}"
    tasks = sorted(f for f in os.listdir(task_dir) if f.endswith(".py"))

    results = {}
    t0 = time.time()
    for i, tf in enumerate(tasks):
        path = os.path.join(task_dir, tf)
        name = tf.replace(".py", "")
        try:
            r = bench_task(path, i)
            results[name] = r
            status = f'{r.get("median_ms", "?")} ms' if "median_ms" in r else r.get("error", "?")
        except Exception as ex:
            results[name] = {"error": str(ex)[:200]}
            status = f"ERROR: {str(ex)[:80]}"
        print(f"[{i+1}/{len(tasks)}] {name}: {status}")
        torch.cuda.empty_cache()

    elapsed = time.time() - t0
    gpu_name = torch.cuda.get_device_name(0).replace(" ", "_")
    out_path = f"baselines_{gpu_name}_{level}.json"
    with open(out_path, "w") as f:
        json.dump(
            {"gpu": torch.cuda.get_device_name(0), "torch": torch.__version__, "results": results},
            f,
            indent=2,
        )

    ok = [v for v in results.values() if "median_ms" in v]
    err = [v for v in results.values() if "error" in v]
    print(f"\nDone in {elapsed:.1f}s. Passed: {len(ok)}/{len(results)}, Errors: {len(err)}")
    if ok:
        medians = [v["median_ms"] for v in ok]
        print(f"Median range: {min(medians):.3f} - {max(medians):.3f} ms")
    print(f"Saved to {out_path}")
