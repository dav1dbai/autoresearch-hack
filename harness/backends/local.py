"""Local CPU backend for kernel benchmarks."""
from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

_EVAL_SCRIPT = r"""
import sys, json, time, importlib.util
import numpy as np

kernel_path, M, N, K, reps = sys.argv[1], int(sys.argv[2]), int(sys.argv[3]), int(sys.argv[4]), int(sys.argv[5])

spec = importlib.util.spec_from_file_location("_kernel", kernel_path)
mod = importlib.util.module_from_spec(spec)
spec.loader.exec_module(mod)

rng = np.random.default_rng(42)
A = rng.random((M, K))
B = rng.random((K, N))
ref = A @ B

try:
    out = mod.matmul(A.copy(), B.copy())
    correct = bool(np.allclose(out, ref, atol=1e-6, rtol=1e-5))
except Exception as e:
    print(json.dumps({"gflops": 0.0, "correct": False, "seconds": 0.0, "error": str(e)}))
    sys.exit(0)

if not correct:
    print(json.dumps({"gflops": 0.0, "correct": False, "seconds": 0.0}))
    sys.exit(0)

times = []
for _ in range(reps):
    t0 = time.perf_counter()
    mod.matmul(A.copy(), B.copy())
    times.append(time.perf_counter() - t0)
times.sort()
seconds = times[len(times) // 2]
gflops = 2 * M * N * K / seconds / 1e9
print(json.dumps({"gflops": gflops, "correct": True, "seconds": seconds}))
"""


class LocalBackend:
    """Runs the benchmark in a subprocess on the current machine (CPU)."""

    def run(self, kernel_path: Path, problem: dict) -> dict:
        M = problem.get("M", 128)
        N = problem.get("N", 128)
        K = problem.get("K", 128)
        reps = problem.get("reps", 20)
        result = subprocess.run(
            [sys.executable, "-c", _EVAL_SCRIPT,
             str(kernel_path), str(M), str(N), str(K), str(reps)],
            capture_output=True,
            text=True,
            timeout=120,
        )
        if result.returncode != 0 or not result.stdout.strip():
            return {"gflops": 0.0, "correct": False, "seconds": 0.0,
                    "error": result.stderr[:500]}
        return json.loads(result.stdout.strip())
