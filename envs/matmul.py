"""Active inner task (matmul kernel). Same Env contract as nanochat; swap shapes/target for harder variants.

Editable surface: kernel.py (copied into agent workdir on reset).
Evaluator: NOT in workdir — runs out-of-process (subprocess), integrity-clean.

Reward normalization:
  - WRONG kernel (np.allclose fails, any exception) → reward = 0.0
  - CORRECT kernel → reward = clip(baseline_gflops / measured_seconds / target_gflops, 0, 1)
    where gflops = 2*M*N*K / measured_seconds / 1e9, normalized to [0, 1] by target_gflops.
    baseline (slow Python triple-loop starter) earns ~0.01–0.2; np.dot can reach 1.0.

STUB mode (MATMUL_STUB=1): skip real timing; return a fixed STUB_SECONDS so tests are
fast and deterministic.

GPU hook: to run on Vast.ai / Modal, set:
  MATMUL_RUNNER=gpu  (or inject a custom runner via MatmulEnv(runner=...))
  The runner callable has signature:
      runner(kernel_path: Path, M: int, N: int, K: int, reps: int) -> float
  and must return median wall-time in seconds.  Default runner is _cpu_runner (numpy).
"""
from __future__ import annotations

import os
import shutil
import subprocess
import sys
import tempfile
import textwrap
from pathlib import Path
from typing import Callable, Literal

import numpy as np

from envs.base import BaseEnv
from harness.contracts import Split, Submission, StepResult, TaskSpec

# ---------------------------------------------------------------------------
# Constants / env-var toggles
# ---------------------------------------------------------------------------

_STUB_MODE: bool = os.environ.get("MATMUL_STUB", "0") == "1"
_STUB_SECONDS: float = float(os.environ.get("MATMUL_STUB_SECONDS", "0.01"))

# Default matrix shape and benchmark parameters
_DEFAULT_M: int = int(os.environ.get("MATMUL_M", "128"))
_DEFAULT_N: int = int(os.environ.get("MATMUL_N", "128"))
_DEFAULT_K: int = int(os.environ.get("MATMUL_K", "128"))
_DEFAULT_REPS: int = int(os.environ.get("MATMUL_REPS", "20"))

# target_gflops: kernel achieving this or better earns reward >= 1.0.
# For small CPU matrices, naive numpy easily exceeds 1 GFLOP/s, so we set a
# generous target.  Scale up for GPU variants.
_DEFAULT_TARGET_GFLOPS: float = float(os.environ.get("MATMUL_TARGET_GFLOPS", "5.0"))

# ---------------------------------------------------------------------------
# Default starter kernel (editable surface — copied into workdir)
# ---------------------------------------------------------------------------

_STARTER_KERNEL = textwrap.dedent("""\
    \"\"\"matmul kernel — edit this file to maximize speed while keeping correctness.

    Contract: matmul(A, B) must return a 2-D array numerically close to A @ B.
    Benchmark harness calls matmul repeatedly; median wall-time is scored.

    Starter is a pure-Python triple loop (deliberately slow) so the inner curve
    has headroom to climb naive → vectorized → np.dot. See proof/DECISIONS.md path (A).
    \"\"\"
    import numpy as np


    def matmul(A, B):
        M, K = A.shape
        _, N = B.shape
        C = np.zeros((M, N), dtype=A.dtype)
        for i in range(M):
            for j in range(N):
                s = 0.0
                for k in range(K):
                    s += A[i, k] * B[k, j]
                C[i, j] = s
        return C
""")

_GPU_STARTER_KERNEL = textwrap.dedent("""\
    \"\"\"GPU matmul kernel — optimize for CUDA throughput.

    Contract: matmul(A, B) must return a numpy array close to A @ B.
    Inputs are float64 numpy arrays; you may use torch/cuda inside.
    Starter uses a slow CUDA elementwise loop — headroom to optimize.
    \"\"\"
    import numpy as np
    import torch


    def matmul(A, B):
        ta = torch.from_numpy(np.asarray(A, dtype=np.float64)).cuda()
        tb = torch.from_numpy(np.asarray(B, dtype=np.float64)).cuda()
        m, k = ta.shape
        n = tb.shape[1]
        out = torch.zeros(m, n, device=ta.device, dtype=ta.dtype)
        for i in range(m):
            for j in range(n):
                out[i, j] = (ta[i, :] * tb[:, j]).sum()
        return out.cpu().numpy()
""")

# ---------------------------------------------------------------------------
# Out-of-process benchmark runner (the evaluator — NEVER placed in workdir)
# ---------------------------------------------------------------------------

# Evaluation script executed as a subprocess.  It imports the submitted kernel,
# checks correctness, then times it.  stdout = JSON line: {"gflops": ..., "correct": ...,"seconds": ...}
_EVAL_SCRIPT = textwrap.dedent("""\
    import sys, json, time, importlib.util, pathlib
    import numpy as np

    kernel_path = pathlib.Path(sys.argv[1])
    M, N, K, reps = int(sys.argv[2]), int(sys.argv[3]), int(sys.argv[4]), int(sys.argv[5])
    stub = sys.argv[6] == "1"
    stub_seconds = float(sys.argv[7])

    spec = importlib.util.spec_from_file_location("_kernel", kernel_path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)

    rng = np.random.default_rng(42)
    A = rng.random((M, K), dtype=np.float64)
    B = rng.random((K, N), dtype=np.float64)
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

    if stub:
        seconds = stub_seconds
    else:
        times = []
        for _ in range(reps):
            t0 = time.perf_counter()
            mod.matmul(A.copy(), B.copy())
            times.append(time.perf_counter() - t0)
        times.sort()
        seconds = times[len(times) // 2]  # median

    flops = 2 * M * N * K
    gflops = flops / seconds / 1e9
    print(json.dumps({"gflops": gflops, "correct": True, "seconds": seconds}))
""")


def _cpu_runner(kernel_path: Path, M: int, N: int, K: int, reps: int) -> dict:
    """Run _EVAL_SCRIPT out-of-process; return parsed dict."""
    stub_flag = "1" if _STUB_MODE else "0"
    result = subprocess.run(
        [
            sys.executable, "-c", _EVAL_SCRIPT,
            str(kernel_path), str(M), str(N), str(K), str(reps),
            stub_flag, str(_STUB_SECONDS),
        ],
        capture_output=True,
        text=True,
        timeout=120,
    )
    if result.returncode != 0 or not result.stdout.strip():
        return {"gflops": 0.0, "correct": False, "seconds": 0.0,
                "error": result.stderr[:500]}
    import json
    return json.loads(result.stdout.strip())


def _default_runner() -> Runner:
    return _cpu_runner


# ---------------------------------------------------------------------------
# Env
# ---------------------------------------------------------------------------

Runner = Callable[[Path, int, int, int, int], dict]


class MatmulEnv(BaseEnv):
    """Env that tasks the agent with optimizing a matmul kernel.

    reset()  copies starter kernel.py into an isolated tmpdir; agent edits only that dir.
    score()  evaluates kernel.py OUT-OF-PROCESS: correctness check + wall-time benchmark.
             Reward = 0.0 if wrong; else clip(measured_gflops / target_gflops, 0, 1).

    The evaluator script (_EVAL_SCRIPT) is injected at call time and is never written
    to the agent workdir — integrity boundary maintained.
    """

    id: str
    split: Split

    def __init__(
        self,
        split: Split = "train",
        M: int = _DEFAULT_M,
        N: int = _DEFAULT_N,
        K: int = _DEFAULT_K,
        reps: int = _DEFAULT_REPS,
        target_gflops: float = _DEFAULT_TARGET_GFLOPS,
        runner: Runner | None = None,
    ) -> None:
        self.id = f"matmul-{split}-{M}x{N}x{K}"
        self.split = split
        self.M = M
        self.N = N
        self.K = K
        self.reps = reps
        self.target_gflops = target_gflops
        self._runner: Runner = runner or _default_runner()
        self._use_gpu_starter = os.environ.get("MATMUL_GPU_STARTER", "0") == "1"
        self._workdir: Path | None = None

    # ------------------------------------------------------------------
    # Env protocol
    # ------------------------------------------------------------------

    def reset(self) -> TaskSpec:
        """Copy starter kernel.py into a fresh tmpdir (the editable surface only)."""
        tmp = Path(tempfile.mkdtemp(prefix="matmul_"))
        kernel_dst = tmp / "kernel.py"
        starter = _GPU_STARTER_KERNEL if self._use_gpu_starter else _STARTER_KERNEL
        kernel_dst.write_text(starter)
        self._workdir = tmp
        return TaskSpec(
            env_id=self.id,
            split=self.split,
            prompt=(
                "Optimize this matmul kernel; it must stay numerically correct. "
                "Edit kernel.py to maximize throughput (GFLOP/s). "
                "The function signature must remain: def matmul(A, B) -> array."
            ),
            workdir=tmp,
            payload={"M": self.M, "N": self.N, "K": self.K, "editable_file": "kernel.py"},
        )

    def score(self, sub: Submission) -> StepResult:
        """Evaluate submitted kernel.py out-of-process; normalize reward to [0, 1]."""
        kernel_path = sub.workdir / "kernel.py"
        if not kernel_path.exists():
            return StepResult(
                reward=0.0,
                raw={"correct": False, "error": "kernel.py not found"},
                feedback="kernel.py missing",
                done=False,  # recoverable: agent can recreate kernel.py next iter (D-00)
            )

        metrics = self._runner(kernel_path, self.M, self.N, self.K, self.reps)
        correct: bool = bool(metrics.get("correct", False))
        gflops: float = float(metrics.get("gflops", 0.0))
        seconds: float = float(metrics.get("seconds", 0.0))

        if not correct:
            reward = 0.0
        else:
            reward = float(np.clip(gflops / self.target_gflops, 0.0, 1.0))

        raw = {"gflops": gflops, "correct": correct, "seconds": seconds}
        if "error" in metrics:
            raw["error"] = metrics["error"]

        return StepResult(
            reward=reward,
            raw=raw,
            feedback=(
                f"correct={correct}  gflops={gflops:.3f}  "
                f"target={self.target_gflops:.1f}  reward={reward:.4f}"
            ),
            done=False,  # no natural terminal; let solve() keep iterating (D-00)
        )
