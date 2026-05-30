"""Modal GPU backend — ships kernel source to a Modal GPU container."""
from __future__ import annotations

import importlib.util
import os
import tempfile
import time
from pathlib import Path

import modal

from infra.modal.images import app, matmul_gpu_image


def _modal_gpu_eval(kernel_src: str, problem: dict) -> dict:
    """Benchmark kernel source on GPU (runs inside _modal_gpu_run container)."""
    import numpy as np

    with tempfile.NamedTemporaryFile(suffix=".py", mode="w", delete=False) as f:
        f.write(kernel_src)
        kpath = f.name

    try:
        M = problem.get("M", 128)
        N = problem.get("N", 128)
        K = problem.get("K", 128)
        reps = problem.get("reps", 20)

        spec = importlib.util.spec_from_file_location("_kernel", kpath)
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
            return {"gflops": 0.0, "correct": False, "seconds": 0.0, "error": str(e)}

        if not correct:
            return {"gflops": 0.0, "correct": False, "seconds": 0.0}

        times = []
        for _ in range(reps):
            t0 = time.perf_counter()
            mod.matmul(A.copy(), B.copy())
            times.append(time.perf_counter() - t0)
        times.sort()
        seconds = times[len(times) // 2]
        gflops = 2 * M * N * K / seconds / 1e9
        return {"gflops": gflops, "correct": True, "seconds": seconds}
    finally:
        import os as _os
        _os.unlink(kpath)


@app.function(
    image=matmul_gpu_image,
    gpu="A10G",
    timeout=300,
    max_containers=8,
)
def _modal_gpu_run(kernel_src: str, problem: dict) -> dict:
    """Run the kernel benchmark inside a Modal GPU container."""
    return _modal_gpu_eval(kernel_src, problem)


def _invoke_modal_gpu_run(kernel_src: str, problem: dict) -> dict:
    """Call _modal_gpu_run from host or from inside another Modal container."""
    on_modal = os.environ.get("MODAL_ENVIRONMENT") is not None
    if not on_modal:
        from harness.cloud.session import deployed_enabled, ensure_app_deployed

        if deployed_enabled():
            ensure_app_deployed(app)
            fn = modal.Function.from_name(app.name, "_modal_gpu_run")
            return fn.remote(kernel_src, problem)
    return _modal_gpu_run.remote(kernel_src, problem)


class ModalGPUBackend:
    """Ships kernel source to a Modal GPU container for benchmarking."""

    def run(self, kernel_path: Path, problem: dict) -> dict:
        if os.environ.get("MODAL_ENVIRONMENT") is None:
            from infra.modal.images import assert_hackathon_profile
            assert_hackathon_profile()
        kernel_src = kernel_path.read_text()
        try:
            return _invoke_modal_gpu_run(kernel_src, problem)
        except Exception as e:
            return {"gflops": 0.0, "correct": False, "seconds": 0.0, "error": str(e)}
