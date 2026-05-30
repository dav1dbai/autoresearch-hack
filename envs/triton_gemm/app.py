"""Persistent Modal app for the triton-gemm kernel sidequest.

DEPLOY ONCE (no ephemeral litter):
    cd ~/Desktop/autoresearch-hack
    export MODAL_PROFILE=... MODAL_TOKEN_ID=... MODAL_TOKEN_SECRET=...   # from .env
    .venv/bin/modal deploy sidequest/triton_gemm/app.py

Then invoke the DEPLOYED function via modal.Function.from_name(
"ar2-triton", "triton_run").remote(kernel_src, problem).  No top-level dotenv
import — the container image has no python-dotenv, and auth is a host-side concern
at deploy/invoke time.

This is the GPU-eval backend the triton-gemm env will call:
triton_run(kernel_src, problem) -> {gflops, correct, seconds}.
"""
from __future__ import annotations

import modal

app = modal.App("ar2-triton")

image = (
    modal.Image.debian_slim(python_version="3.12")
    .pip_install("torch", "triton", "numpy",
                 extra_options="--extra-index-url https://download.pytorch.org/whl/cu124")
)


@app.function(image=image, gpu="H100", timeout=900)
def triton_run(kernel_src: str, problem: dict | None = None) -> dict:
    """Import the agent's Triton kernel module, run it on random fp16 tensors,
    and return {gflops, correct, seconds}. This is the fixed referee — the agent
    supplies ONLY the kernel; setup/timing/correctness live here and are never
    given to the agent."""
    import importlib.util
    import statistics
    import tempfile
    from pathlib import Path

    import torch

    problem = problem or {}
    M = int(problem.get("M", 1024))
    N = int(problem.get("N", 1024))
    K = int(problem.get("K", 1024))

    # Static check: the kernel file must contain @triton.jit (not just torch.matmul)
    if "@triton.jit" not in kernel_src:
        return {"gflops": 0.0, "correct": False, "seconds": 0.0,
                "error": "kernel must define an @triton.jit kernel"}

    # Write the agent's module to a temp file and import it
    try:
        with tempfile.NamedTemporaryFile(suffix=".py", delete=False, mode="w") as f:
            f.write(kernel_src)
            tmp_path = f.name
        spec = importlib.util.spec_from_file_location("agent_kernel", tmp_path)
        mod = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(mod)
        matmul_fn = mod.matmul
    except Exception as e:
        return {"gflops": 0.0, "correct": False, "seconds": 0.0,
                "error": f"import error: {e}"}

    # Build random fp16 inputs on GPU
    torch.manual_seed(42)
    A = torch.randn(M, K, dtype=torch.float16, device="cuda")
    B = torch.randn(K, N, dtype=torch.float16, device="cuda")

    # Correctness: compare against torch.matmul (fp16 reference)
    try:
        ref = torch.matmul(A, B)
        out = matmul_fn(A, B)
        correct = bool(torch.allclose(out.float(), ref.float(), atol=1e-1, rtol=1e-2))
    except Exception as e:
        return {"gflops": 0.0, "correct": False, "seconds": 0.0,
                "error": f"correctness check failed: {e}"}

    if not correct:
        return {"gflops": 0.0, "correct": False, "seconds": 0.0}

    # Warmup
    for _ in range(5):
        _ = matmul_fn(A, B)
    torch.cuda.synchronize()

    # Timed runs with CUDA events
    times: list[float] = []
    for _ in range(20):
        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)
        start.record()
        matmul_fn(A, B)
        end.record()
        torch.cuda.synchronize()
        times.append(start.elapsed_time(end) / 1000.0)  # seconds

    median_s = statistics.median(times)
    gflops = 2.0 * M * N * K / median_s / 1e9

    return {"gflops": gflops, "correct": True, "seconds": median_s}
