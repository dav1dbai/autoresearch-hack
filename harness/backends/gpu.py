"""AR² GPU backend — pluggable compute backend for kernel benchmarks.

Protocol
────────
GPUBackend.run(kernel_path, problem) -> dict
  kernel_path: Path to the kernel file to benchmark.
  problem:     dict with at least {"M": int, "N": int, "K": int, "reps": int}.
  Returns:     {"gflops": float, "correct": bool, "seconds": float, ...}.

Selection: AR2_GPU_BACKEND ∈ {local (default), modal, vast}.

Implementations live in local.py, modal_gpu.py, vast.py. This module exposes
the protocol, factory, and matmul_runner() wiring only.
"""
from __future__ import annotations

import os
from pathlib import Path
from typing import Protocol, runtime_checkable

from harness.backends.local import LocalBackend
from harness.backends.modal_gpu import ModalGPUBackend
from harness.backends.vast import VastBackend

_GPU_BACKEND_ENV = os.environ.get("AR2_GPU_BACKEND", "local")


@runtime_checkable
class GPUBackend(Protocol):
    def run(self, kernel_path: Path, problem: dict) -> dict: ...


def gpu_backend_name() -> str:
    """Read AR2_GPU_BACKEND at call time (Modal injects env after module import)."""
    return os.environ.get("AR2_GPU_BACKEND", _GPU_BACKEND_ENV)


def make_gpu_backend(backend: str | None = None) -> GPUBackend:
    """Return a GPUBackend for the given backend name."""
    name = backend or gpu_backend_name()
    if name == "local":
        return LocalBackend()
    if name == "modal":
        return ModalGPUBackend()
    if name == "vast":
        return VastBackend()
    raise ValueError(f"Unknown AR2_GPU_BACKEND: {name!r}. Choose local, modal, or vast.")


def matmul_runner():
    """Return MatmulEnv(runner=...) callable wired from AR2_GPU_BACKEND."""
    name = gpu_backend_name()
    backend = make_gpu_backend(name)
    inst = os.environ.get("VAST_INSTANCE_ID")
    if inst and name == "vast":
        backend = VastBackend(instance_id=int(inst))

    def _run(kernel_path: Path, M: int, N: int, K: int, reps: int) -> dict:
        return backend.run(kernel_path, {"M": M, "N": N, "K": K, "reps": reps})

    return _run


__all__ = [
    "GPUBackend",
    "LocalBackend",
    "ModalGPUBackend",
    "VastBackend",
    "gpu_backend_name",
    "make_gpu_backend",
    "matmul_runner",
]
