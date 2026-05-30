"""Default train / held-out env pools for the outer loop CLI."""
from __future__ import annotations

import os

from envs.matmul import MatmulEnv
from harness.contracts import Env

# Shapes differ by split so train≫heldout gap is meaningful (DECISIONS env pools).
_TRAIN_SHAPES = ((64, 64, 64), (128, 128, 128))
_HELDOUT_SHAPE = (96, 96, 96)


def default_matmul_pools(
    *,
    stub: bool | None = None,
    target_gflops: float | None = None,
) -> tuple[list[Env], list[Env]]:
    """Return (train_envs, heldout_envs) for matmul CPU pipeline smoke + demo."""
    if stub is None:
        stub = os.environ.get("MATMUL_STUB", "0") == "1"
    if stub:
        os.environ["MATMUL_STUB"] = "1"
    if target_gflops is None:
        target_gflops = float(os.environ.get("MATMUL_TARGET_GFLOPS", "5.0"))

    train = [
        MatmulEnv(split="train", M=m, N=n, K=k, target_gflops=target_gflops)
        for m, n, k in _TRAIN_SHAPES
    ]
    m, n, k = _HELDOUT_SHAPE
    heldout = [
        MatmulEnv(split="heldout", M=m, N=n, K=k, target_gflops=target_gflops)
    ]
    return train, heldout


# GPU / Vast pools — larger shapes, realistic GFLOP targets (not CPU-saturated).
_GPU_TRAIN_SHAPES = ((1024, 1024, 1024), (2048, 512, 2048))
_GPU_HELDOUT_SHAPE = (1536, 1536, 1536)
_GPU_SMOKE_TRAIN_SHAPES = ((256, 256, 256), (512, 256, 512))
_GPU_SMOKE_HELDOUT_SHAPE = (384, 384, 384)


def gpu_matmul_pools(
    *,
    target_gflops: float | None = None,
    smoke: bool | None = None,
    runner=None,
) -> tuple[list[Env], list[Env]]:
    """Matmul envs scored on AR2_GPU_BACKEND (vast/modal), not local CPU.

    ``runner`` must be supplied by the CLI (harness/__main__.py) — env package
    does not wire GPU backends directly.
    """
    if runner is None:
        raise ValueError("gpu_matmul_pools requires runner= from harness CLI")
    os.environ.setdefault("MATMUL_RUNNER", "gpu")
    os.environ.setdefault("MATMUL_GPU_STARTER", "1")
    if smoke is None:
        smoke = os.environ.get("AR2_GPU_SMOKE", "0") == "1"
    if smoke:
        train_shapes = _GPU_SMOKE_TRAIN_SHAPES
        heldout_shape = _GPU_SMOKE_HELDOUT_SHAPE
        reps = 3
        default_target = 50.0
    else:
        train_shapes = _GPU_TRAIN_SHAPES
        heldout_shape = _GPU_HELDOUT_SHAPE
        reps = 10
        default_target = 400.0
    if target_gflops is None:
        target_gflops = float(os.environ.get("MATMUL_TARGET_GFLOPS", str(default_target)))

    train = [
        MatmulEnv(
            split="train", M=m, N=n, K=k,
            target_gflops=target_gflops, reps=reps, runner=runner,
        )
        for m, n, k in train_shapes
    ]
    m, n, k = heldout_shape
    heldout = [
        MatmulEnv(
            split="heldout", M=m, N=n, K=k,
            target_gflops=target_gflops, reps=reps, runner=runner,
        )
    ]
    return train, heldout
