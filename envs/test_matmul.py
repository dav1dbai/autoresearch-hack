"""Unit tests for envs/matmul.py — offline with MATMUL_STUB=1."""
from __future__ import annotations

import os
from pathlib import Path

import numpy as np
import pytest

os.environ.setdefault("MATMUL_STUB", "1")

from envs.matmul import MatmulEnv
from harness.contracts import Submission


def _make_env(split="train", **kw) -> MatmulEnv:
    return MatmulEnv(split=split, M=32, N=32, K=32, **kw)


def _correct_kernel() -> str:
    return (
        "import numpy as np\n"
        "def matmul(A, B):\n"
        "    return np.dot(A, B)\n"
    )


def _wrong_kernel() -> str:
    return (
        "import numpy as np\n"
        "def matmul(A, B):\n"
        "    return np.zeros((A.shape[0], B.shape[1]))\n"
    )


def _broken_kernel() -> str:
    return (
        "def matmul(A, B):\n"
        "    raise RuntimeError('intentional error')\n"
    )


def _write_kernel(workdir: Path, code: str) -> None:
    (workdir / "kernel.py").write_text(code)


class TestMatmulEnv:
    def test_reset_creates_kernel_in_workdir(self):
        env = _make_env()
        task = env.reset()
        assert (task.workdir / "kernel.py").exists()
        assert task.env_id.startswith("matmul-")
        assert task.split == "train"
        assert "kernel.py" in task.payload.get("editable_file", "")

    def test_evaluator_not_in_workdir(self):
        env = _make_env()
        task = env.reset()
        names = [f.name for f in task.workdir.iterdir()]
        assert names == ["kernel.py"], f"Unexpected files in workdir: {names}"

    def test_wrong_kernel_reward_zero(self):
        env = _make_env()
        task = env.reset()
        _write_kernel(task.workdir, _wrong_kernel())
        result = env.score(Submission(workdir=task.workdir))
        assert result.reward == 0.0
        assert result.raw["correct"] is False

    def test_broken_kernel_reward_zero(self):
        env = _make_env()
        task = env.reset()
        _write_kernel(task.workdir, _broken_kernel())
        result = env.score(Submission(workdir=task.workdir))
        assert result.reward == 0.0

    def test_correct_kernel_reward_in_range(self):
        env = _make_env()
        task = env.reset()
        _write_kernel(task.workdir, _correct_kernel())
        result = env.score(Submission(workdir=task.workdir))
        assert result.raw["correct"] is True
        assert 0.0 <= result.reward <= 1.0

    def test_missing_kernel_reward_zero(self):
        env = _make_env()
        task = env.reset()
        (task.workdir / "kernel.py").unlink()
        result = env.score(Submission(workdir=task.workdir))
        assert result.reward == 0.0

    def test_reward_normalization_formula(self):
        M = N = K = 32
        stub_sec = float(os.environ.get("MATMUL_STUB_SECONDS", "0.01"))
        expected_gflops = 2 * M * N * K / stub_sec / 1e9
        target = expected_gflops * 2.0
        env = MatmulEnv(split="train", M=M, N=N, K=K, target_gflops=target)
        task = env.reset()
        _write_kernel(task.workdir, _correct_kernel())
        result = env.score(Submission(workdir=task.workdir))
        assert result.raw["correct"] is True
        expected_reward = float(np.clip(expected_gflops / target, 0.0, 1.0))
        assert abs(result.reward - expected_reward) < 0.05

    def test_heldout_split(self):
        env = _make_env(split="heldout")
        task = env.reset()
        assert task.split == "heldout"
        assert "heldout" in env.id
