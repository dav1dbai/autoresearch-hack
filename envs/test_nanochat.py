"""Tests for envs/nanochat.py — offline with NANOCHAT_STUB=1."""
from __future__ import annotations

import os

import pytest

os.environ.setdefault("NANOCHAT_STUB", "1")

from harness.contracts import Env, Submission, TaskSpec, StepResult
from envs.nanochat import NanoChatEnv, _parse_bpb


def test_nanochat_satisfies_env_protocol():
    env = NanoChatEnv(split="train", stub=True)
    assert isinstance(env, Env), "NanoChatEnv must satisfy the runtime-checkable Env protocol"


def test_nanochat_has_required_attrs():
    env = NanoChatEnv(split="heldout", stub=True)
    assert env.id == "nanochat-heldout"
    assert env.split == "heldout"


def test_reset_returns_taskspec():
    env = NanoChatEnv(split="train", stub=True)
    spec = env.reset()
    assert isinstance(spec, TaskSpec)
    assert spec.env_id == "nanochat-train"
    assert spec.split == "train"
    assert spec.workdir.exists()


def test_reset_workdir_contains_only_train_py():
    env = NanoChatEnv(split="train", stub=True)
    spec = env.reset()
    names = {f.name for f in spec.workdir.iterdir()}
    assert "train.py" in names, "train.py must exist in workdir"
    assert "prepare.py" not in names, "prepare.py (evaluator) must NOT be in workdir"


def test_reset_workdir_does_not_contain_evaluator():
    env = NanoChatEnv(split="train", stub=True)
    spec = env.reset()
    evaluator_files = list(spec.workdir.rglob("prepare.py"))
    assert not evaluator_files, "Evaluator prepare.py must not appear in the agent workdir"


def test_score_parses_and_normalizes(tmp_path):
    env = NanoChatEnv(split="train", stub=True, baseline_bpb=1.0)
    spec = env.reset()
    sub = Submission(workdir=spec.workdir)
    result = env.score(sub)

    assert isinstance(result, StepResult)
    assert 0.0 <= result.reward <= 1.0, "reward must be in [0, 1]"
    assert "val_bpb" in result.raw
    assert result.done is False


def test_score_reward_at_baseline_is_zero():
    env = NanoChatEnv(split="train", stub=True, baseline_bpb=1.0)
    env._run = lambda wd: 1.0  # type: ignore[method-assign]
    spec = env.reset()
    result = env.score(Submission(workdir=spec.workdir))
    assert result.reward == pytest.approx(0.0)


def test_score_reward_below_baseline_is_positive():
    env = NanoChatEnv(split="train", stub=True, baseline_bpb=1.0)
    env._run = lambda wd: 0.8  # type: ignore[method-assign]
    spec = env.reset()
    result = env.score(Submission(workdir=spec.workdir))
    assert result.reward > 0.0


def test_score_reward_clipped_at_one():
    env = NanoChatEnv(split="train", stub=True, baseline_bpb=1.0)
    env._stub = False  # bypass stub, feed a synthetic log instead
    spec = env.reset()
    env._run = lambda wd: 0.0  # type: ignore[method-assign]
    result = env.score(Submission(workdir=spec.workdir))
    assert result.reward == pytest.approx(1.0)


def test_score_reward_clipped_at_zero():
    env = NanoChatEnv(split="train", stub=True, baseline_bpb=1.0)
    env._run = lambda wd: 9999.0  # type: ignore[method-assign]
    spec = env.reset()
    result = env.score(Submission(workdir=spec.workdir))
    assert result.reward == pytest.approx(0.0)


def test_parse_bpb_valid():
    log = "---\nval_bpb:          0.997900\ntraining_seconds: 300.1\n"
    assert _parse_bpb(log) == pytest.approx(0.9979)


def test_parse_bpb_missing_raises():
    with pytest.raises(ValueError, match="val_bpb not found"):
        _parse_bpb("no metric here")
