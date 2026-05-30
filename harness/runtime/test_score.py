"""Offline tests for harness/sandbox.py, harness/referee.py, harness/score_repo.py.

Modal is mocked entirely — no network, no credentials required.
"""
from __future__ import annotations

import pickle
import sys
import types
from pathlib import Path
from typing import Literal
from unittest.mock import MagicMock, patch

import pytest

# ---------------------------------------------------------------------------
# Build a minimal modal stub so imports succeed without the real Modal client
# ---------------------------------------------------------------------------

def _make_modal_stub() -> types.ModuleType:
    stub = types.ModuleType("modal")
    # App
    app_cls = MagicMock()
    app_cls.lookup = MagicMock(return_value=MagicMock())
    stub.App = app_cls
    # Image
    img_cls = MagicMock()
    img_cls.debian_slim = MagicMock(return_value=MagicMock())
    stub.Image = img_cls
    # Sandbox
    sandbox_cls = MagicMock()
    stub.Sandbox = sandbox_cls
    # Secret (needed when backends.gpu is imported after this module)
    stub.Secret = MagicMock()
    stub.Secret.from_name = MagicMock(return_value=MagicMock())
    stub.Secret.from_dotenv = MagicMock(return_value=MagicMock())
    return stub


# Inject before any harness imports
_modal_stub = _make_modal_stub()
sys.modules.setdefault("modal", _modal_stub)

# Now safe to import harness modules
from harness.contracts import (  # noqa: E402
    Budget,
    Env,
    Rollout,
    StepResult,
    Submission,
    TaskSpec,
)
from harness.runtime.sandbox import exec, make_sandbox, make_spawn, read_file, write_file, _make_spawn_local  # noqa: E402
from harness.runtime.referee import make_referee  # noqa: E402
from harness.runtime.score import score_repo  # noqa: E402

# ---------------------------------------------------------------------------
# Helpers / fixtures
# ---------------------------------------------------------------------------

DUMMY_WORKDIR = Path("/tmp/dummy_workdir")


def _make_task() -> TaskSpec:
    return TaskSpec(
        env_id="test-env",
        split="train",
        prompt="maximize reward",
        workdir=DUMMY_WORKDIR,
    )


class DummyEnv:
    """Minimal Env implementation for testing."""
    id = "test-env"
    split: Literal["train", "heldout"] = "train"

    def reset(self) -> TaskSpec:
        return _make_task()

    def score(self, sub: Submission) -> StepResult:
        return StepResult(reward=0.75, raw={"test": True})


class CrashingEnv:
    id = "crash-env"
    split: Literal["train", "heldout"] = "train"

    def reset(self) -> TaskSpec:
        return _make_task()

    def score(self, sub: Submission) -> StepResult:
        raise RuntimeError("grader exploded")


# ---------------------------------------------------------------------------
# sandbox.py tests
# ---------------------------------------------------------------------------

class TestMakeSandbox:
    def test_calls_sandbox_create(self):
        fake_sb = MagicMock()
        _modal_stub.Sandbox.create.return_value = fake_sb
        _modal_stub.App.lookup.return_value = MagicMock()

        img = MagicMock()
        sb = make_sandbox(img, timeout_s=60)

        _modal_stub.Sandbox.create.assert_called_once()
        call_kwargs = _modal_stub.Sandbox.create.call_args[1]
        assert call_kwargs["timeout"] == 60
        assert call_kwargs["block_network"] is False

    def test_block_network_forwarded(self):
        fake_sb = MagicMock()
        _modal_stub.Sandbox.create.return_value = fake_sb

        make_sandbox(MagicMock(), timeout_s=30, block_network=True)

        call_kwargs = _modal_stub.Sandbox.create.call_args[1]
        assert call_kwargs["block_network"] is True


class TestExecHelper:
    def test_exec_returns_stdout_stderr_rc(self):
        proc = MagicMock()
        proc.stdout.read.return_value = "hello"
        proc.stderr.read.return_value = ""
        proc.returncode = 0
        proc.wait.return_value = None

        sb = MagicMock()
        sb.exec.return_value = proc

        stdout, stderr, rc = exec(sb, "echo", "hello")

        sb.exec.assert_called_once_with("echo", "hello")
        assert stdout == "hello"
        assert rc == 0


class TestFileHelpers:
    def test_read_file(self):
        sb = MagicMock()
        sb.filesystem.read_text.return_value = "content"
        assert read_file(sb, "/work/trace.jsonl") == "content"
        sb.filesystem.read_text.assert_called_once_with("/work/trace.jsonl")

    def test_write_file(self):
        sb = MagicMock()
        write_file(sb, "/work/out.txt", "data")
        sb.filesystem.write_text.assert_called_once_with("data", "/work/out.txt")


class TestMakeSpawn:
    def test_empty_list(self):
        spawn = make_spawn()
        assert spawn(lambda x: x, []) == []

    def test_runs_all_items(self):
        spawn = make_spawn()
        results = spawn(lambda x: x * 2, [1, 2, 3])
        assert sorted(results) == [2, 4, 6]

    def test_respects_concurrency(self):
        import threading
        active = []
        peak = []
        lock = threading.Lock()

        def work(x):
            with lock:
                active.append(x)
                peak.append(len(active))
            import time; time.sleep(0.05)
            with lock:
                active.remove(x)
            return x

        spawn = _make_spawn_local(2)
        spawn(work, list(range(6)))
        assert max(peak) <= 2

    def test_single_arg_callable(self):
        spawn = _make_spawn_local(2)
        results = spawn(lambda x: x + 10, [5, 6])
        assert sorted(results) == [15, 16]

    def test_tuple_args_unpacked(self):
        spawn = _make_spawn_local(2)
        results = spawn(lambda a, b: a + b, [(1, 2), (3, 4)])
        assert sorted(results) == [3, 7]


# ---------------------------------------------------------------------------
# referee.py tests
# ---------------------------------------------------------------------------

class TestMakeReferee:
    def test_returns_correct_reward(self):
        env = DummyEnv()
        score = make_referee(env)
        sub = Submission(workdir=DUMMY_WORKDIR)
        result = score(sub)
        assert isinstance(result, StepResult)
        assert abs(result.reward - 0.75) < 1e-9

    def test_crash_returns_zero_reward(self):
        env = CrashingEnv()
        score = make_referee(env)
        sub = Submission(workdir=DUMMY_WORKDIR)
        result = score(sub)
        assert result.reward == 0.0

    def test_referee_is_isolated(self):
        # The referee runs in a separate process; mutating env locally
        # after make_referee should not affect grading (env is pickled at creation).
        env = DummyEnv()
        score = make_referee(env)
        env.id = "mutated"  # type: ignore[attr-defined] — should have no effect
        sub = Submission(workdir=DUMMY_WORKDIR)
        result = score(sub)
        assert result.reward == 0.75


# ---------------------------------------------------------------------------
# score_repo.py tests
# ---------------------------------------------------------------------------

def _make_ar_dir(tmp_path: Path, solve_src: str) -> Path:
    ar_dir = tmp_path / "ar"
    ar_dir.mkdir()
    (ar_dir / "entrypoint.py").write_text(solve_src)
    return ar_dir


SOLVE_GOOD = """
from harness.contracts import TaskSpec, Budget, Submission, StepResult
from pathlib import Path

def solve(task, budget, score, spawn):
    sub = Submission(workdir=task.workdir)
    score(sub)
    return sub
"""

SOLVE_CRASH = """
def solve(task, budget, score, spawn):
    raise RuntimeError("inner loop exploded")
"""


class TestScoreRepo:
    def test_returns_one_rollout_per_env(self, tmp_path):
        ar_dir = _make_ar_dir(tmp_path, SOLVE_GOOD)
        envs = [DummyEnv()]
        budget = Budget(wall_seconds=10)
        rollouts = score_repo(ar_dir, envs, budget)
        assert len(rollouts) == 1
        assert isinstance(rollouts[0], Rollout)

    def test_captures_inner_curve_rewards(self, tmp_path):
        ar_dir = _make_ar_dir(tmp_path, SOLVE_GOOD)
        budget = Budget(wall_seconds=10)
        rollouts = score_repo(ar_dir, [DummyEnv()], budget)
        assert rollouts[0].rewards == [0.75]
        assert rollouts[0].final_reward == 0.75

    def test_crash_returns_zero_and_hack_flag(self, tmp_path):
        ar_dir = _make_ar_dir(tmp_path, SOLVE_CRASH)
        budget = Budget(wall_seconds=10)
        rollouts = score_repo(ar_dir, [DummyEnv()], budget)
        assert rollouts[0].final_reward == 0.0
        assert "crash" in rollouts[0].hack_flags

    def test_inject_callback_called(self, tmp_path):
        ar_dir = _make_ar_dir(tmp_path, SOLVE_GOOD)
        budget = Budget(wall_seconds=10)
        injected: list[str] = []

        def inject(env, trace_id=""):
            injected.append(env.id)
            return {"TRACE_ID": trace_id or "abc"}

        score_repo(ar_dir, [DummyEnv()], budget, inject=inject)
        assert injected == ["test-env"]

    def test_multiple_envs(self, tmp_path):
        ar_dir = _make_ar_dir(tmp_path, SOLVE_GOOD)
        budget = Budget(wall_seconds=10)
        envs = [DummyEnv(), DummyEnv()]
        rollouts = score_repo(ar_dir, envs, budget)
        assert len(rollouts) == 2
        assert all(r.final_reward == 0.75 for r in rollouts)

    def test_env_id_and_split_propagated(self, tmp_path):
        ar_dir = _make_ar_dir(tmp_path, SOLVE_GOOD)
        budget = Budget(wall_seconds=10)
        rollouts = score_repo(ar_dir, [DummyEnv()], budget)
        assert rollouts[0].env_id == "test-env"
        assert rollouts[0].split == "train"
