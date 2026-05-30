"""Shared Modal stubs for harness/cloud tests."""
from __future__ import annotations

import sys
import types
import uuid
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import MagicMock

import pytest

from harness.contracts import Budget, Rollout

BUDGET = Budget(wall_seconds=10.0, max_concurrency=4)

_SOLVE_SRC = """
from harness.contracts import Submission
def solve(task, budget, score, spawn):
    sub = Submission(workdir=task.workdir)
    score(sub)
    return sub
"""


@pytest.fixture(autouse=True)
def _modal_ephemeral_for_tests(monkeypatch):
    monkeypatch.setenv("AR2_MODAL_DEPLOYED", "0")
    monkeypatch.setenv("AR2_MODAL_REUSE", "1")


class _FakeEnv:
    def __init__(self, env_id: str, split: str = "train"):
        self.id = env_id
        self.split = split

    def reset(self):
        from harness.contracts import TaskSpec
        return TaskSpec(env_id=self.id, split=self.split,
                        prompt="x", workdir=Path("/tmp"))

    def score(self, sub):
        from harness.contracts import StepResult
        return StepResult(reward=0.5)


def _make_ar_dir(tmp_path: Path) -> Path:
    d = tmp_path / "ar"
    d.mkdir(exist_ok=True)
    (d / "entrypoint.py").write_text(_SOLVE_SRC)
    return d


def _make_rollout(env_id: str, reward: float = 0.5) -> Rollout:
    return Rollout(
        env_id=env_id,
        split="train",
        rewards=[reward],
        final_reward=reward,
        cost=BUDGET,
        trace_id=str(uuid.uuid4()),
    )


def _make_full_modal_stub() -> types.ModuleType:
    modal_mod = types.ModuleType("modal")

    class _Image:
        def debian_slim(self, python_version="3.12"): return self
        def apt_install(self, *a): return self
        def run_commands(self, *a): return self
        def pip_install(self, *a): return self
        def add_local_python_source(self, *a, **kw): return self
        def add_local_dir(self, *a, **kw): return self

    class _BatchUploadCtx:
        def __enter__(self): return self
        def __exit__(self, *a): pass
        def put_directory(self, local: str, remote: str): pass
        def put_file(self, local, remote: str): pass

    class _Volume:
        @staticmethod
        def from_name(name: str, create_if_missing: bool = False) -> "_Volume":
            return _Volume()
        def commit(self): pass
        def reload(self): pass
        def batch_upload(self, force: bool = False): return _BatchUploadCtx()

    class _FunctionWrapper:
        def __init__(self, fn):
            self._fn = fn
            self.starmap = MagicMock(side_effect=lambda inputs: [
                fn(*args) for args in inputs
            ])
            self.map = MagicMock(side_effect=lambda inputs: [fn(a) for a in inputs])
            self.remote = MagicMock(side_effect=lambda *a, **kw: fn(*a, **kw))
            self.spawn = MagicMock(side_effect=lambda *a, **kw: SimpleNamespace(
                get=lambda: fn(*a, **kw)
            ))
        def __call__(self, *a, **kw):
            return self._fn(*a, **kw)

    def _function_decorator(**kwargs):
        def decorator(fn):
            return _FunctionWrapper(fn)
        return decorator

    class _RunCtx:
        def __enter__(self): return self
        def __exit__(self, *a): return False

    class _App:
        def __init__(self, name: str):
            self.name = name
        lookup = MagicMock(side_effect=lambda name, **kw: None)
        def function(self, **kwargs):
            return _function_decorator(**kwargs)
        def run(self):
            return _RunCtx()
        def deploy(self):
            pass

    modal_mod.Image = _Image()
    modal_mod.App = _App
    modal_mod.Volume = _Volume
    modal_mod.Sandbox = MagicMock()
    modal_mod.Sandbox.create = MagicMock(return_value=MagicMock())
    modal_mod.Secret = MagicMock()
    modal_mod.Secret.from_dotenv = MagicMock(return_value=MagicMock())
    modal_mod.Secret.from_name = MagicMock(return_value=MagicMock())

    class _FunctionFromName:
        @staticmethod
        def from_name(app_name: str, fn_name: str):
            wrapper = MagicMock()
            wrapper.starmap = MagicMock(return_value=[])
            return wrapper

    modal_mod.Function = _FunctionFromName
    return modal_mod


def _fresh_modal_runner(monkeypatch):
    dotenv_mod = types.ModuleType("dotenv")
    dotenv_mod.load_dotenv = lambda *a, **kw: None  # type: ignore[attr-defined]
    monkeypatch.setitem(sys.modules, "dotenv", dotenv_mod)

    stub = _make_full_modal_stub()
    monkeypatch.setitem(sys.modules, "modal", stub)

    for key in list(sys.modules):
        if key in (
            "harness.cloud.runner",
            "harness.cloud.register",
            "harness.backends.gpu",
            "harness.backends.local",
            "harness.backends.modal_gpu",
            "harness.backends.vast",
            "harness.runtime.score",
            "harness.runtime.sandbox",
            "infra.modal.images",
            "infra.modal.secrets",
            "infra",
        ):
            monkeypatch.delitem(sys.modules, key, raising=False)

    from harness.cloud.register import register_all

    register_all()
    import harness.cloud.runner as mr
    return mr
