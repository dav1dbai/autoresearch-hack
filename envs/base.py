"""envs/base.py — re-exports Env from contracts and provides BaseEnv mixin + a simple registry."""
from __future__ import annotations

from typing import Literal

from harness.contracts import Env, Split, TaskSpec, Submission, StepResult

__all__ = ["Env", "BaseEnv", "register", "list_envs"]

_registry: list[Env] = []


class BaseEnv:
    """Mixin that satisfies the structural id/split fields required by the Env protocol."""
    id: str
    split: Split

    def reset(self) -> TaskSpec:
        raise NotImplementedError

    def score(self, sub: Submission) -> StepResult:
        raise NotImplementedError


def register(env: Env) -> Env:
    """Add env to the global registry. Returns env (usable as a decorator target)."""
    _registry.append(env)
    return env


def list_envs(split: Literal["train", "heldout"] | None = None) -> list[Env]:
    """Return all registered envs, optionally filtered by split."""
    if split is None:
        return list(_registry)
    return [e for e in _registry if e.split == split]
