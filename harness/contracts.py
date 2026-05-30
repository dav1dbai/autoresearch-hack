"""AR² contracts — single source of truth for types and protocols (host-only).

The mutable stack (`ar/` + snapshot `harness/runtime/`) implements Solve and Improve.
Selection, sandbox boundaries, obs injection, hack detection, and the outer driver
live in other harness subpackages and are not editable inside a version snapshot.
See proof/DESIGN.md §3 and proof/DECISIONS.md D-15.
"""
from __future__ import annotations

from pathlib import Path
from typing import Callable, Literal, Protocol, runtime_checkable

from pydantic import BaseModel, Field

Split = Literal["train", "heldout"]
Reward = float  # normalized to [0,1] by the Env; 1.0 = perfect. Comparable across envs.


class Budget(BaseModel):
    wall_seconds: float = 300.0
    usd: float | None = None
    tokens: int | None = None


class EvalProtocol(BaseModel):
    """Harness eval rigor — separate from agent Budget (D-08)."""
    inner_max_iters: int = 8
    seeds: int = 1


class TaskSpec(BaseModel):
    env_id: str
    split: Split
    prompt: str
    workdir: Path  # the editable solution surface (e.g. a train.py); the referee is NOT here
    payload: dict = Field(default_factory=dict)


class Submission(BaseModel):
    workdir: Path  # AR's final edited solution surface
    notes: str = ""


class StepResult(BaseModel):
    reward: Reward
    raw: dict = Field(default_factory=dict)  # raw metric, e.g. {"val_bpb": 0.979}
    feedback: str | None = None
    # done=True means STOP the inner search — the target reward was reached or an
    # unrecoverable terminal state was hit. It does NOT mean "this score() call
    # finished" (that is always true). Envs with no natural terminal state must
    # return done=False so solve() keeps iterating until budget. See proof/DECISIONS.md D-00.
    done: bool = False


class Rollout(BaseModel):
    env_id: str
    split: Split
    rewards: list[Reward] = Field(default_factory=list)  # per inner iteration -> dR/dt (the inner curve)
    final_reward: Reward = 0.0
    cost: Budget
    trace_id: str
    trace_path: str = ""  # local path to trace.jsonl when telemetry pull ran (optional)
    hack_flags: list[str] = Field(default_factory=list)


class Attempt(BaseModel):
    """One node in the version archive (one evaluated AR snapshot)."""
    version: int
    parent: int | None = None
    diff_summary: str = ""
    train_reward: Reward = 0.0
    heldout_reward: Reward = 0.0  # the outer-curve y; selection signal
    hack_flags: list[str] = Field(default_factory=list)
    cost: Budget
    trace_id: str = ""
    source_ref: str = ""  # version snapshot root (versions/v_*/ with ar/ + harness/runtime/)
    train_rollouts: list[Rollout] = Field(default_factory=list)
    heldout_rollouts: list[Rollout] = Field(default_factory=list)


class Archive(BaseModel):
    attempts: list[Attempt] = Field(default_factory=list)

    def add(self, a: Attempt) -> None:
        self.attempts.append(a)

    def best(self) -> Attempt | None:
        """Best by held-out reward, preferring non-hacked attempts."""
        clean = [a for a in self.attempts if not a.hack_flags]
        pool = clean or self.attempts
        return max(pool, key=lambda a: a.heldout_reward, default=None)

    def frontier(self, k: int = 3) -> list[Attempt]:
        """Top-k non-hacked attempts by held-out reward (greedy legacy)."""
        ranked = sorted((a for a in self.attempts if not a.hack_flags),
                        key=lambda a: a.heldout_reward, reverse=True)
        return ranked[:k] or self.attempts[-k:]

    def sample_parents(self, m: int, *, seed: int | None = None) -> list[Attempt]:
        """D-02: weighted sample of non-hacked parents (not greedy top-k)."""
        import random

        rng = random.Random(seed)
        clean = [a for a in self.attempts if not a.hack_flags]
        pool = clean or list(self.attempts)
        if not pool:
            return []
        if len(pool) <= m:
            return pool

        child_counts: dict[int, int] = {}
        for a in self.attempts:
            if a.parent is not None:
                child_counts[a.parent] = child_counts.get(a.parent, 0) + 1

        weights = [
            max(a.heldout_reward, 1e-6) / (1 + child_counts.get(a.version, 0))
            for a in pool
        ]
        total = sum(weights)
        probs = [w / total for w in weights]

        chosen: list[Attempt] = []
        remaining = list(pool)
        remaining_weights = list(probs)
        for _ in range(min(m, len(remaining))):
            pick = rng.choices(remaining, weights=remaining_weights, k=1)[0]
            chosen.append(pick)
            idx = remaining.index(pick)
            remaining.pop(idx)
            remaining_weights.pop(idx)
            if remaining_weights:
                s = sum(remaining_weights)
                remaining_weights = [w / s for w in remaining_weights]
        return chosen


@runtime_checkable
class Env(Protocol):
    """The referee. Owned by envs/ (Alex). Lives OUTSIDE ar/. `score` is
    deterministic, pure, and runs out-of-process — never editable by the agent."""
    id: str
    split: Split

    def reset(self) -> TaskSpec: ...
    def score(self, sub: Submission) -> StepResult: ...


# --- AR entrypoint contract: the ONLY surface harness/ requires of ar/ ----------
# ar/entrypoint.py exposes module-level callables matching these. BOTH evolve across versions.

ScoreFn = Callable[[Submission], StepResult]  # the isolated referee, injected by the harness
SpawnFn = Callable[..., list]                 # the capped fanout primitive, injected by the harness


class Solve(Protocol):
    """Do research: edit task.workdir to maximize score(). v0 = Karpathy's inner loop;
    later versions may parallelize via spawn(), add critics, change the search."""
    def __call__(self, task: TaskSpec, budget: Budget, score: ScoreFn, spawn: SpawnFn) -> Submission: ...


class Improve(Protocol):
    """Emit the SUCCESSOR ar/ folder (a new dir) given the repo + archive. v0 invokes a
    coding agent; later versions may rewrite this, including how they fan out the search."""
    def __call__(self, archive: Archive, budget: Budget, spawn: SpawnFn) -> Path: ...
