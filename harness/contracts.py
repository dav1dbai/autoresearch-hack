"""AR² contracts — the single source of truth. IMMUTABLE.

Everything in harness/ and envs/ imports these. The mutable ar/ folder implements
the two entrypoint callables (Solve, Improve). The referee, selection rule,
sandboxing, obs injection, and hack detection all live in harness/ and are never
reachable from ar/.  See proof/DESIGN.md §3.
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
    max_concurrency: int = 1  # cap on agent-spawned fanout; ENFORCED by the harness, not the agent


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
    """One node in the evolutionary archive (a single AR version's evaluation)."""
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
        """DGM-style: branch from the top-k non-hacked ancestors by held-out reward."""
        ranked = sorted((a for a in self.attempts if not a.hack_flags),
                        key=lambda a: a.heldout_reward, reverse=True)
        return ranked[:k] or self.attempts[-k:]


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
