"""Build Attempt records from rollouts — single choke point for evaluate()."""
from __future__ import annotations

import uuid
from pathlib import Path

from harness.contracts import Attempt, Budget, Rollout


def _mean_final(rolls: list[Rollout]) -> float:
    return sum(r.final_reward for r in rolls) / len(rolls) if rolls else 0.0


def mean_final_reward(rolls: list[Rollout]) -> float:
    return _mean_final(rolls)


def attempt_from_rollouts(
    train_rolls: list[Rollout],
    heldout_rolls: list[Rollout],
    *,
    hack_flags: list[str] | None = None,
    version: int = 0,
    parent: int | None = None,
    diff_summary: str = "",
    source_ref: str = "",
) -> Attempt:
    """Collapse rollouts into an Attempt while preserving inner curves (D-03)."""
    total_cost = Budget(
        wall_seconds=sum(r.cost.wall_seconds for r in train_rolls + heldout_rolls),
        usd=sum((r.cost.usd or 0.0) for r in train_rolls + heldout_rolls) or None,
        tokens=sum((r.cost.tokens or 0) for r in train_rolls + heldout_rolls) or None,
    )
    trace_ids = [r.trace_id for r in train_rolls + heldout_rolls if r.trace_id]
    trace_id = ",".join(trace_ids) if trace_ids else str(uuid.uuid4())

    return Attempt(
        version=version,
        parent=parent,
        diff_summary=diff_summary,
        train_reward=_mean_final(train_rolls),
        heldout_reward=_mean_final(heldout_rolls),
        hack_flags=hack_flags or [],
        cost=total_cost,
        trace_id=trace_id,
        source_ref=source_ref,
        train_rollouts=train_rolls,
        heldout_rollouts=heldout_rolls,
    )
