"""Tests for D-02 sample_parents."""
from __future__ import annotations

from harness.contracts import Archive, Attempt, Budget


def _attempt(v: int, held: float, parent: int | None = None) -> Attempt:
    return Attempt(
        version=v,
        parent=parent,
        train_reward=held,
        heldout_reward=held,
        cost=Budget(wall_seconds=1.0),
    )


def test_sample_parents_prefers_under_sampled():
    archive = Archive(attempts=[
        _attempt(0, 0.9),
        _attempt(1, 0.5, parent=0),
        _attempt(2, 0.48, parent=0),
        _attempt(3, 0.47, parent=0),
    ])
    picks = archive.sample_parents(1, seed=0)
    assert len(picks) == 1
    # v1 has high heldout but many children; v2/v3 may win weight lottery
    assert picks[0].version in {1, 2, 3}
