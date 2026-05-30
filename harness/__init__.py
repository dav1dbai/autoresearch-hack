"""AR² harness — immutable meta-loop driver (see proof/DESIGN.md §3).

Subpackages:
  loop/     — outer meta-loop (drive, evaluate, archive)
  runtime/  — in-process rollout execution (score_repo, referee, sandbox)
  modal/    — Modal fan-out and app session
  backends/ — GPU compute (local, Modal, Vast)
  tracing/  — telemetry injection and db sync
  util/     — shared helpers

Public contract types live in harness.contracts (single source of truth).
"""
from harness.contracts import (
    Archive,
    Attempt,
    Budget,
    Env,
    Improve,
    Rollout,
    ScoreFn,
    Solve,
    SpawnFn,
    Split,
    StepResult,
    Submission,
    TaskSpec,
)

__all__ = [
    "Archive",
    "Attempt",
    "Budget",
    "Env",
    "Improve",
    "Rollout",
    "ScoreFn",
    "Solve",
    "SpawnFn",
    "Split",
    "StepResult",
    "Submission",
    "TaskSpec",
]
