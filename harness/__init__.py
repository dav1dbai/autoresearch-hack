"""AR² harness — host outer-loop driver and rollout infrastructure.

Subpackages:
  loop/     — outer meta-loop (drive, evaluate, archive) — host-only
  runtime/  — rollout execution (score_repo, referee, sandbox) — host default;
              also copied into version snapshots for meta-editing (D-15)
  cloud/    — Modal fan-out and app session — host-only
  backends/ — GPU compute (local, Modal, Vast) — host-only
  tracing/  — telemetry injection and db sync — host-only
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
