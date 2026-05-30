"""Pydantic models + YAML loader for harness task definitions.

A *task* is the RL-environment unit of this project: a prompt, a submission
contract, and a verifier that scores a submission and returns a scalar reward.
Tasks are defined in YAML (see ``tasks/*.yaml``) and parsed into ``TaskDef``.

The verifier is intentionally described as a *command run in a separate
sandbox* (see ``harness/executor.py``) so the agent that produces the
submission can never reach the grader — the one invariant that keeps the reward
honest.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Literal

import yaml
from pydantic import BaseModel, Field

Domain = Literal["legal", "kernel", "terminal", "multi_agent", "sre", "misc"]
VerifierKind = Literal["deterministic", "llm_judge", "blended"]


class FileMap(BaseModel):
    """Copy ``src`` (relative to the repo root, or absolute) to ``dest``
    (relative to the sandbox working dir)."""

    src: str
    dest: str


class ImageSpec(BaseModel):
    """Declarative Modal image. ``base`` is either ``debian_slim`` or a registry
    reference like ``nvidia/cuda:12.4.1-devel-ubuntu22.04``."""

    base: str = "debian_slim"
    python_version: str = "3.12"
    apt: list[str] = Field(default_factory=list)
    pip: list[str] = Field(default_factory=list)
    run: list[str] = Field(default_factory=list)


class Budget(BaseModel):
    """Caps. ``wall_clock_s`` bounds a whole rollout; everything fits in ~20 min
    for the hackathon."""

    max_iters: int = 6
    wall_clock_s: int = 20 * 60
    agent_turn_timeout_s: int = 8 * 60
    verifier_timeout_s: int = 5 * 60


class SubmissionSpec(BaseModel):
    """The single artifact the agent edits, relative to the agent workdir.

    ``template`` is written into the workdir before the agent starts so the
    submission path always exists (and the agent sees the expected schema).
    """

    path: str
    template: str | None = None


class Verifier(BaseModel):
    """How a submission is graded.

    ``command`` is argv executed inside a *separate* sandbox. The token
    ``{submission}`` is replaced with the submission filename as staged in the
    verifier sandbox. The verifier must print JSON to stdout containing
    ``reward_key`` (a float, ideally normalized to [0, 1]).
    """

    kind: VerifierKind = "deterministic"
    files: list[FileMap] = Field(default_factory=list)
    command: list[str]
    reward_key: str = "reward"
    image: ImageSpec | None = None
    gpu: str | None = None
    # Documentation for llm_judge / blended verifiers; the judge call itself
    # lives inside the verifier command so it stays out of the agent's reach.
    judge_model: str | None = None
    notes: str | None = None


class AgentSpec(BaseModel):
    """The headless coding agent that produces submissions. ``cmd`` has the
    task prompt appended as its final argument by the executor."""

    cmd: str = "codex exec"
    image: ImageSpec | None = None
    gpu: str | None = None


class TaskDef(BaseModel):
    id: str
    domain: Domain
    description: str = ""
    prompt: str
    inputs: list[FileMap] = Field(default_factory=list)
    submission: SubmissionSpec
    verifier: Verifier
    budget: Budget = Field(default_factory=Budget)
    agent: AgentSpec = Field(default_factory=AgentSpec)
    metadata: dict[str, Any] = Field(default_factory=dict)


TASKS_DIR = Path(__file__).resolve().parents[1] / "tasks"


def load_task(path: str | Path) -> TaskDef:
    data = yaml.safe_load(Path(path).read_text())
    return TaskDef(**data)


def load_tasks(directory: str | Path = TASKS_DIR) -> list[TaskDef]:
    directory = Path(directory)
    return [load_task(p) for p in sorted(directory.glob("*.yaml"))]


def find_task(task_id: str, directory: str | Path = TASKS_DIR) -> TaskDef:
    for task in load_tasks(directory):
        if task.id == task_id:
            return task
    available = ", ".join(t.id for t in load_tasks(directory))
    raise ValueError(f"Unknown task id {task_id!r}. Available: {available}")
