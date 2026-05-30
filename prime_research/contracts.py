from __future__ import annotations

from pathlib import Path
from typing import Any, Literal

from pydantic import BaseModel, Field


Domain = Literal["kernel", "legal"]


class Budget(BaseModel):
    max_iters: int = 8
    wall_clock_s: int = 20 * 60
    poll_s: int = 10
    final_score_timeout_s: int = 5 * 60


class TaskSpec(BaseModel):
    name: str
    domain: Domain
    prime_env_id: str | None = None
    gpu: str | None = None
    task_path: str | None = None
    task_level: str | None = None
    task_index: int | None = None
    budget: Budget = Field(default_factory=Budget)
    metadata: dict[str, Any] = Field(default_factory=dict)


class PreparedTask(BaseModel):
    spec: TaskSpec
    workdir: Path
    verifier_dir: Path
    artifact_path: Path
    score_cmd: list[str]
    prompt: str


class ExperimentRecord(BaseModel):
    iter: int
    reward: float
    summary: str = ""
    score: dict[str, Any] = Field(default_factory=dict)


class RolloutResult(BaseModel):
    task: TaskSpec
    reward: float
    best_iter: int | None
    experiments: list[ExperimentRecord]
    final_score: dict[str, Any] = Field(default_factory=dict)
    workdir: str
    artifact_path: str
    wall_clock_s: float
    crashed: bool = False
    error: str | None = None
    agent_returncode: int | None = None
    agent_stdout_tail: str = ""
    agent_stderr_tail: str = ""
