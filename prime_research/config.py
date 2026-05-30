from __future__ import annotations

import tomllib
from pathlib import Path
from typing import Any

from .contracts import Budget, TaskSpec


DEFAULT_CONFIG = Path(__file__).parent / "configs" / "kernel_legal.toml"


def _budget(data: dict[str, Any], default: Budget) -> Budget:
    merged = default.model_dump()
    merged.update(data)
    return Budget(**merged)


def load_tasks(path: str | Path = DEFAULT_CONFIG) -> list[TaskSpec]:
    config_path = Path(path)
    data = tomllib.loads(config_path.read_text())
    default_budget = Budget(**data.get("budget", {}))
    tasks: list[TaskSpec] = []
    for row in data.get("tasks", []):
        row = dict(row)
        row["budget"] = _budget(row.pop("budget", {}), default_budget)
        tasks.append(TaskSpec(**row))
    return tasks


def select_tasks(path: str | Path, names: list[str] | None) -> list[TaskSpec]:
    tasks = load_tasks(path)
    if not names:
        return tasks
    wanted = set(names)
    selected = [task for task in tasks if task.name in wanted]
    missing = wanted - {task.name for task in selected}
    if missing:
        raise ValueError(f"Unknown task(s): {', '.join(sorted(missing))}")
    return selected

