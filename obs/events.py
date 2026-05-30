"""Append-only run event log for the live dashboard (obs/run_events.jsonl)."""
from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Any

_DEFAULT_PATH = Path("obs/run_events.jsonl")


def log_event(
    phase: str,
    message: str,
    *,
    version: int | None = None,
    parent: int | None = None,
    path: Path | None = None,
    extra: dict[str, Any] | None = None,
    events_path: Path = _DEFAULT_PATH,
) -> None:
    """Record one harness/improve/evaluate milestone."""
    events_path.parent.mkdir(parents=True, exist_ok=True)
    row = {
        "ts": time.time(),
        "phase": phase,
        "message": message,
        "version": version,
        "parent": parent,
        "path": str(path) if path else None,
    }
    if extra:
        row.update(extra)
    with events_path.open("a", encoding="utf-8") as fh:
        fh.write(json.dumps(row, default=str) + "\n")
