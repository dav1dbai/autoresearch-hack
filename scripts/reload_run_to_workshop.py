#!/usr/bin/env python3
"""Reload a saved AR² run into the active Raindrop Workshop UI."""
from __future__ import annotations

import argparse
import sqlite3
import sys
from pathlib import Path

_root = Path(__file__).resolve().parents[1]
if str(_root) not in sys.path:
    sys.path.insert(0, str(_root))

from harness.tracing.sync import push_spans_live, sync_archive_jsonl


def _safe_run_root(run_id: str) -> Path:
    if not run_id or "/" in run_id or "\\" in run_id or run_id in {".", ".."}:
        raise SystemExit(f"invalid run id: {run_id!r}")
    root = Path("obs/runs") / run_id
    if not root.is_dir():
        raise SystemExit(f"run not found: {root}")
    return root


def _load_spans(db_path: Path) -> list[dict]:
    if not db_path.exists():
        return []
    con = sqlite3.connect(str(db_path))
    con.row_factory = sqlite3.Row
    try:
        return [dict(row) for row in con.execute("SELECT * FROM spans ORDER BY ts")]
    finally:
        con.close()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("run_id", help="Run id under obs/runs/<run_id>")
    args = parser.parse_args()

    run_root = _safe_run_root(args.run_id)
    archive = run_root / "archive.jsonl"
    span_db = run_root / "ar2_workshop.db"
    if not span_db.exists():
        span_db = run_root / "traces.db"

    spans = _load_spans(span_db)
    pushed = push_spans_live(spans)
    attempts = sync_archive_jsonl(archive) if archive.exists() else 0
    print(
        f"Reloaded {args.run_id}: pushed {pushed}/{len(spans)} span(s), "
        f"synced {attempts} attempt(s)"
    )


if __name__ == "__main__":
    main()
