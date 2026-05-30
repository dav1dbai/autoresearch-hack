#!/usr/bin/env python3
"""Print post-run summary + build obs/report.html."""
from __future__ import annotations

import json
import os
import sqlite3
import sys
from pathlib import Path

_root = Path(__file__).resolve().parents[1]
if str(_root) not in sys.path:
    sys.path.insert(0, str(_root))

from harness.contracts import Attempt
from obs.dashboard import build_report, _resolve_cost_db


def main() -> None:
    root = Path(__file__).resolve().parents[1]
    archive_path = Path(os.environ.get("AR2_DASHBOARD_ARCHIVE", root / "obs" / "archive.jsonl"))
    if not archive_path.exists():
        print(f"No {archive_path} — run harness first.", file=sys.stderr)
        sys.exit(1)

    attempts: list[Attempt] = []
    with archive_path.open() as fh:
        for line in fh:
            line = line.strip()
            if line:
                attempts.append(Attempt.model_validate_json(line))
    attempts.sort(key=lambda a: a.version)

    cost_db = _resolve_cost_db(Path(os.environ.get("AR2_DASHBOARD_DB", root / "obs" / "traces.db")))
    costs: dict[int, float] = {}
    if cost_db.exists():
        con = sqlite3.connect(str(cost_db))
        try:
            rows = con.execute(
                "SELECT version, SUM(cost_usd), COUNT(*) FROM spans GROUP BY version"
            ).fetchall()
            costs = {int(r[0]): float(r[1] or 0) for r in rows}
            span_counts = {int(r[0]): int(r[2]) for r in rows}
        except sqlite3.OperationalError:
            span_counts = {}
        finally:
            con.close()
    else:
        span_counts = {}

    report = build_report(
        archive_path=archive_path,
        db_path=cost_db,
        out=Path(os.environ.get("AR2_DASHBOARD_OUT", root / "obs" / "report.html")),
        events_path=Path(os.environ.get("AR2_DASHBOARD_EVENTS", root / "obs" / "run_events.jsonl")),
        versions_root=Path(os.environ.get("AR2_DASHBOARD_VERSIONS", root / "versions")),
    )

    print("=" * 60)
    print("AR² K=3 run summary")
    print("=" * 60)
    print(f"Attempts: {len(attempts)}")
    print(f"Report:   {report}")
    print(f"Cost DB:  {cost_db}")
    print()
    print(f"{'ver':>4}  {'train':>8}  {'heldout':>8}  {'parent':>6}  {'inner pts':>10}  {'spans':>6}  source")
    print("-" * 60)
    for a in attempts:
        inner_pts = sum(len(r.rewards) for r in a.train_rollouts + a.heldout_rollouts)
        src = a.source_ref
        if len(src) > 40:
            src = "…" + src[-37:]
        print(
            f"{a.version:>4}  {a.train_reward:8.4f}  {a.heldout_reward:8.4f}  "
            f"{str(a.parent):>6}  {inner_pts:>10}  {span_counts.get(a.version, 0):>6}  {src}"
        )
        for r in a.train_rollouts + a.heldout_rollouts:
            curve = ", ".join(f"{x:.3f}" for x in r.rewards) or "—"
            print(f"      {r.split:7} {r.env_id}: [{curve}] → {r.final_reward:.4f}")

    if attempts:
        best = max(attempts, key=lambda a: a.heldout_reward)
        print()
        print(f"Best held-out: v{best.version} = {best.heldout_reward:.4f}")

    versions_root = Path(os.environ.get("AR2_DASHBOARD_VERSIONS", root / "versions"))
    versions = sorted(versions_root.glob("v_*")) if versions_root.is_dir() else []
    print(f"\nVersion snapshots: {len(versions)} under {versions_root}/")
    for v in versions[-5:]:
        has_spawn = any(
            "spawn(" in (v / "ar" / "entrypoint.py").read_text(errors="replace")
            for p in [v / "ar" / "entrypoint.py"]
            if p.is_file()
        )
        flag = " spawn wired" if has_spawn else ""
        print(f"  {v.name}{flag}")


if __name__ == "__main__":
    main()
