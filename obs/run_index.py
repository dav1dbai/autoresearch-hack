"""Build a master index for namespaced AR² runs under obs/runs/."""
from __future__ import annotations

import html
import json
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path

from harness.contracts import Attempt


@dataclass(frozen=True)
class RunSummary:
    run_id: str
    root: Path
    report: Path
    attempts: int
    latest_version: int | None
    best_heldout: float | None
    updated_at: float


def _load_attempts(path: Path) -> list[Attempt]:
    if not path.exists():
        return []
    attempts: list[Attempt] = []
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if line:
            attempts.append(Attempt.model_validate_json(line))
    return attempts


def _summarize_run(run_root: Path) -> RunSummary | None:
    archive = run_root / "archive.jsonl"
    report = run_root / "report.html"
    if not archive.exists() and not report.exists():
        return None
    attempts = _load_attempts(archive)
    mtimes = [p.stat().st_mtime for p in (archive, report) if p.exists()]
    latest = max((a.version for a in attempts), default=None)
    best = max((a.heldout_reward for a in attempts), default=None)
    return RunSummary(
        run_id=run_root.name,
        root=run_root,
        report=report,
        attempts=len(attempts),
        latest_version=latest,
        best_heldout=best,
        updated_at=max(mtimes) if mtimes else run_root.stat().st_mtime,
    )


def build_index(runs_root: Path = Path("obs/runs"), out: Path | None = None) -> Path:
    out = out or runs_root / "index.html"
    runs_root.mkdir(parents=True, exist_ok=True)
    summaries = [
        summary
        for run_dir in sorted(runs_root.iterdir(), key=lambda p: p.stat().st_mtime, reverse=True)
        if run_dir.is_dir()
        if (summary := _summarize_run(run_dir)) is not None
    ]

    cards: list[str] = []
    for summary in summaries:
        report_rel = html.escape(summary.report.name)
        best = "n/a" if summary.best_heldout is None else f"{summary.best_heldout:.4f}"
        latest = "n/a" if summary.latest_version is None else f"v{summary.latest_version}"
        updated = datetime.fromtimestamp(summary.updated_at).strftime("%Y-%m-%d %H:%M:%S")
        cards.append(
            f"""
<article class="card">
  <h2>{html.escape(summary.run_id)}</h2>
  <p><strong>Attempts:</strong> {summary.attempts}</p>
  <p><strong>Latest:</strong> {latest}</p>
  <p><strong>Best heldout:</strong> {best}</p>
  <p class="muted">Updated {html.escape(updated)}</p>
  <p><a href="{report_rel if summary.root == runs_root else html.escape(summary.run_id) + '/report.html'}">Open report</a></p>
</article>
"""
        )

    payload = json.dumps(
        [
            {
                "run_id": s.run_id,
                "attempts": s.attempts,
                "latest_version": s.latest_version,
                "best_heldout": s.best_heldout,
                "updated_at": s.updated_at,
            }
            for s in summaries
        ],
        indent=2,
    )
    built_at = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    doc = f"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="utf-8"/>
<meta http-equiv="refresh" content="15"/>
<title>AR² Runs</title>
<style>
  body {{ font-family: system-ui, sans-serif; margin: 32px; background: #fafafa; color: #222; }}
  h1 {{ margin-bottom: 4px; }}
  .muted {{ color: #666; font-size: 0.85rem; }}
  .grid {{ display: grid; grid-template-columns: repeat(auto-fill, minmax(240px, 1fr)); gap: 16px; margin-top: 24px; }}
  .card {{ background: white; border: 1px solid #ddd; border-radius: 10px; padding: 16px; box-shadow: 0 1px 2px rgba(0,0,0,0.04); }}
  .card h2 {{ margin: 0 0 10px; font-size: 1rem; }}
  .card p {{ margin: 6px 0; }}
  pre {{ display: none; }}
</style>
</head>
<body>
<h1>AR² Runs</h1>
<p class="muted">Built {html.escape(built_at)}. Refreshes every 15s.</p>
<section class="grid">
{''.join(cards) or '<p>No runs yet.</p>'}
</section>
<pre id="runs-json">{html.escape(payload)}</pre>
</body>
</html>
"""
    out.write_text(doc, encoding="utf-8")
    return out


if __name__ == "__main__":
    print(f"Run index written to {build_index()}")
