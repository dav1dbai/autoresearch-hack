"""Cross-package smoke tests — run via `uv run pytest -m smoke -q`."""
from __future__ import annotations

import sqlite3

import pytest


@pytest.mark.smoke
def test_full_e2e_offline(tmp_path):
    """Full offline E2E: outer loop + telemetry sync + report generation."""
    from smoke.e2e_matmul import run
    run(out_dir=tmp_path)

    assert (tmp_path / "traces.db").exists()
    assert (tmp_path / "archive.jsonl").exists()
    assert (tmp_path / "report.html").exists()

    con = sqlite3.connect(str(tmp_path / "traces.db"))
    count = con.execute("SELECT COUNT(*) FROM spans").fetchone()[0]
    con.close()
    assert count > 0
