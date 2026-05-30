"""Tests for obs/dashboard.py.

Synthesizes a 4-version archive (v0–v3, v2 hacked), writes archive.jsonl + a tiny
traces.db to a temp dir, calls build_report(), and asserts the HTML output is correct.
"""
import json
import sqlite3
import tempfile
from pathlib import Path

import pytest

from harness.contracts import Archive, Attempt, Budget
from obs.dashboard import build_report, _load_attempts, _cost_per_version


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

def _make_budget(usd: float = 0.01) -> Budget:
    return Budget(wall_seconds=60.0, usd=usd, tokens=1000, max_concurrency=1)


def _make_archive() -> tuple[Archive, list[Attempt]]:
    attempts = [
        Attempt(
            version=0,
            parent=None,
            diff_summary="initial v0",
            train_reward=0.30,
            heldout_reward=0.28,
            hack_flags=[],
            cost=_make_budget(0.01),
            trace_id="trace-v0",
            source_ref="abc001",
        ),
        Attempt(
            version=1,
            parent=0,
            diff_summary="add critic",
            train_reward=0.50,
            heldout_reward=0.47,
            hack_flags=[],
            cost=_make_budget(0.03),
            trace_id="trace-v1",
            source_ref="abc002",
        ),
        # v2: hacked — train spikes, heldout flat
        Attempt(
            version=2,
            parent=1,
            diff_summary="exploit train env quirk",
            train_reward=0.95,
            heldout_reward=0.40,
            hack_flags=["train_heldout_gap>0.4", "fabricated_log"],
            cost=_make_budget(0.05),
            trace_id="trace-v2-hacked",
            source_ref="abc003",
        ),
        Attempt(
            version=3,
            parent=1,
            diff_summary="add fanout",
            train_reward=0.72,
            heldout_reward=0.69,
            hack_flags=[],
            cost=_make_budget(0.08),
            trace_id="trace-v3",
            source_ref="abc004",
        ),
    ]
    archive = Archive(attempts=attempts)
    return archive, attempts


@pytest.fixture()
def tmp_obs(tmp_path: Path) -> tuple[Path, Path, Path]:
    """Write archive.jsonl + traces.db to tmp_path, return (archive, db, out) paths."""
    _, attempts = _make_archive()

    archive_path = tmp_path / "archive.jsonl"
    with archive_path.open("w") as f:
        for a in attempts:
            f.write(a.model_dump_json() + "\n")

    db_path = tmp_path / "traces.db"
    con = sqlite3.connect(db_path)
    con.execute(
        "CREATE TABLE spans (version INTEGER, cost_usd REAL, trace_id TEXT)"
    )
    con.executemany(
        "INSERT INTO spans VALUES (?,?,?)",
        [(0, 0.01, "trace-v0"),
         (1, 0.03, "trace-v1"),
         (2, 0.05, "trace-v2-hacked"),
         (3, 0.08, "trace-v3")],
    )
    con.commit()
    con.close()

    out_path = tmp_path / "report.html"
    return archive_path, db_path, out_path


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------

class TestLoadAttempts:
    def test_round_trips_all_fields(self, tmp_path: Path) -> None:
        _, attempts = _make_archive()
        p = tmp_path / "archive.jsonl"
        with p.open("w") as f:
            for a in attempts:
                f.write(a.model_dump_json() + "\n")
        loaded = _load_attempts(p)
        assert len(loaded) == 4
        assert [a.version for a in loaded] == [0, 1, 2, 3]
        assert loaded[2].hack_flags == ["train_heldout_gap>0.4", "fabricated_log"]

    def test_missing_file_returns_empty(self, tmp_path: Path) -> None:
        result = _load_attempts(tmp_path / "nonexistent.jsonl")
        assert result == []


class TestCostPerVersion:
    def test_sums_correctly(self, tmp_obs: tuple) -> None:
        _, db_path, _ = tmp_obs
        costs = _cost_per_version(db_path)
        assert costs[2] == pytest.approx(0.05)
        assert costs[3] == pytest.approx(0.08)

    def test_missing_db_returns_empty(self, tmp_path: Path) -> None:
        result = _cost_per_version(tmp_path / "no.db")
        assert result == {}

    def test_resolve_cost_db_prefers_raindrop(self, tmp_path: Path, monkeypatch) -> None:
        from obs.dashboard import _resolve_cost_db

        rd = tmp_path / "raindrop.db"
        rd.touch()
        monkeypatch.setattr(
            "harness.tracing.sync.raindrop_db_path",
            lambda: rd,
        )
        assert _resolve_cost_db(None) == rd
        explicit = tmp_path / "custom.db"
        assert _resolve_cost_db(explicit) == explicit


class TestBuildReport:
    def test_creates_html_file(self, tmp_obs: tuple) -> None:
        archive_path, db_path, out_path = tmp_obs
        result = build_report(archive_path, db_path, out_path)
        assert result == out_path
        assert out_path.exists()
        content = out_path.read_text()
        assert content.startswith("<!DOCTYPE html>")

    def test_all_versions_present(self, tmp_obs: tuple) -> None:
        archive_path, db_path, out_path = tmp_obs
        build_report(archive_path, db_path, out_path)
        content = out_path.read_text()
        for v in range(4):
            assert f"v{v}" in content

    def test_hacked_version_marked_red(self, tmp_obs: tuple) -> None:
        archive_path, db_path, out_path = tmp_obs
        build_report(archive_path, db_path, out_path)
        content = out_path.read_text()
        # Red color code must appear (hacked version v2)
        assert "#c0392b" in content

    def test_clean_versions_marked_green(self, tmp_obs: tuple) -> None:
        archive_path, db_path, out_path = tmp_obs
        build_report(archive_path, db_path, out_path)
        content = out_path.read_text()
        assert "#27ae60" in content

    def test_hack_flags_appear_in_drilldown(self, tmp_obs: tuple) -> None:
        archive_path, db_path, out_path = tmp_obs
        build_report(archive_path, db_path, out_path)
        content = out_path.read_text()
        assert "train_heldout_gap" in content
        assert "fabricated_log" in content

    def test_trace_id_anchor_for_hacked_version(self, tmp_obs: tuple) -> None:
        archive_path, db_path, out_path = tmp_obs
        build_report(archive_path, db_path, out_path)
        content = out_path.read_text()
        # Drilldown anchor for hacked trace
        assert "trace-v2-hacked" in content

    def test_reward_values_present(self, tmp_obs: tuple) -> None:
        archive_path, db_path, out_path = tmp_obs
        build_report(archive_path, db_path, out_path)
        content = out_path.read_text()
        # heldout rewards should appear (formatted to 2 or 3 dp)
        assert "0.28" in content   # v0
        assert "0.47" in content   # v1
        assert "0.69" in content   # v3

    def test_empty_archive(self, tmp_path: Path) -> None:
        archive_path = tmp_path / "empty.jsonl"
        archive_path.write_text("")
        db_path = tmp_path / "traces.db"
        out_path = tmp_path / "report.html"
        build_report(archive_path, db_path, out_path)
        assert out_path.exists()
        content = out_path.read_text()
        assert "No attempts yet." in content

    def test_cost_values_from_db(self, tmp_obs: tuple) -> None:
        archive_path, db_path, out_path = tmp_obs
        build_report(archive_path, db_path, out_path)
        content = out_path.read_text()
        # cost values from traces.db appear in chart tooltips / labels
        assert "0.080" in content or "0.08" in content   # v3 cost

    def test_lineage_parent_child(self, tmp_obs: tuple) -> None:
        archive_path, db_path, out_path = tmp_obs
        build_report(archive_path, db_path, out_path)
        content = out_path.read_text()
        # lineage SVG must contain an edge (dashed line for parent-child)
        assert "stroke-dasharray" in content
