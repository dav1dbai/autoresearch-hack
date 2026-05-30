"""Offline tests for harness/tracing/sync.py."""
from __future__ import annotations

import json
import sqlite3
import time
import uuid
from pathlib import Path
from typing import Any

import pytest

import harness.tracing.sync as sync_mod
from harness.contracts import Attempt, Budget, Rollout
from harness.tracing.sync import (
    _cost_usd,
    ensure_workshop_db,
    load_archive_from_raindrop,
    mirror_to_workshop,
    prepare_improve_context,
    sync,
    sync_all,
    sync_attempt,
)


@pytest.fixture()
def tmp(tmp_path: Path) -> Path:
    return tmp_path


def _make_span(
    trace_id: str = "t1",
    version: int = 0,
    candidate: str = "c0",
    env_id: str = "e0",
    split: str = "train",
    model: str = "gpt-4o-mini",
    prompt_tokens: int = 100,
    completion_tokens: int = 50,
    latency_ms: float = 120.0,
    tool_name: str | None = None,
    tool_input: str | None = None,
    ts: float | None = None,
) -> dict:
    return {
        "trace_id": trace_id,
        "version": version,
        "candidate": candidate,
        "env_id": env_id,
        "split": split,
        "model": model,
        "prompt_tokens": prompt_tokens,
        "completion_tokens": completion_tokens,
        "latency_ms": latency_ms,
        "tool_name": tool_name,
        "tool_input": tool_input,
        "ts": ts or time.time(),
    }


def _write_trace(path: Path, spans: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w") as f:
        for s in spans:
            f.write(json.dumps(s) + "\n")


def _rollout(
    trace_id: str = "t1",
    env_id: str = "e0",
    split: str = "train",
    final_reward: float = 0.8,
    rewards: list[float] | None = None,
) -> Rollout:
    return Rollout(
        env_id=env_id,
        split=split,
        rewards=rewards or [0.5, final_reward],
        final_reward=final_reward,
        cost=Budget(wall_seconds=60.0),
        trace_id=trace_id,
    )


class TestCostUsd:
    def test_known_model(self):
        cost = _cost_usd("gpt-4o-mini", 1000, 500)
        assert cost > 0
        assert abs(cost - 0.000450) < 1e-9

    def test_prefix_match(self):
        assert _cost_usd("gpt-4o-2024-08-06", 1000, 0) > 0

    def test_unknown_model_uses_fallback(self):
        cost = _cost_usd("unknown-model-xyz", 1000, 1000)
        assert cost > 0


class TestSync:
    def test_inserts_spans(self, tmp: Path):
        tf = tmp / "run0" / "trace.jsonl"
        spans = [_make_span(trace_id="t1", model="gpt-4o-mini"), _make_span(trace_id="t2")]
        _write_trace(tf, spans)
        db = tmp / "obs" / "traces.db"
        inserted = sync([tf], canonical=db)
        assert inserted == 2

        con = sqlite3.connect(str(db))
        rows = con.execute("SELECT trace_id, cost_usd FROM spans").fetchall()
        con.close()
        assert len(rows) == 2
        assert all(r[1] > 0 for r in rows)

    def test_cost_computed(self, tmp: Path):
        tf = tmp / "trace.jsonl"
        _write_trace(tf, [_make_span(model="gpt-4o", prompt_tokens=1000, completion_tokens=500)])
        db = tmp / "traces.db"
        sync([tf], canonical=db)
        con = sqlite3.connect(str(db))
        cost = con.execute("SELECT cost_usd FROM spans").fetchone()[0]
        con.close()
        expected = 1000 * 0.0025 / 1000 + 500 * 0.010 / 1000
        assert abs(cost - expected) < 1e-9

    def test_multiple_files(self, tmp: Path):
        files = []
        for i in range(3):
            tf = tmp / f"run{i}" / "trace.jsonl"
            _write_trace(tf, [_make_span(trace_id=f"t{i}")])
            files.append(tf)
        db = tmp / "traces.db"
        inserted = sync(files, canonical=db)
        assert inserted == 3

    def test_empty_files_ok(self, tmp: Path):
        tf = tmp / "trace.jsonl"
        tf.parent.mkdir(parents=True, exist_ok=True)
        tf.write_text("")
        db = tmp / "traces.db"
        assert sync([tf], canonical=db) == 0

    def test_push_spans_live_includes_run_id(self, monkeypatch: pytest.MonkeyPatch):
        captured: list[dict[str, Any]] = []
        monkeypatch.setenv("RAINDROP_WORKSHOP", "1")
        monkeypatch.setenv("AR2_RUN_ID", "raindrop-k2")

        class FakeResponse:
            status = 200

            def __enter__(self):
                return self

            def __exit__(self, exc_type, exc, tb):
                return False

            def read(self) -> bytes:
                return b'{"spansIngested": 1}'

        def fake_urlopen(req, timeout=5.0):
            captured.append(json.loads(req.data.decode("utf-8")))
            return FakeResponse()

        monkeypatch.setattr(sync_mod.urllib.request, "urlopen", fake_urlopen)

        pushed = sync_mod.push_spans_live([
            _make_span(tool_name="evaluate", tool_input="smoke", version=0)
        ])

        assert pushed == 1
        span = captured[0]["resourceSpans"][0]["scopeSpans"][0]["spans"][0]
        assert span["name"].startswith("[raindrop-k2] v0 evaluate")
        attrs = {a["key"]: a["value"] for a in span["attributes"]}
        assert attrs["ar2.run_id"]["stringValue"] == "raindrop-k2"
        assert attrs["ar2.display_name"]["stringValue"].startswith("[raindrop-k2]")


class TestSyncAllLocal:
    def test_local_db_when_workshop_disabled(self, tmp: Path, monkeypatch: pytest.MonkeyPatch):
        monkeypatch.setenv("RAINDROP_WORKSHOP", "0")
        tf = tmp / "trace.jsonl"
        _write_trace(tf, [_make_span(trace_id="local1")])
        traces_db = tmp / "traces.db"
        archive_db = tmp / "archive.db"
        attempt = Attempt(
            version=0,
            train_reward=0.5,
            heldout_reward=0.4,
            diff_summary="smoke",
            source_ref="ar",
            trace_id="local1",
            cost=Budget(wall_seconds=1.0),
        )
        result = sync_all(
            trace_files=[tf],
            attempt=attempt,
            traces_db=traces_db,
            archive_db=archive_db,
        )
        assert result["spans_inserted"] == 1
        assert result["attempts_synced"] == 1
        assert traces_db.exists()
        assert archive_db.exists()


class TestRaindropRead:
    def test_load_archive_from_raindrop(self, tmp: Path) -> None:
        db = tmp / "raindrop.db"
        attempt = Attempt(
            version=0,
            train_reward=0.5,
            heldout_reward=0.4,
            diff_summary="seed",
            source_ref="ar",
            trace_id="t0",
            cost=Budget(wall_seconds=1.0),
            train_rollouts=[
                Rollout(
                    env_id="e1",
                    split="train",
                    rewards=[0.1, 0.5],
                    final_reward=0.5,
                    cost=Budget(wall_seconds=1.0),
                    trace_id="t0",
                )
            ],
        )
        sync_attempt(attempt, db=db)
        loaded = load_archive_from_raindrop(db=db)
        assert len(loaded.attempts) == 1
        assert loaded.attempts[0].train_rollouts[0].rewards == [0.1, 0.5]

        archive, digest = prepare_improve_context(loaded, db=db)
        assert len(archive.attempts) == 1
        assert isinstance(digest, str)


class TestMirrorToWorkshop:
    def test_skips_when_disabled(self, tmp: Path, monkeypatch: pytest.MonkeyPatch):
        monkeypatch.setenv("RAINDROP_WORKSHOP", "0")
        db = tmp / "traces.db"
        sync([], canonical=db)
        counts = mirror_to_workshop(db)
        assert counts == {"spans": 0, "attempts": 0}

    def test_mirrors_rows(self, tmp: Path, monkeypatch: pytest.MonkeyPatch):
        monkeypatch.setenv("RAINDROP_WORKSHOP", "1")
        monkeypatch.setenv("RAINDROP_WORKSHOP_DB_PATH", str(tmp / "ar2_workshop.db"))
        workshop = ensure_workshop_db()
        assert workshop.exists()
        tf = tmp / "trace.jsonl"
        _write_trace(tf, [_make_span(trace_id="tw1")])
        canonical = tmp / "obs" / "traces.db"
        sync([tf], canonical=canonical)
        counts = mirror_to_workshop(canonical)
        assert counts["spans"] >= 1
        con = sqlite3.connect(str(workshop))
        rows = con.execute("SELECT trace_id FROM spans").fetchall()
        con.close()
        assert ("tw1",) in rows


class TestSyncAttempt:
    def test_roundtrip(self, tmp: Path):
        attempt = Attempt(
            version=0,
            train_reward=0.5,
            heldout_reward=0.4,
            cost=Budget(wall_seconds=1.0),
            train_rollouts=[
                Rollout(
                    env_id="e1",
                    split="train",
                    final_reward=0.5,
                    cost=Budget(wall_seconds=1.0),
                    trace_id="t1",
                    rewards=[0.1, 0.5],
                ),
            ],
        )
        db = tmp / "archive.db"
        sync_attempt(attempt, canonical=db)
        con = sqlite3.connect(str(db))
        row = con.execute("SELECT version, train_reward, train_rollouts_json FROM attempts").fetchone()
        con.close()
        assert row[0] == 0
        assert row[1] == pytest.approx(0.5)
        assert "0.1" in row[2]


# ---------------------------------------------------------------------------
# Raindrop aggregation (from former test_e2e_matmul.py)
# ---------------------------------------------------------------------------

def _make_trace_file(path: Path, spans: list[dict[str, Any]]) -> None:
    with path.open("w") as fh:
        for s in spans:
            fh.write(json.dumps(s) + "\n")


def _span(
    *,
    trace_id: str,
    version: int,
    split: str = "train",
    model: str = "gpt-4o-mini",
    prompt_tokens: int = 100,
    completion_tokens: int = 40,
    latency_ms: float = 80.0,
) -> dict[str, Any]:
    return {
        "trace_id": trace_id,
        "version": version,
        "candidate": f"cand-{version}",
        "env_id": f"matmul-{split}",
        "split": split,
        "model": model,
        "prompt_tokens": prompt_tokens,
        "completion_tokens": completion_tokens,
        "latency_ms": latency_ms,
        "tool_name": "score",
        "tool_input": "{}",
        "ts": time.time(),
    }


@pytest.mark.smoke
class TestRaindropAggregation:
    def _build_spans_for_versions(self) -> tuple[list[dict], dict[int, dict]]:
        specs = {
            0: dict(model="gpt-4o-mini", prompt_tokens=200, completion_tokens=80, latency_ms=100.0, n=3),
            1: dict(model="gpt-4o-mini", prompt_tokens=350, completion_tokens=130, latency_ms=150.0, n=2),
        }
        all_spans = []
        expected = {}
        for version, s in specs.items():
            tid = str(uuid.uuid4())
            for _ in range(s["n"]):
                all_spans.append(_span(
                    trace_id=tid,
                    version=version,
                    model=s["model"],
                    prompt_tokens=s["prompt_tokens"],
                    completion_tokens=s["completion_tokens"],
                    latency_ms=s["latency_ms"],
                ))
            expected[version] = {
                "total_prompt": s["prompt_tokens"] * s["n"],
                "total_completion": s["completion_tokens"] * s["n"],
                "mean_latency": s["latency_ms"],
            }
        return all_spans, expected

    def test_group_by_version_token_totals(self, tmp_path):
        spans, expected = self._build_spans_for_versions()
        tf = tmp_path / "trace.jsonl"
        _make_trace_file(tf, spans)

        db = tmp_path / "traces.db"
        inserted = sync([tf], canonical=db)
        assert inserted == len(spans)

        con = sqlite3.connect(str(db))
        rows = con.execute("""
            SELECT version,
                   SUM(prompt_tokens),
                   SUM(completion_tokens),
                   AVG(latency_ms)
            FROM spans
            GROUP BY version
            ORDER BY version
        """).fetchall()
        con.close()

        result = {int(r[0]): {"total_prompt": int(r[1]), "total_completion": int(r[2]),
                               "mean_latency": float(r[3])} for r in rows}

        for version, exp in expected.items():
            assert version in result
            got = result[version]
            assert got["total_prompt"] == exp["total_prompt"]
            assert got["total_completion"] == exp["total_completion"]
            assert abs(got["mean_latency"] - exp["mean_latency"]) < 1.0

    def test_cost_usd_computed_from_tokens(self, tmp_path):
        tf = tmp_path / "trace.jsonl"
        _make_trace_file(tf, [_span(trace_id=str(uuid.uuid4()), version=0,
                                   model="gpt-4o-mini", prompt_tokens=1000, completion_tokens=500)])
        db = tmp_path / "traces.db"
        sync([tf], canonical=db)

        con = sqlite3.connect(str(db))
        cost = con.execute("SELECT SUM(cost_usd) FROM spans WHERE version=0").fetchone()[0]
        con.close()

        expected_usd = (1000 * 0.000150 + 500 * 0.000600) / 1000.0
        assert cost is not None and cost > 0.0
        assert abs(cost - expected_usd) < 1e-7

    def test_multiple_trace_files_merged(self, tmp_path):
        spans_v0 = [_span(trace_id=str(uuid.uuid4()), version=0) for _ in range(4)]
        spans_v1 = [_span(trace_id=str(uuid.uuid4()), version=1) for _ in range(3)]
        tf0 = tmp_path / "trace_v0.jsonl"
        tf1 = tmp_path / "trace_v1.jsonl"
        _make_trace_file(tf0, spans_v0)
        _make_trace_file(tf1, spans_v1)

        db = tmp_path / "traces.db"
        total = sync([tf0, tf1], canonical=db)
        assert total == 7

        con = sqlite3.connect(str(db))
        count = con.execute("SELECT COUNT(*) FROM spans").fetchone()[0]
        con.close()
        assert count == 7
