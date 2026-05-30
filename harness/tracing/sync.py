"""harness/tracing/sync.py — AR² span/attempt sync + Raindrop Workshop UI push.

Pull model: Modal/local rollouts return trace.jsonl → merge into obs/ar2_workshop.db
(AR² SQL SOT for improve() read path) + OTLP push to Raindrop Workshop (:5899 UI).
Never write AR² custom tables into ~/.raindrop/raindrop_workshop.db — that breaks
Raindrop's migrations and leaves the UI empty.

Env:
  RAINDROP_WORKSHOP=1          — OTLP push + AR² SQL sync (default on)
  RAINDROP_WORKSHOP_DB_PATH    — AR² SQL path (default obs/ar2_workshop.db)
  RAINDROP_WORKSHOP_URL        — Workshop OTLP endpoint (default :5899)
  AR2_OBS_CACHE=1              — also mirror spans/attempts to obs/traces.db + obs/archive.db
"""
from __future__ import annotations

import json
import os
import sqlite3
import time
import urllib.error
import urllib.request
from pathlib import Path
from typing import Any

from harness.contracts import Attempt, Archive, Rollout
from harness.tracing.telemetry import parse_trace_file

_COST_TABLE: dict[str, tuple[float, float]] = {
    "gpt-5-codex":               (0.00125, 0.010),
    "gpt-4o-mini":               (0.000150, 0.000600),
    "gpt-4o":                    (0.0025,  0.010),
    "gpt-4-turbo":               (0.010,   0.030),
    "gpt-3.5-turbo":             (0.0005,  0.0015),
    "claude-opus-4-5":           (0.015,   0.075),
    "claude-sonnet-4-5":         (0.003,   0.015),
    "claude-haiku-4-5":          (0.00025, 0.00125),
    "claude-3-5-sonnet-20241022": (0.003,   0.015),
    "claude-3-5-haiku-20241022": (0.00025, 0.00125),
    "o1":                        (0.015,   0.060),
    "o1-mini":                   (0.003,   0.012),
    "o3-mini":                   (0.0011,  0.0044),
}
_DEFAULT_COST = (0.001, 0.003)

_SPANS_DDL = """
CREATE TABLE IF NOT EXISTS spans (
    trace_id        TEXT,
    version         INTEGER,
    candidate       TEXT,
    env_id          TEXT,
    split           TEXT,
    model           TEXT,
    prompt_tokens   INTEGER,
    completion_tokens INTEGER,
    cost_usd        REAL,
    latency_ms      REAL,
    tool_name       TEXT,
    tool_input      TEXT,
    ts              REAL
);
CREATE INDEX IF NOT EXISTS idx_trace ON spans(trace_id);
CREATE INDEX IF NOT EXISTS idx_version ON spans(version);
"""

_ATTEMPTS_DDL = """
CREATE TABLE IF NOT EXISTS attempts (
    version             INTEGER PRIMARY KEY,
    parent              INTEGER,
    train_reward        REAL,
    heldout_reward      REAL,
    hack_flags          TEXT,
    diff_summary        TEXT,
    source_ref          TEXT,
    trace_id            TEXT,
    cost_wall_seconds   REAL,
    cost_usd            REAL,
    train_rollouts_json TEXT,
    heldout_rollouts_json TEXT,
    synced_at           REAL
);
CREATE INDEX IF NOT EXISTS idx_attempt_parent ON attempts(parent);
"""

_SPANS_INSERT = """
INSERT INTO spans
  (trace_id, version, candidate, env_id, split, model,
   prompt_tokens, completion_tokens, cost_usd, latency_ms,
   tool_name, tool_input, ts)
VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?)
"""

_ATTEMPT_UPSERT = """
INSERT INTO attempts (
    version, parent, train_reward, heldout_reward, hack_flags, diff_summary,
    source_ref, trace_id, cost_wall_seconds, cost_usd,
    train_rollouts_json, heldout_rollouts_json, synced_at
) VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?)
ON CONFLICT(version) DO UPDATE SET
    parent=excluded.parent,
    train_reward=excluded.train_reward,
    heldout_reward=excluded.heldout_reward,
    hack_flags=excluded.hack_flags,
    diff_summary=excluded.diff_summary,
    source_ref=excluded.source_ref,
    trace_id=excluded.trace_id,
    cost_wall_seconds=excluded.cost_wall_seconds,
    cost_usd=excluded.cost_usd,
    train_rollouts_json=excluded.train_rollouts_json,
    heldout_rollouts_json=excluded.heldout_rollouts_json,
    synced_at=excluded.synced_at
"""


def _cost_usd(model: str, prompt_tokens: int, completion_tokens: int) -> float:
    model_key = model.lower()
    prices = next(
        (v for k, v in _COST_TABLE.items() if model_key.startswith(k)),
        _DEFAULT_COST,
    )
    return (prompt_tokens * prices[0] + completion_tokens * prices[1]) / 1000.0


def workshop_enabled() -> bool:
    return os.environ.get("RAINDROP_WORKSHOP", "1") == "1"


_AR2_WORKSHOP_DB = Path("obs/ar2_workshop.db")


def raindrop_db_path() -> Path:
    """AR² SQL mirror for spans + attempts (NOT Raindrop's ~/.raindrop DB)."""
    override = os.environ.get("RAINDROP_WORKSHOP_DB_PATH")
    if override:
        return Path(override).expanduser()
    return _AR2_WORKSHOP_DB


def workshop_db_path() -> Path:
    """Alias for raindrop_db_path() (back-compat)."""
    return raindrop_db_path()


def workshop_url() -> str:
    return os.environ.get("RAINDROP_WORKSHOP_URL", "http://127.0.0.1:5899").rstrip("/")


def obs_cache_enabled() -> bool:
    return os.environ.get("AR2_OBS_CACHE", "0") == "1"


def _open(path: Path) -> sqlite3.Connection:
    path.parent.mkdir(parents=True, exist_ok=True)
    con = sqlite3.connect(str(path))
    con.executescript(_SPANS_DDL + _ATTEMPTS_DDL)
    con.commit()
    return con


def ensure_raindrop_db() -> Path:
    path = raindrop_db_path()
    _open(path)
    return path


ensure_workshop_db = ensure_raindrop_db


def _row(span: dict[str, Any]) -> tuple:
    pt = int(span.get("prompt_tokens") or 0)
    ct = int(span.get("completion_tokens") or 0)
    model = str(span.get("model") or "")
    return (
        str(span.get("trace_id") or ""),
        int(span.get("version") or 0),
        str(span.get("candidate") or ""),
        str(span.get("env_id") or ""),
        str(span.get("split") or ""),
        model,
        pt,
        ct,
        _cost_usd(model, pt, ct),
        float(span.get("latency_ms") or 0.0),
        span.get("tool_name"),
        span.get("tool_input"),
        float(span.get("ts") or 0.0),
    )


def _attempt_row(attempt: Attempt) -> tuple:
    return (
        attempt.version,
        attempt.parent,
        attempt.train_reward,
        attempt.heldout_reward,
        json.dumps(attempt.hack_flags),
        attempt.diff_summary,
        attempt.source_ref,
        attempt.trace_id,
        attempt.cost.wall_seconds,
        attempt.cost.usd,
        json.dumps([r.model_dump() for r in attempt.train_rollouts]),
        json.dumps([r.model_dump() for r in attempt.heldout_rollouts]),
        time.time(),
    )


def sync_spans(
    trace_files: list[Path],
    *,
    db: Path | None = None,
) -> int:
    """Insert spans into target DB. Returns rows inserted."""
    if not trace_files:
        return 0
    if db is None and not workshop_enabled():
        return 0
    target = db or raindrop_db_path()
    con = _open(target)
    inserted = 0
    with con:
        for tf in trace_files:
            for span in parse_trace_file(tf):
                con.execute(_SPANS_INSERT, _row(span))
                inserted += 1
    con.close()
    return inserted


def sync(
    trace_files: list[Path],
    canonical: Path | None = None,
) -> int:
    """Back-compat: sync spans to canonical path (tests) or Raindrop SOT."""
    if canonical is not None:
        return sync_spans(trace_files, db=canonical)
    if not workshop_enabled():
        return 0
    return sync_spans(trace_files, db=raindrop_db_path())


def sync_attempt(
    attempt: Attempt,
    *,
    db: Path | None = None,
    canonical: Path | None = None,
) -> None:
    """Upsert Attempt into attempts table."""
    target = db or canonical
    if target is None:
        if not workshop_enabled():
            return
        target = raindrop_db_path()
    con = _open(target)
    with con:
        con.execute(_ATTEMPT_UPSERT, _attempt_row(attempt))
    con.close()
    if obs_cache_enabled() and target != Path("obs/archive.db"):
        cache = Path("obs/archive.db")
        con = _open(cache)
        with con:
            con.execute(_ATTEMPT_UPSERT, _attempt_row(attempt))
        con.close()


def sync_archive_jsonl(
    jsonl_path: Path = Path("obs/archive.jsonl"),
    *,
    db: Path | None = None,
) -> int:
    if not jsonl_path.exists():
        return 0
    count = 0
    with jsonl_path.open() as fh:
        for line in fh:
            line = line.strip()
            if line:
                sync_attempt(Attempt.model_validate_json(line), db=db)
                count += 1
    return count


def _attempt_from_db_row(row: tuple) -> Attempt:
    from harness.contracts import Budget

    (
        version, parent, train_reward, heldout_reward, hack_flags_json,
        diff_summary, source_ref, trace_id, cost_wall_seconds, cost_usd,
        train_rollouts_json, heldout_rollouts_json,
    ) = row
    train_rollouts = [
        Rollout.model_validate(r)
        for r in json.loads(train_rollouts_json or "[]")
    ]
    heldout_rollouts = [
        Rollout.model_validate(r)
        for r in json.loads(heldout_rollouts_json or "[]")
    ]

    return Attempt(
        version=int(version),
        parent=parent,
        train_reward=float(train_reward or 0.0),
        heldout_reward=float(heldout_reward or 0.0),
        hack_flags=json.loads(hack_flags_json or "[]"),
        diff_summary=diff_summary or "",
        source_ref=source_ref or "",
        trace_id=trace_id or "",
        cost=Budget(
            wall_seconds=float(cost_wall_seconds or 0.0),
            usd=cost_usd,
        ),
        train_rollouts=train_rollouts,
        heldout_rollouts=heldout_rollouts,
    )


def load_archive_from_raindrop(db: Path | None = None) -> Archive:
    """Read Attempt history from Raindrop SOT (for outer-loop improve() context)."""
    path = db or raindrop_db_path()
    if not path.exists():
        return Archive()
    con = sqlite3.connect(str(path))
    try:
        rows = con.execute(
            """
            SELECT version, parent, train_reward, heldout_reward, hack_flags,
                   diff_summary, source_ref, trace_id, cost_wall_seconds, cost_usd,
                   train_rollouts_json, heldout_rollouts_json
            FROM attempts ORDER BY version
            """
        ).fetchall()
    except sqlite3.OperationalError:
        return Archive()
    finally:
        con.close()
    archive = Archive()
    for row in rows:
        archive.add(_attempt_from_db_row(row))
    return archive


def span_trajectory_digest(
    archive: Archive,
    *,
    db: Path | None = None,
    max_spans: int = 40,
) -> str:
    """Recent agent/score spans for meta-agent triage (Raindrop spans table)."""
    if not archive.attempts:
        return ""
    versions = [a.version for a in archive.attempts]
    path = db or raindrop_db_path()
    if not path.exists():
        return ""
    placeholders = ",".join("?" * len(versions))
    con = sqlite3.connect(str(path))
    try:
        rows = con.execute(
            f"""
            SELECT version, env_id, split, model, tool_name, tool_input,
                   cost_usd, latency_ms, ts
            FROM spans
            WHERE version IN ({placeholders})
            ORDER BY ts DESC
            LIMIT ?
            """,
            (*versions, max_spans),
        ).fetchall()
    except sqlite3.OperationalError:
        return ""
    finally:
        con.close()
    if not rows:
        return "(no spans synced yet — run with RAINDROP_WORKSHOP=1)"
    lines: list[str] = []
    for ver, env_id, split, model, tool, tool_input, cost, latency, ts in reversed(rows):
        tip = (tool_input or "")[:120].replace("\n", " ")
        lines.append(
            f"  v{ver} {split}/{env_id} {tool or 'span'} "
            f"model={model} cost=${float(cost or 0):.4f} "
            f"lat={float(latency or 0):.0f}ms | {tip}"
        )
    return "\n".join(lines)


def prepare_improve_context(
    archive: Archive,
    *,
    db: Path | None = None,
) -> tuple[Archive, str]:
    """Raindrop read path: reload archive + span digest before improve().

    Only replaces in-memory archive when Raindrop rows match the current run
    (same latest version + heldout reward) so unrelated workshop history is ignored.
    """
    if not workshop_enabled() or not archive.attempts:
        return archive, ""
    rd = load_archive_from_raindrop(db=db)
    if len(rd.attempts) >= len(archive.attempts):
        latest_mem = archive.attempts[-1]
        latest_rd = rd.attempts[len(archive.attempts) - 1]
        if (
            latest_rd.version == latest_mem.version
            and abs(latest_rd.heldout_reward - latest_mem.heldout_reward) < 1e-9
        ):
            merged = Archive()
            for a in rd.attempts[: len(archive.attempts)]:
                merged.add(a)
            archive = merged
    return archive, span_trajectory_digest(archive, db=db)


def _otlp_trace_id(trace_id: str) -> str:
    """32-char hex trace id for OTLP (strip UUID dashes)."""
    hex_id = trace_id.replace("-", "")
    if len(hex_id) >= 32:
        return hex_id[:32]
    return hex_id.ljust(32, "0")


def push_spans_live(spans: list[dict[str, Any]]) -> int:
    """OTLP JSON push to Raindrop Workshop UI. Returns spans ingested (best-effort)."""
    if not workshop_enabled() or not spans:
        return 0

    otlp_spans: list[dict[str, Any]] = []
    for i, span in enumerate(spans):
        ts_ms = float(span.get("ts") or time.time()) * 1000.0
        latency_ms = float(span.get("latency_ms") or 0.0)
        ts_ns = int(ts_ms * 1e6)
        tool = span.get("tool_name") or "score"
        model = str(span.get("model") or "")
        otlp_spans.append({
            "traceId": _otlp_trace_id(str(span.get("trace_id") or uuid_hex(i))),
            "spanId": f"{i + 1:016x}",
            "name": tool if tool != "agent" else (model or "llm"),
            "kind": 1,
            "startTimeUnixNano": str(ts_ns),
            "endTimeUnixNano": str(ts_ns + int(latency_ms * 1e6)),
            "attributes": [
                {"key": "model", "value": {"stringValue": model}},
                {"key": "env_id", "value": {"stringValue": str(span.get("env_id") or "")}},
                {"key": "split", "value": {"stringValue": str(span.get("split") or "")}},
                {"key": "ar2.version", "value": {"intValue": str(span.get("version") or 0)}},
                {"key": "ar2.candidate", "value": {"stringValue": str(span.get("candidate") or "")[:200]}},
                {"key": "ar2.tool_input", "value": {"stringValue": str(span.get("tool_input") or "")[:500]}},
            ],
        })

    payload = json.dumps({
        "resourceSpans": [{
            "resource": {"attributes": [{"key": "service.name", "value": {"stringValue": "ar2"}}]},
            "scopeSpans": [{"spans": otlp_spans}],
        }]
    }).encode()

    req = urllib.request.Request(
        f"{workshop_url()}/v1/traces",
        data=payload,
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    try:
        with urllib.request.urlopen(req, timeout=5.0) as resp:
            if 200 <= resp.status < 300:
                body = json.loads(resp.read().decode())
                return int(body.get("spansIngested", len(spans)))
    except (urllib.error.URLError, TimeoutError, OSError, json.JSONDecodeError, ValueError):
        pass
    return 0


def uuid_hex(seed: int) -> str:
    return f"{seed:032x}"


def cost_per_version(db: Path | None = None) -> dict[int, float]:
    """Sum cost_usd from Raindrop spans grouped by version."""
    path = db or raindrop_db_path()
    if not path.exists():
        return {}
    con = sqlite3.connect(str(path))
    try:
        rows = con.execute(
            "SELECT version, SUM(cost_usd) FROM spans GROUP BY version"
        ).fetchall()
        return {int(r[0]): float(r[1] or 0.0) for r in rows}
    except sqlite3.OperationalError:
        return {}
    finally:
        con.close()


def mirror_to_workshop(
    traces_canonical: Path = Path("obs/traces.db"),
    archive_canonical: Path = Path("obs/archive.db"),
) -> dict[str, int]:
    """Deprecated: Raindrop is already SOT. Copies obs cache → Raindrop if needed."""
    if not workshop_enabled():
        return {"spans": 0, "attempts": 0}
    rd = raindrop_db_path()
    counts = {"spans": 0, "attempts": 0}
    if traces_canonical.exists() and traces_canonical != rd:
        counts["spans"] = sync_spans([], db=rd)  # no-op placeholder
        con_src = sqlite3.connect(str(traces_canonical))
        rows = con_src.execute("SELECT * FROM spans").fetchall()
        con_src.close()
        if rows:
            con = _open(rd)
            existing = set(con.execute("SELECT trace_id, ts FROM spans").fetchall())
            new = [r for r in rows if (r[0], r[12]) not in existing]
            with con:
                con.executemany(_SPANS_INSERT, new)
            con.close()
            counts["spans"] = len(new)
    return counts


def sync_all(
    *,
    trace_files: list[Path] | None = None,
    attempt: Attempt | None = None,
    archive_jsonl: Path | None = None,
    live_push: bool = True,
    traces_db: Path | None = Path("obs/traces.db"),
    archive_db: Path | None = Path("obs/archive.db"),
) -> dict[str, Any]:
    """Sync rollouts → Raindrop SOT and/or local obs/*.db (+ optional OTLP UI push)."""
    result: dict[str, Any] = {
        "spans_inserted": 0,
        "attempts_synced": 0,
        "otlp_pushed": 0,
        "raindrop_db": str(raindrop_db_path()),
        "local_traces_db": str(traces_db) if traces_db else None,
        "local_archive_db": str(archive_db) if archive_db else None,
    }
    workshop = workshop_enabled()

    if workshop:
        ensure_raindrop_db()
        if trace_files:
            result["spans_inserted"] = sync_spans(trace_files)
            if live_push:
                pushed = 0
                for tf in trace_files:
                    pushed += push_spans_live(parse_trace_file(tf))
                result["otlp_pushed"] = pushed
        if attempt is not None:
            sync_attempt(attempt)
            result["attempts_synced"] = 1
        elif archive_jsonl is not None and archive_jsonl.exists():
            result["attempts_synced"] = sync_archive_jsonl(archive_jsonl)

    if traces_db is not None and trace_files:
        local_spans = sync_spans(trace_files, db=traces_db)
        if not workshop:
            result["spans_inserted"] = local_spans

    if archive_db is not None:
        if attempt is not None:
            sync_attempt(attempt, db=archive_db)
            if not workshop:
                result["attempts_synced"] = 1
        elif archive_jsonl is not None and archive_jsonl.exists() and not workshop:
            result["attempts_synced"] = sync_archive_jsonl(archive_jsonl, db=archive_db)

    if obs_cache_enabled() and workshop:
        if traces_db is not None and trace_files:
            sync_spans(trace_files, db=traces_db)
        if archive_db is not None and attempt is not None:
            sync_attempt(attempt, db=archive_db)

    return result
