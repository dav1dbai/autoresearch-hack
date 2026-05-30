"""Smoke E2E for the matmul inner task.

Runs a minimal outer-loop with stubbed compute (no billing, no GPU, no Codex),
then exercises the full Raindrop telemetry pipeline:
  trace.jsonl -> db_sync.sync -> obs/traces.db -> GROUP BY aggregation
  + obs.dashboard.build_report

Usage:
    python -m smoke.e2e_matmul          # full offline run
    INNER_AGENT_CMD="codex exec" MUTATE_AGENT_CMD="codex exec" python -m smoke.e2e_matmul
        # ^ real Codex inner agent (bills tokens)

For a real GPU run on Vast.ai / Modal:
  1. Set MATMUL_STUB=0 and MATMUL_RUNNER=gpu (or implement a custom runner and inject
     it into MatmulEnv(..., runner=my_gpu_runner)).
  2. Set INNER_AGENT_CMD / MUTATE_AGENT_CMD to your Codex / Claude Code command.
  3. Supply OPENAI_API_KEY / ANTHROPIC_API_KEY.
  4. Set MATMUL_TARGET_GFLOPS to a realistic GPU target (e.g. 500.0 for A100).
"""
from __future__ import annotations

import json
import os
import tempfile
import time
import uuid
from pathlib import Path

from envs.matmul import MatmulEnv
from harness.contracts import Archive, Attempt, Budget, Rollout, Submission
from harness.tracing import sync as db_sync
from harness.loop.archive import save as save_archive
from obs.dashboard import build_report

# ---------------------------------------------------------------------------
# Stub score_repo and load_ar — deterministic, no network, no billing
# ---------------------------------------------------------------------------

def _stub_score_repo(ar_dir: Path, envs: list, budget: Budget, **kwargs) -> list[Rollout]:
    """Deterministic stub: score the current kernel in each env, emit synthetic spans."""
    rollouts: list[Rollout] = []
    for env in envs:
        task = env.reset()
        # Use the kernel that is already in the workdir (starter kernel → correct numpy).
        sub = Submission(workdir=task.workdir)
        result = env.score(sub)
        trace_id = str(uuid.uuid4())
        rollouts.append(Rollout(
            env_id=env.id,
            split=env.split,
            rewards=[result.reward],
            final_reward=result.reward,
            cost=Budget(wall_seconds=0.05, tokens=10),
            trace_id=trace_id,
        ))
    return rollouts


class _StubArModule:
    """Stub ar/ module: improve() returns the same ar_dir (no actual mutation)."""

    def __init__(self, ar_dir: Path) -> None:
        self._ar_dir = ar_dir

    def improve(self, archive: Archive, budget: Budget, spawn) -> Path:
        return self._ar_dir


def _stub_load_ar(source_ref: str):
    return _StubArModule(Path(source_ref))

# ---------------------------------------------------------------------------
# Real agent mode (gated behind env vars — not invoked here)
# ---------------------------------------------------------------------------
# To use real Codex/Claude Code, set:
#   INNER_AGENT_CMD=<codex exec or claude code cmd>
#   MUTATE_AGENT_CMD=<same>
# drive() will call ar/entrypoint.solve and improve via those commands.

_USE_REAL_AGENT = bool(os.environ.get("INNER_AGENT_CMD"))

# ---------------------------------------------------------------------------
# Telemetry helpers
# ---------------------------------------------------------------------------

def _write_synthetic_spans(
    trace_path: Path,
    *,
    trace_id: str,
    version: int,
    env_id: str,
    split: str,
    model: str,
    prompt_tokens: int,
    completion_tokens: int,
    latency_ms: float,
    n_spans: int = 3,
) -> None:
    """Write n_spans representative span lines to trace_path (jsonl)."""
    with trace_path.open("a") as fh:
        for i in range(n_spans):
            span = {
                "trace_id": trace_id,
                "version": version,
                "candidate": f"cand-{version}",
                "env_id": env_id,
                "split": split,
                "model": model,
                "prompt_tokens": prompt_tokens,
                "completion_tokens": completion_tokens,
                "latency_ms": latency_ms + i * 5.0,
                "tool_name": "score" if i == n_spans - 1 else "edit",
                "tool_input": f"iteration_{i}",
                "ts": time.time() + i * 0.1,
            }
            fh.write(json.dumps(span) + "\n")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def run(out_dir: Path | None = None) -> None:
    os.environ.setdefault("MATMUL_STUB", "1")  # ensure offline unless explicitly unset

    tmp_root = Path(tempfile.mkdtemp(prefix="e2e_matmul_")) if out_dir is None else out_dir
    tmp_root.mkdir(parents=True, exist_ok=True)

    db_path = tmp_root / "traces.db"
    archive_path = tmp_root / "archive.jsonl"
    report_path = tmp_root / "report.html"

    print(f"[e2e] workdir: {tmp_root}")

    # --- Build env pools -------------------------------------------------------
    train_envs = [
        MatmulEnv(split="train", M=64, N=64, K=64),
        MatmulEnv(split="train", M=128, N=128, K=128),
    ]
    heldout_envs = [
        MatmulEnv(split="heldout", M=96, N=96, K=96),
    ]

    # --- Drive outer loop (K=1 generation) -------------------------------------
    from harness.loop.outer import drive

    ar0_dir = Path(__file__).parent.parent / "ar"

    archive = drive(
        ar0_dir=ar0_dir,
        train=train_envs,
        heldout=heldout_envs,
        budget=Budget(wall_seconds=30.0),
        K=1,
        score_repo=_stub_score_repo,
        load_ar=_stub_load_ar,
        _persist_path=archive_path,
    )

    best = archive.best()
    print(
        f"[e2e] archive: {len(archive.attempts)} attempt(s), "
        f"best heldout_reward={f'{best.heldout_reward:.4f}' if best is not None else 'n/a'}"
    )

    # --- Synthesize per-version telemetry spans ---------------------------------
    # One trace.jsonl per attempt — mirrors what score_repo would pull from sandboxes.
    trace_files: list[Path] = []
    for attempt in archive.attempts:
        tf = tmp_root / f"trace_v{attempt.version}.jsonl"
        # Each trace_id in the attempt may be comma-joined from multiple rollouts.
        trace_ids = attempt.trace_id.split(",") if attempt.trace_id else [str(uuid.uuid4())]
        for i, tid in enumerate(trace_ids):
            _write_synthetic_spans(
                tf,
                trace_id=tid.strip(),
                version=attempt.version,
                env_id=f"matmul-v{attempt.version}-env{i}",
                split="train" if i % 2 == 0 else "heldout",
                model="gpt-4o-mini",
                prompt_tokens=200 + attempt.version * 50,
                completion_tokens=80 + attempt.version * 20,
                latency_ms=120.0 + attempt.version * 15.0,
                n_spans=3,
            )
        trace_files.append(tf)

    # --- Sync to canonical DB ---------------------------------------------------
    inserted = db_sync.sync(trace_files, canonical=db_path)
    print(f"[e2e] db_sync: inserted {inserted} span(s) into {db_path}")

    # --- Raindrop aggregation: GROUP BY version ---------------------------------
    import sqlite3
    con = sqlite3.connect(str(db_path))
    rows = con.execute("""
        SELECT
            version,
            SUM(prompt_tokens)     AS total_prompt,
            SUM(completion_tokens) AS total_completion,
            SUM(cost_usd)          AS total_cost,
            AVG(latency_ms)        AS mean_latency_ms
        FROM spans
        GROUP BY version
        ORDER BY version
    """).fetchall()
    con.close()

    print("\n[e2e] Raindrop aggregation (GROUP BY version):")
    print(f"  {'version':>7}  {'prompt_tok':>10}  {'compl_tok':>9}  {'cost_usd':>10}  {'mean_lat_ms':>11}")
    for version, total_prompt, total_completion, total_cost, mean_lat in rows:
        print(
            f"  {version:>7}  {total_prompt:>10}  {total_completion:>9}"
            f"  {total_cost:>10.6f}  {mean_lat:>11.1f}"
        )

    # --- Build dashboard --------------------------------------------------------
    report = build_report(archive_path=archive_path, db_path=db_path, out=report_path)
    print(f"\n[e2e] dashboard: {report}")
    print("[e2e] DONE")


if __name__ == "__main__":
    run()
