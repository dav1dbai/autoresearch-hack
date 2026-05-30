"""End-to-end integration (offline, zero billing).

The REAL outer loop + Archive + dashboard compose; only the compute layer
(sandbox / coding agents) is stubbed. Hack review is deferred to the outer-loop
agent — harness evaluate() leaves Attempt.hack_flags empty.

Proves: the loop turns (archive grows across versions), held-out reward drives
selection, and report.html renders.
"""
from __future__ import annotations

import tempfile
from pathlib import Path
from types import SimpleNamespace

import pytest

from harness.loop.archive import outer_curve, save
from harness.contracts import Budget, Rollout
from harness.loop.outer import drive
from obs.dashboard import build_report


def _stub_score_repo(ar_dir, envs, budget, **kwargs):
    """Reward rises with version; v2's candidate overfits — held-out collapses."""
    marker = Path(ar_dir) / "_v.txt"
    v = int(marker.read_text()) if marker.exists() else 0
    rolls = []
    for e in envs:
        r = 0.4 + 0.1 * v
        if v == 2 and e.split == "heldout":
            r = 0.1  # train climbs but held-out tanks -> overfit signature
        rolls.append(Rollout(env_id=e.id, split=e.split, rewards=[r],
                             final_reward=r, cost=budget, trace_id=f"t{v}"))
    return rolls


class _FakeAR:
    def improve(self, archive, budget, spawn):
        v = len(archive.attempts)  # next version index
        d = Path(tempfile.mkdtemp(prefix="ar_v_"))
        (d / "_v.txt").write_text(str(v))
        return d


@pytest.mark.smoke
def test_loop_turns_and_report_renders(tmp_path):
    ar0 = tmp_path / "ar0"
    ar0.mkdir()  # no marker -> version 0

    train = [SimpleNamespace(id="train_env", split="train")]
    heldout = [SimpleNamespace(id="heldout_env", split="heldout")]

    archive = drive(
        ar0, train, heldout, Budget(wall_seconds=1.0, max_concurrency=1),
        K=2, M=1,
        score_repo=_stub_score_repo,
        load_ar=lambda source_ref: _FakeAR(),
    )

    by_v = {a.version: a for a in archive.attempts}
    assert 0 in by_v and len(archive.attempts) >= 3      # loop turned across generations
    assert all(not a.hack_flags for a in archive.attempts)
    best = archive.best()
    assert best is not None and best.version == 1
    assert best.heldout_reward == pytest.approx(0.5)

    curve = outer_curve(archive)
    bests = [c[1] for c in curve]
    for a, b in zip(bests, bests[1:]):
        assert b >= a - 1e-9

    jsonl = tmp_path / "archive.jsonl"
    save(archive, jsonl)
    report = tmp_path / "report.html"
    build_report(jsonl, tmp_path / "traces.db", report)  # missing db degrades gracefully
    assert report.exists()
    assert "heldout" in report.read_text().lower()


def test_score_repo_inner_loop_accumulates_rewards(tmp_path, monkeypatch):
    """D-00 regression: with envs returning done=False, a real score_repo rollout
    records >=2 inner rewards (baseline + >=1 agent edit) — not just the baseline.

    The Modal sandbox + out-of-process referee are mocked (their isolation is
    covered elsewhere); this asserts the inner loop actually turns through the
    real ar/solve and a real Env. Would fail under the old done=True envs.
    """
    import harness.runtime.score as sr
    from envs.matmul import MatmulEnv

    # Trivial always-correct inner "agent": append a comment to kernel.py (cwd=workdir).
    agent = tmp_path / "agent.sh"
    agent.write_text('echo "# noop edit $1" >> kernel.py\n')
    monkeypatch.setenv("INNER_AGENT_CMD", f"bash {agent}")

    # Neutralize out-of-process referee; D-00 is about the loop turning.
    monkeypatch.setenv("MATMUL_STUB", "1")
    monkeypatch.setattr("harness.runtime.rollout.make_referee", lambda env: env.score)

    ar_dir = Path(__file__).parent.parent / "ar"
    env = MatmulEnv(split="train", M=64, N=64, K=64, reps=3)
    rolls = sr.score_repo(ar_dir, [env], Budget(wall_seconds=10.0, max_concurrency=1))

    assert len(rolls) == 1
    assert len(rolls[0].rewards) >= 2  # baseline + >=1 inner iteration
