"""Offline tests for harness/outer_loop.py and harness/archive.py.

All network/Modal/agent dependencies are replaced with deterministic fakes.
"""
from __future__ import annotations

import tempfile
from pathlib import Path

import pytest

from harness.contracts import Archive, Attempt, Budget, Rollout
from harness.loop.archive import save, load, outer_curve
from harness.loop.evaluate import attempt_from_rollouts
from harness.loop.outer import evaluate, drive


# ── shared budget ────────────────────────────────────────────────────────────

BUDGET = Budget(wall_seconds=1.0)


# ── fake Env (satisfies Env protocol without import) ─────────────────────────

class FakeEnv:
    def __init__(self, env_id: str, split: str):
        self.id = env_id
        self.split = split

    def reset(self):
        from harness.contracts import TaskSpec
        return TaskSpec(
            env_id=self.id,
            split=self.split,
            prompt="optimize",
            workdir=Path("/tmp"),
        )

    def score(self, sub):
        from harness.contracts import StepResult
        return StepResult(reward=0.5)


TRAIN_ENVS = [FakeEnv("env-train-1", "train"), FakeEnv("env-train-2", "train")]
HELDOUT_ENVS = [FakeEnv("env-ho-1", "heldout")]


# ── fake score_repo factories ─────────────────────────────────────────────────

def make_score_repo(train_reward: float, heldout_reward: float):
    """Returns a score_repo stub that gives fixed rewards regardless of split."""
    def _score_repo(ar_dir: Path, envs, budget: Budget, **kwargs) -> list[Rollout]:
        results = []
        for env in envs:
            reward = train_reward if env.split == "train" else heldout_reward
            results.append(Rollout(
                env_id=env.id,
                split=env.split,
                rewards=[reward],
                final_reward=reward,
                cost=Budget(wall_seconds=0.1),
                trace_id=f"trace-{env.id}",
            ))
        return results
    return _score_repo


def clean_score_repo(ar_dir: Path, envs, budget: Budget, **kwargs) -> list[Rollout]:
    """A candidate with decent, coherent train/heldout rewards."""
    return make_score_repo(0.6, 0.6)(ar_dir, envs, budget)


def hacked_score_repo(ar_dir: Path, envs, budget: Budget, **kwargs) -> list[Rollout]:
    """Train ≫ heldout — overfit signature without harness hack detection."""
    return make_score_repo(0.99, 0.1)(ar_dir, envs, budget)


# ── fake load_ar ──────────────────────────────────────────────────────────────

class _FakeArObj:
    """improve() returns a version snapshot root (smoke-checkable)."""
    def __init__(self, quality: float):
        self.quality = quality

    def improve(self, archive: Archive, budget: Budget, spawn) -> Path:
        import os

        vr = os.environ.get("AR2_VERSION_ROOT")
        d = Path(vr) if vr else Path(tempfile.mkdtemp())
        ar = d / "ar"
        ar.mkdir(parents=True, exist_ok=True)
        ep = ar / "entrypoint.py"
        if not ep.exists():
            ep.write_text("def solve(t,b,s,sp): pass\ndef improve(a,b,sp): pass\n")
        rt = d / "harness" / "runtime"
        rt.mkdir(parents=True, exist_ok=True)
        if not (rt / "score.py").exists():
            (rt / "score.py").write_text("# stub\n")
        return d


def make_load_ar(quality: float):
    obj = _FakeArObj(quality)
    def _load_ar(source_ref: str):
        return obj
    return _load_ar


# ─────────────────────────────────────────────────────────────────────────────
# Tests
# ─────────────────────────────────────────────────────────────────────────────

class TestEvaluate:
    def test_basic_fields(self, tmp_path):
        a = evaluate(
            tmp_path, TRAIN_ENVS, HELDOUT_ENVS, BUDGET,
            score_repo=clean_score_repo,
            version=1,
            parent=0,
        )
        assert a.version == 1
        assert a.parent == 0
        assert a.train_reward == pytest.approx(0.6)
        assert a.heldout_reward == pytest.approx(0.6)
        assert a.hack_flags == []

    def test_overfit_rewards_recorded(self, tmp_path):
        a = evaluate(
            tmp_path, TRAIN_ENVS, HELDOUT_ENVS, BUDGET,
            score_repo=hacked_score_repo,
        )
        assert a.train_reward > a.heldout_reward + 0.3

    def test_mean_aggregation(self, tmp_path):
        # Two train envs at 0.6 each -> mean 0.6; one heldout at 0.6 -> 0.6
        a = evaluate(
            tmp_path, TRAIN_ENVS, HELDOUT_ENVS, BUDGET,
            score_repo=clean_score_repo,
        )
        assert a.train_reward == pytest.approx(0.6)
        assert a.heldout_reward == pytest.approx(0.6)

    def test_rollouts_persisted_on_attempt(self, tmp_path):
        a = evaluate(
            tmp_path, TRAIN_ENVS, HELDOUT_ENVS, BUDGET,
            score_repo=clean_score_repo,
            sync_traces=None,
        )
        assert len(a.train_rollouts) == len(TRAIN_ENVS)
        assert len(a.heldout_rollouts) == len(HELDOUT_ENVS)
        assert all(r.final_reward == pytest.approx(0.6) for r in a.train_rollouts)


class TestAttemptFromRollouts:
    def test_roundtrip_fields(self):
        train = [
            Rollout(env_id="t1", split="train", rewards=[0.1, 0.2], final_reward=0.2,
                    cost=BUDGET, trace_id="a"),
        ]
        held = [
            Rollout(env_id="h1", split="heldout", rewards=[0.3], final_reward=0.3,
                    cost=BUDGET, trace_id="b"),
        ]
        attempt = attempt_from_rollouts(
            train, held, hack_flags=[], version=2, parent=1, diff_summary="x", source_ref="/ar",
        )
        assert attempt.version == 2
        assert attempt.train_reward == pytest.approx(0.2)
        assert attempt.heldout_reward == pytest.approx(0.3)
        assert attempt.train_rollouts == train
        assert attempt.heldout_rollouts == held
        assert "a" in attempt.trace_id and "b" in attempt.trace_id
    def test_save_load_empty(self, tmp_path):
        path = tmp_path / "archive.jsonl"
        save(Archive(), path)
        loaded = load(path)
        assert loaded.attempts == []

    def test_save_load_attempts(self, tmp_path):
        archive = Archive()
        for i in range(3):
            archive.add(Attempt(
                version=i,
                parent=None if i == 0 else i - 1,
                train_reward=0.5 + i * 0.1,
                heldout_reward=0.4 + i * 0.1,
                hack_flags=[],
                cost=BUDGET,
                trace_id=f"t{i}",
                source_ref=f"ref{i}",
            ))
        path = tmp_path / "archive.jsonl"
        save(archive, path)
        loaded = load(path)
        assert len(loaded.attempts) == 3
        for orig, rt in zip(archive.attempts, loaded.attempts):
            assert orig.version == rt.version
            assert orig.train_reward == pytest.approx(rt.train_reward)
            assert orig.heldout_reward == pytest.approx(rt.heldout_reward)
            assert orig.source_ref == rt.source_ref

    def test_save_load_default_path(self, tmp_path, monkeypatch):
        """save/load with default path (obs/archive.jsonl) relative to cwd."""
        monkeypatch.chdir(tmp_path)
        archive = Archive()
        archive.add(Attempt(
            version=0, train_reward=0.5, heldout_reward=0.5,
            hack_flags=[], cost=BUDGET, trace_id="x",
        ))
        save(archive)
        loaded = load()
        assert len(loaded.attempts) == 1


class TestOuterCurve:
    def _make_archive(self, specs: list[tuple[float, list[str]]]) -> Archive:
        a = Archive()
        for i, (hr, flags) in enumerate(specs):
            a.add(Attempt(
                version=i, train_reward=hr, heldout_reward=hr,
                hack_flags=flags, cost=BUDGET, trace_id=f"t{i}",
            ))
        return a

    def test_monotonic_non_decreasing(self):
        archive = self._make_archive([(0.3, []), (0.5, []), (0.4, []), (0.7, [])])
        curve = outer_curve(archive)
        bests = [c[1] for c in curve]
        for a, b in zip(bests, bests[1:]):
            assert b >= a, f"curve not non-decreasing: {bests}"

    def test_hacked_does_not_advance_best(self):
        # v0=0.3 clean, v1=0.9 hacked, v2=0.5 clean
        archive = self._make_archive([(0.3, []), (0.9, ["gap"]), (0.5, [])])
        curve = outer_curve(archive)
        # After v1 (hacked), best_clean should still be 0.3, not 0.9
        assert curve[1][2] is True       # v1 is hacked
        assert curve[1][1] == pytest.approx(0.3)   # best-so-far not advanced
        assert curve[2][1] == pytest.approx(0.5)   # v2 clean advances to 0.5

    def test_all_clean(self):
        archive = self._make_archive([(0.1, []), (0.2, []), (0.3, [])])
        curve = outer_curve(archive)
        assert [c[2] for c in curve] == [False, False, False]
        assert [c[1] for c in curve] == pytest.approx([0.1, 0.2, 0.3])

    def test_empty_archive(self):
        assert outer_curve(Archive()) == []


class TestDrive:
    @pytest.fixture(autouse=True)
    def _seed_ar0(self, tmp_path):
        ep = tmp_path / "entrypoint.py"
        if not ep.exists():
            ep.write_text("def solve(t,b,s,sp): pass\ndef improve(a,b,sp): pass\n")

    def test_v0_seeded(self, tmp_path):
        archive = drive(
            tmp_path, TRAIN_ENVS, HELDOUT_ENVS, BUDGET,
            K=0,
            score_repo=clean_score_repo,
            load_ar=make_load_ar(0.6),
        )
        assert len(archive.attempts) == 1
        assert archive.attempts[0].version == 0
        assert archive.attempts[0].parent is None

    def test_k_generations_produces_candidates(self, tmp_path):
        # 2 generations, one improve per selected parent.
        archive = drive(
            tmp_path, TRAIN_ENVS, HELDOUT_ENVS, BUDGET,
            K=2,
            score_repo=clean_score_repo,
            load_ar=make_load_ar(0.6),
        )
        assert len(archive.attempts) >= 3  # v0 + at least 1 per generation

    def test_overfit_candidate_not_best_by_reward(self, tmp_path):
        """A train≫heldout candidate loses on held-out reward, not hack flags."""
        call_count = {"n": 0}

        def mixed_score_repo(ar_dir: Path, envs, budget: Budget, **kwargs) -> list[Rollout]:
            call_count["n"] += 1
            if call_count["n"] <= 2:
                return clean_score_repo(ar_dir, envs, budget)
            return hacked_score_repo(ar_dir, envs, budget)

        archive = drive(
            tmp_path, TRAIN_ENVS, HELDOUT_ENVS, BUDGET,
            K=1,
            score_repo=mixed_score_repo,
            load_ar=make_load_ar(0.6),
        )
        best = archive.best()
        assert best is not None
        assert best.heldout_reward == pytest.approx(0.6)

    def test_outer_curve_monotonic_over_drive(self, tmp_path):
        """outer_curve on the drive output is non-decreasing in best-heldout."""
        # Each successive candidate gets a slightly better heldout score.
        gen_count = {"n": 0}

        def improving_score_repo(ar_dir, envs, budget, **kwargs):
            results = []
            for env in envs:
                if env.split == "train":
                    r = 0.5 + gen_count["n"] * 0.05
                else:
                    r = 0.4 + gen_count["n"] * 0.05
                    gen_count["n"] += 1
                results.append(Rollout(
                    env_id=env.id, split=env.split,
                    rewards=[r], final_reward=r,
                    cost=Budget(wall_seconds=0.1),
                    trace_id=f"t-{env.id}-{gen_count['n']}",
                ))
            return results

        archive = drive(
            tmp_path, TRAIN_ENVS, HELDOUT_ENVS, BUDGET,
            K=2,
            score_repo=improving_score_repo,
            load_ar=make_load_ar(0.6),
        )
        curve = outer_curve(archive)
        bests = [c[1] for c in curve]
        for a, b in zip(bests, bests[1:]):
            assert b >= a - 1e-9, f"outer curve not non-decreasing: {bests}"

    def test_persistence(self, tmp_path):
        persist_path = tmp_path / "test_archive.jsonl"
        drive(
            tmp_path, TRAIN_ENVS, HELDOUT_ENVS, BUDGET,
            K=1,
            score_repo=clean_score_repo,
            load_ar=make_load_ar(0.6),
            _persist_path=persist_path,
        )
        assert persist_path.exists()
        loaded = load(persist_path)
        assert len(loaded.attempts) >= 1


# ---------------------------------------------------------------------------
# Matmul orchestration stub (from former test_e2e_matmul.py)
# ---------------------------------------------------------------------------

class TestOrchestrationStub:
    def test_drive_grows_archive(self, tmp_path):
        import os
        import uuid
        from pathlib import Path

        os.environ.setdefault("MATMUL_STUB", "1")
        from envs.matmul import MatmulEnv
        from harness.contracts import Budget, Rollout, Submission

        def _make_env(split="train", **kw) -> MatmulEnv:
            return MatmulEnv(split=split, M=32, N=32, K=32, **kw)

        train_envs = [_make_env("train")]
        heldout_envs = [_make_env("heldout")]

        def _score_repo(ar_dir, envs, budget, **kwargs):
            rollouts = []
            for env in envs:
                task = env.reset()
                sub = Submission(workdir=task.workdir)
                result = env.score(sub)
                rollouts.append(Rollout(
                    env_id=env.id,
                    split=env.split,
                    rewards=[result.reward],
                    final_reward=result.reward,
                    cost=Budget(wall_seconds=0.01),
                    trace_id=str(uuid.uuid4()),
                ))
            return rollouts

        class _StubAr:
            def improve(self, archive, budget, spawn):
                return Path(__file__).resolve().parents[2] / "ar"

        def _load_ar(source_ref):
            return _StubAr()

        ar0 = Path(__file__).resolve().parents[2] / "ar"
        archive = drive(
            ar0_dir=ar0,
            train=train_envs,
            heldout=heldout_envs,
            budget=Budget(wall_seconds=10.0),
            K=1,
            score_repo=_score_repo,
            load_ar=_load_ar,
        )

        assert len(archive.attempts) >= 2
        for a in archive.attempts:
            assert 0.0 <= a.train_reward <= 1.0
            assert 0.0 <= a.heldout_reward <= 1.0
