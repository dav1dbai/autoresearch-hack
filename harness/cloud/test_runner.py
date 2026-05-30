"""Offline tests for harness/cloud/runner.py Modal parallel fan-out."""
from __future__ import annotations

import sys
import threading
import uuid
from unittest.mock import MagicMock, patch

import pytest

from harness.cloud.conftest import (
    BUDGET,
    _FakeEnv,
    _fresh_modal_runner,
    _make_ar_dir,
    _make_rollout,
)
from harness.contracts import Budget, Rollout


class TestModalFanOut:
    def test_workshop_trace_ids_are_valid_hex(self, monkeypatch):
        mr = _fresh_modal_runner(monkeypatch)

        trace_id = mr._new_workshop_trace_id()

        assert len(trace_id) == 32
        assert mr._valid_workshop_trace_id(trace_id) == trace_id
        with pytest.raises(ValueError):
            mr._valid_workshop_trace_id("raindrop-k1-mcp")

    def test_score_repo_modal_calls_starmap(self, tmp_path, monkeypatch):
        monkeypatch.setenv("AR2_BACKEND", "modal")
        monkeypatch.setenv("MODAL_PROFILE", "autoresearch-hack")

        mr = _fresh_modal_runner(monkeypatch)
        envs = [_FakeEnv(f"env-{i}") for i in range(3)]
        captured_inputs: list[list] = []

        def fake_starmap(inputs):
            captured_inputs.extend(list(inputs))
            return [_make_rollout(f"env-{i}").model_dump() for i in range(3)]

        mr.run_rollout.starmap = MagicMock(side_effect=fake_starmap)

        with patch.object(mr, "upload_snapshot", return_value="snap_abc"):
            rollouts = mr.run_rollouts_parallel(_make_ar_dir(tmp_path), envs, BUDGET)

        assert len(captured_inputs) == 3
        for item in captured_inputs:
            snap_ref, env_spec, budget_dict, inject_env, trace_id = item
            assert snap_ref == "snap_abc"
            assert "id" in env_spec
            assert "wall_seconds" in budget_dict
            assert isinstance(inject_env, dict)
            assert trace_id

        assert len(rollouts) == 3
        assert all(isinstance(r, Rollout) for r in rollouts)

    def test_concurrency_cap_threaded_through(self, tmp_path, monkeypatch):
        monkeypatch.setenv("AR2_BACKEND", "modal")
        monkeypatch.setenv("MODAL_PROFILE", "autoresearch-hack")

        mr = _fresh_modal_runner(monkeypatch)
        budget = Budget(wall_seconds=30.0)
        envs = [_FakeEnv("env-x"), _FakeEnv("env-y")]
        captured: list[dict] = []

        def fake_starmap(inputs):
            for _snap_ref, _env_spec, budget_dict, _inject_env, _trace_id in inputs:
                captured.append(budget_dict)
            return [_make_rollout(e.id).model_dump() for e in envs]

        mr.run_rollout.starmap = MagicMock(side_effect=fake_starmap)

        with patch.object(mr, "upload_snapshot", return_value="snap_xyz"):
            mr.run_rollouts_parallel(_make_ar_dir(tmp_path), envs, budget)

        assert all(b["wall_seconds"] == 30.0 for b in captured)

    def test_profile_guard_fires(self, tmp_path, monkeypatch):
        monkeypatch.setenv("AR2_BACKEND", "modal")
        monkeypatch.setenv("MODAL_PROFILE", "wrong-profile")

        mr = _fresh_modal_runner(monkeypatch)

        with pytest.raises(RuntimeError, match="autoresearch-hack"):
            mr.run_rollouts_parallel(_make_ar_dir(tmp_path), [_FakeEnv("e")], BUDGET)

    def test_evaluate_uses_run_evaluate_on_modal(self, tmp_path, monkeypatch):
        monkeypatch.setenv("AR2_BACKEND", "modal")
        monkeypatch.setenv("AR2_MODAL_RUN_EVALUATE", "1")

        for key in list(sys.modules):
            if key in ("harness.loop.outer", "harness.runtime.score"):
                monkeypatch.delitem(sys.modules, key, raising=False)

        from harness.loop.outer import evaluate

        train = [_FakeEnv("t1", "train"), _FakeEnv("t2", "train")]
        heldout = [_FakeEnv("h1", "heldout")]
        ar_dir = _make_ar_dir(tmp_path)

        def fake_run_evaluate_on_modal(ar_dir, train, heldout, budget, *, version=0, candidate=""):
            return (
                [_make_rollout("t1", 0.8), _make_rollout("t2", 0.7)],
                [_make_rollout("h1", 0.6)],
            )

        with patch("harness.cloud.runner.run_evaluate_on_modal", side_effect=fake_run_evaluate_on_modal):
            attempt = evaluate(ar_dir, train, heldout, BUDGET, sync_traces=None)

        assert attempt.train_reward == pytest.approx(0.75)
        assert attempt.heldout_reward == pytest.approx(0.6)
        assert len(attempt.train_rollouts) == 2

    def test_starmap_receives_one_tuple_per_env(self, tmp_path, monkeypatch):
        monkeypatch.setenv("AR2_BACKEND", "modal")
        monkeypatch.setenv("MODAL_PROFILE", "autoresearch-hack")

        mr = _fresh_modal_runner(monkeypatch)
        n_envs = 5
        envs = [_FakeEnv(f"e{i}", "train") for i in range(n_envs)]
        call_args: list[list] = []

        def fake_starmap(inputs):
            all_inputs = list(inputs)
            call_args.append(all_inputs)
            return [_make_rollout(f"e{i}").model_dump() for i in range(n_envs)]

        mr.run_rollout.starmap = MagicMock(side_effect=fake_starmap)

        with patch.object(mr, "upload_snapshot", return_value="snap_r"):
            mr.run_rollouts_parallel(_make_ar_dir(tmp_path), envs, BUDGET)

        assert len(call_args) == 1
        assert len(call_args[0]) == n_envs


class TestLocalFanOut:
    def test_make_spawn_local_concurrency(self, monkeypatch):
        monkeypatch.setenv("AR2_BACKEND", "local")
        for key in list(sys.modules):
            if key == "harness.runtime.sandbox":
                monkeypatch.delitem(sys.modules, key, raising=False)

        from harness.runtime.sandbox import _make_spawn_local

        active: list[int] = []
        peak: list[int] = []
        lock = threading.Lock()

        def work(x):
            with lock:
                active.append(x)
                peak.append(len(active))
            import time; time.sleep(0.02)
            with lock:
                active.remove(x)
            return x

        spawn = _make_spawn_local(2)
        results = spawn(work, list(range(6)))
        assert max(peak) <= 2
        assert sorted(results) == list(range(6))

    def test_score_repo_local_no_starmap(self, tmp_path, monkeypatch):
        monkeypatch.setenv("AR2_BACKEND", "local")

        modal_mod = sys.modules.get("modal")
        if modal_mod is not None:
            fake_sb = MagicMock()
            fake_sb.filesystem.read_text.side_effect = FileNotFoundError
            fake_sb.terminate.return_value = None
            fresh_create = MagicMock(return_value=fake_sb)
            monkeypatch.setattr(modal_mod.Sandbox, "create", fresh_create)

        for key in list(sys.modules):
            if key == "harness.runtime.score":
                monkeypatch.delitem(sys.modules, key, raising=False)

        from harness.runtime.score import score_repo
        ar_dir = _make_ar_dir(tmp_path)
        envs = [_FakeEnv("env-a"), _FakeEnv("env-b")]
        rollouts = score_repo(ar_dir, envs, BUDGET)
        assert len(rollouts) == 2
        assert all(isinstance(r, Rollout) for r in rollouts)


class TestResultAssembly:
    def test_rollout_fields_preserved(self, tmp_path, monkeypatch):
        monkeypatch.setenv("AR2_BACKEND", "modal")
        monkeypatch.setenv("MODAL_PROFILE", "autoresearch-hack")

        mr = _fresh_modal_runner(monkeypatch)
        envs = [_FakeEnv("env-train-1", "train"), _FakeEnv("env-ho-1", "heldout")]
        expected = [
            _make_rollout("env-train-1", 0.8),
            _make_rollout("env-ho-1", 0.6),
        ]

        mr.run_rollout.starmap = MagicMock(
            side_effect=lambda inputs: [r.model_dump() for r in expected]
        )

        with patch.object(mr, "upload_snapshot", return_value="snap_q"):
            rollouts = mr.run_rollouts_parallel(_make_ar_dir(tmp_path), envs, BUDGET)

        assert len(rollouts) == 2
        assert rollouts[0].env_id == "env-train-1"
        assert rollouts[0].final_reward == pytest.approx(0.8)
        assert rollouts[1].env_id == "env-ho-1"
        assert rollouts[1].final_reward == pytest.approx(0.6)

    def test_drive_modal_evaluates_candidates_in_parallel(self, tmp_path, monkeypatch):
        monkeypatch.setenv("AR2_BACKEND", "modal")

        for key in list(sys.modules):
            if key == "harness.loop.outer":
                monkeypatch.delitem(sys.modules, key, raising=False)

        lock = threading.Lock()
        active: list[str] = []

        def fake_score_repo(ar_dir, envs, budget, **kwargs):
            import time
            with lock:
                active.append(str(ar_dir))
            time.sleep(0.01)
            with lock:
                active.remove(str(ar_dir))
            return [_make_rollout(e.id) for e in envs]

        class _FakeArObj:
            def improve(self, archive, budget, spawn):
                import os

                vr = os.environ.get("AR2_VERSION_ROOT")
                d = tmp_path / f"cand_{uuid.uuid4().hex[:8]}"
                if vr:
                    d = Path(vr)
                else:
                    d.mkdir()
                ar = d / "ar"
                ar.mkdir(parents=True, exist_ok=True)
                (ar / "entrypoint.py").write_text(
                    "def solve(t,b,s,sp): pass\ndef improve(a,b,sp): pass\n"
                )
                rt = d / "harness" / "runtime"
                rt.mkdir(parents=True, exist_ok=True)
                (rt / "score.py").write_text("# stub\n")
                return d

        def fake_improve_on_modal(parent_source_ref, archive, budget, workshop_traj=""):
            return _FakeArObj().improve(archive, budget, lambda f, a: [])

        from harness.loop.outer import drive
        with patch("harness.cloud.runner.run_improve_on_modal", side_effect=fake_improve_on_modal):
            archive = drive(
            tmp_path / "ar0",
            [_FakeEnv("train-e", "train")],
            [_FakeEnv("ho-e", "heldout")],
            BUDGET,
            K=1,
            score_repo=fake_score_repo,
            load_ar=lambda ref: _FakeArObj(),
            )

        assert len(archive.attempts) >= 2

    def test_archive_versions_monotonic(self, tmp_path, monkeypatch):
        monkeypatch.setenv("AR2_BACKEND", "modal")

        for key in list(sys.modules):
            if key == "harness.loop.outer":
                monkeypatch.delitem(sys.modules, key, raising=False)

        def fake_score_repo(ar_dir, envs, budget, **kwargs):
            return [_make_rollout(e.id) for e in envs]

        class _FakeArObj:
            def improve(self, archive, budget, spawn):
                d = tmp_path / f"c_{uuid.uuid4().hex[:6]}"
                d.mkdir()
                return d

        def fake_improve_on_modal(parent_source_ref, archive, budget, workshop_traj=""):
            return _FakeArObj().improve(archive, budget, lambda f, a: [])

        from harness.loop.outer import drive
        with patch("harness.cloud.runner.run_improve_on_modal", side_effect=fake_improve_on_modal):
            archive = drive(
            tmp_path / "ar0",
            [_FakeEnv("t")],
            [_FakeEnv("h", "heldout")],
            BUDGET,
            K=2,
            score_repo=fake_score_repo,
            load_ar=lambda ref: _FakeArObj(),
            )

        versions = [a.version for a in archive.attempts]
        assert len(set(versions)) == len(versions)
        assert min(versions) == 0
