"""Tests for ar/entrypoint.py — fully offline, no real LLM calls.

All subprocess invocations are monkeypatched so no codex/claude/openai traffic
is produced.  Tests cover:
  - keep/revert logic in solve (higher reward kept, lower discarded)
  - budget/iter cap respected
  - improve produces a new directory distinct from source
  - improve feeds archive history into the agent prompt
"""

from __future__ import annotations

import os
import shutil
import tempfile
from pathlib import Path
from unittest.mock import MagicMock, patch, call

import pytest

from harness.contracts import (
    Archive,
    Attempt,
    Budget,
    Submission,
    StepResult,
    TaskSpec,
)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture()
def workdir(tmp_path: Path) -> Path:
    """Minimal workdir with one file, simulating task.workdir."""
    d = tmp_path / "workdir"
    d.mkdir()
    (d / "solution.py").write_text("# baseline\n")
    return d


@pytest.fixture()
def task(workdir: Path) -> TaskSpec:
    return TaskSpec(
        env_id="test-env",
        split="train",
        prompt="minimize val_bpb",
        workdir=workdir,
    )


@pytest.fixture()
def small_budget() -> Budget:
    return Budget(wall_seconds=60.0)


def noop_spawn(*args, **kwargs) -> list:
    return []


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _noop_run_agent(cmd, prompt, cwd=None, timeout=None):
    """Replacement for _run_agent — does nothing (no LLM call)."""


def _writing_run_agent(write_path: Path, new_content: str):
    """Factory: returns a _run_agent stub that writes new_content to write_path."""
    def _inner(cmd, prompt, cwd=None, timeout=None):
        write_path.write_text(new_content)
    return _inner


# ---------------------------------------------------------------------------
# solve() — keep/revert logic
# ---------------------------------------------------------------------------

class TestSolveKeepRevert:
    def test_keeps_higher_reward(self, task: TaskSpec, small_budget: Budget, workdir: Path):
        """Agent edit improves reward — solve should keep the new file content."""
        improved_content = "# improved\n"

        rewards = iter([0.5, 0.8])  # baseline=0.5, then agent edit scores 0.8

        def score(sub: Submission) -> StepResult:
            return StepResult(reward=next(rewards), done=False)

        def writing_agent(cmd, prompt, cwd=None, timeout=None):
            (workdir / "solution.py").write_text(improved_content)

        import ar.entrypoint as ep
        orig = ep._MAX_ITERS_PER_SECOND
        ep._MAX_ITERS_PER_SECOND = 1.0  # 1 iter per second × 60s budget = 60 iters max
        try:
            with patch("ar.entrypoint._run_agent", side_effect=writing_agent):
                sub = ep.solve(task, small_budget, score, noop_spawn)
        finally:
            ep._MAX_ITERS_PER_SECOND = orig

        assert (workdir / "solution.py").read_text() == improved_content
        assert "reward=0.8000" in sub.notes or sub.notes.startswith("iter=")

    def test_reverts_lower_reward(self, task: TaskSpec, small_budget: Budget, workdir: Path):
        """Agent edit worsens reward — solve should revert to baseline content."""
        baseline_content = (workdir / "solution.py").read_text()
        bad_content = "# worse\n"

        rewards = iter([0.7, 0.3])  # baseline=0.7, agent edit=0.3

        def score(sub: Submission) -> StepResult:
            return StepResult(reward=next(rewards), done=False)

        def degrading_agent(cmd, prompt, cwd=None, timeout=None):
            (workdir / "solution.py").write_text(bad_content)

        import ar.entrypoint as ep
        orig = ep._MAX_ITERS_PER_SECOND
        ep._MAX_ITERS_PER_SECOND = 1.0
        try:
            with patch("ar.entrypoint._run_agent", side_effect=degrading_agent):
                sub = ep.solve(task, small_budget, score, noop_spawn)
        finally:
            ep._MAX_ITERS_PER_SECOND = orig

        assert (workdir / "solution.py").read_text() == baseline_content
        assert sub.notes == "baseline"

    def test_keeps_best_across_multiple_iters(self, task: TaskSpec, workdir: Path):
        """Over multiple iterations, solve tracks the highest reward seen."""
        budget = Budget(wall_seconds=120.0)
        contents = ["# v1\n", "# v2\n", "# v3\n"]
        iter_idx = [0]

        # rewards: baseline=0.4, v1=0.6 (keep), v2=0.5 (revert), v3=0.9 (keep)
        reward_seq = iter([0.4, 0.6, 0.5, 0.9])

        def score(sub: Submission) -> StepResult:
            return StepResult(reward=next(reward_seq), done=False)

        def cycling_agent(cmd, prompt, cwd=None, timeout=None):
            i = iter_idx[0] % len(contents)
            (workdir / "solution.py").write_text(contents[i])
            iter_idx[0] += 1

        import ar.entrypoint as ep
        orig = ep._MAX_ITERS_PER_SECOND
        ep._MAX_ITERS_PER_SECOND = 1.0 / 40.0  # 3 iters in 120s
        try:
            with patch("ar.entrypoint._run_agent", side_effect=cycling_agent):
                sub = ep.solve(task, budget, score, noop_spawn)
        finally:
            ep._MAX_ITERS_PER_SECOND = orig

        assert (workdir / "solution.py").read_text() == "# v3\n"
        assert "0.9000" in sub.notes

    def test_score_exception_reverts(self, task: TaskSpec, small_budget: Budget, workdir: Path):
        """If score() raises after an agent edit, workdir is reverted."""
        baseline_content = (workdir / "solution.py").read_text()
        call_count = [0]

        def score(sub: Submission) -> StepResult:
            call_count[0] += 1
            if call_count[0] == 1:
                return StepResult(reward=0.5)
            raise RuntimeError("scorer exploded")

        def writing_agent(cmd, prompt, cwd=None, timeout=None):
            (workdir / "solution.py").write_text("# crash edit\n")

        import ar.entrypoint as ep
        orig = ep._MAX_ITERS_PER_SECOND
        ep._MAX_ITERS_PER_SECOND = 1.0
        try:
            with patch("ar.entrypoint._run_agent", side_effect=writing_agent):
                sub = ep.solve(task, small_budget, score, noop_spawn)
        finally:
            ep._MAX_ITERS_PER_SECOND = orig

        assert (workdir / "solution.py").read_text() == baseline_content


# ---------------------------------------------------------------------------
# solve() — budget / iter cap
# ---------------------------------------------------------------------------

class TestSolveBudget:
    def test_respects_max_iters(self, task: TaskSpec, workdir: Path):
        """Iter cap derived from budget.wall_seconds is not exceeded."""
        budget = Budget(wall_seconds=3.0)
        call_count = [0]

        def score(sub: Submission) -> StepResult:
            call_count[0] += 1
            return StepResult(reward=0.0)

        import ar.entrypoint as ep
        orig = ep._MAX_ITERS_PER_SECOND
        ep._MAX_ITERS_PER_SECOND = 1.0  # 3 max iters (floor of 3.0 * 1.0)
        try:
            with patch("ar.entrypoint._run_agent", side_effect=_noop_run_agent):
                ep.solve(task, budget, score, noop_spawn)
        finally:
            ep._MAX_ITERS_PER_SECOND = orig

        # 1 baseline + up to 3 agent iters = 4 score calls max
        assert call_count[0] <= 4

    def test_done_flag_stops_loop(self, task: TaskSpec, small_budget: Budget):
        """If score returns done=True on the baseline, no agent iters run."""
        call_count = [0]

        def score(sub: Submission) -> StepResult:
            call_count[0] += 1
            return StepResult(reward=1.0, done=True)

        with patch("ar.entrypoint._run_agent", side_effect=_noop_run_agent) as mock_agent:
            import ar.entrypoint as ep
            ep.solve(task, small_budget, score, noop_spawn)

        assert call_count[0] == 1
        mock_agent.assert_not_called()

    def test_env_done_false_runs_iterations(self, task: TaskSpec, small_budget: Budget):
        """D-00: when score returns done=False, solve runs >=1 agent edit, not just baseline."""
        call_count = [0]

        def score(sub: Submission) -> StepResult:
            call_count[0] += 1
            return StepResult(reward=0.5, done=False)

        import ar.entrypoint as ep
        orig = ep._MAX_ITERS_PER_SECOND
        ep._MAX_ITERS_PER_SECOND = 1.0
        try:
            with patch("ar.entrypoint._run_agent", side_effect=_noop_run_agent) as mock_agent:
                ep.solve(task, small_budget, score, noop_spawn)
        finally:
            ep._MAX_ITERS_PER_SECOND = orig

        assert mock_agent.call_count >= 1   # at least one agent edit attempted
        assert call_count[0] >= 2           # baseline + >=1 post-edit score

    def test_returns_submission_with_workdir(self, task: TaskSpec, small_budget: Budget):
        """solve() always returns a Submission whose workdir matches task.workdir."""
        def score(sub: Submission) -> StepResult:
            return StepResult(reward=0.5)

        import ar.entrypoint as ep
        with patch("ar.entrypoint._run_agent", side_effect=_noop_run_agent):
            sub = ep.solve(task, small_budget, score, noop_spawn)

        assert sub.workdir == task.workdir


# ---------------------------------------------------------------------------
# improve() — new dir, distinct from source, archive fed to prompt
# ---------------------------------------------------------------------------

class TestImprove:
    @pytest.fixture(autouse=True)
    def _repo_root(self, monkeypatch):
        repo = Path(__file__).resolve().parent.parent
        monkeypatch.setenv("AR2_REPO_ROOT", str(repo))

    def test_returns_new_directory(self, tmp_path: Path):
        """improve() creates a NEW directory, not the same as ar/."""
        import ar.entrypoint as ep

        archive = Archive()
        budget = Budget(wall_seconds=10.0)
        captured_prompts: list[str] = []

        def capturing_agent(cmd, prompt, cwd=None, timeout=None):
            captured_prompts.append(prompt)

        with patch("ar.entrypoint._run_agent", side_effect=capturing_agent):
            result = ep.improve(archive, budget, noop_spawn)

        try:
            ar_dir = Path(ep.__file__).parent
            assert result != ar_dir
            assert result.exists()
            assert (result / "ar" / "entrypoint.py").exists()
            assert (result / "harness" / "runtime" / "score.py").exists()
        finally:
            shutil.rmtree(result, ignore_errors=True)

    def test_new_dir_is_copy_of_ar(self, tmp_path: Path):
        """The returned directory is a copy of the ar/ folder."""
        import ar.entrypoint as ep

        archive = Archive()
        budget = Budget(wall_seconds=10.0)

        with patch("ar.entrypoint._run_agent", side_effect=_noop_run_agent):
            result = ep.improve(archive, budget, noop_spawn)

        try:
            ar_dir = Path(ep.__file__).parent
            assert (result / "ar" / "entrypoint.py").exists()
        finally:
            shutil.rmtree(result, ignore_errors=True)

    def test_archive_history_in_prompt(self):
        """Archive attempts are serialized into the improve prompt."""
        import ar.entrypoint as ep

        archive = Archive(attempts=[
            Attempt(
                version=1,
                parent=None,
                diff_summary="added critic",
                train_reward=0.7,
                heldout_reward=0.65,
                hack_flags=[],
                cost=Budget(wall_seconds=300.0),
                trace_id="abc",
                source_ref="ref1",
            ),
            Attempt(
                version=2,
                parent=1,
                diff_summary="HACKED: removed test",
                train_reward=0.95,
                heldout_reward=0.30,
                hack_flags=["overfit"],
                cost=Budget(wall_seconds=300.0),
                trace_id="def",
                source_ref="ref2",
            ),
        ])
        budget = Budget(wall_seconds=10.0)
        captured: list[str] = []

        def capturing_agent(cmd, prompt, cwd=None, timeout=None):
            captured.append(prompt)

        with patch("ar.entrypoint._run_agent", side_effect=capturing_agent):
            result = ep.improve(archive, budget, noop_spawn)

        shutil.rmtree(result, ignore_errors=True)

        assert len(captured) == 1
        prompt = captured[0]
        assert "Attempts so far: 2" in prompt
        assert "Latest version: v2" in prompt
        assert "Best known version: v1 heldout=0.6500" in prompt
        assert "raindrop.query_traces" in prompt
        assert "raindrop.get_run_outline" in prompt
        assert "raindrop.search_run" in prompt
        assert "train=0.7000" not in prompt
        assert "HACKED" not in prompt

    def test_improve_prompt_scopes_workshop_to_run_id(self, monkeypatch):
        """When AR2_RUN_ID is set, improve prompt requires same-run Workshop evidence."""
        import ar.entrypoint as ep

        monkeypatch.setenv("AR2_RUN_ID", "raindrop-k2")
        prompt = ep._build_improve_prompt(Archive(), Path("/tmp/version-root"))

        assert "Current AR2_RUN_ID: `raindrop-k2`" in prompt
        assert "metadata or OTLP attributes contain this" in prompt
        assert "`runId` or `ar2.run_id`" in prompt
        assert "Ignore traces from other run ids" in prompt
        assert "identify which same-run traces you inspected" in prompt

    def test_empty_archive_no_crash(self):
        """improve() with no prior attempts completes without error."""
        import ar.entrypoint as ep

        archive = Archive()
        budget = Budget(wall_seconds=10.0)

        with patch("ar.entrypoint._run_agent", side_effect=_noop_run_agent):
            result = ep.improve(archive, budget, noop_spawn)

        try:
            assert result.exists()
        finally:
            shutil.rmtree(result, ignore_errors=True)

    def test_agent_receives_cmd_from_env(self, monkeypatch):
        """MUTATE_AGENT_CMD env var is passed to subprocess via _run_agent."""
        import ar.entrypoint as ep

        monkeypatch.setenv("MUTATE_AGENT_CMD", "fake-mutate-agent --headless")
        archive = Archive()
        budget = Budget(wall_seconds=10.0)
        captured_cmds: list[list[str]] = []

        def capturing_agent(cmd, prompt, cwd=None, timeout=None):
            captured_cmds.append(cmd)

        with patch("ar.entrypoint._run_agent", side_effect=capturing_agent):
            result = ep.improve(archive, budget, noop_spawn)

        shutil.rmtree(result, ignore_errors=True)
        assert captured_cmds[0] == ["fake-mutate-agent", "--headless"]

    def test_solve_agent_cmd_from_env(self, task: TaskSpec, small_budget: Budget, monkeypatch):
        """INNER_AGENT_CMD env var is forwarded to the agent subprocess."""
        import ar.entrypoint as ep

        monkeypatch.setenv("INNER_AGENT_CMD", "fake-inner-agent --exec")
        captured_cmds: list[list[str]] = []

        def capturing_agent(cmd, prompt, cwd=None, timeout=None):
            captured_cmds.append(cmd)

        def score(sub: Submission) -> StepResult:
            return StepResult(reward=0.5, done=True)

        # done=True on baseline means no agent call; set done=False to force one iter.
        call_n = [0]
        def score2(sub: Submission) -> StepResult:
            call_n[0] += 1
            return StepResult(reward=0.5, done=call_n[0] > 1)

        orig = ep._MAX_ITERS_PER_SECOND
        ep._MAX_ITERS_PER_SECOND = 1.0
        try:
            with patch("ar.entrypoint._run_agent", side_effect=capturing_agent):
                ep.solve(task, small_budget, score2, noop_spawn)
        finally:
            ep._MAX_ITERS_PER_SECOND = orig

        assert len(captured_cmds) >= 1
        assert captured_cmds[0] == ["fake-inner-agent", "--exec"]
