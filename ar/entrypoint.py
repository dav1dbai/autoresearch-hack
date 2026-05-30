"""AR v0 entrypoints — implements Solve and Improve from harness/contracts.py.

solve:   Karpathy-style inner loop — repeatedly invoke a headless coding agent
         to propose edits to task.workdir, score each, keep the best, revert on
         regression.  Single-agent in v0; spawn is wired but unused so a future
         version can fan out without restructuring.

improve: DGM-style self-modification — copy this ar/ directory, build an archive
         digest, invoke a coding agent over the copy with instructions to raise
         held-out reward.  Returns path to the new ar/ snapshot.

Agent invocation is fully controlled by two env vars:
    INNER_AGENT_CMD  — command prefix for solve (default: "codex exec")
    MUTATE_AGENT_CMD — command prefix for improve (default: "codex exec")

The full shell command is: <CMD> <prompt_file>
where prompt_file is a temp file containing the task prompt for the agent.
"""

from __future__ import annotations

import os
import re
import shlex
import shutil
import subprocess
import tempfile
import time
from pathlib import Path

from harness.contracts import (
    Archive,
    Budget,
    Submission,
    StepResult,
    TaskSpec,
    ScoreFn,
    SpawnFn,
)

_INNER_AGENT_DEFAULT = "codex exec"
_MUTATE_AGENT_DEFAULT = "codex exec"

# Derived from budget: run at most this many iterations regardless of wall time.
# Keeps a single slow run from burning the whole budget on one edit.
_MAX_ITERS_PER_SECOND = 1.0 / 30.0  # ~2 per minute; cap at budget.wall_seconds * factor
_REWARD_CEILING = float(os.environ.get("AR2_REWARD_CEILING", "0.999"))
_STALE_ITERS = int(os.environ.get("AR2_STALE_ITERS", "2"))
_AGENT_LOG = "agent.log"


def _agent_cmd(env_var: str, default: str) -> list[str]:
    raw = os.environ.get(env_var, default)
    return shlex.split(raw)


def _run_agent(cmd: list[str], prompt: str, cwd: Path | None = None, timeout: float | None = None) -> None:
    """Write prompt to a temp file, invoke agent, append output to agent.log."""
    with tempfile.NamedTemporaryFile(mode="w", suffix=".md", delete=False) as f:
        f.write(prompt)
        prompt_path = f.name
    log_path = (cwd / _AGENT_LOG) if cwd else Path(tempfile.gettempdir()) / _AGENT_LOG
    try:
        with log_path.open("a", encoding="utf-8") as log_f:
            log_f.write(f"\n--- agent run {time.time():.0f} ---\n")
            log_f.flush()
            subprocess.run(
                cmd + [prompt_path],
                cwd=str(cwd) if cwd else None,
                timeout=timeout,
                check=False,
                stdout=log_f,
                stderr=subprocess.STDOUT,
            )
    finally:
        os.unlink(prompt_path)


def _parse_codex_tokens(log_path: Path) -> int | None:
    """Parse trailing 'tokens used' line from codex stdout."""
    if not log_path.exists():
        return None
    text = log_path.read_text(encoding="utf-8", errors="replace")
    matches = re.findall(r"tokens used\s*\n\s*([\d,]+)", text, flags=re.IGNORECASE)
    if not matches:
        return None
    return int(matches[-1].replace(",", ""))


def solve(
    task: TaskSpec,
    budget: Budget,
    score: ScoreFn,
    spawn: SpawnFn,
) -> Submission:
    """Karpathy inner loop: edit task.workdir → score → keep if better → repeat.

    v0 is single-agent.  spawn is accepted but unused; a future version can
    replace this body entirely and use spawn for parallel fanout.
    """
    deadline = time.monotonic() + budget.wall_seconds
    max_iters = max(1, int(budget.wall_seconds * _MAX_ITERS_PER_SECOND))
    cmd = _agent_cmd("INNER_AGENT_CMD", _INNER_AGENT_DEFAULT)

    # Score the unedited baseline first.
    best_sub = Submission(workdir=task.workdir, notes="baseline")
    best_result: StepResult = score(best_sub)

    if best_result.reward >= _REWARD_CEILING:
        return best_sub

    stale_iters = 0

    # Snapshot the workdir so we can revert on regression.
    with tempfile.TemporaryDirectory(prefix="ar_snapshot_") as snap_root:
        snap_dir = Path(snap_root) / "best"
        shutil.copytree(task.workdir, snap_dir, dirs_exist_ok=False)

        for iteration in range(max_iters):
            if time.monotonic() >= deadline:
                break
            if best_result.done:
                break
            if best_result.reward >= _REWARD_CEILING:
                break
            if stale_iters >= _STALE_ITERS:
                break

            remaining = deadline - time.monotonic()
            prompt = _build_solve_prompt(task, best_result, iteration)

            try:
                _run_agent(cmd, prompt, cwd=task.workdir, timeout=remaining * 0.8)
            except subprocess.TimeoutExpired:
                break

            candidate = Submission(workdir=task.workdir, notes=f"iter={iteration}")
            try:
                result = score(candidate)
            except Exception:
                # Revert on crash — restore snapshot.
                _restore(snap_dir, task.workdir)
                continue

            if result.reward > best_result.reward:
                best_sub = Submission(workdir=task.workdir, notes=f"iter={iteration} reward={result.reward:.4f}")
                best_result = result
                stale_iters = 0
                # Advance snapshot.
                shutil.rmtree(snap_dir)
                shutil.copytree(task.workdir, snap_dir, dirs_exist_ok=False)
            else:
                stale_iters += 1
                # Revert to last-best state.
                _restore(snap_dir, task.workdir)

    return best_sub


def improve(
    archive: Archive,
    budget: Budget,
    spawn: SpawnFn,
) -> Path:
    """DGM-style meta-improvement: copy ar/ to a new dir and invoke a coding agent.

    The agent sees:
    - The full ar/ source (in the new copy, which it may freely edit).
    - A digest of archive.attempts (rewards, diffs, hack flags) as context.

    Returns the path to the new ar/ snapshot.
    """
    ar_dir = Path(__file__).parent  # this file lives in ar/
    repo_root = ar_dir.parent
    default_cache = repo_root / "versions"
    cache_root = Path(os.environ.get("AR2_CACHE_DIR", default_cache))
    cache_root.mkdir(parents=True, exist_ok=True)

    # Fresh copy under versions/ (gitignored) — keeps repo root clean.
    new_ar = Path(tempfile.mkdtemp(prefix="ar_v_", dir=cache_root))
    shutil.copytree(ar_dir, new_ar, dirs_exist_ok=True)

    cmd = _agent_cmd("MUTATE_AGENT_CMD", _MUTATE_AGENT_DEFAULT)
    remaining = budget.wall_seconds
    prompt = _build_improve_prompt(archive, new_ar)

    try:
        _run_agent(cmd, prompt, cwd=new_ar, timeout=remaining * 0.9)
    except subprocess.TimeoutExpired:
        pass  # partial edit is still a valid candidate

    return new_ar


# ---------------------------------------------------------------------------
# Prompt builders
# ---------------------------------------------------------------------------

def _build_solve_prompt(task: TaskSpec, last: StepResult, iteration: int) -> str:
    feedback_block = f"\nFeedback from last score: {last.feedback}" if last.feedback else ""
    return f"""# AR v0 — inner research loop (iteration {iteration})

## Task
Env: {task.env_id}  Split: {task.split}
Goal: {task.prompt}

## Current best reward
{last.reward:.6f}  (raw: {last.raw}){feedback_block}

## Instructions
You are editing the solution files in the current working directory.
Propose ONE concrete change that may raise the reward.  Edit the files directly.
Do NOT modify anything outside this directory.
Do NOT touch the grader, harness, or evaluation scripts.
Make your edit and stop — the harness scores and keeps/reverts automatically.
"""


def _build_improve_prompt(archive: Archive, new_ar: Path) -> str:
    digest = _archive_digest(archive)
    workshop_traj = os.environ.get("AR2_WORKSHOP_TRAJECTORY", "").strip()
    traj_block = ""
    if workshop_traj:
        traj_block = f"""
## Workshop trajectory (Raindrop spans — triage inner + outer runs)
{workshop_traj}
"""
    return f"""# AR² — meta-improvement loop

## Your task
You are editing the autoresearch (AR) codebase located at:
  {new_ar}

Your goal: modify ar/entrypoint.py (and any other file in this ar/ directory)
so that future `solve()` runs achieve higher HELD-OUT reward.

Ideas: add parallelism via spawn(), add a critic agent, improve search strategy,
tune the prompt, add memory/reflection between iterations.

## HARD RULES
- Edit ONLY files inside {new_ar}
- Do NOT touch harness/, envs/, pyproject.toml, uv.lock, or .venv
- Do NOT attempt to access or modify the grader
- Do NOT fabricate scores or logs

## Archive history (past attempts)
{digest}
{traj_block}
Make your edits, then stop.
"""


def _archive_digest(archive: Archive) -> str:
    if not archive.attempts:
        return "(no prior attempts)"
    lines: list[str] = []
    for a in archive.attempts:
        hack = f" [HACKED: {', '.join(a.hack_flags)}]" if a.hack_flags else ""
        lines.append(
            f"  v{a.version} parent={a.parent} train={a.train_reward:.4f} "
            f"heldout={a.heldout_reward:.4f}{hack} | {a.diff_summary}"
        )
        for r in a.train_rollouts:
            curve = ", ".join(f"{x:.4f}" for x in r.rewards) or "—"
            lines.append(f"    train {r.env_id}: inner=[{curve}] final={r.final_reward:.4f}")
        for r in a.heldout_rollouts:
            curve = ", ".join(f"{x:.4f}" for x in r.rewards) or "—"
            lines.append(f"    heldout {r.env_id}: inner=[{curve}] final={r.final_reward:.4f}")
    best = archive.best()
    if best:
        lines.append(f"\nBest so far: v{best.version} heldout={best.heldout_reward:.4f}")
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _restore(snap_dir: Path, target: Path) -> None:
    """Overwrite target with the contents of snap_dir."""
    for item in target.iterdir():
        if item.is_dir():
            shutil.rmtree(item)
        else:
            item.unlink()
    shutil.copytree(snap_dir, target, dirs_exist_ok=True)
