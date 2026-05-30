"""AR v0 entrypoints — implements Solve and Improve from harness/contracts.py.

solve:   Inner research loop (v0: Karpathy autoresearch shape) — invoke a headless
         coding agent to propose edits to task.workdir, score each, keep the best,
         revert on regression. spawn() is injected by the harness; v0 ignores it.
         A later ar/ version should wire spawn itself when it invents parallel fanout.

improve: outer meta-loop step — snapshot mutable stack (ar/ + harness/runtime/),
         invoke a coding agent to edit the copy, return the version root path.

Agent invocation is fully controlled by two env vars:
    INNER_AGENT_CMD  — command prefix for solve (default: scripts/raindrop_codex_exec.sh exec)
    MUTATE_AGENT_CMD — command prefix for improve (default: scripts/raindrop_codex_exec.sh exec)

Defaults wire Codex through Raindrop Workshop MCP (``raindrop workshop mcp``) so token/tool
traces land in Workshop — same as ``raindrop setup`` + Workshop's codex chat pane.

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

def _raindrop_codex_default() -> str:
    """Absolute path to raindrop_codex_exec.sh when available."""
    extra = os.environ.get(
        "AR2_CODEX_EXTRA_FLAGS",
        "-m gpt-5-codex -c preferred_auth_method=apikey",
    ).strip()
    candidates: list[Path] = []
    if raw := os.environ.get("AR2_REPO_ROOT"):
        candidates.append(Path(raw))
    candidates.append(Path(__file__).resolve().parent.parent)
    for root in candidates:
        script = (root / "scripts" / "raindrop_codex_exec.sh").resolve()
        if script.is_file():
            return f"{script} exec {extra}".strip()
    return f"codex exec {extra}".strip()


_INNER_AGENT_DEFAULT = _raindrop_codex_default()
_MUTATE_AGENT_DEFAULT = _raindrop_codex_default()

# Derived from budget: run at most this many iterations regardless of wall time.
# Keeps a single slow run from burning the whole budget on one edit.
_MAX_ITERS_PER_SECOND = 1.0 / 30.0  # ~2 per minute; cap at budget.wall_seconds * factor
_REWARD_CEILING = float(os.environ.get("AR2_REWARD_CEILING", "0.999"))
_STALE_ITERS = int(os.environ.get("AR2_STALE_ITERS", "2"))
_AGENT_LOG = "agent.log"


def _agent_cmd(env_var: str, default: str | None = None) -> list[str]:
    if default is None:
        default = _raindrop_codex_default()
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

    v0 is single-agent and does not call spawn().
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
    """Outer-loop mutation step: run meta-agent on a version snapshot.

    Snapshot creation is host-owned (``harness.loop.snapshot``) when
    ``AR2_VERSION_ROOT`` is set. Otherwise copies from ``AR2_REPO_ROOT`` using
    only stdlib + env (Modal path — no harness import from ar/).
    """
    ar_dir = Path(__file__).parent
    existing = os.environ.get("AR2_VERSION_ROOT")
    if existing:
        version_root = Path(existing)
    else:
        version_root = _create_version_snapshot(ar_dir)

    cmd = _agent_cmd("MUTATE_AGENT_CMD", _MUTATE_AGENT_DEFAULT)
    remaining = budget.wall_seconds
    prompt = _build_improve_prompt(archive, version_root)
    (version_root / "improve_prompt.md").write_text(prompt, encoding="utf-8")

    try:
        _run_agent(cmd, prompt, cwd=version_root, timeout=remaining * 0.9)
    except subprocess.TimeoutExpired:
        pass

    return version_root


def _create_version_snapshot(ar_dir: Path) -> Path:
    """Copy ar/ + harness/runtime/ into versions/v_* (no harness imports)."""
    raw = os.environ.get("AR2_REPO_ROOT")
    if not raw:
        raise RuntimeError(
            "AR2_REPO_ROOT must be set by drive() or Modal run_improve before improve()"
        )
    repo_root = Path(raw).resolve()
    cache_root = Path(os.environ.get("AR2_CACHE_DIR", repo_root / "versions"))
    cache_root.mkdir(parents=True, exist_ok=True)
    version_root = Path(tempfile.mkdtemp(prefix="v_", dir=cache_root))

    def _ignore(_d: str, names: list[str]) -> set[str]:
        skip = {"__pycache__", ".pytest_cache"}
        return {n for n in names if n in skip or n.startswith("test_")}

    shutil.copytree(ar_dir, version_root / "ar", dirs_exist_ok=True)
    runtime_src = repo_root / "harness" / "runtime"
    if runtime_src.is_dir():
        shutil.copytree(
            runtime_src,
            version_root / "harness" / "runtime",
            dirs_exist_ok=True,
            ignore=_ignore,
        )
    return version_root


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


def _build_improve_prompt(archive: Archive, version_root: Path) -> str:
    context = _minimal_improve_context(archive)
    run_id = os.environ.get("AR2_RUN_ID", "").strip()
    workshop_run_id = os.environ.get("RAINDROP_WORKSHOP_TRACE_ID", "").strip()
    workshop_lookup = (
        f"Pass `{workshop_run_id}` as the Workshop `run_id` when a Raindrop MCP tool requires one."
        if workshop_run_id
        else "If a Raindrop MCP tool requires a Workshop `run_id`, first use "
        "`raindrop.query_traces` to find the concrete run id for this AR2 tag."
    )
    run_scope = (
        f"""
Current AR2_RUN_ID: `{run_id}`
Workshop run id for this improve invocation: `{workshop_run_id or "(not set)"}`

Only inspect Workshop rows/events where metadata or OTLP attributes contain this
exact AR2 run tag (`runId` or `ar2.run_id`). Ignore traces from other run tags.
`AR2_RUN_ID` is a harness tag, not the Workshop run id; do not pass it as the
`run_id` argument to `get_run_outline` or `search_run`.
{workshop_lookup}
Before editing, identify which same-run traces you inspected and what evidence
they provided."""
        if run_id
        else """
No AR2_RUN_ID is set for this improve step; use Workshop traces that clearly
belong to this current run context."""
    )
    return f"""# AR² — meta-improvement

You are editing a **version snapshot** of the autoresearch stack at:
  {version_root}

## Editable (this snapshot)
- `ar/` — inner research loop (`solve`) and how the next meta-edit runs (`improve`)
- `harness/runtime/` — rollout orchestration (`score_repo`, `rollout`, `sandbox`)

## Frozen (not in this snapshot — host-only; do not try to reach or patch)
- `envs/` (referees), `harness/contracts.py`, `harness/tracing/`, `harness/loop/` (outer driver),
  `harness/cloud/`, `harness/backends/`, `infra/`

Read the code in this snapshot. Change what you think will raise held-out reward on the
next evaluation. Your edits here are the candidate.

## Minimal run context
{context}
{run_scope}

## Required Workshop lookup
Before editing, inspect Raindrop Workshop with the configured MCP tools. Use at least one of:
- `raindrop.query_traces`
- `raindrop.get_run_outline`
- `raindrop.search_run`

These are Codex MCP tool calls on the `raindrop` server, not shell commands.
Do not try `raindrop query_traces`, `curl /v1/query_traces`, or local Python
MCP clients unless the native MCP tool call fails and you explain the failure.

Treat Workshop as the source of truth for archive history, rollout traces, score curves, and
failure evidence. Do not rely on a pasted archive digest; it is intentionally omitted. Base
your edit on evidence you retrieve from Workshop, then make one focused change and stop.

Make your edits, then stop.
"""


def _minimal_improve_context(archive: Archive) -> str:
    best = archive.best()
    best_line = (
        f"Best known version: v{best.version} heldout={best.heldout_reward:.4f}"
        if best
        else "Best known version: none"
    )
    latest = max((a.version for a in archive.attempts), default=-1)
    return f"Attempts so far: {len(archive.attempts)}\nLatest version: v{latest}\n{best_line}"


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
