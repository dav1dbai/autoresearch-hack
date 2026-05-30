"""Meta-autoresearch: optimize program.md (the research *strategy*), not train.py.

autoresearch optimizes train.py to minimize val_bpb. The strategy that drives it
lives in program.md as English prose, executed by a coding agent. This outer loop
treats that prose as the artifact under optimization: each evaluation is a full
inner autoresearch run, scored by the best val_bpb a downstream agent reaches
under that program.md within a fixed budget. Keep the edit if downstream research
got better; roll back otherwise. Same keep/rollback shape as the inner loop, one
level up.

The inner loop's intelligence is an LLM coding agent (irreducibly stochastic).
Everything wrapping it -- bounding, scoring, keep/rollback -- is determinate code.
"""

from __future__ import annotations

import csv
import os
import subprocess
import time
from pathlib import Path
from statistics import mean

from dotenv import load_dotenv
from pydantic import BaseModel

load_dotenv()  # folder-scoped .env: MODAL_*, OPENAI_API_KEY, *_AGENT_CMD

REPO = Path(__file__).parent / "repo"
RUNS = Path(__file__).parent / "runs"

# The one irreducible dependency: a headless coding agent that can read program.md
# and execute its LOOP FOREVER. Swap for `codex exec ...` etc. via env.
INNER_AGENT_CMD = os.environ.get("INNER_AGENT_CMD", "claude -p --dangerously-skip-permissions")
MUTATE_AGENT_CMD = os.environ.get("MUTATE_AGENT_CMD", "claude -p")


class Budget(BaseModel):
    max_inner_iters: int = 6      # downstream experiments per evaluation
    wall_clock_s: int = 30 * 60   # hard cap on a single inner run
    poll_s: int = 10              # how often we check the inner run's progress
    seeds: int = 1                # average over N inner runs to fight 5-min noise


class InnerResult(BaseModel):
    best_val_bpb: float           # the score the outer loop minimizes; inf if no keep
    n_experiments: int
    crashed: int
    wall_clock_s: float
    work_dir: str


class MetaStep(BaseModel):
    iter: int
    parent_score: float
    child_score: float
    kept: bool
    rationale: str


# --- pure, determinate scoring of an inner run ---------------------------------

def row_count(tsv: Path) -> int:
    if not tsv.exists():
        return 0
    return max(0, sum(1 for _ in tsv.open()) - 1)  # minus header


def score(tsv: Path, wall_s: float, work_dir: str) -> InnerResult:
    best, n, crashed = float("inf"), 0, 0
    if tsv.exists():
        for r in csv.DictReader(tsv.open(), delimiter="\t"):
            n += 1
            v = float(r["val_bpb"])
            if r["status"] == "crash" or v <= 0:
                crashed += 1
                continue
            best = min(best, v)
    return InnerResult(best_val_bpb=best, n_experiments=n, crashed=crashed,
                       wall_clock_s=wall_s, work_dir=work_dir)


# --- inner run: bound the unbounded agent loop, then score it ------------------

def _git(args: list[str], cwd: Path) -> None:
    subprocess.run(["git", *args], cwd=cwd, check=True, capture_output=True, text=True)


def run_inner(program_md: str, tag: str, budget: Budget) -> InnerResult:
    """Isolated checkout, swap in program_md, run the agent, cap it, score results.tsv.

    The cap is enforced externally (row count / wall clock) so program.md's
    'NEVER STOP' semantics stay untouched -- we don't edit the methodology to
    bound it, we just stop watching.
    """
    work = RUNS / tag
    if work.exists():
        _git(["worktree", "remove", "--force", str(work)], cwd=REPO)
    work.parent.mkdir(parents=True, exist_ok=True)
    _git(["worktree", "add", "--detach", str(work)], cwd=REPO)
    (work / "program.md").write_text(program_md)
    results = work / "results.tsv"

    proc = subprocess.Popen(
        [*INNER_AGENT_CMD.split(),
         f"Read program.md and run the autoresearch loop on branch autoresearch/{tag}. "
         f"Do the setup, then begin experimenting. Do not ask me anything."],
        cwd=work, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
    )
    t0 = time.time()
    try:
        while proc.poll() is None:
            done = row_count(results) >= budget.max_inner_iters
            timed_out = time.time() - t0 > budget.wall_clock_s
            if done or timed_out:
                break
            time.sleep(budget.poll_s)
    finally:
        proc.terminate()
    return score(results, time.time() - t0, str(work))


def evaluate(program_md: str, tag: str, budget: Budget) -> float:
    """Mean best-val_bpb over `seeds` inner runs. Lower is better."""
    runs = [run_inner(program_md, f"{tag}-s{s}", budget) for s in range(budget.seeds)]
    return mean(r.best_val_bpb for r in runs)


# --- mutation: propose a better methodology ------------------------------------

MUTATE_PROMPT = """You are editing the research *methodology* of an autoresearch system.

`program.md` instructs a downstream coding agent that autonomously edits `train.py`
to minimize val_bpb (validation bits-per-byte, lower is better) under a fixed
per-experiment time budget. Propose ONE concrete edit to `program.md` that makes
the downstream agent reach a lower val_bpb within its experiment budget -- e.g.
better prioritization of ideas, smarter rollback rules, sharper search guidance.

Current program.md:
<program>
{program}
</program>

Meta-history (past edits and the best val_bpb they produced; lower is better):
{history}

Output the FULL new program.md and nothing else."""


def propose_program(current: str, history: list[MetaStep]) -> str:
    hist = "\n".join(
        f"- iter {s.iter}: score={s.child_score:.6f} kept={s.kept} :: {s.rationale}"
        for s in history[-10:]
    ) or "(none yet)"
    out = subprocess.run(
        [*MUTATE_AGENT_CMD.split(), MUTATE_PROMPT.format(program=current, history=hist)],
        capture_output=True, text=True, check=True,
    )
    return out.stdout.strip()


# --- outer loop: keep/rollback on program.md -----------------------------------

def meta_loop(iters: int, budget: Budget | None = None) -> None:
    budget = budget or Budget()
    current = (REPO / "program.md").read_text()
    best_score = evaluate(current, "meta-base", budget)
    history: list[MetaStep] = []

    log = Path(__file__).parent / "meta_results.tsv"
    with log.open("w") as f:
        f.write("iter\tscore\tbest\tkept\trationale\n")

    for i in range(iters):
        candidate = propose_program(current, history)
        s = evaluate(candidate, f"meta-{i}", budget)
        kept = s < best_score
        if kept:
            current, best_score = candidate, s
            (Path(__file__).parent / f"program.kept.{i}.md").write_text(candidate)
        step = MetaStep(iter=i, parent_score=best_score, child_score=s, kept=kept,
                        rationale=candidate.splitlines()[0][:120])
        history.append(step)
        with log.open("a") as f:
            f.write(f"{i}\t{s:.6f}\t{best_score:.6f}\t{kept}\t{step.rationale}\n")


if __name__ == "__main__":
    meta_loop(iters=20)
