"""AR² harness — version archive persistence (jsonl).

save / load round-trip the Archive through jsonl (one Attempt per line).
outer_curve extracts the (version, best-heldout-so-far, hacked?) series
for the two-colored outer-loop plot.
"""
from __future__ import annotations

import json
from pathlib import Path

from harness.contracts import Archive, Attempt, Budget

_DEFAULT_PATH = Path("obs/archive.jsonl")


def save(archive: Archive, path: Path = _DEFAULT_PATH) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w") as fh:
        for attempt in archive.attempts:
            fh.write(attempt.model_dump_json() + "\n")


def load(path: Path = _DEFAULT_PATH) -> Archive:
    archive = Archive()
    if not path.exists():
        return archive
    with path.open() as fh:
        for line in fh:
            line = line.strip()
            if line:
                archive.add(Attempt.model_validate_json(line))
    return archive


def outer_curve(archive: Archive) -> list[tuple[int, float, bool]]:
    """Return (version, best-heldout-so-far, hacked?) for each attempt in order.

    best-heldout-so-far is the running maximum of heldout_reward over clean
    (non-hacked) attempts seen up to and including this version.  For a hacked
    attempt the running max does NOT advance — the caller uses the hacked flag
    to colour the point red without raising the baseline.
    """
    curve: list[tuple[int, float, bool]] = []
    best_clean: float = 0.0
    for a in archive.attempts:
        hacked = bool(a.hack_flags)
        if not hacked:
            best_clean = max(best_clean, a.heldout_reward)
        curve.append((a.version, best_clean, hacked))
    return curve


def efficiency_curve(archive: Archive) -> list[dict]:
    """Per-version inner-loop sample efficiency — the second-derivative instrument.

    For each archived version, aggregate its TRAIN rollouts (the climb the inner
    agent actually runs) into: attempts (scored edits), baseline (first reward),
    final (best reward reached), gain (final-baseline) and gain_per_attempt — the
    climb rate dR/dattempt.  A later AR version that reaches more reward in fewer
    attempts shows up as a higher gain_per_attempt: the meta-loop made the climb
    more EFFICIENT, not merely higher.  Computed from already-persisted rewards;
    no extra instrumentation."""
    out: list[dict] = []
    for a in archive.attempts:
        attempts: list[int] = []
        baselines: list[float] = []
        finals: list[float] = []
        for r in a.train_rollouts or []:
            rw = list(r.rewards or [])
            if not rw:
                continue
            attempts.append(len(rw))
            baselines.append(rw[0])
            finals.append(max(rw))
        if not attempts:
            continue
        n = sum(attempts) / len(attempts)
        base = sum(baselines) / len(baselines)
        fin = sum(finals) / len(finals)
        out.append({
            "version": a.version,
            "attempts": n,
            "baseline": base,
            "final": fin,
            "gain": fin - base,
            "gain_per_attempt": (fin - base) / max(n - 1.0, 1.0),
            "reward_per_attempt": fin / max(n, 1.0),
        })
    return out
