"""Inner-curve metrics for AR² dashboard (D-04)."""
from __future__ import annotations

import statistics

from harness.contracts import Archive, Attempt, Rollout


def inner_slope(rewards: list[float]) -> float:
    """OLS slope of reward vs step index t (0..n-1)."""
    n = len(rewards)
    if n < 2:
        return 0.0
    xs = list(range(n))
    x_mean = sum(xs) / n
    y_mean = sum(rewards) / n
    num = sum((x - x_mean) * (y - y_mean) for x, y in zip(xs, rewards))
    den = sum((x - x_mean) ** 2 for x in xs)
    return num / den if den else 0.0


def rollout_slope(r: Rollout) -> float:
    return inner_slope(list(r.rewards))


def S(attempt: Attempt) -> float:
    """Mean inner slope over held-out rollouts."""
    if not attempt.heldout_rollouts:
        return 0.0
    return statistics.mean(rollout_slope(r) for r in attempt.heldout_rollouts)


def delta_S(archive: Archive, version: int) -> float | None:
    """S(N) - S(N-1) for consecutive versions by parent lineage best-effort."""
    by_v = {a.version: a for a in archive.attempts}
    if version not in by_v or version == 0:
        return None
    prev_versions = [v for v in by_v if v < version]
    if not prev_versions:
        return None
    prev = max(prev_versions)
    return S(by_v[version]) - S(by_v[prev])


def bootstrap_ci(values: list[float], n_boot: int = 200, alpha: float = 0.05) -> tuple[float, float]:
    """Simple bootstrap CI for the mean."""
    if not values:
        return 0.0, 0.0
    if len(values) == 1:
        return values[0], values[0]
    import random

    rng = random.Random(0)
    means: list[float] = []
    for _ in range(n_boot):
        sample = [values[rng.randrange(len(values))] for _ in values]
        means.append(statistics.mean(sample))
    means.sort()
    lo = means[int((alpha / 2) * n_boot)]
    hi = means[int((1 - alpha / 2) * n_boot) - 1]
    return lo, hi
