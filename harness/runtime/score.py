"""AR² harness — black-box sandbox runner.

score_repo:
  For each Env:
    1. Optionally call inject(env, trace_id) for telemetry env-vars.
    2. Import ar.entrypoint.solve from ar_dir and run it with:
         score  = make_referee(env)   (out-of-process grader)
         spawn  = make_spawn()  (harness-internal cap if ar/ uses it)
    3. Record the inner-curve rewards from each score() call.
    4. Append score spans to a local trace.jsonl (local backend).
    5. Return a Rollout (final_reward = best reward seen).

Crash / timeout / cap-breach => Rollout(final_reward=0.0, hack_flags=["crash"]).

Backend selection via AR2_BACKEND:
  local (default) — in-process sequential runner; used for all offline tests.
  modal           — fans all (ar_dir × env) rollouts out in parallel via
                    harness.cloud.runner.run_rollouts_parallel.
"""
from __future__ import annotations

import inspect
import os
from collections.abc import Callable
from pathlib import Path

from harness.contracts import Budget, Env, Rollout
from harness.runtime.rollout import InjectFn, run_rollout_once


def _backend() -> str:
    return os.environ.get("AR2_BACKEND", "local")


def _normalize_inject(
    inject: Callable[..., dict[str, str]] | None,
    *,
    version: int,
    candidate: str,
) -> InjectFn | None:
    """Accept inject(env) or inject(env, trace_id); default to inject_for_rollout."""
    if inject is None:
        from harness.tracing.telemetry import inject_for_rollout

        def _default(env: Env, trace_id: str) -> dict[str, str]:
            return inject_for_rollout(
                env, trace_id=trace_id, version=version, candidate=candidate,
            )

        return _default

    params = inspect.signature(inject).parameters

    if len(params) >= 2:

        def _two_arg(env: Env, trace_id: str) -> dict[str, str]:
            return inject(env, trace_id)

        return _two_arg

    def _one_arg(env: Env, trace_id: str) -> dict[str, str]:
        return inject(env)

    return _one_arg


def score_repo(
    ar_dir: Path,
    envs: list[Env],
    budget: Budget,
    inject: Callable[..., dict[str, str]] | None = None,
    *,
    version: int = 0,
    candidate: str = "",
) -> list[Rollout]:
    """Run the AR repo on each Env, return one Rollout per Env.

    AR2_BACKEND=local  (default): runs each env sequentially in-process.
    AR2_BACKEND=modal : fans all (ar_dir × env) rollouts out in parallel via Modal.
    """
    fn = _normalize_inject(inject, version=version, candidate=candidate or str(ar_dir))

    if _backend() == "modal":
        from harness.cloud.runner import run_rollouts_parallel
        return run_rollouts_parallel(
            ar_dir, envs, budget, fn, version=version, candidate=candidate or str(ar_dir),
        )

    return [
        run_rollout_once(
            ar_dir,
            env,
            budget,
            inject=fn,
            version=version,
            candidate=candidate or str(ar_dir),
        )
        for env in envs
    ]


def evaluate_rollouts(
    ar_dir: Path,
    train: list[Env],
    heldout: list[Env],
    budget: Budget,
    inject: Callable[..., dict[str, str]] | None = None,
    *,
    version: int = 0,
    candidate: str = "",
) -> tuple[list[Rollout], list[Rollout]]:
    """Score train + heldout envs; Modal batches both when AR2_MODAL_RUN_EVALUATE=1."""
    cand = candidate or str(ar_dir)
    from harness.runtime.dynamic import force_host_runtime, has_snapshot_runtime, load_runtime_module

    if has_snapshot_runtime(cand):
        snap_score = load_runtime_module(cand, "score")
        fn = getattr(snap_score, "evaluate_rollouts", None)
        if callable(fn) and fn.__module__ != __name__:
            with force_host_runtime():
                return fn(
                    ar_dir, train, heldout, budget, inject,
                    version=version, candidate=cand,
                )
    return _evaluate_rollouts_host(
        ar_dir, train, heldout, budget, inject, version=version, candidate=cand,
    )


def _evaluate_rollouts_host(
    ar_dir: Path,
    train: list[Env],
    heldout: list[Env],
    budget: Budget,
    inject: Callable[..., dict[str, str]] | None = None,
    *,
    version: int = 0,
    candidate: str = "",
) -> tuple[list[Rollout], list[Rollout]]:
    """Host evaluate_rollouts implementation (D-15b entry after snapshot dispatch)."""
    cand = candidate or str(ar_dir)
    if _backend() == "modal" and os.environ.get("AR2_MODAL_RUN_EVALUATE", "1") == "1":
        from harness.cloud.runner import run_evaluate_on_modal

        return run_evaluate_on_modal(
            ar_dir, train, heldout, budget, version=version, candidate=cand,
        )
    return (
        score_repo(ar_dir, train, budget, inject, version=version, candidate=cand),
        score_repo(ar_dir, heldout, budget, inject, version=version, candidate=cand),
    )


def modal_backend_label() -> str | None:
    """Return Modal app mode string for logging, or None when not on modal backend."""
    if _backend() != "modal":
        return None
    from harness.cloud.session import modal_app_mode

    return modal_app_mode()
