"""Run one (ar_dir × env) rollout — shared by local and Modal paths."""
from __future__ import annotations

import inspect
import os
import tempfile
import time
import traceback
import uuid
from collections.abc import Callable
from pathlib import Path

from harness.contracts import Budget, Env, Rollout, Submission
from harness.runtime.referee import make_referee
from harness.runtime.sandbox import make_spawn
from harness.tracing.telemetry import inject_for_rollout, write_span

InjectFn = Callable[[Env, str], dict[str, str]]


def _call_inject(
    inject: InjectFn | Callable[[Env], dict[str, str]],
    env: Env,
    trace_id: str,
    trace_path: Path,
    *,
    version: int,
    candidate: str,
) -> dict[str, str]:
    params = inspect.signature(inject).parameters
    if len(params) >= 2:
        obs = inject(env, trace_id)
    else:
        obs = inject(env)  # type: ignore[call-arg]
    merged = dict(obs)
    merged.setdefault("AR2_TRACE_FILE", str(trace_path))
    merged.setdefault("AR2_TRACE_ID", trace_id)
    return merged


def run_rollout_once(
    ar_dir: Path,
    env: Env,
    budget: Budget,
    *,
    inject: InjectFn | Callable[[Env], dict[str, str]] | None = None,
    version: int = 0,
    candidate: str = "",
) -> Rollout:
    """Execute solve() for one env; record inner-curve rewards and optional trace."""
    trace_id = str(uuid.uuid4())
    rewards: list[float] = []
    trace_path = Path(tempfile.mkdtemp(prefix="ar2_trace_")) / "trace.jsonl"

    if inject is not None:
        obs_env = _call_inject(
            inject, env, trace_id, trace_path, version=version, candidate=candidate,
        )
    else:
        obs_env = inject_for_rollout(
            env, trace_id=trace_id, version=version, candidate=candidate, trace_file=trace_path,
        )

    saved_env: dict[str, str | None] = {}
    for key, val in obs_env.items():
        saved_env[key] = os.environ.get(key)
        os.environ[key] = val

    try:
        gpu_scoring = os.environ.get("MATMUL_RUNNER", "cpu").lower() in (
            "gpu", "modal", "vast",
        ) or os.environ.get("AR2_GPU_BACKEND", "local") != "local"
        raw_score = env.score if gpu_scoring else make_referee(env)

        def tracked_score(sub: Submission):
            t0 = time.perf_counter()
            result = raw_score(sub)
            rewards.append(result.reward)
            write_span(
                model=os.environ.get("AR2_MODEL", "agent"),
                prompt_tokens=0,
                completion_tokens=0,
                latency_ms=(time.perf_counter() - t0) * 1000.0,
                tool_name="score",
                tool_input=sub.notes or "",
                trace_file=trace_path,
            )
            return result

        spawn = make_spawn(budget.max_concurrency)
        task = env.reset()
        from harness.runtime.loader import load_ar

        solve = load_ar(str(ar_dir)).solve
        solve(task, budget, tracked_score, spawn)

        final_reward = max(rewards) if rewards else 0.0
        has_trace = trace_path.exists() and trace_path.stat().st_size > 0
        return Rollout(
            env_id=env.id,
            split=env.split,
            rewards=rewards,
            final_reward=final_reward,
            cost=budget,
            trace_id=trace_id,
            trace_path=str(trace_path) if has_trace else "",
        )
    except Exception:
        traceback.print_exc()
        return Rollout(
            env_id=env.id,
            split=env.split,
            rewards=rewards,
            final_reward=0.0,
            cost=budget,
            trace_id=trace_id,
            trace_path=str(trace_path) if trace_path.exists() else "",
            hack_flags=["crash"],
        )
    finally:
        for key, prev in saved_env.items():
            if prev is None:
                os.environ.pop(key, None)
            else:
                os.environ[key] = prev
