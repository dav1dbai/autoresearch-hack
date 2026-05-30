"""AR² harness — out-of-process referee.

Runs Env.score in a child process so that the AR repo's code can never reach
or modify the grader (DESIGN §2 integrity invariant).  Any exception (crash,
timeout, serialisation error) returns StepResult(reward=0.0).
"""
from __future__ import annotations

import multiprocessing
import pickle

from harness.contracts import Env, ScoreFn, StepResult, Submission


def _score_worker(env_pickle: bytes, sub_pickle: bytes, out: multiprocessing.Queue) -> None:
    try:
        env: Env = pickle.loads(env_pickle)
        sub: Submission = pickle.loads(sub_pickle)
        result = env.score(sub)
        out.put(pickle.dumps(result))
    except Exception as exc:
        out.put(pickle.dumps(StepResult(reward=0.0, raw={"error": str(exc)})))


def make_referee(env: Env) -> ScoreFn:
    """Return a score(Submission) -> StepResult that runs env.score out-of-process."""
    env_pickle = pickle.dumps(env)

    def score(sub: Submission) -> StepResult:
        ctx = multiprocessing.get_context("spawn")
        q: multiprocessing.Queue = ctx.Queue()
        sub_pickle = pickle.dumps(sub)
        p = ctx.Process(target=_score_worker, args=(env_pickle, sub_pickle, q), daemon=True)
        p.start()
        try:
            result_pickle = q.get(timeout=300)
            p.join(timeout=5)
            return pickle.loads(result_pickle)
        except Exception as exc:
            p.kill()
            return StepResult(reward=0.0, raw={"error": str(exc)})

    return score
