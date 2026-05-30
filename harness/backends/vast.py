"""Vast.ai GPU backend — SSH kernel to a rented instance."""
from __future__ import annotations

import json
import os
import tempfile
import time
from pathlib import Path

import modal

from infra.modal.images import app, vast_scorer_image
from infra.modal.secrets import VAST_SECRET_NAME, vast_secret

_VAST_EVAL_SCRIPT = r"""
import sys, json, time, importlib.util
import numpy as np

kernel_path, M, N, K, reps = sys.argv[1], int(sys.argv[2]), int(sys.argv[3]), int(sys.argv[4]), int(sys.argv[5])

spec = importlib.util.spec_from_file_location("_kernel", kernel_path)
mod = importlib.util.module_from_spec(spec)
spec.loader.exec_module(mod)

rng = np.random.default_rng(42)
A = rng.random((M, K), dtype=np.float64)
B = rng.random((K, N), dtype=np.float64)
ref = A @ B

try:
    out = mod.matmul(A.copy(), B.copy())
    correct = bool(np.allclose(out, ref, atol=1e-6, rtol=1e-5))
except Exception as e:
    print(json.dumps({"gflops": 0.0, "correct": False, "seconds": 0.0, "error": str(e)}))
    sys.exit(0)

if not correct:
    print(json.dumps({"gflops": 0.0, "correct": False, "seconds": 0.0}))
    sys.exit(0)

times = []
for _ in range(reps):
    t0 = time.perf_counter()
    mod.matmul(A.copy(), B.copy())
    times.append(time.perf_counter() - t0)
times.sort()
seconds = times[len(times) // 2]
gflops = 2 * M * N * K / seconds / 1e9
print(json.dumps({"gflops": gflops, "correct": True, "seconds": seconds}))
"""


def vast_gpu_eval(
    kernel_src: str,
    problem: dict,
    instance_id: int,
    *,
    api_key: str,
) -> dict:
    """Benchmark kernel on a Vast instance via SSH (host or Modal _vast_gpu_run)."""
    from infra.vast.pool import scp_to, ssh_conn, ssh_run

    user_host, port = ssh_conn(instance_id, api_key=api_key)

    with tempfile.NamedTemporaryFile(suffix=".py", mode="w", delete=False) as kf:
        kf.write(kernel_src)
        kernel_path = kf.name
    with tempfile.NamedTemporaryFile(suffix=".py", mode="w", delete=False) as tf:
        tf.write(_VAST_EVAL_SCRIPT)
        eval_script_path = tf.name

    M = problem.get("M", 128)
    N = problem.get("N", 128)
    K = problem.get("K", 128)
    reps = problem.get("reps", 20)
    retries = 3
    retry_delay = 5.0

    try:
        for local_f, remote_f in [
            (kernel_path, "/tmp/kernel.py"),
            (eval_script_path, "/tmp/eval_kernel.py"),
        ]:
            scp_to(
                local_f,
                user_host,
                port,
                remote_f,
                retries=retries,
                retry_delay=retry_delay,
            )

        remote_cmd = f"python3 /tmp/eval_kernel.py /tmp/kernel.py {M} {N} {K} {reps}"
        last_err = ""
        for attempt in range(retries):
            result = ssh_run(user_host, port, remote_cmd, timeout=120)
            out = result.stdout.strip()
            if result.returncode == 0 and out:
                return json.loads(out)
            last_err = (result.stderr or out or "empty stdout").strip()
            if attempt < retries - 1:
                time.sleep(retry_delay)

        return {
            "gflops": 0.0,
            "correct": False,
            "seconds": 0.0,
            "error": last_err[:500],
        }
    except Exception as e:
        return {"gflops": 0.0, "correct": False, "seconds": 0.0, "error": str(e)}
    finally:
        import os as _os
        _os.unlink(kernel_path)
        _os.unlink(eval_script_path)


@app.function(
    image=vast_scorer_image,
    secrets=[vast_secret()],
    timeout=300,
    max_containers=8,
)
def _vast_gpu_run(kernel_src: str, problem: dict, instance_id: int) -> dict:
    """Score a kernel on Vast from Modal cloud (SSH; VAST_API_KEY from Modal secret)."""
    api_key = os.environ.get("VAST_API_KEY", "")
    if not api_key:
        return {
            "gflops": 0.0,
            "correct": False,
            "seconds": 0.0,
            "error": (
                f"VAST_API_KEY missing in Modal secret {VAST_SECRET_NAME!r} — "
                f"run: modal secret create {VAST_SECRET_NAME} VAST_API_KEY=... --force"
            ),
        }
    return vast_gpu_eval(kernel_src, problem, instance_id, api_key=api_key)


def _invoke_vast_gpu_run(kernel_src: str, problem: dict, instance_id: int) -> dict:
    """Call _vast_gpu_run from inside a Modal rollout container (or host when deployed)."""
    on_modal = os.environ.get("MODAL_ENVIRONMENT") is not None
    if not on_modal:
        from harness.cloud.session import deployed_enabled, ensure_app_deployed

        if deployed_enabled():
            ensure_app_deployed(app)
            fn = modal.Function.from_name(app.name, "_vast_gpu_run")
            return fn.remote(kernel_src, problem, instance_id)
    return _vast_gpu_run.remote(kernel_src, problem, instance_id)


class VastBackend:
    """Benchmark on a Vast.ai GPU instance via SSH.

    Requires VAST_API_KEY and a pre-rented instance (VAST_INSTANCE_ID or
    instance_id=).  Rent with `uv run python -m harness --gpu --vast-rent`.
    """

    def __init__(self, instance_id: int | None = None) -> None:
        env_inst = os.environ.get("VAST_INSTANCE_ID")
        self._instance_id = instance_id if instance_id is not None else (
            int(env_inst) if env_inst else None
        )
        self._api_key = os.environ.get("VAST_API_KEY", "")

    def run(self, kernel_path: Path, problem: dict) -> dict:
        if self._instance_id is None:
            raise RuntimeError(
                "VastBackend requires an instance_id — rent an instance first "
                "with `python -m harness --vast-rent` or set VAST_INSTANCE_ID"
            )
        on_modal = os.environ.get("MODAL_ENVIRONMENT") is not None
        if not on_modal and not self._api_key:
            raise RuntimeError("VAST_API_KEY not set")
        kernel_src = kernel_path.read_text()
        if on_modal:
            try:
                return _invoke_vast_gpu_run(kernel_src, problem, self._instance_id)
            except Exception as e:
                return {"gflops": 0.0, "correct": False, "seconds": 0.0, "error": str(e)}
        return vast_gpu_eval(
            kernel_src, problem, self._instance_id, api_key=self._api_key,
        )

    def _run_on_instance(self, kernel_path: Path, problem: dict) -> dict:
        """Back-compat alias — prefer run()."""
        return self.run(kernel_path, problem)
