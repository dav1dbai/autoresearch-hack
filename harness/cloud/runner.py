"""AR² Modal runner — app, parallel rollout fan-out, and snapshot shipping.

Architecture
────────────
App:        modal.App("ar2")  — same name as infra.modal.images; shares workspace registry.

Volume:     modal.Volume("ar2-snapshots") mounted at /snapshots.
            Before each parallel rollout batch, upload_snapshot() copies the ar/
            candidate directory into the volume via vol.batch_upload().put_directory().
            Containers call vol.reload() so they see the freshly committed snapshot.

Function:   run_rollout(snapshot_ref, env_spec, budget_dict) -> dict
            CPU-only Modal container (no GPU).  Loads the ar/ snapshot from the
            volume, reconstructs the Env, runs solve() with an isolated referee and
            a capped SpawnFn, returns Rollout.model_dump().
            Heavy GPU work (e.g. matmul benchmark) is delegated to harness.backends.gpu
            via the env's runner= seam — NOT baked into run_rollout itself.

Function:   run_evaluate(payload) -> dict
            One Attempt — fans out train envs, then heldout envs, each via run_rollout.
            Groups all rollouts for a version in the Modal call graph (easier triage).

    Fan-out:    run_rollouts_parallel() builds tuples per env; inner spawn cap is
                harness-internal (_SPAWN_FANOUT_CAP in sandbox.py) if ar/ uses spawn().

Backend switch
──────────────
AR2_BACKEND ∈ {local (default), modal} selects the execution path.
infra.modal.images.assert_hackathon_profile() is called before any remote work.

Dynamic snapshot shipping
─────────────────────────
Harness code + envs are baked into sandbox_image at image build time
(add_local_python_source / add_local_dir during Modal image construction).
Candidate ar/ snapshots change every generation and are NOT baked in — they
are written into the ar2-snapshots Volume by upload_snapshot() and read back
from /snapshots/<snapshot_ref> inside run_rollout.  vol.reload() before the
solve() call ensures containers see the latest committed snapshot.
"""
from __future__ import annotations

import importlib.util
import traceback
import uuid
from pathlib import Path
from typing import Any

import modal

from harness.contracts import Archive, Budget, Rollout

# ── Volume for dynamic candidate snapshots ───────────────────────────────────

SNAPSHOT_VOLUME_NAME = "ar2-snapshots"
SNAPSHOT_MOUNT = "/snapshots"

_vol: modal.Volume | None = None


def _snapshot_volume() -> modal.Volume:
    global _vol
    if _vol is None:
        _vol = modal.Volume.from_name(SNAPSHOT_VOLUME_NAME, create_if_missing=True)
    return _vol


# ── Modal app + function ──────────────────────────────────────────────────────

from infra.modal.images import app, sandbox_image  # noqa: E402  — shared app with gpu_backend

_MAX_CONTAINERS = 32  # hard ceiling; Modal will schedule up to this many in parallel


@app.function(
    image=sandbox_image,
    # No GPU here — orchestration only.  GPU work is delegated to gpu_backend
    # via the env's runner= seam inside solve().
    timeout=600,
    max_containers=_MAX_CONTAINERS,
    secrets=[
        modal.Secret.from_dotenv(),                       # INNER_AGENT_CMD / MUTATE_AGENT_CMD
        modal.Secret.from_name("autoresearch-openai"),    # OpenAI key (workspace secret, authoritative)
    ],
    # Volume is referenced by name; the lambda defers evaluation so test stubs
    # do not need to materialise the Volume at import time.
    volumes={SNAPSHOT_MOUNT: modal.Volume.from_name(SNAPSHOT_VOLUME_NAME, create_if_missing=True)},
)
def run_rollout(
    snapshot_ref: str,
    env_spec: dict,
    budget_dict: dict,
    inject_env: dict | None = None,
    trace_id: str | None = None,
) -> dict:
    """Execute one (snapshot × env) rollout inside a Modal container.

    Args:
        snapshot_ref: subdirectory name under /snapshots containing the ar/ snapshot.
        env_spec:     serialised Env — {"module": str, "class": str, "id": str,
                      "split": str, "kwargs": dict}.  Module must be importable
                      inside sandbox_image (harness + envs are baked in).
        budget_dict:  Budget.model_dump().

    Returns:
        Rollout.model_dump() — JSON-serialisable.
    """
    import os

    _snapshot_volume().reload()

    budget = Budget.model_validate(budget_dict)
    trace_id = trace_id or str(uuid.uuid4())
    env_id = env_spec.get("id", "unknown")
    split = env_spec.get("split", "train")

    try:
        if inject_env:
            for key, val in inject_env.items():
                os.environ[key] = val
        os.environ.setdefault("AR2_TRACE_ID", trace_id)

        _ensure_raindrop_codex_env()
        _codex_login()
        env = _load_env(env_spec)
        snap_dir = Path(SNAPSHOT_MOUNT) / snapshot_ref

        inject_fn = None
        if inject_env:

            def inject_fn(_e, _tid):
                return dict(inject_env)

        from harness.runtime.rollout import run_rollout_once

        rollout = run_rollout_once(
            snap_dir,
            env,
            budget,
            inject=inject_fn,
            version=int(os.environ.get("AR2_VERSION", "0")),
            candidate=os.environ.get("AR2_CANDIDATE", ""),
        )
        data = rollout.model_dump()
        if rollout.trace_path:
            tp = Path(rollout.trace_path)
            if tp.exists():
                data["_trace_jsonl"] = tp.read_text()
            else:
                data["_trace_jsonl"] = ""
        else:
            data["_trace_jsonl"] = ""
        return data

    except Exception:
        traceback.print_exc()
        return Rollout(
            env_id=env_id,
            split=split,
            rewards=[],
            final_reward=0.0,
            cost=budget,
            trace_id=trace_id,
            hack_flags=["crash"],
        ).model_dump()


@app.function(
    image=sandbox_image,
    timeout=900,
    max_containers=4,
    secrets=[
        modal.Secret.from_dotenv(),
        modal.Secret.from_name("autoresearch-openai"),
    ],
    volumes={SNAPSHOT_MOUNT: modal.Volume.from_name(SNAPSHOT_VOLUME_NAME, create_if_missing=True)},
)
def run_evaluate(payload: dict) -> dict:
    """Evaluate one candidate (train + heldout) — parent node for Modal triage."""
    import os

    _snapshot_volume().reload()

    gpu_env = payload.get("gpu_env") or {}
    for key, val in gpu_env.items():
        os.environ[key] = str(val)

    snapshot_ref = payload["snapshot_ref"]
    budget_dict = payload["budget"]
    version = int(payload.get("version", 0))
    candidate = str(payload.get("candidate", ""))
    train_specs = payload["train_specs"]
    heldout_specs = payload["heldout_specs"]

    from harness.tracing.telemetry import inject_for_rollout

    class _EnvShim:
        def __init__(self, spec: dict) -> None:
            self.id = spec["id"]
            self.split = spec["split"]

    def _inputs(specs: list[dict]) -> list[tuple]:
        out: list[tuple] = []
        for spec in specs:
            tid = str(uuid.uuid4())
            shim = _EnvShim(spec)
            inject_env = inject_for_rollout(
                shim, trace_id=tid, version=version, candidate=candidate,
            )
            out.append((snapshot_ref, spec, budget_dict, inject_env, tid))
        return out

    train_raw = list(run_rollout.starmap(_inputs(train_specs)))
    heldout_raw = list(run_rollout.starmap(_inputs(heldout_specs)))
    return {"train": train_raw, "heldout": heldout_raw}


# ── env_spec helpers ─────────────────────────────────────────────────────────

def env_to_spec(env: Any) -> dict:
    """Serialise an Env to a dict for run_rollout."""
    kwargs = {
        k: v for k, v in vars(env).items()
        if not k.startswith("_") and not callable(v)
    }
    return {
        "module": type(env).__module__,
        "class": type(env).__qualname__,
        "id": env.id,
        "split": env.split,
        "kwargs": kwargs,
    }


def _load_env(env_spec: dict) -> Any:
    """Rebuild the Env via its real constructor so __init__ runs (sets _runner etc.).

    env_to_spec captures vars(env), which includes computed attrs like `id` that are
    NOT __init__ params; passing those would raise TypeError and force a __new__
    fallback that skips __init__. Filter to actual constructor params instead."""
    import inspect
    import os

    mod = importlib.import_module(env_spec["module"])
    cls = getattr(mod, env_spec["class"].split(".")[-1])
    kwargs = env_spec.get("kwargs", {})
    params = set(inspect.signature(cls.__init__).parameters) - {"self"}
    valid = {k: v for k, v in kwargs.items() if k in params}
    env = cls(**valid)
    if os.environ.get("MATMUL_RUNNER", "cpu").lower() in ("gpu", "modal", "vast"):
        from harness.backends.gpu import matmul_runner

        env._runner = matmul_runner()
    return env


def _codex_login() -> None:
    """Register OPENAI_API_KEY with codex (writes auth.json) so `codex exec` sends a
    Bearer token. Codex ignores the bare env var for its responses transport unless
    `codex login --with-api-key` has run."""
    import os
    import subprocess
    key = os.environ.get("OPENAI_API_KEY")
    if not key:
        return
    try:
        subprocess.run(["codex", "login", "--with-api-key"], input=key,
                       text=True, capture_output=True, timeout=60)
    except Exception:
        pass


def _ensure_raindrop_codex_env() -> None:
    """Wire INNER/MUTATE agent cmds through Raindrop MCP (overrides bare codex exec in .env)."""
    import os
    from pathlib import Path

    import harness

    repo_root = Path(harness.__file__).resolve().parent.parent
    os.environ["AR2_REPO_ROOT"] = str(repo_root)
    os.environ["AR2_MODAL"] = "1"
    os.environ.setdefault("RAINDROP_BIN", "/root/.raindrop/bin/raindrop")
    workshop = os.environ.get("RAINDROP_WORKSHOP_URL", "http://127.0.0.1:5899").rstrip("/")
    os.environ.setdefault("RAINDROP_WORKSHOP_URL", workshop)
    os.environ.setdefault("RAINDROP_LOCAL_DEBUGGER", f"{workshop}/v1/")

    if os.environ.get("RAINDROP_WORKSHOP", "1") != "1":
        return

    wrapper = repo_root / "scripts" / "raindrop_codex_exec.sh"
    if not wrapper.is_file():
        return

    extra = os.environ.get(
        "AR2_CODEX_EXTRA_FLAGS",
        "-m gpt-5-codex -c preferred_auth_method=apikey",
    ).strip()
    cmd = f"{wrapper} exec {extra}".strip()
    os.environ["INNER_AGENT_CMD"] = cmd
    os.environ["MUTATE_AGENT_CMD"] = cmd


@app.function(
    image=sandbox_image,
    timeout=600,
    max_containers=8,
    secrets=[
        modal.Secret.from_dotenv(),
        modal.Secret.from_name("autoresearch-openai"),
    ],
    volumes={SNAPSHOT_MOUNT: modal.Volume.from_name(SNAPSHOT_VOLUME_NAME, create_if_missing=True)},
)
def run_improve(payload: dict) -> dict:
    """Meta improve() in a Modal container — returns volume ref for new version snapshot."""
    import os
    import tempfile
    from pathlib import Path

    from harness.contracts import Archive, Budget

    _snapshot_volume().reload()
    _ensure_raindrop_codex_env()
    _codex_login()

    parent_ar = Path(SNAPSHOT_MOUNT) / payload["parent_ar_ref"]
    archive = Archive.model_validate(payload["archive"])
    budget = Budget.model_validate(payload["budget"])
    traj = payload.get("workshop_traj") or ""
    if traj:
        os.environ["AR2_WORKSHOP_TRAJECTORY"] = str(traj)
    else:
        os.environ.pop("AR2_WORKSHOP_TRAJECTORY", None)
    run_id = str(payload.get("run_id") or "").strip()
    if run_id:
        os.environ["AR2_RUN_ID"] = run_id
    else:
        os.environ.pop("AR2_RUN_ID", None)

    cache = Path("/tmp/ar2_cache")
    cache.mkdir(parents=True, exist_ok=True)
    os.environ["AR2_CACHE_DIR"] = str(cache)
    import harness

    os.environ["AR2_REPO_ROOT"] = str(Path(harness.__file__).resolve().parent.parent)

    from harness.runtime.loader import load_ar

    mod = load_ar(str(parent_ar))

    def _noop_spawn(fn, list_of_args):
        return [fn(*((a,) if not isinstance(a, tuple) else a)) for a in list_of_args]

    version_root = Path(mod.improve(archive, budget, _noop_spawn))
    import base64
    import io
    import tarfile

    buf = io.BytesIO()
    with tarfile.open(fileobj=buf, mode="w:gz") as tar:
        tar.add(version_root, arcname=".")
    return {"tar_b64": base64.b64encode(buf.getvalue()).decode("ascii")}


# ── Volume snapshot upload ────────────────────────────────────────────────────

def upload_snapshot(ar_dir: Path) -> str:
    """Copy ar_dir into the snapshot Volume; return the snapshot_ref key.

    Uses vol.batch_upload() — the canonical Modal Volume batch write API.
    The returned snap_ref is the directory name under SNAPSHOT_MOUNT.
    Call this from the harness host before dispatching a rollout batch.
    """
    vol = _snapshot_volume()
    snap_ref = f"snap_{uuid.uuid4().hex}"
    dest_in_vol = f"/{snap_ref}"  # path INSIDE the volume root

    with vol.batch_upload(force=True) as batch:
        batch.put_directory(str(ar_dir), dest_in_vol)

    return snap_ref


def upload_version_snapshot(version_root: Path) -> str:
    """Upload full version snapshot (ar/ + harness/runtime/) into the Volume."""
    vol = _snapshot_volume()
    snap_ref = f"snap_{uuid.uuid4().hex}"
    dest = f"/{snap_ref}"
    with vol.batch_upload(force=True) as batch:
        batch.put_directory(str(version_root / "ar"), f"{dest}/ar")
        runtime = version_root / "harness" / "runtime"
        if runtime.is_dir():
            batch.put_directory(str(runtime), f"{dest}/harness/runtime")
    return snap_ref


def _extract_version_tar(tar_b64: str, dest: Path) -> Path:
    """Unpack run_improve tarball into a local version snapshot root."""
    import base64
    import io
    import tarfile

    dest.mkdir(parents=True, exist_ok=True)
    data = base64.b64decode(tar_b64)
    with tarfile.open(fileobj=io.BytesIO(data), mode="r:gz") as tar:
        tar.extractall(dest, filter="data")
    return dest


def run_improve_on_modal(
    parent_source_ref: str,
    archive: Archive,
    budget: Budget,
    workshop_traj: str = "",
) -> Path:
    """Host entry: run improve() on Modal, download version snapshot locally."""
    import os
    import tempfile

    from infra.modal.images import assert_hackathon_profile
    from harness.runtime.loader import resolve_ar_dir

    assert_hackathon_profile()
    parent_ar = resolve_ar_dir(parent_source_ref)
    parent_ref = upload_snapshot(parent_ar)
    payload = {
        "parent_ar_ref": parent_ref,
        "archive": archive.model_dump(),
        "workshop_traj": workshop_traj,
        "budget": budget.model_dump(),
        "run_id": os.environ.get("AR2_RUN_ID", ""),
    }
    from harness.cloud.session import invoke_run_improve

    raw = invoke_run_improve(app, run_improve, payload)
    cache = Path(os.environ.get("AR2_CACHE_DIR", "versions"))
    cache.mkdir(parents=True, exist_ok=True)
    local_root = Path(tempfile.mkdtemp(prefix="v_", dir=cache))
    return _extract_version_tar(raw["tar_b64"], local_root)


def _build_rollout_inputs(
    snap_ref: str,
    envs: list[Any],
    budget_dict: dict,
    inject: Any,
    *,
    version: int,
    candidate: str,
) -> list[tuple]:
    inputs: list[tuple] = []
    for env in envs:
        tid = str(uuid.uuid4())
        if inject is not None:
            inject_env = inject(env, tid)
        else:
            from harness.tracing.telemetry import inject_for_rollout
            inject_env = inject_for_rollout(
                env, trace_id=tid, version=version, candidate=candidate,
            )
        inputs.append((snap_ref, env_to_spec(env), budget_dict, inject_env, tid))
    return inputs


def unpack_rollout_results(raw_results: list[dict]) -> list[Rollout]:
    """Convert Modal run_rollout dicts → Rollouts with local trace paths."""
    import tempfile

    rollouts: list[Rollout] = []
    for raw in raw_results:
        data = dict(raw)
        trace_jsonl = data.pop("_trace_jsonl", "")
        rollout = Rollout.model_validate(data)
        if trace_jsonl.strip():
            trace_path = Path(tempfile.gettempdir()) / f"trace_{rollout.trace_id}.jsonl"
            trace_path.write_text(trace_jsonl)
            rollout = rollout.model_copy(update={"trace_path": str(trace_path)})
        rollouts.append(rollout)
    return rollouts


def run_evaluate_on_modal(
    ar_dir: Path,
    train: list[Any],
    heldout: list[Any],
    budget: Budget,
    *,
    version: int = 0,
    candidate: str = "",
) -> tuple[list[Rollout], list[Rollout]]:
    """Host entry: one run_evaluate.remote() per Attempt (single snapshot upload)."""
    from infra.modal.images import assert_hackathon_profile
    from harness.tracing.telemetry import forward_gpu_env

    assert_hackathon_profile()
    snap_ref = upload_snapshot(ar_dir)
    cand = candidate or str(ar_dir)
    payload = {
        "snapshot_ref": snap_ref,
        "train_specs": [env_to_spec(e) for e in train],
        "heldout_specs": [env_to_spec(e) for e in heldout],
        "budget": budget.model_dump(),
        "version": version,
        "candidate": cand,
        "gpu_env": forward_gpu_env(),
    }
    from harness.cloud.session import invoke_run_evaluate

    raw = invoke_run_evaluate(app, run_evaluate, payload)
    return (
        unpack_rollout_results(raw["train"]),
        unpack_rollout_results(raw["heldout"]),
    )


# ── Parallel fan-out entry point (used by score_repo when AR2_BACKEND=modal) ─

def run_rollouts_parallel(
    ar_dir: Path,
    envs: list[Any],
    budget: Budget,
    inject: Any = None,
    *,
    version: int = 0,
    candidate: str = "",
) -> list[Rollout]:
    """Fan out all (ar_dir × env) rollouts concurrently via Modal.

    inject(env, trace_id) -> dict[str, str] is forwarded into each container's
    environment before solve() runs; score spans append to AR2_TRACE_FILE inside
    the container and are pulled back as _trace_jsonl for local db_sync.
    """
    from infra.modal.images import assert_hackathon_profile
    assert_hackathon_profile()

    snap_ref = upload_snapshot(ar_dir)
    budget_dict = budget.model_dump()
    cand = candidate or str(ar_dir)
    inputs = _build_rollout_inputs(
        snap_ref, envs, budget_dict, inject, version=version, candidate=cand,
    )

    from harness.cloud.session import starmap_run_rollout

    raw_results = starmap_run_rollout(app, run_rollout, inputs)
    return unpack_rollout_results(list(raw_results))
