"""CLI: run the fixed outer loop on matmul env pools.

Usage:
    uv run python -m harness --help
    uv run python -m harness --stub -K 0

    # GPU kernel opt — Modal agents score on Vast GPU via _vast_gpu_run:
    AR2_BACKEND=modal AR2_GPU_BACKEND=vast uv run python -m harness --gpu --vast-rent -K 1

    # Modal agents + Modal GPU matmul:
    AR2_BACKEND=modal uv run python -m harness --gpu -K 1 -M 1
"""
from __future__ import annotations

import argparse
import atexit
import os
from pathlib import Path

from dotenv import load_dotenv

_REPO_ROOT = Path(__file__).resolve().parents[1]
load_dotenv(_REPO_ROOT / ".env")

os.environ.setdefault("PYTHONUNBUFFERED", "1")

from envs.pools import default_matmul_pools, gpu_matmul_pools
from harness.tracing import sync as db_sync
from harness.contracts import Budget
from harness.loop.outer import drive
from harness.util.progress import progress

_vast_instance: int | None = None


def _vast_cleanup() -> None:
    if _vast_instance is not None:
        from infra.vast.pool import destroy
        progress(f"vast: destroying instance {_vast_instance}")
        destroy(_vast_instance)


def main() -> None:
    global _vast_instance

    parser = argparse.ArgumentParser(description="AR² outer loop driver")
    parser.add_argument("--ar-dir", type=Path, default=Path("ar"), help="Path to ar/ snapshot")
    parser.add_argument("-K", type=int, default=0, help="Meta generations after v0")
    parser.add_argument("-M", type=int, default=1, help="Candidates per frontier parent")
    parser.add_argument(
        "--budget-seconds",
        type=float,
        default=float(os.environ.get("AR2_BUDGET_SECONDS", "300")),
        help="Per-rollout wall budget",
    )
    parser.add_argument(
        "--archive",
        type=Path,
        default=Path("obs/archive.jsonl"),
        help="Where to persist Attempt records (JSONL)",
    )
    parser.add_argument(
        "--traces-db",
        type=Path,
        default=Path("obs/traces.db"),
        help="Canonical SQLite trace store",
    )
    parser.add_argument(
        "--archive-db",
        type=Path,
        default=Path("obs/archive.db"),
        help="Canonical SQLite archive store (queryable curves)",
    )
    parser.add_argument("--stub", action="store_true", help="MATMUL_STUB=1 (fast CPU stub)")
    parser.add_argument(
        "--gpu",
        action="store_true",
        help="GPU matmul pools (large shapes, slow CUDA starter, MATMUL_RUNNER=gpu)",
    )
    parser.add_argument(
        "--vast-rent",
        action="store_true",
        help="Rent a Vast.ai GPU for scoring (sets AR2_GPU_BACKEND=vast + VAST_INSTANCE_ID)",
    )
    parser.add_argument(
        "--modal-ephemeral",
        action="store_true",
        help="Use ephemeral app.run() instead of deployed app (AR2_MODAL_DEPLOYED=0)",
    )
    parser.add_argument(
        "--gpu-smoke",
        action="store_true",
        help="Small GPU matmul shapes (256³) for fast inner-loop smoke",
    )
    parser.add_argument(
        "--no-workshop",
        action="store_true",
        help="Skip Raindrop/OTLP; still writes obs/traces.db + obs/archive.db",
    )
    args = parser.parse_args()

    if args.no_workshop:
        os.environ["RAINDROP_WORKSHOP"] = "0"

    if args.stub:
        os.environ["MATMUL_STUB"] = "1"

    if args.gpu:
        os.environ.setdefault("MATMUL_RUNNER", "gpu")
        os.environ.setdefault("MATMUL_GPU_STARTER", "1")

    if args.vast_rent:
        if not os.environ.get("VAST_API_KEY"):
            parser.error("VAST_API_KEY must be set for --vast-rent")
        os.environ["AR2_GPU_BACKEND"] = "vast"
        from infra.modal.secrets import ensure_vast_modal_secret
        from infra.vast.pool import bootstrap_instance, rent_gpu, ssh_conn

        secret_name = ensure_vast_modal_secret()
        progress(f"vast: Modal secret synced → {secret_name}")

        gpu = os.environ.get("VAST_GPU_NAME", "RTX_4090")
        max_price = os.environ.get("VAST_MAX_PRICE", "0.60")
        progress(f"vast: searching {gpu} offers (dph<={max_price})...")
        _vast_instance = rent_gpu()
        os.environ["VAST_INSTANCE_ID"] = str(_vast_instance)
        atexit.register(_vast_cleanup)
        progress("vast: bootstrapping remote (numpy, torch)...")
        bootstrap_instance(_vast_instance)
        user_host, port = ssh_conn(_vast_instance)
        progress(
            f"vast: instance {_vast_instance} ready — "
            f"Modal rollouts will score via _vast_gpu_run (ssh -p {port} {user_host})"
        )

    if args.gpu and os.environ.get("AR2_BACKEND") == "modal" and not args.vast_rent:
        os.environ.setdefault("AR2_GPU_BACKEND", "modal")

    if args.modal_ephemeral:
        os.environ["AR2_MODAL_DEPLOYED"] = "0"
        os.environ.setdefault("AR2_MODAL_REUSE", "1")
    elif os.environ.get("AR2_BACKEND") == "modal":
        os.environ.setdefault("AR2_MODAL_DEPLOYED", "1")
        if os.environ.get("VAST_API_KEY"):
            from infra.modal.secrets import ensure_vast_modal_secret

            ensure_vast_modal_secret()
        from harness.cloud.register import register_all

        register_all()
        from harness.cloud.runner import app as modal_app
        from harness.cloud.session import ensure_app_deployed, modal_app_mode

        progress(f"modal: deploying app {modal_app.name}...")
        ensure_app_deployed(modal_app)
        progress(f"modal: app mode={modal_app_mode()}")

    progress(
        f"backend={os.environ.get('AR2_BACKEND', 'local')} "
        f"gpu={os.environ.get('AR2_GPU_BACKEND', 'local')} "
        f"workshop={db_sync.workshop_enabled()}"
    )

    if args.gpu_smoke:
        os.environ["AR2_GPU_SMOKE"] = "1"

    if args.gpu and not args.stub:
        from harness.backends.gpu import matmul_runner

        train, heldout = gpu_matmul_pools(runner=matmul_runner(), smoke=args.gpu_smoke)
    else:
        train, heldout = default_matmul_pools(stub=args.stub)

    budget = Budget(wall_seconds=args.budget_seconds, max_concurrency=1)

    archive = drive(
        args.ar_dir.resolve(),
        train,
        heldout,
        budget,
        args.K,
        args.M,
        _persist_path=args.archive,
        _traces_db=args.traces_db,
        _archive_db=args.archive_db,
    )

    if db_sync.workshop_enabled():
        synced = db_sync.sync_archive_jsonl(args.archive, db=db_sync.raindrop_db_path())
        progress(f"raindrop: synced {synced} attempt(s) → {db_sync.raindrop_db_path()}")
    else:
        synced = db_sync.sync_archive_jsonl(args.archive, db=args.archive_db)
        progress(f"local: synced {synced} attempt(s) → {args.archive_db}")

    print(f"Done: {len(archive.attempts)} attempts → {args.archive.resolve()}", flush=True)
    if db_sync.workshop_enabled():
        ws = db_sync.raindrop_db_path()
        print(f"Raindrop DB:   {ws}", flush=True)
        print(f"Raindrop UI:   http://localhost:5899", flush=True)
        print(f"Dashboard:     uv run python -m obs.dashboard", flush=True)
    if args.traces_db.exists():
        print(f"Traces DB:     {args.traces_db.resolve()}", flush=True)
    if args.archive_db.exists():
        print(f"Archive DB:    {args.archive_db.resolve()}", flush=True)


if __name__ == "__main__":
    main()
