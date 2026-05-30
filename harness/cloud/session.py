"""Modal app lifecycle — deployed (default) or ephemeral app.run() fallback.

Default (AR2_MODAL_DEPLOYED=1): `app.deploy()` once per driver process, then
invoke via Function.from_name("ar2", "run_rollout"). Shows as a deployed app in
Modal dashboard — not ephemeral.

Fallback (AR2_MODAL_DEPLOYED=0): shared app.run() session (AR2_MODAL_REUSE=1) or
per-batch ephemeral app.run() (AR2_MODAL_REUSE=0).
"""
from __future__ import annotations

import atexit
import os
import threading
from typing import Any

import modal

APP_NAME = "ar2"
_LOCK = threading.Lock()
_RUN_CTX: Any | None = None
_ATExit_REGISTERED = False
_DEPLOYED_ONCE = False


def deployed_enabled() -> bool:
    return os.environ.get("AR2_MODAL_DEPLOYED", "1") == "1"


def reuse_enabled() -> bool:
    if deployed_enabled():
        return False
    return os.environ.get("AR2_MODAL_REUSE", "1") == "1"


def modal_app_mode() -> str:
    if deployed_enabled():
        return "deployed"
    if reuse_enabled():
        return "shared-session"
    return "ephemeral-per-batch"


def ensure_app_deployed(app: modal.App) -> None:
    """Deploy the ar2 app once per process (idempotent)."""
    global _DEPLOYED_ONCE
    if not deployed_enabled():
        return
    with _LOCK:
        if not _DEPLOYED_ONCE:
            deploy_app(app)
            _DEPLOYED_ONCE = True


def ensure_app_session(app: modal.App) -> None:
    """Enter app.run() once; subsequent calls are no-ops (ephemeral fallback only)."""
    global _RUN_CTX, _ATExit_REGISTERED
    if deployed_enabled() or not reuse_enabled():
        return
    with _LOCK:
        if _RUN_CTX is None:
            _RUN_CTX = app.run()
            _RUN_CTX.__enter__()
            if not _ATExit_REGISTERED:
                atexit.register(close_app_session)
                _ATExit_REGISTERED = True


def close_app_session() -> None:
    """Tear down the shared app.run() context (tests / explicit shutdown)."""
    global _RUN_CTX
    with _LOCK:
        if _RUN_CTX is not None:
            try:
                _RUN_CTX.__exit__(None, None, None)
            except Exception:  # noqa: BLE001
                pass
            _RUN_CTX = None


class ephemeral_run:
    """Context manager: deployed (no-op), shared session, or per-call app.run()."""

    def __init__(self, app: modal.App) -> None:
        self._app = app
        self._local_ctx: Any | None = None

    def __enter__(self) -> modal.App:
        if deployed_enabled():
            ensure_app_deployed(self._app)
            return self._app
        if reuse_enabled():
            ensure_app_session(self._app)
            return self._app
        self._local_ctx = self._app.run()
        return self._local_ctx.__enter__()

    def __exit__(self, exc_type, exc, tb) -> bool:
        if deployed_enabled() or reuse_enabled():
            return False
        if self._local_ctx is not None:
            return bool(self._local_ctx.__exit__(exc_type, exc, tb))
        return False


def starmap_run_rollout(
    app: modal.App,
    run_rollout_fn: Any,
    inputs: list[tuple],
) -> list[Any]:
    """Dispatch run_rollout.starmap via deployed app or ephemeral app.run()."""
    if deployed_enabled():
        ensure_app_deployed(app)
        fn = modal.Function.from_name(APP_NAME, "run_rollout")
        return list(fn.starmap(inputs))
    with ephemeral_run(app):
        return list(run_rollout_fn.starmap(inputs))


def invoke_run_evaluate(
    app: modal.App,
    run_evaluate_fn: Any,
    payload: dict,
) -> dict:
    """Dispatch run_evaluate.remote via deployed app or ephemeral app.run()."""
    if deployed_enabled():
        ensure_app_deployed(app)
        fn = modal.Function.from_name(APP_NAME, "run_evaluate")
        return fn.remote(payload)
    with ephemeral_run(app):
        return run_evaluate_fn.remote(payload)


def invoke_run_improve(
    app: modal.App,
    run_improve_fn: Any,
    payload: dict,
) -> dict:
    """Dispatch run_improve.remote via deployed app or ephemeral app.run()."""
    if deployed_enabled():
        ensure_app_deployed(app)
        fn = modal.Function.from_name(APP_NAME, "run_improve")
        return fn.remote(payload)
    with ephemeral_run(app):
        return run_improve_fn.remote(payload)


def deploy_app(app: modal.App) -> None:
    """Deploy the ar2 app persistently (modal deploy equivalent)."""
    from infra.modal.images import assert_hackathon_profile

    assert_hackathon_profile()
    app.deploy()
