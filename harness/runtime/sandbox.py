"""AR² harness — Modal sandbox wrapper + capped fanout primitive.

Modal API (2026):
  Sandbox.create(image, app=, timeout=, block_network=, workdir=, env=)
  sb.exec(*args) -> ContainerProcess  (.stdout.read(), .wait(), .returncode)
  sb.filesystem.read_text(path) / .write_text(data, path)
  sb.terminate()

SpawnFn backend selection via AR2_BACKEND:
  local (default) — ThreadPoolExecutor capped at max_concurrency.
  modal           — Function.spawn() calls fanned out behind a semaphore so
                    at most max_concurrency are in-flight at once; each call
                    returns a FunctionCall whose .get() is awaited in a thread.
                    The cap lives here in the harness; ar/ receives a plain
                    callable and cannot exceed it.
"""
from __future__ import annotations

import concurrent.futures
import os
from collections.abc import Callable
from typing import Any

import modal

from harness.contracts import SpawnFn

_BACKEND = os.environ.get("AR2_BACKEND", "local")


def make_sandbox(
    image: modal.Image,
    timeout_s: int,
    block_network: bool = False,
    workdir: str = "/work",
    env: dict[str, str] | None = None,
) -> modal.Sandbox:
    """Create a gVisor-isolated Modal sandbox (gVisor is the Modal default)."""
    app = modal.App.lookup("ar2", create_if_missing=True)
    return modal.Sandbox.create(
        image=image,
        app=app,
        timeout=timeout_s,
        block_network=block_network,
        workdir=workdir,
        env=env or {},
    )


def exec(sb: modal.Sandbox, *args: str) -> tuple[str, str, int]:
    """Run a command in the sandbox, returning (stdout, stderr, returncode)."""
    proc = sb.exec(*args)
    stdout = proc.stdout.read()
    stderr = proc.stderr.read()
    proc.wait()
    return stdout, stderr, proc.returncode


def read_file(sb: modal.Sandbox, path: str) -> str:
    return sb.filesystem.read_text(path)


def write_file(sb: modal.Sandbox, path: str, text: str) -> None:
    sb.filesystem.write_text(text, path)


# Hard ceiling if a future ar/ calls spawn(); not exposed on Budget or CLI.
_SPAWN_FANOUT_CAP = 32


def make_spawn() -> SpawnFn:
    """Return spawn(fn, args) capped at _SPAWN_FANOUT_CAP (harness-internal only)."""
    if _BACKEND == "modal":
        return _make_spawn_modal(_SPAWN_FANOUT_CAP)
    return _make_spawn_local(_SPAWN_FANOUT_CAP)


def _make_spawn_local(max_concurrency: int) -> SpawnFn:
    def spawn(fn: Callable[..., Any], list_of_args: list[Any]) -> list[Any]:
        if not list_of_args:
            return []
        with concurrent.futures.ThreadPoolExecutor(max_workers=max_concurrency) as pool:
            futures = [
                pool.submit(fn, *((a,) if not isinstance(a, tuple) else a))
                for a in list_of_args
            ]
            return [f.result() for f in concurrent.futures.as_completed(futures)]

    return spawn


def _make_spawn_modal(max_concurrency: int) -> SpawnFn:
    import threading

    sem = threading.Semaphore(max_concurrency)

    def spawn(fn: Any, list_of_args: list[Any]) -> list[Any]:
        if not list_of_args:
            return []

        def _call_one(args):
            unpacked = (args,) if not isinstance(args, tuple) else args
            sem.acquire()
            try:
                fc = fn.spawn(*unpacked)
                return fc.get()
            finally:
                sem.release()

        with concurrent.futures.ThreadPoolExecutor(max_workers=max_concurrency) as pool:
            futures = [pool.submit(_call_one, a) for a in list_of_args]
            return [f.result() for f in concurrent.futures.as_completed(futures)]

    return spawn
