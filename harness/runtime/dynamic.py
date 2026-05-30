"""Load harness/runtime modules from version snapshots when present (D-15b)."""
from __future__ import annotations

import importlib.util
import threading
from contextlib import contextmanager
from pathlib import Path
from types import ModuleType

from harness.runtime.loader import resolve_version_root

_tls = threading.local()


def using_host_runtime() -> bool:
    return bool(getattr(_tls, "force_host", False))


@contextmanager
def force_host_runtime():
    """Prevent re-loading snapshot score while executing snapshot score module."""
    prev = getattr(_tls, "force_host", False)
    _tls.force_host = True
    try:
        yield
    finally:
        _tls.force_host = prev


def _load_module_from_path(path: Path, name: str) -> ModuleType:
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise ImportError(f"Cannot load {path}")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)  # type: ignore[attr-defined]
    return mod


def snapshot_runtime_dir(candidate: str | Path) -> Path | None:
    """Return snapshot harness/runtime/ if it exists."""
    if not candidate:
        return None
    root = resolve_version_root(candidate)
    rt = root / "harness" / "runtime"
    return rt if rt.is_dir() else None


def has_snapshot_runtime(candidate: str | Path) -> bool:
    rt = snapshot_runtime_dir(candidate)
    return rt is not None and (rt / "score.py").is_file()


def load_runtime_module(candidate: str | Path, module_name: str) -> ModuleType:
    """Prefer snapshot harness/runtime/{module}.py; fall back to host package."""
    if not using_host_runtime():
        rt = snapshot_runtime_dir(candidate)
        if rt is not None:
            snap_path = rt / f"{module_name}.py"
            if snap_path.is_file():
                tag = abs(hash(str(snap_path.resolve())))
                return _load_module_from_path(snap_path, f"snap_rt_{module_name}_{tag}")
    return importlib.import_module(f"harness.runtime.{module_name}")
