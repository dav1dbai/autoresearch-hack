"""Load mutable ar/ snapshots and version roots from Attempt.source_ref."""
from __future__ import annotations

import importlib.util
from pathlib import Path

# Re-export for tests and legacy imports — canonical implementation in harness.loop.snapshot.
from harness.loop.snapshot import copy_version_snapshot  # noqa: F401


def resolve_ar_dir(source_ref: str | Path) -> Path:
    """Return the ar/ directory inside a version snapshot (or the path itself)."""
    root = Path(source_ref)
    nested = root / "ar" / "entrypoint.py"
    if nested.is_file():
        return root / "ar"
    if (root / "entrypoint.py").is_file():
        return root
    raise FileNotFoundError(f"No ar entrypoint under {root}")


def resolve_version_root(source_ref: str | Path) -> Path:
    """Return the version snapshot root (parent of ar/ when nested)."""
    root = Path(source_ref)
    if (root / "ar" / "entrypoint.py").is_file():
        return root
    if (root / "entrypoint.py").is_file():
        return root.parent if root.name == "ar" else root
    return root


def load_ar(source_ref: str):
    """Load an ar/ snapshot's entrypoint module (exposes solve and improve)."""
    ar_dir = resolve_ar_dir(source_ref)
    ep_path = ar_dir / "entrypoint.py"
    spec = importlib.util.spec_from_file_location(
        f"ar._snap_{abs(hash(str(ar_dir.resolve())))}", ep_path,
    )
    if spec is None or spec.loader is None:
        raise ImportError(f"Cannot load ar entrypoint from {ep_path}")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)  # type: ignore[attr-defined]
    return mod
