"""Load mutable ar/ snapshots by source_ref (path to an ar/ directory)."""
from __future__ import annotations

import importlib.util
from pathlib import Path


def load_ar(source_ref: str):
    """Load an ar/ snapshot's entrypoint module (exposes solve and improve).

    source_ref is the path to an ar/ snapshot (what the Archive stores).
    """
    ar_dir = Path(source_ref)
    ep_path = ar_dir / "entrypoint.py"
    spec = importlib.util.spec_from_file_location(f"ar._snap_{abs(hash(source_ref))}", ep_path)
    if spec is None or spec.loader is None:
        raise ImportError(f"Cannot load ar entrypoint from {ep_path}")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)  # type: ignore[attr-defined]
    return mod
