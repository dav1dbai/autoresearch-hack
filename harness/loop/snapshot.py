"""Version snapshot creation and validation (D-15, D-07).

Host-only — not copied into candidate snapshots for meta-agent editing.
"""
from __future__ import annotations

import os
import shutil
import tempfile
from pathlib import Path

_MUTABLE_HARNESS_DIRS = ("runtime",)


def resolve_repo_root() -> Path:
    """Host repository root — never infer from ar/ location inside a snapshot."""
    raw = os.environ.get("AR2_REPO_ROOT")
    if raw:
        root = Path(raw).resolve()
        if (root / "harness").is_dir() and (root / "ar").is_dir():
            return root
        raise RuntimeError(f"AR2_REPO_ROOT invalid (missing harness/ or ar/): {root}")
    # Fallback for tests / standalone improve without explicit env.
    here = Path(__file__).resolve()
    for parent in here.parents:
        if (parent / "harness").is_dir() and (parent / "ar").is_dir():
            return parent
    raise RuntimeError("Set AR2_REPO_ROOT to the host repo root before improve()")


def _ignore_pycache(_dir: str, names: list[str]) -> set[str]:
    skip = {"__pycache__", ".pytest_cache"}
    return {n for n in names if n in skip or n.startswith("test_")}


def copy_version_snapshot(
    ar_dir: Path,
    repo_root: Path,
    *,
    dest: Path,
) -> Path:
    """Copy mutable autoresearch slice into a fresh version directory."""
    dest.mkdir(parents=True, exist_ok=True)
    shutil.copytree(ar_dir, dest / "ar", dirs_exist_ok=True)
    for part in _MUTABLE_HARNESS_DIRS:
        src = repo_root / "harness" / part
        if src.is_dir():
            shutil.copytree(
                src,
                dest / "harness" / part,
                dirs_exist_ok=True,
                ignore=_ignore_pycache,
            )
    return dest


def create_version_snapshot(
    source_ref: str | Path,
    *,
    repo_root: Path | None = None,
    cache_root: Path | None = None,
) -> Path:
    """Create a new version snapshot from an existing ar/ or version root."""
    from harness.runtime.loader import resolve_ar_dir

    repo = repo_root or resolve_repo_root()
    ar_dir = resolve_ar_dir(source_ref)
    cache = cache_root or Path(os.environ.get("AR2_CACHE_DIR", repo / "versions"))
    cache.mkdir(parents=True, exist_ok=True)
    version_root = Path(tempfile.mkdtemp(prefix="v_", dir=cache))
    return copy_version_snapshot(ar_dir, repo, dest=version_root)


def smoke_check_version_snapshot(version_root: Path) -> tuple[bool, str]:
    """D-07: reject broken snapshots before evaluate()."""
    try:
        from harness.runtime.loader import load_ar

        mod = load_ar(str(version_root))
        if not callable(getattr(mod, "solve", None)):
            return False, "entrypoint missing callable solve"
        if not callable(getattr(mod, "improve", None)):
            return False, "entrypoint missing callable improve"
        return True, ""
    except Exception as exc:
        return False, str(exc)
