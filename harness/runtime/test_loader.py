"""Tests for version snapshot loading (D-15)."""
from __future__ import annotations

from pathlib import Path

import pytest

from harness.runtime.loader import copy_version_snapshot, load_ar, resolve_ar_dir


def test_resolve_ar_dir_flat_legacy(tmp_path: Path) -> None:
    ar = tmp_path / "ar"
    ar.mkdir()
    (ar / "entrypoint.py").write_text("def solve(): ...\n")
    assert resolve_ar_dir(ar) == ar


def test_resolve_ar_dir_version_root(tmp_path: Path) -> None:
    root = tmp_path / "v_abc"
    (root / "ar").mkdir(parents=True)
    (root / "ar" / "entrypoint.py").write_text(
        "def solve(t,b,s,sp): pass\ndef improve(a,b,sp): pass\n"
    )
    assert resolve_ar_dir(root) == root / "ar"


def test_copy_version_snapshot_layout(tmp_path: Path) -> None:
    ar = tmp_path / "seed" / "ar"
    ar.mkdir(parents=True)
    (ar / "entrypoint.py").write_text("# seed\n")
    repo = tmp_path / "repo"
    rt = repo / "harness" / "runtime"
    rt.mkdir(parents=True)
    (rt / "score.py").write_text("# score\n")

    dest = tmp_path / "v_out"
    copy_version_snapshot(ar, repo, dest=dest)

    assert (dest / "ar" / "entrypoint.py").exists()
    assert (dest / "harness" / "runtime" / "score.py").exists()


def test_load_ar_from_version_root(tmp_path: Path) -> None:
    root = tmp_path / "v1"
    ar = root / "ar"
    ar.mkdir(parents=True)
    (ar / "entrypoint.py").write_text(
        "def solve(t,b,s,sp):\n"
        "    from harness.contracts import Submission\n"
        "    return Submission(workdir=t.workdir)\n"
        "def improve(a,b,sp):\n"
        "    return __import__('pathlib').Path('.')\n"
    )
    mod = load_ar(str(root))
    assert callable(mod.solve)
    assert callable(mod.improve)
