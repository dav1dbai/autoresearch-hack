"""Tests for harness/loop/snapshot.py and D-07 smoke check."""
from __future__ import annotations

import os
from pathlib import Path

import pytest

from harness.loop.snapshot import (
    copy_version_snapshot,
    create_version_snapshot,
    resolve_repo_root,
    smoke_check_version_snapshot,
)


def test_resolve_repo_root_from_env(tmp_path, monkeypatch):
    repo = tmp_path / "repo"
    (repo / "harness").mkdir(parents=True)
    (repo / "ar").mkdir()
    monkeypatch.setenv("AR2_REPO_ROOT", str(repo))
    assert resolve_repo_root() == repo.resolve()


def test_smoke_check_rejects_missing_solve(tmp_path):
    root = tmp_path / "v_bad"
    (root / "ar").mkdir(parents=True)
    (root / "ar" / "entrypoint.py").write_text("def improve(a,b,sp): pass\n")
    ok, err = smoke_check_version_snapshot(root)
    assert not ok
    assert "solve" in err


def test_smoke_check_accepts_valid_snapshot(tmp_path):
    root = tmp_path / "v_ok"
    ar = root / "ar"
    ar.mkdir(parents=True)
    (ar / "entrypoint.py").write_text(
        "def solve(t,b,s,sp): pass\ndef improve(a,b,sp): pass\n"
    )
    ok, err = smoke_check_version_snapshot(root)
    assert ok
    assert err == ""


def test_create_version_snapshot_layout(tmp_path, monkeypatch):
    repo = tmp_path / "repo"
    ar = repo / "ar"
    ar.mkdir(parents=True)
    (ar / "entrypoint.py").write_text("# seed\n")
    rt = repo / "harness" / "runtime"
    rt.mkdir(parents=True)
    (rt / "score.py").write_text("# score\n")
    monkeypatch.setenv("AR2_REPO_ROOT", str(repo))
    monkeypatch.setenv("AR2_CACHE_DIR", str(tmp_path / "versions"))

    root = create_version_snapshot(ar)
    assert (root / "ar" / "entrypoint.py").exists()
    assert (root / "harness" / "runtime" / "score.py").exists()
