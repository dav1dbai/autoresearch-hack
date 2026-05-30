"""Shared fixtures for env unit tests."""
from __future__ import annotations

import os

import pytest

# Must run before envs/__init__.py imports matmul (test_base loads first alphabetically).
os.environ["MATMUL_STUB"] = "1"


@pytest.fixture(autouse=True)
def _matmul_stub_mode(monkeypatch):
    monkeypatch.setenv("MATMUL_STUB", "1")
    import envs.matmul as matmul_mod
    matmul_mod._STUB_MODE = True
