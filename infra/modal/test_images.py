"""Offline tests for infra/images.py and infra/collector.py.

All Modal I/O is blocked by monkeypatching; no network calls are made.
"""
from __future__ import annotations

import os
import sys
import types
from unittest.mock import MagicMock, patch


# ---------------------------------------------------------------------------
# Helpers — build a minimal modal stub so imports work without a token.
# We patch at the sys.modules level before importing infra.modal.images so the
# module-level `import modal` resolves to our stub.
# ---------------------------------------------------------------------------

def _make_modal_stub() -> types.ModuleType:
    modal = types.ModuleType("modal")

    class _Image:
        def debian_slim(self, python_version: str = "3.12") -> "_Image":
            return self
        def apt_install(self, *pkgs: str) -> "_Image":
            return self
        def run_commands(self, *cmds: str) -> "_Image":
            return self
        def pip_install(self, *pkgs: str) -> "_Image":
            return self
        def add_local_python_source(self, *modules: str) -> "_Image":
            return self
        def add_local_dir(self, *a, **kw) -> "_Image":
            return self

    class _App:
        def __init__(self, name: str) -> None:
            self.name = name
        def function(self, **kwargs):
            def decorator(fn):
                return fn
            return decorator

    class _Volume:
        @staticmethod
        def from_name(name: str, create_if_missing: bool = False) -> "_Volume":
            return _Volume()
        def commit(self) -> None:
            pass

    def fastapi_endpoint(method: str = "GET", **kwargs):
        def decorator(fn):
            return fn
        return decorator

    modal.Image = _Image()
    modal.App = _App
    modal.Volume = _Volume
    modal.fastapi_endpoint = fastapi_endpoint
    return modal


# ---------------------------------------------------------------------------
# Fixtures — inject the stub before any infra import.
# ---------------------------------------------------------------------------

import pytest


@pytest.fixture(autouse=True)
def _stub_modal(monkeypatch):
    stub = _make_modal_stub()
    monkeypatch.setitem(sys.modules, "modal", stub)
    # Stub load_dotenv to a no-op so the real .env doesn't override monkeypatched env vars.
    dotenv_mod = sys.modules.get("dotenv") or types.ModuleType("dotenv")
    dotenv_mod.load_dotenv = lambda *a, **kw: None  # type: ignore[attr-defined]
    monkeypatch.setitem(sys.modules, "dotenv", dotenv_mod)
    # Ensure infra modules are re-imported fresh with the stub each test.
    for key in list(sys.modules):
        if key.startswith("infra"):
            monkeypatch.delitem(sys.modules, key, raising=False)
    yield stub


# ---------------------------------------------------------------------------
# infra.modal.images — assert_hackathon_profile
# ---------------------------------------------------------------------------

class TestAssertHackathonProfile:
    def test_passes_on_correct_profile(self, monkeypatch):
        monkeypatch.setenv("MODAL_PROFILE", "autoresearch-hack")
        import infra.modal.images as img
        img.assert_hackathon_profile()  # must not raise

    def test_raises_on_wrong_profile(self, monkeypatch):
        monkeypatch.setenv("MODAL_PROFILE", "my-work-account")
        import infra.modal.images as img
        with pytest.raises(RuntimeError, match="autoresearch-hack"):
            img.assert_hackathon_profile()

    def test_raises_when_profile_unset(self, monkeypatch):
        monkeypatch.delenv("MODAL_PROFILE", raising=False)
        import infra.modal.images as img
        with pytest.raises(RuntimeError, match="autoresearch-hack"):
            img.assert_hackathon_profile()


# ---------------------------------------------------------------------------
# infra.modal.images — image objects are constructed (not None, not raising)
# ---------------------------------------------------------------------------

class TestImageObjects:
    def test_base_image_constructed(self):
        import infra.modal.images as img
        assert img.base_image is not None

    def test_nanochat_gpu_image_constructed(self):
        import infra.modal.images as img
        assert img.nanochat_gpu_image is not None

    def test_sandbox_image_constructed(self):
        import infra.modal.images as img
        assert img.sandbox_image is not None

    def test_sandbox_image_is_base(self):
        import infra.modal.images as img
        assert img.sandbox_image is img.base_image

    def test_app_name(self):
        import infra.modal.images as img
        assert img.app.name == "ar2"


# ---------------------------------------------------------------------------
# infra.collector — module imports cleanly and SpanBatch validates
# ---------------------------------------------------------------------------

class TestCollector:
    def test_imports_cleanly(self):
        import infra.collector  # noqa: F401

    def test_span_batch_model(self):
        import infra.collector as col
        batch = col.SpanBatch(
            trace_id="abc123",
            version=1,
            candidate=0,
            env_id="nanochat",
            split="train",
            spans=[{"op": "llm.call", "tokens": 100}],
        )
        assert batch.trace_id == "abc123"
        assert batch.version == 1
        assert batch.split == "train"
        assert len(batch.spans) == 1

    def test_span_batch_candidate_optional(self):
        import infra.collector as col
        batch = col.SpanBatch(
            trace_id="xyz",
            version=2,
            env_id="legal",
            split="heldout",
            spans=[],
        )
        assert batch.candidate is None

    def test_ingest_endpoint_defined(self):
        import infra.collector as col
        assert callable(col.ingest)
