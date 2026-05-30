"""Tests for harness/modal_session.py — shared Modal app.run() session."""
from __future__ import annotations

import sys
import types
from unittest.mock import MagicMock

import pytest


def _install_modal_stub(monkeypatch, *, run_call_count: list[int]):
    modal_mod = types.ModuleType("modal")

    class _RunCtx:
        def __enter__(self):
            return self
        def __exit__(self, *a):
            return False

    class _App:
        def __init__(self, name: str):
            self.name = name

        def run(self):
            run_call_count[0] += 1
            return _RunCtx()

        def deploy(self):
            pass

    class _Fn:
        @staticmethod
        def from_name(app_name: str, fn_name: str):
            m = MagicMock()
            m.starmap = MagicMock(return_value=[{"env_id": "x"}])
            return m

    modal_mod.App = _App
    modal_mod.Function = _Fn
    monkeypatch.setitem(sys.modules, "modal", modal_mod)


def test_reuse_enters_app_run_once(monkeypatch):
    counts = [0]
    _install_modal_stub(monkeypatch, run_call_count=counts)
    monkeypatch.setenv("AR2_MODAL_REUSE", "1")
    monkeypatch.setenv("AR2_MODAL_DEPLOYED", "0")
    monkeypatch.delitem(sys.modules, "harness.cloud.session", raising=False)

    import harness.cloud.session as ms

    ms.close_app_session()
    app = ms.modal.App("ar2")
    fn = MagicMock()
    fn.starmap = MagicMock(return_value=[1, 2])

    ms.starmap_run_rollout(app, fn, [("a",), ("b",)])
    ms.starmap_run_rollout(app, fn, [("c",)])

    assert counts[0] == 1
    assert fn.starmap.call_count == 2
    ms.close_app_session()


def test_deployed_uses_function_from_name(monkeypatch):
    counts = [0]
    _install_modal_stub(monkeypatch, run_call_count=counts)
    monkeypatch.setenv("AR2_MODAL_DEPLOYED", "1")
    monkeypatch.setenv("MODAL_PROFILE", "autoresearch-hack")
    monkeypatch.delitem(sys.modules, "harness.cloud.session", raising=False)

    import harness.cloud.session as ms

    ms._DEPLOYED_ONCE = False
    app = ms.modal.App("ar2")
    fn = MagicMock()
    fn.starmap = MagicMock(return_value=[1])

    ms.starmap_run_rollout(app, fn, [("a",)])
    ms.starmap_run_rollout(app, fn, [("b",)])

    assert counts[0] == 0
    assert fn.starmap.call_count == 0


def test_no_reuse_enters_each_batch(monkeypatch):
    counts = [0]
    _install_modal_stub(monkeypatch, run_call_count=counts)
    monkeypatch.setenv("AR2_MODAL_REUSE", "0")
    monkeypatch.setenv("AR2_MODAL_DEPLOYED", "0")
    monkeypatch.delitem(sys.modules, "harness.cloud.session", raising=False)

    import harness.cloud.session as ms

    ms.close_app_session()
    app = ms.modal.App("ar2")
    fn = MagicMock()
    fn.starmap = MagicMock(return_value=[1])

    ms.starmap_run_rollout(app, fn, [("a",)])
    ms.starmap_run_rollout(app, fn, [("b",)])

    assert counts[0] == 2
    ms.close_app_session()
