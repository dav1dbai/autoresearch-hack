"""Tests for infra/modal_secrets.py."""
from __future__ import annotations

from unittest.mock import MagicMock, patch

import pytest


def test_ensure_vast_modal_secret_calls_modal_cli(monkeypatch):
    monkeypatch.setenv("MODAL_PROFILE", "autoresearch-hack")
    monkeypatch.setenv("VAST_API_KEY", "test-key-123")

    with patch("subprocess.run") as mock_run:
        mock_run.return_value = MagicMock(returncode=0)
        from infra.modal.secrets import ensure_vast_modal_secret

        name = ensure_vast_modal_secret()

    assert name == "autoresearch-vast"
    mock_run.assert_called_once()
    cmd = mock_run.call_args.args[0]
    assert cmd[0:3] == ["modal", "secret", "create"]
    assert "VAST_API_KEY=test-key-123" in cmd
    assert "--force" in cmd


def test_ensure_vast_modal_secret_requires_key(monkeypatch):
    monkeypatch.delenv("VAST_API_KEY", raising=False)
    monkeypatch.setenv("MODAL_PROFILE", "autoresearch-hack")

    from infra.modal.secrets import ensure_vast_modal_secret

    with pytest.raises(RuntimeError, match="VAST_API_KEY"):
        ensure_vast_modal_secret()
