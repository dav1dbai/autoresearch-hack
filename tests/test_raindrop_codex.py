"""Raindrop-wired Codex argv builder."""
from __future__ import annotations

import json
import os
from pathlib import Path

import pytest

from harness.agents.raindrop_codex import (
    _live,
    _push_root_span,
    build_raindrop_codex_argv,
    default_wrapper_cmd,
    resolve_workshop_url,
)


def test_build_argv_wires_raindrop_mcp(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    fake_raindrop = tmp_path / "raindrop"
    fake_raindrop.write_text("#!/bin/sh\n", encoding="utf-8")
    fake_raindrop.chmod(0o755)
    monkeypatch.setenv("RAINDROP_BIN", str(fake_raindrop))
    monkeypatch.setenv("RAINDROP_WORKSHOP_URL", "http://127.0.0.1:5899")

    argv = build_raindrop_codex_argv("exec", "/tmp/prompt.md", cwd="/work")

    assert argv[0] == "codex"
    assert "--dangerously-bypass-approvals-and-sandbox" in argv
    assert "-C" in argv and "/work" in argv

    cmd_idx = argv.index("-c")
    assert json.dumps(str(fake_raindrop)) in argv[cmd_idx + 1]
    assert 'mcp_servers.raindrop.args=["workshop","mcp"]' in argv

    env_idx = next(i for i, v in enumerate(argv) if v.startswith("mcp_servers.raindrop.env="))
    env_val = argv[env_idx].split("=", 1)[1]
    assert "RAINDROP_WORKSHOP_URL" in env_val
    assert "codex" in env_val
    assert "--ignore-user-config" not in argv
    assert "--ephemeral" in argv
    assert "--json" in argv
    assert argv[-1] == "/tmp/prompt.md"
    exec_idx = argv.index("exec")
    assert argv[exec_idx : exec_idx + 3] == ["exec", "--json", "--ephemeral"]


def test_ignore_user_config_opt_in(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("RAINDROP_WORKSHOP_CODEX_IGNORE_USER_CONFIG", "1")

    argv = build_raindrop_codex_argv("exec", "prompt.md")

    assert "--ignore-user-config" in argv


def test_bypass_can_be_disabled(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("RAINDROP_WORKSHOP_CODEX_BYPASS_PERMISSIONS", "0")
    monkeypatch.setenv("RAINDROP_WORKSHOP_CODEX_SANDBOX", "workspace-write")

    argv = build_raindrop_codex_argv("exec", "prompt.md")
    assert "--dangerously-bypass-approvals-and-sandbox" not in argv
    assert "-a" in argv and "never" in argv
    assert "--sandbox" in argv and "workspace-write" in argv


def test_default_wrapper_cmd_uses_repo_root(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    scripts = tmp_path / "scripts"
    scripts.mkdir()
    wrapper = scripts / "raindrop_codex_exec.sh"
    wrapper.write_text("#!/bin/sh\n", encoding="utf-8")
    monkeypatch.setenv("AR2_REPO_ROOT", str(tmp_path))

    cmd = default_wrapper_cmd()
    assert cmd == f"{wrapper} exec"


def test_resolve_workshop_url_strips_trailing_slash(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("RAINDROP_WORKSHOP_URL", "http://127.0.0.1:5899/")
    assert resolve_workshop_url() == "http://127.0.0.1:5899"


def test_live_events_include_run_id_metadata(monkeypatch: pytest.MonkeyPatch) -> None:
    captured: list[tuple[str, dict]] = []
    monkeypatch.setenv("AR2_RUN_ID", "raindrop-k2")
    monkeypatch.setattr(
        "harness.agents.raindrop_codex._post_json",
        lambda path, payload, timeout=1.5: captured.append((path, payload)),
    )

    _live("trace-1", "status", "hello", metadata={"ok": True})

    assert captured[0][0] == "/v1/live"
    metadata = captured[0][1]["metadata"]
    assert metadata["runId"] == "raindrop-k2"
    assert metadata["ar2.run_id"] == "raindrop-k2"
    assert metadata["ok"] is True


def test_root_span_prefixes_name_and_attributes(monkeypatch: pytest.MonkeyPatch) -> None:
    captured: list[tuple[str, dict]] = []
    monkeypatch.setenv("AR2_RUN_ID", "raindrop-k2")
    monkeypatch.setattr(
        "harness.agents.raindrop_codex._post_json",
        lambda path, payload, timeout=1.5: captured.append((path, payload)),
    )

    _push_root_span("trace-1", "Codex improve")

    span = captured[0][1]["resourceSpans"][0]["scopeSpans"][0]["spans"][0]
    assert span["name"] == "[raindrop-k2] Codex improve"
    attrs = {a["key"]: a["value"] for a in span["attributes"]}
    assert attrs["ar2.run_id"]["stringValue"] == "raindrop-k2"
    assert attrs["ar2.display_name"]["stringValue"] == "[raindrop-k2] Codex improve"
