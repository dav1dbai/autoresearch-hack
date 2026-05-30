"""Launch `codex exec` with Raindrop Workshop MCP — mirrors workshop buildCodexArgs.

See raindrop-ai/workshop src/codex-cli-chat.ts for the canonical wiring.
"""
from __future__ import annotations

import json
import hashlib
import os
import shutil
import subprocess
import sys
import time
import urllib.error
import urllib.request
import uuid
from pathlib import Path
from typing import Any


def resolve_raindrop_bin() -> str:
    override = os.environ.get("RAINDROP_BIN")
    if override and Path(override).is_file():
        return override
    home_bin = Path.home() / ".raindrop" / "bin" / "raindrop"
    if home_bin.is_file():
        return str(home_bin)
    found = shutil.which("raindrop")
    if found:
        return found
    return "raindrop"


def resolve_workshop_url() -> str:
    return os.environ.get("RAINDROP_WORKSHOP_URL", "http://127.0.0.1:5899").rstrip("/")


def _ar2_run_id() -> str:
    return os.environ.get("AR2_RUN_ID", "").strip()


def _prefix_run_name(name: str, run_id: str) -> str:
    if not run_id or name.startswith(f"[{run_id}]"):
        return name
    return f"[{run_id}] {name}"


def _mcp_env_value(workshop_url: str) -> str:
    """Codex -c value for mcp_servers.raindrop.env (Workshop inline table syntax)."""
    return (
        "{"
        f"RAINDROP_WORKSHOP_URL={json.dumps(workshop_url)},"
        'RAINDROP_WORKSHOP_AGENT_PROVIDER="codex",'
        'RAINDROP_WORKSHOP_ANNOTATION_SOURCE="codex"'
        "}"
    )


def _repo_roots() -> list[Path]:
    roots: list[Path] = []
    if raw := os.environ.get("AR2_REPO_ROOT"):
        roots.append(Path(raw))
    here = Path(__file__).resolve()
    roots.append(here.parent.parent.parent)
    return roots


def resolve_codex_home() -> str | None:
    override = os.environ.get("CODEX_HOME")
    if override:
        return override
    for root in _repo_roots():
        inner = root / ".codex-inner"
        if (inner / "auth.json").is_file() or (inner / "config.toml").is_file():
            return str(inner)
    return None


def build_raindrop_codex_argv(*extra: str, cwd: str | None = None) -> list[str]:
    """Build argv for codex with Raindrop MCP configured like Workshop UI."""
    raindrop_bin = resolve_raindrop_bin()
    workshop_url = resolve_workshop_url()
    codex_bin = os.environ.get("RAINDROP_WORKSHOP_CODEX_BIN", "codex")
    work_cwd = cwd or os.getcwd()

    args: list[str] = [codex_bin]

    if os.environ.get("RAINDROP_WORKSHOP_CODEX_BYPASS_PERMISSIONS", "1") != "0":
        args.append("--dangerously-bypass-approvals-and-sandbox")
    else:
        args.extend(["-a", os.environ.get("RAINDROP_WORKSHOP_CODEX_APPROVAL_POLICY", "never")])
        sandbox = os.environ.get("RAINDROP_WORKSHOP_CODEX_SANDBOX")
        if sandbox:
            args.extend(["--sandbox", sandbox])

    exec_flags: list[str] = []
    if os.environ.get("RAINDROP_WORKSHOP_CODEX_JSON", "1") != "0":
        exec_flags.append("--json")
    if os.environ.get("RAINDROP_WORKSHOP_CODEX_IGNORE_USER_CONFIG", "0") == "1":
        exec_flags.append("--ignore-user-config")
    if os.environ.get("RAINDROP_WORKSHOP_CODEX_EPHEMERAL", "1") != "0":
        exec_flags.append("--ephemeral")

    args.extend(
        [
            "-C",
            work_cwd,
            "-c",
            f"mcp_servers.raindrop.command={json.dumps(raindrop_bin)}",
            "-c",
            'mcp_servers.raindrop.args=["workshop","mcp"]',
            "-c",
            f"mcp_servers.raindrop.env={_mcp_env_value(workshop_url)}",
        ]
    )
    if extra and extra[0] == "exec":
        args.extend(["exec", *exec_flags, *extra[1:]])
    else:
        args.extend(extra)
    return args


def default_wrapper_cmd() -> str | None:
    """Path to scripts/raindrop_codex_exec.sh when present under AR2_REPO_ROOT."""
    for root in _repo_roots():
        script = root / "scripts" / "raindrop_codex_exec.sh"
        if script.is_file():
            return f"{script} exec"
    return None


def _post_json(path: str, payload: dict[str, Any], *, timeout: float = 1.5) -> None:
    url = f"{resolve_workshop_url()}{path}"
    req = urllib.request.Request(
        url,
        data=json.dumps(payload).encode("utf-8"),
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    try:
        with urllib.request.urlopen(req, timeout=timeout):
            return
    except (urllib.error.URLError, TimeoutError, OSError):
        return


def _span_id(seed: str) -> str:
    return hashlib.sha1(seed.encode("utf-8", errors="replace")).hexdigest()[:16]


def _root_span(trace_id: str, name: str, *, status: str = "UNSET") -> dict[str, Any]:
    now_ns = int(time.time() * 1e9)
    run_id = _ar2_run_id()
    attributes = [
        {"key": "model", "value": {"stringValue": "codex"}},
        {"key": "ar2.tool", "value": {"stringValue": "improve"}},
        {"key": "ar2.display_name", "value": {"stringValue": name}},
    ]
    if run_id:
        attributes.append({"key": "ar2.run_id", "value": {"stringValue": run_id}})
    return {
        "traceId": trace_id,
        "spanId": "0000000000000001",
        "name": name,
        "kind": 1,
        "startTimeUnixNano": str(now_ns),
        "endTimeUnixNano": str(now_ns),
        "status": {"code": 1 if status == "OK" else 0},
        "attributes": attributes,
    }


def _push_root_span(trace_id: str, name: str, *, status: str = "UNSET") -> None:
    name = _prefix_run_name(name, _ar2_run_id())
    _post_json(
        "/v1/traces",
        {
            "resourceSpans": [
                {
                    "resource": {
                        "attributes": [
                            {"key": "service.name", "value": {"stringValue": "ar2-codex"}}
                        ]
                    },
                    "scopeSpans": [{"spans": [_root_span(trace_id, name, status=status)]}],
                }
            ]
        },
    )


def _live(
    trace_id: str,
    event_type: str,
    content: str = "",
    *,
    span_id: str | None = None,
    metadata: dict[str, Any] | None = None,
) -> None:
    run_id = _ar2_run_id()
    live_metadata = {
        "eventName": "codex improve",
        "provider": "codex",
        **(metadata or {}),
    }
    if run_id:
        live_metadata.update({"runId": run_id, "ar2.run_id": run_id})
    payload: dict[str, Any] = {
        "traceId": trace_id,
        "type": event_type,
        "content": content,
        "timestamp": int(time.time() * 1000),
        "metadata": live_metadata,
    }
    if span_id:
        payload["spanId"] = span_id
    _post_json("/v1/live", payload)


def _object(value: Any) -> dict[str, Any]:
    return value if isinstance(value, dict) else {}


def _codex_event_to_live(trace_id: str, event: dict[str, Any]) -> None:
    event_type = event.get("type")

    if event_type == "thread.started" and isinstance(event.get("thread_id"), str):
        _live(trace_id, "provider_session", event["thread_id"])
        return

    if event_type == "turn.started":
        _live(trace_id, "status", "Codex turn started")
        return

    if event_type == "turn.completed":
        usage = _object(event.get("usage"))
        _live(trace_id, "usage", json.dumps(usage, sort_keys=True), metadata={"usage": usage})
        _live(trace_id, "done", "Codex turn completed")
        return

    if event_type == "error":
        _live(trace_id, "error", str(event.get("message") or "Codex error"))
        return

    if event_type == "event_msg":
        payload = _object(event.get("payload"))
        payload_type = payload.get("type")
        if payload_type == "agent_message" and isinstance(payload.get("message"), str):
            _live(trace_id, "text", payload["message"])
        elif payload_type == "task_complete":
            _live(trace_id, "done", "Codex task complete")
        elif payload_type == "token_count":
            _live(trace_id, "usage", json.dumps(payload.get("info") or {}, sort_keys=True))
        return

    if event_type == "item.started":
        item = _object(event.get("item"))
        item_type = str(item.get("type") or "")
        item_id = str(item.get("id") or item.get("call_id") or item.get("command") or item_type)
        if item_type == "command_execution":
            command = str(item.get("command") or "")
            _live(trace_id, "tool_start", command[:500], span_id=_span_id(item_id), metadata={"tool": "exec_command"})
        elif item_type == "mcp_tool_call":
            server = str(item.get("server") or "mcp")
            tool = str(item.get("tool") or "tool")
            _live(
                trace_id,
                "tool_start",
                json.dumps(item.get("arguments") or {})[:500],
                span_id=_span_id(item_id),
                metadata={"tool": f"{server}.{tool}"},
            )
        return

    if event_type == "item.completed":
        item = _object(event.get("item"))
        item_type = str(item.get("type") or "")
        item_id = str(item.get("id") or item.get("call_id") or item.get("command") or item_type)
        if item_type == "agent_message" and isinstance(item.get("text"), str):
            _live(trace_id, "text", item["text"])
        elif item_type == "command_execution":
            output = str(item.get("aggregated_output") or "")
            ok = item.get("exit_code") in (0, None) and item.get("status") != "failed"
            _live(trace_id, "tool_result", output[:500], span_id=_span_id(item_id), metadata={"ok": ok})
        elif item_type == "mcp_tool_call":
            error = item.get("error")
            result = error if error else item.get("result")
            _live(
                trace_id,
                "tool_result",
                json.dumps(result or {})[:500],
                span_id=_span_id(item_id),
                metadata={"ok": error is None},
            )
        return

    if event_type == "response_item":
        item = _object(event.get("payload"))
        if item.get("type") == "message":
            text = _content_text(item.get("content"))
            if text:
                _live(trace_id, "text", text)
        elif item.get("type") == "function_call":
            item_id = str(item.get("call_id") or item.get("name") or "function_call")
            _live(
                trace_id,
                "tool_start",
                str(item.get("arguments") or "")[:500],
                span_id=_span_id(item_id),
                metadata={"tool": str(item.get("name") or "function_call")},
            )
        elif item.get("type") == "function_call_output":
            item_id = str(item.get("call_id") or "function_call")
            _live(trace_id, "tool_result", str(item.get("output") or "")[:500], span_id=_span_id(item_id))


def _content_text(content: Any) -> str:
    if isinstance(content, str):
        return content
    if not isinstance(content, list):
        return ""
    parts: list[str] = []
    for part in content:
        if isinstance(part, str):
            parts.append(part)
        elif isinstance(part, dict) and isinstance(part.get("text"), str):
            parts.append(part["text"])
    return "\n".join(parts)


def _run_codex_with_workshop_bridge(codex_argv: list[str]) -> int:
    trace_id = os.environ.get("RAINDROP_WORKSHOP_TRACE_ID") or uuid.uuid4().hex
    name = _prefix_run_name(os.environ.get("RAINDROP_WORKSHOP_RUN_NAME", "Codex improve"), _ar2_run_id())
    _push_root_span(trace_id, name)
    _live(trace_id, "status", "Launching Codex")

    proc = subprocess.Popen(
        codex_argv,
        stdin=subprocess.DEVNULL,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        bufsize=1,
    )
    assert proc.stdout is not None
    for line in proc.stdout:
        print(line, end="", flush=True)
        try:
            event = json.loads(line)
        except json.JSONDecodeError:
            continue
        if isinstance(event, dict):
            _codex_event_to_live(trace_id, event)

    code = proc.wait()
    if code == 0:
        _live(trace_id, "done", "Codex exited successfully")
        _push_root_span(trace_id, name, status="OK")
    else:
        _live(trace_id, "error", f"Codex exited with code {code}")
    return code


def main(argv: list[str] | None = None) -> int:
    args = list(argv if argv is not None else sys.argv[1:])
    if not args:
        print("usage: raindrop_codex.py exec [codex-args...]", file=sys.stderr)
        return 2

    workshop_url = resolve_workshop_url()
    os.environ.setdefault("RAINDROP_WORKSHOP_URL", workshop_url)
    os.environ.setdefault("RAINDROP_LOCAL_DEBUGGER", f"{workshop_url}/v1/")
    if codex_home := resolve_codex_home():
        os.environ.setdefault("CODEX_HOME", codex_home)

    codex_argv = build_raindrop_codex_argv(*args, cwd=os.getcwd())
    if os.environ.get("RAINDROP_WORKSHOP_LIVE_BRIDGE", "1") != "0":
        return _run_codex_with_workshop_bridge(codex_argv)
    # Codex exec reads extra stdin after a prompt file; non-TTY parents never EOF.
    return subprocess.run(codex_argv, check=False, stdin=subprocess.DEVNULL).returncode


if __name__ == "__main__":
    raise SystemExit(main())
