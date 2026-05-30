"""harness/telemetry.py — in-harness telemetry injection + in-sandbox span writer.

Two mechanisms are supported (both documented; file-shim is the default):

(a) File shim (default — pull model, DESIGN §6.1):
    inject() sets OPENAI_BASE_URL / ANTHROPIC_BASE_URL to a tiny in-sandbox logging
    proxy (e.g. `python -m harness.tracing.telemetry proxy`) that intercepts LLM calls,
    appends spans to /work/trace.jsonl, then forwards to the real provider.
    score_repo reads trace.jsonl out via sb.filesystem.read_text after the rollout.

(b) Codex OTEL (alternative):
    inject() sets OTEL_EXPORTER_OTLP_ENDPOINT + relevant OTEL envvars so that
    Codex's built-in [otel] collector (configured via ~/.codex/config.toml,
    exporter = "otlp-http") ships spans to an in-sandbox collector instead.
    The collector still writes /work/trace.jsonl so pull stays consistent.

Neither requires call-home networking; both ensure AR can't disable its own telemetry
because injection happens at the sandbox env boundary.
"""
from __future__ import annotations

import json
import time
import uuid
from pathlib import Path
from typing import Any, TYPE_CHECKING

if TYPE_CHECKING:
    from harness.contracts import Env

TRACE_FILE = Path("/work/trace.jsonl")

# Host env keys forwarded into sandboxes (Modal + local) for matmul GPU scoring.
_GPU_ENV_KEYS = (
    "AR2_GPU_BACKEND",
    "MATMUL_RUNNER",
    "MATMUL_STUB",
    "MATMUL_GPU_STARTER",
    "MATMUL_TARGET_GFLOPS",
    "VAST_INSTANCE_ID",
    "MODAL_PROFILE",
)


def forward_gpu_env() -> dict[str, str]:
    """Copy GPU/matmul config from the host into a rollout sandbox."""
    import os
    return {k: v for k in _GPU_ENV_KEYS if (v := os.environ.get(k))}


# ---------------------------------------------------------------------------
# Harness-side: what to inject at the sandbox boundary
# ---------------------------------------------------------------------------

def inject(tags: dict[str, Any]) -> dict[str, str]:
    """Return env vars to set inside a Modal sandbox before running ar/.

    The returned dict is merged into the sandbox environment.  Every span
    written inside the sandbox will carry *tags* so distributed traces remain
    separable after db_sync merges them.

    Required keys in tags: version (int), candidate (str), env_id (str),
    split ("train"|"heldout"), trace_id (str).
    """
    trace_id = tags.get("trace_id") or str(uuid.uuid4())
    env = {
        # Mechanism (a): redirect LLM base URLs to the in-sandbox shim.
        # The shim listens on a loopback port (5100), proxies to the real
        # endpoint (read from OPENAI_BASE_URL_REAL / ANTHROPIC_BASE_URL_REAL),
        # and appends spans to /work/trace.jsonl.
        "OPENAI_BASE_URL": "http://localhost:5100/openai",
        "ANTHROPIC_BASE_URL": "http://localhost:5100/anthropic",
        # Real endpoints (shim reads these to forward).  Sandboxes pull from
        # their own secret store; we just reserve the var names here.
        "OPENAI_BASE_URL_REAL": "https://api.openai.com/v1",
        "ANTHROPIC_BASE_URL_REAL": "https://api.anthropic.com",
        # Mechanism (b) alternative — Codex OTEL.  If the sandbox prefers OTEL,
        # set CODEX_OTEL_ENABLED=1 and point the exporter at the in-sandbox
        # collector.  The collector writes the same /work/trace.jsonl format.
        "OTEL_EXPORTER_OTLP_ENDPOINT": "http://localhost:4318",
        "OTEL_SERVICE_NAME": "ar2-agent",
        # Tag propagation — the in-sandbox span writer reads these.
        "AR2_TRACE_ID": trace_id,
        "AR2_VERSION": str(tags.get("version", 0)),
        "AR2_CANDIDATE": str(tags.get("candidate", "")),
        "AR2_ENV_ID": str(tags.get("env_id", "")),
        "AR2_SPLIT": str(tags.get("split", "")),
        "AR2_TRACE_FILE": str(TRACE_FILE),
    }
    return env


def inject_for_rollout(
    env: "Env",
    *,
    trace_id: str,
    version: int = 0,
    candidate: str = "",
    trace_file: Path | None = None,
) -> dict[str, str]:
    """Build sandbox/local env vars for one rollout (wraps inject(tags))."""
    out = inject({
        "trace_id": trace_id,
        "version": version,
        "candidate": candidate,
        "env_id": env.id,
        "split": env.split,
    })
    if trace_file is not None:
        out["AR2_TRACE_FILE"] = str(trace_file)
    out.update(forward_gpu_env())
    return out

def write_span(
    *,
    model: str,
    prompt_tokens: int,
    completion_tokens: int,
    latency_ms: float,
    tool_name: str | None = None,
    tool_input: str | None = None,
    trace_file: Path | None = None,
) -> None:
    """Append one JSON span line to /work/trace.jsonl.

    Called by the in-sandbox proxy shim after each LLM request/response pair.
    Reads AR2_* env vars for tags so the shim needs zero config beyond env.
    """
    import os

    out = trace_file or Path(os.environ.get("AR2_TRACE_FILE", str(TRACE_FILE)))
    out.parent.mkdir(parents=True, exist_ok=True)

    span: dict[str, Any] = {
        "trace_id": os.environ.get("AR2_TRACE_ID", ""),
        "version": int(os.environ.get("AR2_VERSION", 0)),
        "candidate": os.environ.get("AR2_CANDIDATE", ""),
        "env_id": os.environ.get("AR2_ENV_ID", ""),
        "split": os.environ.get("AR2_SPLIT", ""),
        "model": model,
        "prompt_tokens": prompt_tokens,
        "completion_tokens": completion_tokens,
        "latency_ms": latency_ms,
        "tool_name": tool_name,
        "tool_input": tool_input,
        "ts": time.time(),
    }
    with out.open("a") as fh:
        fh.write(json.dumps(span) + "\n")


def parse_trace_file(path: Path) -> list[dict[str, Any]]:
    """Read a /work/trace.jsonl file and return parsed spans. Skips malformed lines."""
    spans: list[dict[str, Any]] = []
    if not path.exists():
        return spans
    for line in path.read_text().splitlines():
        line = line.strip()
        if not line:
            continue
        try:
            spans.append(json.loads(line))
        except json.JSONDecodeError:
            continue
    return spans
