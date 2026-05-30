#!/usr/bin/env bash
# Headless Codex with Raindrop Workshop MCP — same wiring as Workshop's codex chat pane.
set -euo pipefail

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
WORKSHOP_URL="${RAINDROP_WORKSHOP_URL:-http://127.0.0.1:5899}"
WORKSHOP_URL="${WORKSHOP_URL%/}"
export RAINDROP_WORKSHOP_URL="$WORKSHOP_URL"
export RAINDROP_LOCAL_DEBUGGER="${RAINDROP_LOCAL_DEBUGGER:-${WORKSHOP_URL}/v1/}"
export AR2_REPO_ROOT="${AR2_REPO_ROOT:-$ROOT}"

if command -v uv >/dev/null 2>&1 && [[ -f "$ROOT/pyproject.toml" ]] && [[ -z "${AR2_MODAL:-}" ]]; then
  exec uv run python -m harness.agents.raindrop_codex "$@"
else
  export PYTHONPATH="${ROOT}${PYTHONPATH:+:$PYTHONPATH}"
  exec python3 -m harness.agents.raindrop_codex "$@"
fi
