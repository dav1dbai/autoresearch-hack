#!/usr/bin/env bash
# Modal GPU outer loop: v0 + K generations, real Codex inner + meta agents, Raindrop SOT.
set -euo pipefail
cd "$(dirname "$0")/.."

export AR2_BACKEND=modal
export AR2_GPU_BACKEND=modal
export MATMUL_STUB=0
export RAINDROP_WORKSHOP=1
export AR2_OBS_CACHE=1
export AR2_STALE_ITERS="${AR2_STALE_ITERS:-2}"
# INNER_AGENT_CMD / MUTATE_AGENT_CMD from .env (default: raindrop_codex_exec.sh)

# shellcheck source=scripts/modal_workshop_ngrok.sh
source "$(dirname "$0")/modal_workshop_ngrok.sh"

K="${AR2_K:-1}"
M="${AR2_M:-1}"

exec uv run python -m harness \
  --gpu --gpu-smoke \
  -K "$K" -M "$M" \
  --budget-seconds "${AR2_BUDGET_SECONDS:-300}" \
  --archive obs/archive.jsonl \
  "$@"
