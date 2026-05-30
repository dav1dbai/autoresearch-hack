#!/usr/bin/env bash
# Time-bound Codex inner loop on Modal GPU matmul (smoke shapes) + Raindrop SOT.
set -euo pipefail
cd "$(dirname "$0")/.."

export AR2_BACKEND=modal
export AR2_GPU_BACKEND=modal
export MATMUL_STUB=0
export RAINDROP_WORKSHOP=1
export AR2_OBS_CACHE=1
export AR2_STALE_ITERS="${AR2_STALE_ITERS:-2}"
# Real codex from .env — do not override INNER_AGENT_CMD unless set externally

exec uv run python -m harness \
  --gpu --gpu-smoke \
  -K 0 \
  --budget-seconds "${AR2_BUDGET_SECONDS:-300}" \
  --archive obs/codex_gpu.jsonl \
  "$@"
