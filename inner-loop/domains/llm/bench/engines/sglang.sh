#!/usr/bin/env bash
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
source "$ROOT/engines/common.env"

exec python -m sglang.launch_server \
  --model-path "${SGLANG_MODEL:-$MODEL_ID}" \
  --host 0.0.0.0 \
  --port "${PORT:-8000}" \
  --context-length "${MAX_MODEL_LEN:-32768}" \
  --reasoning-parser "${REASONING_PARSER:-qwen3}" \
  --mem-fraction-static "${GPU_MEMORY_UTILIZATION:-0.85}" \
  "$@"
