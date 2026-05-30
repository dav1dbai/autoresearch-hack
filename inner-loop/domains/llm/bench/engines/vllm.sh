#!/usr/bin/env bash
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
source "$ROOT/engines/common.env"

exec vllm serve "${VLLM_MODEL:-$MODEL_ID}" \
  --host 0.0.0.0 \
  --port "${PORT:-8000}" \
  --max-model-len "${MAX_MODEL_LEN:-32768}" \
  --reasoning-parser "${REASONING_PARSER:-qwen3}" \
  --language-model-only \
  --enable-prefix-caching \
  --gpu-memory-utilization "${GPU_MEMORY_UTILIZATION:-0.85}" \
  "$@"
