#!/usr/bin/env bash
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
source "$ROOT/engines/common.env"

LLAMA_SERVER="${LLAMA_SERVER:-llama-server}"
HF_REPO="${GGUF_REPO:-unsloth/Qwen3.6-27B-GGUF}"
HF_FILE="${GGUF_QUANT:-Q8_0}"

exec "$LLAMA_SERVER" \
  -hf "${HF_REPO}:${HF_FILE}" \
  --host 0.0.0.0 \
  --port "${PORT:-8080}" \
  --ctx-size "${MAX_MODEL_LEN:-32768}" \
  --n-gpu-layers 99 \
  --flash-attn on \
  "$@"
