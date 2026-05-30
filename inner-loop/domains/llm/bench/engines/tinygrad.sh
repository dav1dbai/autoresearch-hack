#!/usr/bin/env bash
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
source "$ROOT/engines/common.env"

export CUDA="${CUDA:-1}"
export JITBEAM="${JITBEAM:-2}"
export DEV="${DEV:-CUDA}"

LLM_RUNNER="${TINYGRAD_LLM:-python examples/llm.py}"
TINYGRAD_HOME="${TINYGRAD_HOME:-.}"

cd "$TINYGRAD_HOME"
exec $LLM_RUNNER \
  --model "${TINYGRAD_MODEL:-qwen3:8b}" \
  --serve "${PORT:-8080}" \
  --max_context "${MAX_MODEL_LEN:-32768}" \
  "$@"
