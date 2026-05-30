#!/usr/bin/env bash
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
source "$ROOT/engines/common.env"

MIRAGE_HOME="${MIRAGE_HOME:-/opt/mirage}"
DEMO_DIR="$MIRAGE_HOME/demo/qwen3"
DEMO_MODEL="${MIRAGE_DEMO_MODEL:-Qwen/Qwen3-8B}"

if [[ ! -f "$DEMO_DIR/demo.py" ]]; then
  echo "Mirage demo missing at $DEMO_DIR/demo.py" >&2
  echo "Install: git clone --recursive --branch mpk https://github.com/mirage-project/mirage $MIRAGE_HOME && pip install -e $MIRAGE_HOME -v" >&2
  exit 1
fi

cd "$DEMO_DIR"
exec python demo.py \
  --use-mirage \
  --model "$DEMO_MODEL" \
  --max-seq-length "${MAX_MODEL_LEN:-2048}" \
  --max-new-tokens "${MIRAGE_MAX_NEW_TOKENS:-128}" \
  --prompt "${MIRAGE_PROMPT:-Explain matrix multiplication briefly.}" \
  "$@"
