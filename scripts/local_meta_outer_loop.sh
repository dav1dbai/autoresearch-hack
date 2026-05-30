#!/usr/bin/env bash
# Local stub loop — offline/tests only. Use Modal for real runs:
#   ./scripts/modal_smoke.sh        # K=0 plumbing check
#   ./scripts/modal_codex_outer_loop.sh
#   AR2_K=3 ./scripts/run_k3_e2e.sh
set -euo pipefail
echo "WARNING: local stub run — no Modal, MATMUL_STUB=1 (no speed signal)." >&2
echo "For real e2e use: ./scripts/modal_codex_outer_loop.sh or ./scripts/run_k3_e2e.sh" >&2
cd "$(dirname "$0")/.."

export AR2_BACKEND=local
export MATMUL_STUB=1
export RAINDROP_WORKSHOP="${RAINDROP_WORKSHOP:-1}"
export AR2_STALE_ITERS="${AR2_STALE_ITERS:-2}"

K="${AR2_K:-1}"
M="${AR2_M:-1}"

exec uv run python -m harness \
  --stub \
  -K "$K" -M "$M" \
  --budget-seconds "${AR2_BUDGET_SECONDS:-120}" \
  --archive obs/archive.jsonl \
  "$@"
