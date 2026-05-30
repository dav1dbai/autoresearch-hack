#!/usr/bin/env bash
# Modal GPU outer loop smoke: v0 + 1 generation (K=1), noop inner + meta agents.
set -euo pipefail
cd "$(dirname "$0")/.."

export AR2_BACKEND=modal
export AR2_GPU_BACKEND=modal
export MATMUL_STUB=0
export AR2_OBS_CACHE=1
# Raindrop SOT for improve() read path (drop --no-workshop)
export RAINDROP_WORKSHOP=1
export AR2_STALE_ITERS=1
export INNER_AGENT_CMD="${INNER_AGENT_CMD:-$(pwd)/scripts/noop_agent.sh}"
export MUTATE_AGENT_CMD="${MUTATE_AGENT_CMD:-$(pwd)/scripts/noop_mutate.sh}"

exec uv run python -m harness \
  --gpu --gpu-smoke \
  -K 1 \
  --budget-seconds "${AR2_BUDGET_SECONDS:-90}" \
  --archive obs/archive.jsonl \
  "$@"
