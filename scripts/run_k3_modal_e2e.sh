#!/usr/bin/env bash
# K=3 Modal GPU smoke outer loop — real timing, rollouts in Modal containers.
set -euo pipefail
cd "$(dirname "$0")/.."

LOG_DIR=obs/run_logs
STAMP=$(date +%Y%m%d_%H%M%S)
RUN_ID="${AR2_RUN_ID:-}"
if [[ -n "$RUN_ID" ]]; then
  RUN_ROOT="obs/runs/$RUN_ID"
  LOG_DIR="$RUN_ROOT/run_logs"
  ARCHIVE_PATH="$RUN_ROOT/archive.jsonl"
  TRACES_DB="$RUN_ROOT/traces.db"
  ARCHIVE_DB="$RUN_ROOT/archive.db"
  WORKSHOP_DB="$RUN_ROOT/ar2_workshop.db"
  REPORT_PATH="$RUN_ROOT/report.html"
  EVENTS_PATH="$RUN_ROOT/run_events.jsonl"
  VERSIONS_ROOT="versions/$RUN_ID"
else
  ARCHIVE_PATH="obs/archive.jsonl"
  TRACES_DB="obs/traces.db"
  ARCHIVE_DB="obs/archive.db"
  WORKSHOP_DB="obs/ar2_workshop.db"
  REPORT_PATH="obs/report.html"
  EVENTS_PATH="obs/run_events.jsonl"
  VERSIONS_ROOT="versions"
fi
mkdir -p "$LOG_DIR" "$VERSIONS_ROOT"
LOG="$LOG_DIR/k3_modal_${STAMP}.log"

if [[ "${AR2_FRESH:-0}" == "1" ]]; then
  if [[ -f "$ARCHIVE_PATH" ]]; then
    cp "$ARCHIVE_PATH" "$LOG_DIR/archive_backup_${STAMP}.jsonl"
  fi
  rm -f "$ARCHIVE_PATH" "$TRACES_DB" "$ARCHIVE_DB" "$WORKSHOP_DB" "$REPORT_PATH" "$EVENTS_PATH"
  rm -rf "$VERSIONS_ROOT"/v_*
fi

export AR2_BACKEND=modal
export AR2_GPU_BACKEND=modal
export MATMUL_STUB=0
export RAINDROP_WORKSHOP=1
export AR2_OBS_CACHE=1
export AR2_STALE_ITERS="${AR2_STALE_ITERS:-2}"
export AR2_K="${AR2_K:-3}"
export AR2_M="${AR2_M:-1}"
export AR2_CACHE_DIR="$VERSIONS_ROOT"
export RAINDROP_WORKSHOP_DB_PATH="$WORKSHOP_DB"
export AR2_DASHBOARD_ARCHIVE="$ARCHIVE_PATH"
export AR2_DASHBOARD_DB="$TRACES_DB"
export AR2_DASHBOARD_OUT="$REPORT_PATH"
export AR2_DASHBOARD_EVENTS="$EVENTS_PATH"
export AR2_DASHBOARD_VERSIONS="$VERSIONS_ROOT"

# shellcheck source=scripts/modal_workshop_ngrok.sh
source "$(dirname "$0")/modal_workshop_ngrok.sh"

echo "Log → $LOG"
echo "Outer loop on Modal: improve + evaluate + GPU scoring"
[[ -n "$RUN_ID" ]] && echo "Run ID → $RUN_ID"
echo "Raindrop UI → http://127.0.0.1:5899"

echo "Dashboard → file://$(pwd)/$REPORT_PATH (refresh every 15s during run)"
(
  while true; do
    AR2_DASHBOARD_REFRESH=15 uv run python -m obs.dashboard 2>/dev/null || true
    uv run python -m obs.run_index >/dev/null 2>&1 || true
    sleep 15
  done
) &
DASH_PID=$!
trap 'kill $DASH_PID 2>/dev/null || true' EXIT

uv run python -m harness \
  --gpu --gpu-smoke \
  -K "$AR2_K" -M "$AR2_M" \
  --budget-seconds "${AR2_BUDGET_SECONDS:-300}" \
  --archive "$ARCHIVE_PATH" \
  --traces-db "$TRACES_DB" \
  --archive-db "$ARCHIVE_DB" \
  2>&1 | tee "$LOG"

uv run python scripts/analyze_run.py
AR2_DASHBOARD_REFRESH=0 uv run python -m obs.dashboard
uv run python -m obs.run_index
kill $DASH_PID 2>/dev/null || true
echo "Open $REPORT_PATH for full run timeline + improve transcripts"
[[ -n "$RUN_ID" ]] && echo "Open obs/runs/index.html for all runs"
