#!/usr/bin/env bash
# Wipe stale AR² obs + Workshop history before a fresh Raindrop-wired run.
set -euo pipefail
cd "$(dirname "$0")/.."

echo "Stopping harness processes…"
pkill -f "python -m harness" 2>/dev/null || true
pkill -f "run_k3_modal_e2e" 2>/dev/null || true

echo "Resetting Raindrop Workshop DB (119+ stale synthetic runs)…"
echo y | "${RAINDROP_BIN:-$HOME/.raindrop/bin/raindrop}" workshop reset 2>/dev/null || true
"${RAINDROP_BIN:-$HOME/.raindrop/bin/raindrop}" workshop start 2>/dev/null || true

STAMP=$(date +%Y%m%d_%H%M%S)
mkdir -p obs/run_logs/archive
for f in obs/run_logs/k3_modal_*.log obs/run_logs/modal_*.log obs/run_logs/k3_*.log; do
  [[ -f "$f" ]] && mv "$f" "obs/run_logs/archive/${STAMP}_$(basename "$f")"
done

rm -f obs/archive.jsonl obs/traces.db obs/archive.db obs/ar2_workshop.db
rm -f obs/report.html obs/run_events.jsonl obs/workshop_public.url
rm -rf versions/v_*

echo "Cleared obs/ + versions/v_*"
echo "Next: ./scripts/workshop_ngrok.sh start && ./scripts/workshop_ngrok.sh sync-env"
echo "Then:  AR2_FRESH=1 AR2_K=1 ./scripts/run_k3_modal_e2e.sh"
