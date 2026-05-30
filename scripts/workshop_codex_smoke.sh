#!/usr/bin/env bash
# Prove Codex → Raindrop Workshop locally (no Modal). Checks live_events > 0.
set -euo pipefail
cd "$(dirname "$0")/.."

"${RAINDROP_BIN:-$HOME/.raindrop/bin/raindrop}" workshop start >/dev/null 2>&1 || true

before="$(sqlite3 "$HOME/.raindrop/raindrop_workshop.db" 'SELECT count(*) FROM live_events;' 2>/dev/null || echo 0)"
prompt="$(mktemp -t codex_smoke).md"
echo "Reply with exactly: workshop-ok" >"$prompt"

echo "Running Codex via raindrop_codex_exec.sh (local Workshop)..."
export RAINDROP_WORKSHOP_URL=http://127.0.0.1:5899
export RAINDROP_LOCAL_DEBUGGER=http://127.0.0.1:5899/v1/
./scripts/raindrop_codex_exec.sh exec -m gpt-5-codex -c preferred_auth_method=apikey "$prompt" >/tmp/codex_smoke_out.txt 2>&1 || true
rm -f "$prompt"

sleep 2
after="$(sqlite3 "$HOME/.raindrop/raindrop_workshop.db" 'SELECT count(*) FROM live_events;' 2>/dev/null || echo 0)"
runs="$(sqlite3 "$HOME/.raindrop/raindrop_workshop.db" 'SELECT count(*) FROM runs;' 2>/dev/null || echo 0)"

echo "live_events: $before → $after  |  runs: $runs"
if [[ "$after" -gt "$before" ]]; then
  echo "OK — open http://127.0.0.1:5899 and pick the latest Codex run"
  exit 0
fi
echo "FAIL — no live_events ingested. tail /tmp/codex_smoke_out.txt:" >&2
tail -30 /tmp/codex_smoke_out.txt >&2
exit 1
