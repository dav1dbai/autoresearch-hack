#!/usr/bin/env bash
# Source from Modal driver scripts: expose local Workshop to Modal via ngrok.
# Opt out: AR2_WORKSHOP_TUNNEL=0 or AR2_WORKSHOP_NGROK=0
set -euo pipefail

if [[ "${AR2_WORKSHOP_TUNNEL:-1}" != "1" && "${AR2_WORKSHOP_NGROK:-1}" != "1" ]]; then
  return 0 2>/dev/null || exit 0
fi

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]:-$0}")" && pwd)"

if ! eval "$("$SCRIPT_DIR/workshop_ngrok.sh" export)"; then
  echo "ERROR: ngrok tunnel failed — Modal Codex cannot reach Workshop." >&2
  echo "       Try: ./scripts/workshop_ngrok.sh start" >&2
  return 1 2>/dev/null || exit 1
fi

"$SCRIPT_DIR/workshop_ngrok.sh" sync-env >/dev/null

echo "Modal → Workshop: ${RAINDROP_WORKSHOP_URL}"
echo "Workshop UI (local): http://127.0.0.1:${RAINDROP_WORKSHOP_PORT:-5899}"
