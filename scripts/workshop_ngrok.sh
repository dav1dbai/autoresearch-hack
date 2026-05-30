#!/usr/bin/env bash
# Expose local Raindrop Workshop (:5899) to Modal via ngrok.
#
#   ./scripts/workshop_ngrok.sh start     # raindrop workshop + ngrok tunnel
#   eval "$(./scripts/workshop_ngrok.sh export)"   # shell env for harness
#   ./scripts/workshop_ngrok.sh sync-env  # write URL into .env (Modal Secret.from_dotenv)
#   ./scripts/workshop_ngrok.sh stop
set -euo pipefail

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
WORKSHOP_PORT="${RAINDROP_WORKSHOP_PORT:-5899}"
NGROK_API="${NGROK_API:-http://127.0.0.1:4040}"
URL_FILE="$ROOT/obs/workshop_public.url"
PID_FILE="$ROOT/obs/workshop_ngrok.pid"
LOG_FILE="$ROOT/obs/workshop_ngrok.log"
ENV_FILE="$ROOT/.env"

_raindrop_bin() {
  if [[ -n "${RAINDROP_BIN:-}" && -x "${RAINDROP_BIN}" ]]; then
    echo "$RAINDROP_BIN"
  elif [[ -x "${HOME}/.raindrop/bin/raindrop" ]]; then
    echo "${HOME}/.raindrop/bin/raindrop"
  else
    command -v raindrop
  fi
}

_workshop_health() {
  curl -fsS "http://127.0.0.1:${WORKSHOP_PORT}/health" >/dev/null 2>&1
}

ensure_workshop() {
  if _workshop_health; then
    return 0
  fi
  local bin
  bin="$(_raindrop_bin)"
  echo "Starting Raindrop Workshop on :${WORKSHOP_PORT}..."
  "$bin" workshop start >/dev/null 2>&1 || "$bin" workshop >/dev/null 2>&1 || true
  for _ in $(seq 1 80); do
    if _workshop_health; then
      return 0
    fi
    sleep 0.25
  done
  echo "Workshop did not become healthy on :${WORKSHOP_PORT}" >&2
  return 1
}

ngrok_public_url() {
  curl -fsS "${NGROK_API}/api/tunnels" | python3 -c '
import json, sys
data = json.load(sys.stdin)
tunnels = data.get("tunnels") or []
for prefer in ("https", "http"):
    for t in tunnels:
        if t.get("proto") == prefer and t.get("public_url"):
            print(t["public_url"].rstrip("/"))
            raise SystemExit(0)
for t in tunnels:
    if t.get("public_url"):
        print(t["public_url"].rstrip("/"))
        break
'
}

ngrok_running() {
  [[ -f "$PID_FILE" ]] && kill -0 "$(cat "$PID_FILE")" 2>/dev/null
}

cmd_start() {
  command -v ngrok >/dev/null 2>&1 || {
    echo "ngrok not found — install from https://ngrok.com/download" >&2
    return 1
  }
  ensure_workshop
  mkdir -p "$(dirname "$URL_FILE")"

  if ! ngrok_running; then
    echo "Starting ngrok → localhost:${WORKSHOP_PORT}..."
    ngrok http "$WORKSHOP_PORT" --host-header="localhost:${WORKSHOP_PORT}" --log=stdout >"$LOG_FILE" 2>&1 &
    echo $! >"$PID_FILE"
  fi

  local url=""
  for _ in $(seq 1 80); do
    url="$(ngrok_public_url 2>/dev/null || true)"
    if [[ -n "$url" ]]; then
      break
    fi
    sleep 0.25
  done
  if [[ -z "$url" ]]; then
    echo "ngrok started but no public URL yet — check $LOG_FILE" >&2
    return 1
  fi

  printf '%s\n' "$url" >"$URL_FILE"
  echo "Workshop UI (local):  http://127.0.0.1:${WORKSHOP_PORT}"
  echo "Workshop URL (Modal): ${url}"
  echo "Saved → ${URL_FILE}"
}

cmd_export() {
  local url
  if [[ -f "$URL_FILE" ]]; then
    url="$(tr -d '\n' <"$URL_FILE")"
  fi
  if [[ -z "${url:-}" ]]; then
    cmd_start >/dev/null
    url="$(tr -d '\n' <"$URL_FILE")"
  fi
  printf 'export RAINDROP_WORKSHOP_URL=%q\n' "$url"
  printf 'export RAINDROP_LOCAL_DEBUGGER=%q\n' "${url}/v1/"
}

cmd_sync_env() {
  local url
  url="$(tr -d '\n' <"$URL_FILE" 2>/dev/null || true)"
  if [[ -z "$url" ]]; then
    cmd_start >/dev/null
    url="$(tr -d '\n' <"$URL_FILE")"
  fi
  touch "$ENV_FILE"
  ENV_FILE="$ENV_FILE" WORKSHOP_URL="$url" python3 - <<'PY'
import os
from pathlib import Path

env_path = Path(os.environ["ENV_FILE"])
url = os.environ["WORKSHOP_URL"].rstrip("/")
debugger = f"{url}/v1/"
lines = env_path.read_text(encoding="utf-8").splitlines() if env_path.exists() else []
keys = {"RAINDROP_WORKSHOP_URL": url, "RAINDROP_LOCAL_DEBUGGER": debugger}
seen: set[str] = set()
out: list[str] = []
for line in lines:
    key = line.split("=", 1)[0].strip() if "=" in line else ""
    if key in keys:
        if key not in seen:
            out.append(f"{key}={keys[key]}")
            seen.add(key)
        continue
    out.append(line)
for key, val in keys.items():
    if key not in seen:
        out.append(f"{key}={val}")
env_path.write_text("\n".join(out).rstrip() + "\n", encoding="utf-8")
print(f"Updated {env_path} → RAINDROP_WORKSHOP_URL={url}")
PY
}

cmd_stop() {
  if ngrok_running; then
    kill "$(cat "$PID_FILE")" 2>/dev/null || true
  fi
  rm -f "$PID_FILE" "$URL_FILE"
  echo "ngrok stopped"
}

cmd_status() {
  if _workshop_health; then
    echo "Workshop: up (:${WORKSHOP_PORT})"
  else
    echo "Workshop: down"
  fi
  if ngrok_running; then
    echo "ngrok: running (pid $(cat "$PID_FILE"))"
    if [[ -f "$URL_FILE" ]]; then
      echo "public: $(tr -d '\n' <"$URL_FILE")"
    fi
  else
    echo "ngrok: stopped"
  fi
}

usage() {
  cat <<EOF
Usage: $(basename "$0") <start|stop|status|export|sync-env|url>

  start      Start Workshop (if needed) + ngrok tunnel to :${WORKSHOP_PORT}
  export     Print export RAINDROP_* lines for eval
  sync-env   Write RAINDROP_WORKSHOP_URL into .env for Modal Secret.from_dotenv()
  url        Print current public URL
  stop       Stop ngrok tunnel
  status     Show Workshop + ngrok state
EOF
}

case "${1:-}" in
  start) cmd_start ;;
  stop) cmd_stop ;;
  export) cmd_export ;;
  sync-env) cmd_sync_env ;;
  url) tr -d '\n' <"$URL_FILE" 2>/dev/null || cmd_start >/dev/null && tr -d '\n' <"$URL_FILE" ;;
  status) cmd_status ;;
  -h|--help|help) usage ;;
  *)
    usage >&2
    exit 2
    ;;
esac
