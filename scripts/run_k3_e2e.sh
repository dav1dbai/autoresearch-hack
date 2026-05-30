#!/usr/bin/env bash
# Canonical outer loop — Modal GPU smoke, real timing, Codex in Modal containers.
# Usage: AR2_K=3 ./scripts/run_k3_e2e.sh
set -euo pipefail
exec "$(dirname "$0")/run_k3_modal_e2e.sh" "$@"
