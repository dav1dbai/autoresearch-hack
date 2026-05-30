#!/usr/bin/env bash
# Run the cuda-oxide (Rust->PTX) kernel envs on the MODAL backend (cloud):
# codex edits kernel.rs INSIDE a rollout container, scored by the deployed
# ar2-cudaoxide H100 grader (nested Function.from_name call); meta candidates (M)
# fan out across containers in parallel, so width is ~free and only K (generations)
# and inner-climb depth cost wall time.
#
# Prereq (once): the grader app must be deployed:
#     modal deploy envs/cuda_oxide/app.py
#
# Usage:
#     scripts/modal_cuda_oxide.sh                 # defaults: reduction, K=3 M=4
#     K=5 M=8 BUDGET=180 AR2_CUDA_OXIDE_KERNELS=reduction scripts/modal_cuda_oxide.sh
set -euo pipefail
cd "$(dirname "$0")/.."

# The harness shells the bare `modal` CLI (ensure_app_deployed) — put .venv on PATH.
export PATH="$PWD/.venv/bin:$PATH"
# Only the clean KEY=VALUE lines (.env also holds command strings with spaces).
export $(grep -E '^(MODAL_(PROFILE|TOKEN_ID|TOKEN_SECRET)|OPENAI_API_KEY)=' .env | xargs)

export AR2_BACKEND=modal
export AR2_CUDA_OXIDE_KERNELS="${AR2_CUDA_OXIDE_KERNELS:-reduction}"  # scope to one kernel
export AR2_STALE_ITERS="${AR2_STALE_ITERS:-4}"

K="${K:-3}"; M="${M:-4}"; BUDGET="${BUDGET:-180}"
echo "modal cuda-oxide: kernels=$AR2_CUDA_OXIDE_KERNELS K=$K M=$M budget=${BUDGET}s"
exec caffeinate -i .venv/bin/python -u -m harness --cuda-oxide \
  -K "$K" -M "$M" --budget-seconds "$BUDGET" --no-workshop "$@"
