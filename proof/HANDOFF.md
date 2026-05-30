# AR² — Handoff (2026-05-30)

Digest of session work. Read `DESIGN.md` (architecture), `DECISIONS.md` (work queue) in this folder.

## TL;DR
AR² = **autoresearch on autoresearch**. Inner loop = agent edits a solution to raise verifiable reward; outer loop = mutate the research stack across versions; the science claim is the **second derivative** (`∂²R/∂N∂t`).

**Status (2026-05-30 ~13:30):**
- **166 offline tests green** (`uv run pytest -q`)
- **Modal rollout pipeline proven** end-to-end (codex in gVisor sandbox, env scores kernel, reward returns)
- **`drive()` has run** — `obs/archive.jsonl` exists with GPU matmul v0 (inner curves persisted on Attempt)
- **CLI wired:** `uv run python -m harness --help` (stub / GPU / vast-rent flags)
- **Still open:** hack_detector not wired into `evaluate()`, D-02 parent sampling, hero dashboard curves, nanochat held-out shift

## What works + how to reproduce

Stub outer loop (fast, no billing):
```bash
cd ~/Desktop/autoresearch-hack
MATMUL_STUB=1 uv run python -m harness --stub -K 0
```

Single real Modal rollout (matmul env, codex inner agent):
```bash
AR2_BACKEND=modal AR2_GPU_BACKEND=local uv run python -c "
from dotenv import load_dotenv; load_dotenv('.env')
from pathlib import Path
from harness.runtime.score import score_repo
from harness.contracts import Budget
from envs.matmul import MatmulEnv
print(score_repo(Path('ar'), [MatmulEnv(split='train')], Budget(wall_seconds=300)))
"
```

GPU + Vast path:
```bash
AR2_BACKEND=modal AR2_GPU_BACKEND=vast uv run python -m harness --gpu --vast-rent -K 0
```

## The hard-won fix-chain — DO NOT re-break these
1. **Modal `.starmap`/`.map` MUST be wrapped in `with app.run():`** (else `Function has not been hydrated`).
2. **`sandbox_image` must `.add_local_python_source("harness","envs","infra")` + `.pip_install("numpy")`**.
3. **Secrets**: `run_rollout` uses `modal.Secret.from_dotenv()` + `autoresearch-openai`.
4. **codex auth**: run `codex login --with-api-key` inside container; use `gpt-5-codex` via `INNER_AGENT_CMD`.
5. **`_load_env` must rebuild Env via real `__init__`** (filter kwargs via `inspect.signature`).
6. **VastBackend**: validate `VAST_API_KEY` before reading kernel file on local path.

## Config / env
- `AR2_BACKEND` ∈ {`local`, `modal`}
- `AR2_GPU_BACKEND` ∈ {`local`, `modal`, `vast`}
- `AR2_MODAL_APP_MODE` — deployed vs ephemeral Modal app (`harness/modal_session.py`)
- `.env` (gitignored): `MODAL_PROFILE=autoresearch-hack`, API keys, agent cmds

## Gotchas
- Never `source .env` — use `load_dotenv("/abs/.env")`
- Modal logs by app-id: `modal app list` → `modal app logs ap-XXXX`
- GPU tests can flake if `MODAL_ENVIRONMENT` leaks between tests — `_fresh_gpu_backend` clears it

## File map
- `ar/entrypoint.py` — MUTABLE: `solve` + `improve`
- `harness/contracts.py` — SSOT types (immutable)
- `harness/loop/` — `outer.py` (drive/evaluate), `archive.py`, `evaluate.py`, `loader.py`
- `harness/runtime/` — `score.py`, `rollout.py`, `referee.py`, `sandbox.py`
- `harness/cloud/` — `runner.py`, `session.py` (Modal; not named `modal/` — SDK name clash)
- `harness/backends/gpu.py` — local / Modal / Vast GPU runners
- `harness/tracing/` — `telemetry.py`, `sync.py`
- `envs/` — `matmul.py`, `nanochat.py`, `pools.py`
- `infra/modal/images.py` — Modal images + profile guard
- `infra/vast/pool.py` — Vast rent/SSH helpers
- `obs/` — `dashboard.py` + runtime-generated `archive.jsonl`, `traces.db`, `report.html` (gitignored)
- `scripts/noop_agent.sh` — minimal inner agent for GPU plumbing tests

## Remaining work (priority)
1. Wire `hack_detector` back into `evaluate()` (removed during refactor)
2. D-02 `sample_parents()` — replace greedy `frontier()`
3. D-04 dashboard hero panel — layered `R(N,t)` from `Attempt.train_rollouts`
4. D-01/D-06 nanochat held-out + harness-controlled scoring
5. D-11 close-out — mark Modal path stable, live smoke in CI optional
