# AR² — autoresearch on autoresearch

Evolve the **whole** autoresearch repo (DGM-style), scored on **held-out** verifiable envs.
The second derivative: optimize how much better each AR *version* is at gaining reward.

**Read [`proof/documentation/DESIGN.md`](proof/documentation/DESIGN.md) for architecture.**
**[`proof/documentation/DECISIONS.md`](proof/documentation/DECISIONS.md) is the SSOT for what to build next**
(work queue, locked decisions, acceptance criteria).

See [`proof/documentation/README.md`](proof/documentation/README.md) for the full doc index.

## Layout
- `ar/` — ★ MUTABLE. AR v0 (seeded from `karpathy/autoresearch`). The artifact that evolves.
- `harness/` — IMMUTABLE. Meta-loop driver + runtime. `contracts.py` is the single source of truth.
  - `loop/` — `drive`, `evaluate`, archive persistence
  - `runtime/` — `score_repo`, rollout, referee, sandbox
  - `cloud/` — Modal fan-out + app session (named `cloud/` to avoid shadowing the `modal` SDK)
  - `backends/` — GPU compute (`local.py`, `modal_gpu.py`, `vast.py`; factory in `gpu.py`)
  - `tracing/` — telemetry injection + db sync
- `envs/` — IMMUTABLE to AR. Verifiable envs (the referees).
- `infra/` — Modal images (`infra/modal/`), Vast pool (`infra/vast/`), push collector.
- `obs/` — `dashboard.py` renders `report.html` from `archive.jsonl` + `traces.db` (generated at runtime, gitignored).
- `proof/documentation/` — design docs, decisions log, architecture plans, handoff notes.
- `versions/` — gitignored AR mutation snapshots (`ar_v_*` dirs from `improve()`).
- `vendor/autoresearch` — pristine karpathy reference (read-only).

## Setup (folder-scoped — never touches global config)
```bash
uv sync
cp .env.example .env     # fill MODAL_* (hackathon-scoped), OPENAI/ANTHROPIC keys
```
Creds load from `.env` via python-dotenv; `MODAL_PROFILE` scopes Modal to the hackathon workspace.

## Run
```bash
uv run python -m harness --help          # outer loop CLI (stub / GPU / vast-rent)
uvx modal run infra/modal/images.py      # Modal image definitions (scoped via .env)

uv run pytest -q                         # full offline suite (166 tests)
uv run pytest -m smoke -q                # fast cross-package smoke checks
uv run python -m obs.dashboard           # render obs/report.html from archive + traces
uv run python -m smoke.e2e_matmul        # offline e2e smoke script
```

### Why `pyproject.toml` only (no Makefile)?

`pyproject.toml` is the Python project manifest: dependencies (`uv sync`), pytest config, and package metadata. There is no separate build/task file — use `uv run …` for commands (see above).
