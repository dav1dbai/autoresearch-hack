# AR² — autoresearch on autoresearch

Evolve the **whole** autoresearch stack across **versions**, scored on **held-out** verifiable envs.
The second derivative: optimize how much better each AR *version* is at gaining reward.

**Read [`proof/DESIGN.md`](proof/DESIGN.md) for architecture.**
**[`proof/DECISIONS.md`](proof/DECISIONS.md) is the SSOT for what to build next**
(work queue, locked decisions, acceptance criteria).

See [`proof/README.md`](proof/README.md) for the full doc index.

## Layout

**Mutation boundaries (D-15):** `proof/DECISIONS.md`.

| Zone | Paths |
|------|--------|
| **Mutable** (meta-agent each generation) | `ar/`, `harness/runtime/` → `versions/v_*/` |
| **Host-only** | `envs/`, `harness/contracts.py`, `harness/tracing/`, `harness/loop/`, `harness/cloud/`, `harness/backends/`, `infra/` |

- `ar/` — `solve` + `improve` policy
- `harness/` — host driver + `runtime/` (snapshot copy is meta-editable)
- `envs/` — referees
- `versions/` — gitignored snapshots (`v_*/ar/`, `v_*/harness/runtime/`)
- `proof/` — design docs + decisions (SSOT)
- `vendor/autoresearch` — read-only reference

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
