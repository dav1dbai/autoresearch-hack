# meta-autoresearch

Autoresearch on autoresearch. Karpathy's [autoresearch](https://github.com/karpathy/autoresearch)
optimizes `train.py` to minimize `val_bpb`. This optimizes the layer above:
`program.md` — the research *strategy* prose that drives the inner agent —
scored by the best `val_bpb` a full inner autoresearch run reaches under that
strategy. Keep/rollback, one level up.

The inner loop's intelligence is a stochastic coding agent; everything wrapping
it (bounding, scoring, keep/rollback) is determinate code.

## Layout

- `meta.py` — outer loop: propose a `program.md` edit, run an inner autoresearch
  session, score it, keep if `val_bpb` improved.
- `modal_app.py` — fan candidate strategies across Modal H100s, one inner run each.
- `repo/` — a working clone of karpathy/autoresearch (gitignored; clone it here
  for local runs).

## Setup

```bash
uv sync                                                   # install deps into .venv
cp .env.example .env                                      # fill in real values (.env is gitignored)
git clone https://github.com/karpathy/autoresearch repo   # for local inner runs
```

Credentials are folder-scoped via `.env` (loaded by `python-dotenv`) so they
never touch your global Modal/OpenAI config.

## Run

```bash
uv run python meta.py            # local outer loop
uv run modal run modal_app.py    # parallel fan-out on Modal
```

## Task harness + sandbox executor (`harness/` + `tasks/`)

The double auto-research loop, its principles, and the per-task RL-environment
contract are documented in [`harness/README.md`](harness/README.md). Tasks are
defined as YAML in [`tasks/`](tasks/) (parsed by `harness/schema.py`) and run by
a Modal-sandbox executor that puts a coding agent (Claude Code / Codex) in one
sandbox and the **verifier in a separate sandbox**, so the grader is out of the
agent's reach.

```bash
uv run modal run harness/executor.py --task terminal-coding-rle   # CPU smoke test
uv run modal run harness/executor.py --task legal-loopholes-demo
uv run modal run harness/executor.py --task kernelbench-square-matmul
```

## Kernel / Legal Autoresearch

For open-ended rollouts where the inner loop should iteratively research a new
artifact rather than solve from a reference transcript, use `prime_research/`.
This path intentionally excludes Terminal-Bench and focuses on GPU kernels plus
grounded legal loophole discovery.

```bash
uv run python -m prime_research.runner --task legal-demo-loopholes
uv run modal run prime_research/modal_app.py --task kernelbench-level1-first
```
