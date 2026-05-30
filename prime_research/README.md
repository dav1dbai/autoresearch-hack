# Kernel/Legal Autoresearch Rollouts

This runner is for open-ended tasks where seeing a transcript or public reference
answer would collapse the experiment. The inner loop does iterative research on
one task artifact and repeatedly calls a verifier. The outer layer fans out those
task rollouts on Modal and aggregates the reward curves.

Terminal-Bench is intentionally excluded from this path.

## Shape

```text
outer Modal fanout
  -> kernel task rollout
       -> agent edits submission/submission.py
       -> verifier scores correctness + speedup
       -> experiments.jsonl records the reward curve
  -> legal task rollout
       -> agent edits submission/findings.json
       -> verifier scores grounded loophole findings
       -> experiments.jsonl records the reward curve
```

The mutable workspace is `work/submission`. Verifiers and protected task inputs
are copied into a separate read-only `verifier/` directory and invoked by command.

## Local smoke run

Legal can run locally without GPU:

```bash
uv run python -m prime_research.runner --task legal-demo-loopholes
```

Kernel tasks require a CUDA GPU and KernelBench checkout:

```bash
KERNELBENCH_ROOT=/path/to/KernelBench \
uv run python -m prime_research.runner --task kernelbench-level1-first
```

## Modal run

```bash
uv run modal run prime_research/modal_app.py
```

Run one task:

```bash
uv run modal run prime_research/modal_app.py --task legal-demo-loopholes
uv run modal run prime_research/modal_app.py --task kernelbench-level1-first
```

Pass a different outer-provided research methodology with:

```bash
uv run modal run prime_research/modal_app.py \
  --task legal-demo-loopholes \
  --program prime_research/programs/kernel_legal_v0.md
```

The Modal GPU task uses `H100!`. Set `INNER_AGENT_CMD` in `.env` for the
headless coding agent.

Rollout logs are written to the Modal Volume
`kernel-legal-autoresearch-logs` under `runs/<task>/work/` and committed during
the inner loop. Inspect while a rollout is running:

```bash
uv run modal volume ls kernel-legal-autoresearch-logs /runs
uv run modal volume get kernel-legal-autoresearch-logs \
  /runs/legal-demo-loopholes/work/experiments.jsonl /tmp/experiments.jsonl
uv run modal volume get kernel-legal-autoresearch-logs \
  /runs/legal-demo-loopholes/work/agent.stderr.log /tmp/agent.stderr.log
```

## Config

Tasks live in `prime_research/configs/kernel_legal.toml`.

Current Prime Intellect env references from `inner-loop/research/`:

- `primeintellect/kernelbench`
- `sinatras/kernelbench-kguard`
- `primeintellect/contract-clause-review`

The runner keeps these as metadata and uses local open-ended verifiers for the
actual iterative research loop.

## Inner / outer interaction

The outer loop owns the markdown methodology program, for example
`prime_research/programs/kernel_legal_v0.md`. Each inner rollout receives that
program inside `work/program.md`, then runs iterative experiments against the
task verifier and records the reward curve in `work/experiments.jsonl`.
