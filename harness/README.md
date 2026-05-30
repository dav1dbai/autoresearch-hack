# Harness — the double auto-research loop

This package is the **task / RL-environment layer** of the project: a YAML task
format, pydantic models that parse it, and a Modal-sandbox executor that runs a
basic optimization loop with the **verifier isolated in its own sandbox**.

It complements the design in [`../DESIGN.md`](../DESIGN.md) (the AR² meta-loop)
and the research survey in
[`../inner-loop/research/harness-design.md`](../inner-loop/research/harness-design.md)
(existing repos/benchmarks per domain). Read those for the full picture; this
file explains the *principles* and the concrete contract.

---

## 1. The two loops

We run optimization at two nested levels.

- **Inner loop** — inside one task/environment, optimize a single *candidate*
  against a verifier reward. The candidate is the artifact being produced: a GPU
  kernel, a set of legal findings, a terminal solution, a multi-agent config. One
  rollout = repeatedly edit the candidate, score it, keep what's better.
- **Outer loop (auto-research)** — propose and mutate *strategies* across
  rollouts — the prompt, the agent topology, search operators, the environment
  config — using the inner reward as fitness. This is where the actual research
  happens (GEPA / AFlow / ADAS / DGM-style archives).

`executor.py` implements a deliberately minimal **inner** loop today (iterate →
verify → feed back). The outer loop wraps many of these rollouts; that lives in
`../prime_research/` and `../meta.py`.

### Why this only works on net-new / unbounded tasks

Auto-research is the right tool **only when there is no reference solution.**
If a gold answer exists, handing the transcript to the agent collapses the
experiment — it just emits the answer and "number go up" is meaningless. The
tasks here are chosen to be open-ended or verifier-bounded with **no leakable
reference**: novel legal loopholes, a faster kernel than any we have, a passing
hidden test suite. When the target is "reach a new place," it doesn't matter if
intermediate signal leaks — higher reward is genuinely better. Pick tasks where
the ceiling is unknown.

---

## 2. A task is an RL environment: the 4 elements

Every task (`tasks/*.yaml`, parsed by [`schema.py`](schema.py) into `TaskDef`)
is one self-contained environment with four parts:

1. **Task prompt** (`prompt`) — what the agent must do *and* the exact
   submission contract (file path, schema, how to submit).
2. **Verifier** (`verifier`) — a command that runs a submission and returns a
   scalar reward. Deterministic code, an LLM judge, or a blend (`kind:`).
3. **Tools / inputs** (`inputs`, image specs) — visible files copied into the
   agent's workspace, plus any non-public dependency the *verifier* requires
   (a GPU, a corpus, a judge model). Default: let the agent find its own tools;
   only pre-provision what the verifier needs.
4. **Submission & run mechanism** (`submission`, the `submit` tool) — the single
   artifact the agent writes, and the handoff that triggers grading.

```yaml
id: terminal-coding-rle
domain: terminal
prompt: |            # element 1
  Implement run-length encoding in ./submission/solution.py ...
inputs: []          # element 3 (visible files)
submission:         # element 4 (the artifact + how it's handed off)
  path: submission/solution.py
  template: "def rle_encode(s): ..."
verifier:           # element 2
  kind: deterministic
  files: [...]      # staged into a SEPARATE sandbox, unreachable by the agent
  command: [python, run_pytest.py, --tests, tests]
  reward_key: reward
```

---

## 3. The one invariant: the verifier is out of reach

> The grader runs in a **separate sandbox** the agent's process never touches.
> The submission is the only thing that crosses the boundary.

`executor.py` enforces this physically: the agent runs in one Modal sandbox; the
verifier files are staged into a **fresh second sandbox** (network-disabled for
deterministic graders), the submission is copied in, the command runs, the
reward comes back, and the sandbox is torn down. The agent cannot read the
hidden tests, edit the scorer, or fake the metric — the cheapest reward hacks
are blocked by construction, not by hope.

---

## 4. Grading & anti-reward-hacking

A competent optimizer **will** find any hole in the verifier — reward hacking is
the *expected* behavior, not a tail risk. Treat the verifier as adversarially as
the agent. Cross-cutting principles (full per-domain detail in
[`harness-design.md`](../inner-loop/research/harness-design.md) §3):

1. **Prefer ground truth over judgment.** Passing hidden tests, a correct
   kernel, a clause that actually exists — cheap, hard to fake. Use LLM judges
   only where the task is irreducibly open-ended, and only as a *secondary*
   signal on top of a deterministic gate.
2. **Gate before you grade.** Run a validity check first (kernel is invoked;
   quote exists verbatim; tests collect). Fail → reward 0 *before* any subjective
   scoring. (Legal verifier gates on quote existence; kernel gates on `allclose`.)
3. **Hide and rotate tests.** Inject tests after the agent stops; rotate eval
   instances across outer-loop generations so the optimizer can't overfit.
4. **Verify the fix, not the symptom.** Re-check the actual artifact/fault, not a
   green health probe or a loose tolerance.
5. **Metrics out-of-band and read-only.** Anything the agent can write to, it
   will eventually edit. Compute reward from a surface its credentials can't
   reach — hence the separate sandbox.
6. **LLM judges are exploitable** ("One Token to Fool LLM-as-a-Judge",
   arXiv:2507.08794). Mitigate: reference-grounded analytic rubrics, a stronger
   judge, ensemble/majority vote, an adversarial second judge, verbosity
   penalties, leak-phrase detection.
7. **Never reward observability/cost directly.** Tokens, latency, Raindrop
   "stuck-in-loop" signals → constraints/penalties, never the primary reward.
8. **Constrain blast radius.** Scoped creds, no destructive ops, step/time/token
   caps; penalize action count so "nuke and restart" loses to a surgical fix.

---

## 5. Domains (status here vs. the survey)

Shipped as runnable YAML in `tasks/`:

| Task | Domain | Verifier | Runs on |
|---|---|---|---|
| `terminal-coding-rle` | terminal | hidden pytest (fraction passing) | **CPU** — smoke test |
| `legal-loopholes-demo` | legal | deterministic quote-existence gate + quality score | **CPU** |
| `kernelbench-square-matmul` | kernel | allclose gate → CUDA-event timing → bounded speedup | **GPU (H100)** |

Surveyed and ready to add as new YAML (bases noted in `harness-design.md`):
multi-agent system optimization (GEPA/AFlow outer loop + TransCoder execution
tests + Raindrop instrumentation), SRE/incident response (AIOpsLab + Chaos Mesh,
single-fault localization), full Terminal-Bench (`tb run`). All are scoped to
fit a **~20-minute** rollout including LLM and execution time.

---

## 6. Run it

```bash
uv sync
cp ../.env.example ../.env   # fill in ANTHROPIC_API_KEY / OPENAI_API_KEY + Modal creds

# CPU smoke test (recommended first run):
uv run modal run harness/executor.py --task terminal-coding-rle

# other tasks:
uv run modal run harness/executor.py --task legal-loopholes-demo
uv run modal run harness/executor.py --task kernelbench-square-matmul

# swap the in-sandbox coding agent:
uv run modal run harness/executor.py --task terminal-coding-rle --agent-cmd "claude -p"
```

The agent CLI is whatever you point `agent.cmd` (or `--agent-cmd`) at — `codex
exec` or `claude -p` are pre-installed in the default agent image. The prompt is
appended as the final argument each turn; feedback from the previous verifier
run is written to `./feedback.json` in the agent's workspace.

### Adding a task

1. Drop a `tasks/<name>.yaml` (copy an existing one).
2. Point `verifier.files` at a scorer that prints JSON containing `reward_key`,
   ideally normalized to `[0, 1]`. Keep correctness a hard gate.
3. Reference any verifier/input assets under `tasks/assets/` or reuse the
   domain verifiers under `../inner-loop/domains/`.
