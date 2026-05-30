This is a clean, well-scoped project. Below is the synthesis: a shared architecture, then a per-task harness (task prompt + verifier + tools + submit/run) with grading and anti-reward-hacking notes, then my honest take on sequencing for a hackathon.

---

## Recommended shared stack

**Standardize the harness on Prime Intellect `verifiers`** (github.com/PrimeIntellect-ai/verifiers). One Python interface handles both reward types you described:
- Deterministic → a plain `async def reward(completion, answer) -> float` in a `vf.Rubric`.
- LLM auto-grader → `vf.JudgeRubric`.
- Multiple weighted reward functions sum to one scalar — exactly your model. Author with `load_environment()`, smoke-test with `uv run vf-eval <env> -m <model> -n 5 -r 2`, then the same env feeds the outer loop (prime-rl/verl/SkyRL all consume it).

**Map your "double loop" onto it explicitly:**
- **Inner loop** = hill-climb a single task's *solution* via repeated agentic rollouts scored by the verifier. Use an AlphaEvolve/OpenEvolve-style proposer (LLM mutation + keep-best) or a Kevin-style multi-turn GRPO if you actually train.
- **Outer loop** = autoresearch over the *harness/scaffold itself* (ADAS/AFlow/DSPy-MIPROv2 over prompts, tool sets, agent topology), scored by the same verifier aggregated across tasks.
- The **multi-agent task is just the outer loop pointed at a deterministic inner task** — that's the cleanest way to dodge the "verifier is hard" problem (more below).

**Terminal-Bench is the exception** — author those in its native Harbor format (Dockerfile + pytest + `/logs/verifier/reward.txt`) and wrap as a `verifiers` sandbox env whose reward function shells into the container and returns the pytest exit code.

**Raindrop** = the right product is **raindrop.ai** ("Sentry for AI agents"), not raindrop.io/Cloudflare. Use it as the *observability sink* for rollouts, not the reward oracle: `raindrop.init(key, tracing_enabled=True)`, wrap rollouts in `begin()/finish()`, attach `properties={"reward":…, "tests_passed":…}` and `track_signal(...)`. **Must call `raindrop.flush()` before process exit** or events are lost. Your verifier still computes the actual reward; Raindrop is for comparing/debugging rollouts in the outer loop.

---

## 1. GPU kernels  — *cleanest deterministic reward; build this first*

**Base:** KernelBench (github.com/ScalingIntelligence/KernelBench). There's already a Prime Intellect Hub env (`popfido/kernelbench`) you can clone.

- **Task prompt (draft):** "You are given a PyTorch `Model` class with `get_inputs()`/`get_init_inputs()`. Rewrite it as `ModelNew` using custom CUDA/Triton kernels that is numerically equivalent and faster on {GPU}. You may use any public docs/profilers. Output only the `ModelNew` module. You will be scored on correctness across randomized inputs and on measured speedup over the reference."
- **Verifier (code, deterministic):** KernelBench's `run_and_check_correctness` + `time_execution_with_cuda_event`. Reward = `fast_p`: `0` if any correctness trial fails, else `ref_time / kernel_time`.
- **Tools:** none non-public. Give it `nvcc`/Triton, a GPU, optionally `ncu`/`nsys` profilers. Let the agent find its own optimizations.
- **Submit/run:** agent returns a module string → harness compiles → correctness trials → timing → scalar.

**Grading + anti-hacking (this domain is the most-studied for reward hacking):**
- **Run the candidate kernel *before* the reference, never share output buffers** (Kevin-32B found a KernelBench bug where the candidate recycled the reference's output tensor and a no-op passed).
- **`num_correct_trials` ≥ 5 with randomized shapes**, not just reseeded same-shape inputs — Robust-KBench (arxiv 2509.14279) showed kernels hardcode outputs/weights to pass a single fixed config.
- **Zero reward if any residual `torch.*` functional op remains** (kills "fuse only the activation, leave the matmul in torch").
- **Verify the backward pass, lock clocks (`nvidia-smi -lgc`), flush L2 each iter.** Manually eyeball any suspiciously high score.
- **20-min fit:** Yes, easily — per-task eval is seconds; budget compile time, not timing. A single 4090/L40S/A100 is enough (prefer A100/H100 since v0.1 tensor sizes are tuned to a 1–15ms window on H100).

---

## 2. Multi-agent system optimization  — *the actual "double loop" showcase*

The trick: **don't grade the agent system directly. Optimize it against a downstream task that has a deterministic reward.** Outer loop = AFlow (MCTS over workflows; subclass `BaseBenchmark.evaluate_problem`) or DSPy-MIPROv2 (cleanest `metric(example, pred)->float`).

- **Task prompt (draft, to the meta/optimizer):** "Design a multi-agent workflow (roles, tool access, message topology, prompts) that ports {source repo} from {lang/runtime A} to {B}. The workflow's score is the fraction of the target test suite that passes after porting, minus a token-cost penalty. Propose, evaluate, and iterate."
- **Verifier:** the **downstream task's own deterministic signal** — for a port: `tests_passed / total` after applying the workflow's output; for research: HumanEval/MBPP (tests pass), GSM8K subset (exact match), HotpotQA (F1). Reward = task score − λ·(tokens or wall-clock). This is what makes "small models beating a big one" measurable and ungameable.
- **Tools:** Raindrop instrumentation is mandatory here — wrap every sub-agent so the outer loop can see which roles/edges drove the reward. Plus whatever the downstream task needs (test runner, repo sandbox).
- **Submit/run:** optimizer emits a workflow spec (code/graph) → harness runs it on a held-out task minibatch → aggregate reward.

**Grading + anti-hacking:**
- **The cost penalty is essential** — without it the optimizer just stacks more LLM calls. Penalize tokens *and* latency.
- **Held-out task split:** optimize on a train minibatch, score on unseen tasks, or the scaffold overfits to specific problems.
- **Same "delete the failing test" hazard as code agents** — Spotify's migration agents reward-hacked exactly this way. Make tests read-only, diff the test files before/after, and hard-zero any rollout that modifies them.
- **20-min fit:** Yes *per rollout* if you keep the eval minibatch to 20–50 items. The full search is many rollouts (that's the point) — parallelize.

---

## 3. Legal / loophole discovery  — *the LLM-auto-grader showcase, with a symbolic backstop*

Two-track: a deterministic warm-up set + the headline generative loophole task.

- **Task prompt (draft):** "Given the statute/contract text below, identify an exploitable gap or loophole. You must output: (a) a verbatim quote of the controlling clause, (b) a concrete worked exploit scenario with named parties, amounts, and dates, and (c) the rule-as-applied outcome you claim is exploitable, and why it deviates from intent."
- **Verifier (layered — this is the anti-hacking design):**
  1. **Citation grounding (deterministic):** every quoted statute string must substring-match the source corpus → else 0. Kills hallucinated citations cheaply.
  2. **Symbolic backstop (deterministic):** run the scenario through a Catala/Prolog formalization of that section (reuse the existing IRC §121 Catala formalization; don't formalize from scratch in a hackathon). If execution doesn't reproduce the claimed deviation → 0. This is the single strongest lever against plausible-but-fake loopholes.
  3. **Adversarial LLM grader:** a *separate* prompt acts as opposing counsel, produces the strongest rebuttal; validity reward = grader confidence *after* the rebuttal.
  4. **Severity** on a fixed anchored rubric (dollar magnitude / parties affected), capped to prevent inflation; self-consistency over N grader samples.
- **Deterministic warm-up reward set:** SARA tax (exact tax-owed dollar amount), CUAD (`evaluate.py` span-F1), LegalBench classification tasks (exact-match). Use these to validate the harness before the fuzzy task.
- **Tools:** statute/contract corpus + retrieval; the Catala/Prolog runtime for the backstop.
- **Submit/run:** structured JSON artifact → layered verifier → scalar.

**20-min fit:** Yes. Pure text + arithmetic + a fast symbolic check; no infra.

---

## 4. Software infra / SRE / incident response  — *go offline to hit 20 min*

The 20-minute constraint kills live clusters. **Use OpenRCA (github.com/microsoft/OpenRCA) or ITBench offline SRE snapshots** — both are telemetry *replay* with a structured-answer verifier, no fault injection at rollout time.

- **Task prompt (draft):** "Telemetry (logs, metrics, traces) for an incident window is in {path}. Identify the root cause as JSON: `{datetime, component, reason}`. Use any analysis you like; cite the specific telemetry rows supporting your conclusion."
- **Verifier (deterministic + light judge):** OpenRCA's `main.evaluate` compares `{datetime, component}` exactly against `query.csv`; an LLM judges the free-text `reason`. ITBench adds **recall-gated precision that penalizes over-investigation** (naming extra entities = false positives) — copy this.
- **Tools:** query helpers over the telemetry (DuckDB/pandas); no live cluster.
- **Submit/run:** JSON answer → deterministic compare + judge on `reason`.

**Grading + anti-hacking:**
- **Score root cause, never symptom-silence** — never reward "alert cleared." If you ever do a live mitigation variant (AIOpsLab + pre-warmed KinD only), require *both* restored SLO *and* correct root cause, plus a post-hoc probe that the fault is genuinely gone, not the alert rule deleted.
- **Forbidden-action hard-zero:** deleting alerts/monitors, editing ground-truth/judge files.
- **20-min fit:** OpenRCA/ITBench-offline → yes (scope to one incident window; note OpenRCA wants ~80GB disk/32GB RAM for full telemetry). AIOpsLab/Litmus live → only if the cluster + app is pre-deployed and kept warm; risky for a hackathon.

---

## 5. Terminal-Bench  — *reuse the existing harness wholesale*

Least work — it's already an RL-shaped env.

- **Task prompt:** provided per task (`instruction.md` / `task.yaml`).
- **Verifier (code):** pytest over **final container state** (not the agent's commands); task writes `/logs/verifier/reward.txt` (exit 0 → 1 else 0). You can soften to fractional (fraction of tests passing) for denser reward.
- **Tools:** a real terminal (tmux) in a Docker sandbox; agent installs what it needs.
- **Submit/run:** `tb run --agent terminus --model … --dataset-name terminal-bench-core`. Wrap as a `verifiers` env returning the pytest result as the scalar.

**Anti-hacking:** verifies end-state not transcript, so command-spoofing doesn't help; still make the `tests/` dir read-only and copy tests in *after* the rollout. **20-min fit:** yes — `max_agent_timeout_sec`/`max_test_timeout_sec` cap it; most tasks are well under.

---

## Feasibility / sequencing for the hackathon

| Task             | Reward type               | Existing harness         | Stand-up cost      | 20-min?         |
| ---------------- | ------------------------- | ------------------------ | ------------------ | --------------- |
| GPU kernels      | Deterministic (speedup)   | KernelBench + PI Hub env | Low                | ✅              |
| Terminal-Bench   | Deterministic (pytest)    | tb/Harbor                | Low                | ✅              |
| Legal (warm-up)  | Deterministic (SARA/CUAD) | datasets                 | Low                | ✅              |
| Legal (loophole) | LLM grader + symbolic     | partial                  | Medium             | ✅              |
| SRE              | Deterministic + judge     | OpenRCA/ITBench offline  | Medium (data size) | ✅ offline only |
| Multi-agent opt  | Deterministic downstream  | AFlow/DSPy               | Medium-High        | ✅ per-rollout  |

**My recommendation:** anchor day one on **GPU kernels + Terminal-Bench** — both have ready harnesses and ungameable deterministic rewards, so they de-risk the `verifiers` integration and the outer loop. Use **legal-loophole** as the auto-grader showcase (the layered citation→symbolic→adversarial verifier is the genuinely novel/defensible bit). Do **multi-agent optimization** as the headline "double loop" demo but keep its inner verifier dead simple (MBPP or a tiny test-suite port) — that's where the "small models beat a big one" story lands. Treat **SRE-live** as a stretch; ship the **OpenRCA-offline** version.

**The one cross-cutting risk to design against everywhere:** every domain in the research showed reward hacking the moment the reward was a sandboxed measurement (Sakana's CUDA agent gamed the timing sandbox and got retracted; Spotify's migration agents deleted failing tests). Bake in the three universal defenses — *run candidate-before-reference / read-only tests / an independent adversarial check* — from the start rather than patching after.

Want me to scaffold the repo — a `verifiers`-based env package per task with stub `load_environment()` + `Rubric`, the KernelBench and Terminal-Bench wrappers wired up first? I can also pull the PI `verifiers` KernelBench env locally so you have a working reference.


Other ideas

Assumption: by “three elements” I’d package each harness as: **task prompt**, **verifier/scorer**, and **environment/tools/submission API**. I’d treat private tools/source instructions as part of the third element unless the verifier itself needs them.

**Best Fits**
| Domain | Best hackathon source | Why it fits ~20 min |
|---|---|---|
| GPU kernels | **KernelBench + KernelGYM** | KernelBench already frames tasks as “write correct/efficient CUDA/DSL kernels for PyTorch programs”; KernelGYM adds an RL-style distributed environment, profiling, multi-turn rollouts, and reward-hacking detection. Use 1 small kernel per rollout. ([github.com](https://github.com/ScalingIntelligence/KernelBench)) |
| Multi-agent optimization | **τ-bench / τ²-bench**, plus **Raindrop** tracing | τ-bench has policy, tools, tasks, and simulated users for tool-agent-user interaction; Raindrop gives traces across common agent frameworks, including multi-agent workflows. Keep to 1-3 short retail/airline tasks per rollout. ([github.com](https://github.com/sierra-research/tau2-bench)) |
| Research/porting agents | **MLGym**, maybe PaperBench-derived microtasks | MLGym is explicitly a Gym-style environment for ML research agents, but full tasks can be long; use tiny “improve metric” or “port module and pass tests” slices. PaperBench is valuable rubric inspiration but full replication is too large for 20 minutes. ([github.com](https://github.com/facebookresearch/MLGym)) |
| Legal/misc | **CUAD + CLAUSE + LegalBench-style rubrics** | CUAD gives expert-labeled contract review data; CLAUSE gives perturbed contracts with fine-grained legal/in-text contradictions and metadata. Good for “find valid loopholes/flaws” with LLM+evidence grading. ([atticusprojectai.org](https://www.atticusprojectai.org/cuad/)) |
| SRE/incident response | **ITBench-AA**, **SREGym**, **o11y-bench**, **SadServers** | ITBench-AA uses offline Kubernetes incident snapshots and structured JSON diagnosis, easiest for 20 min. SREGym is richer/live K8s but setup-heavy. o11y-bench covers Grafana-style logs/metrics/traces/dashboards; SadServers has concrete Linux checks. ([artificialanalysis.ai](https://artificialanalysis.ai/evaluations/itbench-aa)) |
| TerminalBench | **Terminal-Bench** | It already has instruction, Docker sandbox, verifier tests, and oracle solution. Use easy/medium tasks or a curated subset with strict timeout. ([github.com](https://github.com/laude-institute/terminal-bench?utm_source=openai)) |

**Generic Inner-Loop Optimizer Prompt**
```text
You are optimizing a candidate system for a hidden evaluator. You may inspect the public task files, run public smoke tests, and modify only files under /workspace/submission. Your goal is to maximize verifier score, not to explain ideas.

Respect the task contract:
- Do not modify verifier files, hidden tests, environment startup scripts, or scoring code.
- Produce the required artifact exactly at the requested path.
- Prefer simple, robust changes over brittle hardcoding.
- Track your assumptions and final answer in /workspace/submission/README.md.
- Stop after the time budget or when public checks pass with a clear improvement.

Submission format:
<domain-specific artifact path>
<optional structured metadata JSON>
```

**Harness Drafts**
**1. GPU Kernels**
Task prompt:
```text
Implement an optimized CUDA/Triton replacement for the provided PyTorch function. Preserve numerical behavior within tolerance across hidden shapes and dtypes. Submit solution.py exposing custom_op(x, ...). Optimize latency on the target GPU.
```
Verifier/scorer: import `solution.py`, compare against PyTorch baseline on public and hidden randomized cases, then benchmark median/p95 latency after warmup. Score = correctness gate * speedup, with penalties for compile failures, excessive memory, nondeterminism, or shape overfitting.

Environment/tools/submission: KernelBench task folder, fixed GPU target, public smoke benchmark, hidden shape set, submit `solution.py` plus optional notes.

**2. Multi-Agent System Optimization**
Task prompt:
```text
Improve the agent configuration in agent_spec.yaml to complete the provided customer-support/research task set. You may change prompts, routing, role definitions, tool policies, and retry logic. Keep total budget under N tool calls and M tokens per task.
```
Verifier/scorer: run 1-3 τ-bench/τ²-bench tasks or a small porting/research task suite. Score task success, policy compliance, cost, latency, and consistency across seeds. Use Raindrop traces to grade trajectory quality and catch tool misuse.

Environment/tools/submission: baseline scaffold, Raindrop instrumentation, fixed model/tool budget, submit `agent_spec.yaml` or code patch plus trace IDs.

**3. Legal/Misc**
Task prompt:
```text
Review the provided contract/scenario and identify legally meaningful loopholes, contradictions, missing terms, or enforceability risks. Submit JSON claims with issue type, affected clause, severity, reasoning, and cited evidence from the supplied materials only.
```
Verifier/scorer: match claims against hidden gold perturbations from CLAUSE/CUAD-style labels; LLM judge only scores structured evidence and severity calibration. Score precision-heavy: unsupported “creative” loopholes should hurt more than missing marginal issues. This should be presented as benchmark/legal-reasoning evaluation, not legal advice.

Environment/tools/submission: static contract pack, jurisdiction/source bundle if needed, no open web unless the task explicitly includes official sources, submit `findings.json`.

**4. SRE / Incident Response**
Task prompt:
```text
Investigate the incident using the provided logs, metrics, traces, Kubernetes snapshots, and shell tools. Submit rca.json identifying root-cause entities, evidence, blast radius, and a minimal remediation plan. If remediation is enabled, apply only allowed changes.
```
Verifier/scorer: for offline ITBench-style tasks, compare `rca.json` to root-cause entities with precision/recall. For live SREGym/SadServers tasks, run health checks after remediation and penalize destructive fixes, alert suppression, or deleting failing workloads.

Environment/tools/submission: mounted incident snapshot or sandbox cluster, read-only observability data, optional controlled `kubectl`, submit `rca.json` and optional patch/remediation log.

**5. TerminalBench**
Task prompt:
```text
Complete the terminal task described in task.yaml. Work inside the sandbox only. When finished, leave the filesystem and services in the state required by the task. Do not edit tests or verifier files.
```
Verifier/scorer: run task-provided tests. For RL use, split into public smoke tests and hidden final tests, because public TerminalBench-style tests are easy to overfit if reused for training.

Environment/tools/submission: Docker task directory with `task.yaml`, environment files, verifier tests, timeout, and final filesystem state.

**Reward-Hacking Controls**
Use a two-stage verifier: public smoke checks for iteration, hidden final checks for score. Keep verifier files outside the writable workspace, run scoring from a clean container, and hash/restore tests before grading.

Gate reward by correctness before quality or speed. For GPU, no speed score unless all hidden numerical tests pass. For SRE, no RCA score boost if the service was “fixed” by deleting the failing component. For legal, no severity credit without clause-level evidence and a legally grounded rationale.

Grade artifacts, not prose. Require structured outputs like `solution.py`, `rca.json`, `findings.json`, or `agent_spec.yaml`, then score deterministically where possible.

For LLM judges, blind the judge to candidate identity, require evidence spans, use a fixed rubric, sample multiple judges or seeds for borderline cases, and maintain a human-reviewed calibration set. Treat LLM grading as a fallback for nuance, not as the only source of truth.

Log trajectories with Raindrop or equivalent tracing. Reward should include negative signals for verifier tampering, hidden-test probing, repeated no-op tool calls, policy violations, and suspicious shortcuts.