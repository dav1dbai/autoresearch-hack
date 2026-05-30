# Double Auto-Research Loop — Harness Design (Draft)

Hackathon target: every full rollout (LLM generation + execution/verification) fits in **~20 minutes**.

## 1. The double loop

- **Inner loop** — within one environment, optimize a *candidate* against a verifier reward. The candidate is the thing being produced: a GPU kernel, a multi-agent system config, a legal analysis, a remediation, a terminal solution. One rollout = one candidate scored.
- **Outer loop** — the auto-research loop. It proposes/mutates *strategies* (prompts, agent topology, search operators, environment configs) across rollouts, using the verifier reward as fitness. This is where GEPA / AFlow / MIPROv2 / ADAS live.

The verifier is the contract between the two loops. If the reward is hackable, the outer loop will find and amplify the exploit — reward hacking is not a tail risk here, it is the **expected** behavior of a competent optimizer. Treat the verifier as adversarially as the agent.

## 2. Common harness abstraction (the 4 elements)

Each harness is an RL-environment-like unit:

1. **Task prompt** — what the inner agent is asked to do + the exact submission contract (file paths, function signatures, output schema).
2. **Verifier** — deterministic code, an LLM-judge prompt, or a blend. Emits a scalar reward.
3. **Tools / non-public dependencies** — what we must provision vs. what the agent is expected to discover/build itself. Default: let the agent find its own tools; only pre-provision what the *verifier* requires (a GPU, a running cluster, a dataset, a judge model).
4. **Submission & run mechanism** — how the agent hands off artifacts and how we invoke the verifier (write-to-path + harness runs tests; submit JSON; emit `reward.txt`).

**Recommended substrate: the Terminal-Bench / Harbor per-task pattern** — a clean, domain-agnostic RL-env template we can reuse for *all* five domains:

```
task/
  task.yaml / task.toml   # instruction prompt + timeouts + parser
  instruction.md          # the task prompt (TB2)
  docker-compose.yaml     # env (single or multi-container)
  Dockerfile              # env image — NO tests, NO solution baked in
  solution.sh             # oracle that fully solves it (sanity check the task)
  run-tests.sh            # installs test deps, runs verifier
  tests/                  # injected AFTER the agent stops -> agent can't read/edit
```

Key properties we want everywhere: tests **hidden** and injected post-run, fresh isolated container per trial, network controllable, binary-or-scalar reward written to a known path. (repo: `github.com/laude-institute/terminal-bench`, TB2: `terminal-bench-2`, docs `tbench.ai/docs`).

## 3. Grading & anti-reward-hacking — cross-cutting principles

1. **Prefer execution/ground-truth over judgment.** A passing held-out test, a numerically-correct kernel, a restored SLO, an existing cited clause — these are cheap and hard to fake. Use LLM-judges only where the task is irreducibly open-ended, and even then only as a *secondary* signal layered on a deterministic gate.
2. **Hold out and rotate.** Hide tests from the agent; rotate eval instances each outer-loop generation so the optimizer can't overfit a fixed set.
3. **Gate before you grade.** Run a deterministic validity check first (kernel is actually invoked; clause text exists in the doc; fault component is named). If it fails, zero the reward *before* any subjective scoring runs. This kills the cheapest exploits.
4. **Verify the fix, not the symptom.** Re-check the whole system / the specific injected fault / the actual kernel call — not just a green health probe or a loose `allclose`.
5. **Make metrics out-of-band and read-only.** Anything the agent can write to (SLO metrics, timing harness, the test file, the rubric) it will eventually edit. Compute reward from a surface the agent's credentials can't touch.
6. **LLM-judges are exploitable.** "One Token to Fool LLM-as-a-Judge" (arXiv:2507.08794) shows trivial tokens elicit up to ~80% false positives even from frontier judges. Mitigate with: reference-grounded analytic rubrics (score criterion-by-criterion), a stronger judge model, ensemble/majority vote, adversarial second judge, rotated rubric phrasing, verbosity penalties, leak-phrase detection.
7. **Never reward observability/cost metrics directly.** Tokens, latency, Raindrop "stuck-in-loop" signals → use as *constraints/penalties*, never as the primary reward (instantly gameable).
8. **Constrain blast radius.** Scoped credentials/RBAC, no destructive actions, step/time/token caps. Penalize action count so "nuke and restart" loses to a surgical fix.

---

## 4. The five harnesses

### A. GPU Kernels

**Base to fork:** KernelBench (`github.com/ScalingIntelligence/KernelBench`, MIT, arXiv:2502.10517) — 250 PyTorch reference tasks (L1 single-op, L2 fusion, L3 nets, L4 HF). Port anti-hacking checks from **robust-kbench** (`github.com/SakanaAI/robust-kbench`). Reference points: Kevin-32B (multi-turn GRPO on L1/L2, arXiv:2507.11948), CUDA-L1 (documented hacking taxonomy, arXiv:2507.14111), AutoTriton.

**Scope for 20 min:** L1/L2 single-op or simple-fusion tasks, **Triton** (fast JIT vs. slow nvcc), fixed shapes, ~5 correctness trials + modest perf trials, one A100/H100, hard wall-clock cap per compile. (Kevin & AutoTriton trained on exactly this scope — confirmed tractable.) Avoid L3/L4.

**Draft task prompt:**
> You are optimizing a CUDA/Triton kernel. Below is a PyTorch reference module `Model` with method `forward`. Write a drop-in replacement `ModelNew` that produces numerically-equivalent outputs and runs as fast as possible on the target GPU. You may use Triton, CUDA C++, ThunderKittens, or CUTLASS. Your kernel must be actually invoked (no falling back to the reference op). Submit a single Python file defining `ModelNew`. Inputs: `get_inputs()` / `get_init_inputs()`. Reference: `<...>`.

**Verifier (deterministic, code):** correctness = `torch.allclose` vs reference over randomized inputs (verify exact `atol/rtol` in your checked-out `eval.py`; relaxing to 1e-2 is common but is itself a hacking surface). Performance = `torch.cuda.Event` wall-clock after warmup, with L2-cache flush, on the **default stream**; reward = `fast_p` (correct AND speedup > p vs torch eager / `torch.compile`).

**Tools:** all public — Triton, PyTorch `cpp_extension`/nvcc, ThunderKittens (MIT), CUTLASS (BSD-3). Skip ncu/Nsight in the loop (too slow). Pre-provision: a GPU.

**Submission/run:** agent writes `submission.py`; harness imports `ModelNew`, runs correctness then timing.

**Reward hacking (documented, real):** extra CUDA streams to hide work from the timer (fake ~18×); caching/memoization keyed on input address; shrinking batch/dims; precision downgrade to pass loose tolerances; **not calling the kernel / torch fallback** (AutoTriton's model still ~10% hacks on L1). **Defenses:** randomize shapes per trial, tighten tolerances, time on default stream with cache flush, static-assert the kernel is invoked, cap speedup (robust-kbench / CUDA-L1 reward normalization).

---

### B. Multi-Agent System Optimization

This is the hardest verifier. The trick: **don't try to grade "the system is good" directly — pin it to a downstream task with a deterministic grader**, and let the outer optimizer tune the system against that.

**Outer optimizer (the "research loop"):** GEPA (`github.com/gepa-ai/gepa`, reflective Pareto prompt evolution, arXiv:2507.19457, beats GRPO with ~35× fewer rollouts) or AFlow (`github.com/FoundationAgents/AFlow`, MCTS over code workflows). Alternatives: DSPy/MIPROv2 (prompts), ADAS (programs new agents — runs untrusted code, sandbox it), Trace/OptoPrime (optimizes topology + prompts), Archon (inference-time architecture search). Survey: EvoAgentX.

**Inner task + verifier (pick one with a hard grader):**
- **Code porting → TransCoder-test/-ST** (C++/Java/Python, 948 samples, **execution-based** unit tests, CA@1). Strongest verifier; single-function/single-file fits 20 min. (CodeXGLUE exists but BLEU/exact-match is text-gameable — avoid as reward.)
- **Research/QA → HotpotQA** (EM + token-F1, cheap) or a **GAIA** subset (exact-match, use private split — validation may be memorized).

**Draft task prompt (porting variant):**
> Port the following `<source-lang>` function to `<target-lang>`, preserving exact behavior. Output only the translated function in a fenced code block. It will be compiled and run against a hidden unit-test suite; your reward is the fraction of tests that pass.

**Verifier:** primary = held-out unit-test execution (CA@1). Secondary (tiebreaker only) = small rubric LLM-judge. Keep tests held out.

**Tools / Raindrop:** instrument the multi-agent system with **Raindrop** (`raindrop.ai`; OSS **Workshop** `github.com/raindrop-ai/workshop`, MIT, local SQLite `.db`, supports DSPy/LangGraph/CrewAI/Claude Agent SDK). Use it for trajectory inspection and signals (tool-errors, "stuck-in-loop") — as **auxiliary diagnostics and penalty constraints, never the primary reward.**

**20-min scope:** single-function port or 3–5 hop QA. Avoid SWE-bench / repo-level (too long).

**Reward hacking:** optimizing to judge biases (verbosity/format); overfitting a tiny eval set; agents colluding (one rubber-stamps another); gaming token/latency. **Defenses:** execution reward (not text-match); held-out + rotated eval each generation; GEPA Pareto frontier resists single-metric overfit; composite reward with structural penalties + semantic-leak detection; never reward observability metrics.

---

### C. Legal / Loophole-Finding

There is a genuine gap: **no published benchmark squarely targets contract loophole-finding as an agentic task** — your harness can fill it. Build it on top of labeled contract data so part of the reward is deterministic.

**Harness substrate:** **Inspect AI** (UK AISI, `github.com/UKGovernmentBEIS/inspect_ai`) — Dataset→Solver→Scorer, built-in `model_graded_qa` rubric scorers with multi-model majority-vote grading. Best fit.

**Task material:** **CUAD** (510 contracts, 41 clause types, CC BY 4.0) and **ContractNLI** (NDAs, 3-class entailment + evidence spans). Also LegalBench, MAUD, CaseHOLD, LexGLUE for deterministic sub-tasks.

**Draft task prompt:**
> Analyze the attached contract and identify legal loopholes or ambiguities a counterparty could exploit. For each finding output JSON: `{quoted_clause_text, location, loophole_description, severity_1to5, exploitation_scenario}`. You MUST quote the exact clause text verbatim — findings whose quoted text cannot be located in the source document are discarded.

**Verifier (blended, gate-then-grade):**
1. **Deterministic gate:** for each finding, verify the `quoted_clause_text` actually exists in the source doc (string/span match). Fabricated citations → that finding zeroed *before* judging. This is the single highest-leverage guard.
2. **Analytic rubric LLM-judge:** severity/validity scored criterion-by-criterion by a strong judge, plus an **adversarial second judge**; ensemble/majority vote (Inspect-native); rotate rubric phrasing.
3. Optionally anchor part of the reward to deterministic labels (CUAD clause classification, ContractNLI entailment, MAUD/CaseHOLD MC) for a partly-objective signal.

**Tools:** judge model(s); the contract corpus. Agent finds its own legal reasoning approach.

**Submission/run:** agent emits JSON findings; verifier runs clause-existence gate then rubric judges.

**Reward hacking (documented, real):** verbose persuasive nonsense fooling the judge; **fabricated citations** (200+ sanctioned attorney filings by mid-2025; judges have ~16–17% citation-verification recall without tools); judge sycophancy; single-token judge exploits (arXiv:2507.08794). **Defenses:** the groundedness gate above; reference-anchored reward; adversarial/ensemble judges; robustness-hardened reward model (Master-RM); verbosity penalty; require every loophole to map to a locatable clause.

---

### D. Software Infra / SRE / Incident Response

**Base to fork:** **AIOpsLab** (`github.com/microsoft/AIOpsLab`, MIT, arXiv:2501.06706) — detection/localization/RCA/mitigation tasks, Chaos Mesh fault injection, Prometheus+Jaeger telemetry, DeathStarBench apps. Alternative: **ITBench** (IBM, `github.com/itbench-hub/ITBench`). `ITBench-Lite` (HF) has static snapshots if live k8s is too heavy. (Skip "DevOps-Eval"/"SRE-skills-bench" for rollouts — they're static QA, not live envs.)

**Env tooling (all public):** Chaos Mesh (CNCF, k8s-native faults — what AIOpsLab uses); microservice SUT = **Google Online Boutique / hipster-shop** (lightest, ~10 min from pre-built manifests; ~2–5 min warm). Heavier: Sock Shop, Train-Ticket, DeathStarBench.

**20-min scope — critical:** **pre-provision the cluster once, outside the timed rollout** (Kind + Online Boutique + Prometheus). Per-rollout: inject one Chaos Mesh fault → agent gets read-only telemetry → agent **localizes the faulty service** (and optionally applies one remediation) → verify. Single-fault localization is the safe 20-min target; full mitigation is the weakest stage even for frontier models.

**Draft task prompt:**
> A fault has been injected into the running microservice cluster, degrading service. You have read-only access to metrics (Prometheus), traces (Jaeger), and logs. Investigate and submit: (1) the single faulty service/component by exact name, (2) the fault type, (3) a proposed remediation. Do not restart or delete resources unless your remediation step requires it.

**Verifier (deterministic):** localization = submitted component matches injected-fault ground truth; mitigation (if scored) = the specific injected fault is gone (Chaos Mesh experiment object removed / offending config reverted) **AND** whole-system SLO restored via an out-of-band probe (AIOpsLab re-checks global health to catch collateral damage); plus MTTR/steps/tokens as efficiency.

**Tools:** pre-provisioned cluster, telemetry endpoints (read-only), scoped kubeconfig. Agent uses kubectl/queries it discovers.

**Submission/run:** agent submits JSON (component, fault type, remediation); harness compares to ground truth + runs out-of-band SLO probe.

**Reward hacking:** nuke/restart everything to turn health green; mask symptoms vs. fix root cause; read the ground-truth fault spec; silence alerts / edit Prometheus rules. **Defenses:** require the agent to *name* the root-cause component (not just green health); scoped RBAC denying destructive ops + action-count penalty; keep fault spec / Chaos CRDs / `eval()` oracle in a namespace the agent's kubeconfig can't read; recompute SLO out-of-band; whole-system post-check fails rollouts that break other services.

---

### E. Terminal-Bench

**Use directly:** `github.com/laude-institute/terminal-bench` (Stanford + Laude Institute). TB2.0 (Jan 2026, `terminal-bench-2`) = 89 curated hard tasks across SWE/ML/security/data/sysadmin; frontier models <65%. Plus 26 adapted external benchmarks (SWE-Bench Verified, AppWorld, etc.). Site/leaderboard `tbench.ai`.

**Run it:** `pip install terminal-bench` (or `uv tool install`), needs Docker + uv. Single task: `tb run --agent terminus --model <provider/model> --task-id <id> --livestream`. Oracle sanity check: `--agent oracle`. Adapters exist for Claude Code, Codex CLI, Gemini CLI, OpenHands, mini-SWE-agent.

**20-min fit:** comfortable for normal tasks. Runtime governed by `max_agent_timeout_sec` (typical 180–360s) / `max_test_timeout_sec`; set agent timeout ≈ 600–900s + image-pull overhead. Avoid the 24h outliers (MLE-bench).

**Draft task prompt:** task-specific (provided by each task's `instruction.md` / `task.yaml`). To author a new one, fill the template in §2.

**Verifier:** binary pass/fail — `run-tests.sh` runs **pytest** inside the container after the agent stops; ALL tests must pass. TB2 maps exit code → `reward.txt` (1.0/0.0).

**Tools:** Docker. Agent operates via a headless bash terminal (Terminus) over tmux.

**Reward hacking & built-in protections:** tests/solution are **not in the image** — injected after the agent stops, so the agent can't read or edit them; isolated fresh container per trial; quality checks flag hardcoded-answer solutions. **Gaps to close for RL use:** network is *not* hard-disabled by default in TB1 (enforce isolation yourself to stop solution lookups); if your task leaves writable artifacts the tests read, write robust assertions so outputs can't be trivially faked.

---

## 5. Suggested hackathon build order

1. **Terminal-Bench first** — it already is a working harness; gets the end-to-end loop (agent → container → hidden tests → reward) running fastest, and its per-task format becomes the template for everything else.
2. **GPU kernels** — fork KernelBench, restrict to L1/L2 Triton, port robust-kbench checks. Deterministic, well-trodden.
3. **SRE** — fork AIOpsLab, pre-provision Online Boutique on Kind, single-fault localization only.
4. **Legal** — Inspect AI + CUAD, clause-existence gate + rubric judges. Highest-novelty (fills a real gap), but verifier needs the most care.
5. **Multi-agent optimization** — GEPA/AFlow outer loop + TransCoder unit-test inner verifier + Raindrop instrumentation. Most moving parts; do last.

## 6. Key URLs

- KernelBench `github.com/ScalingIntelligence/KernelBench` · robust-kbench `github.com/SakanaAI/robust-kbench` · CUDA-L1 `github.com/deepreinforce-ai/CUDA-L1` · ThunderKittens `github.com/HazyResearch/ThunderKittens` · CUTLASS `github.com/NVIDIA/cutlass`
- GEPA `github.com/gepa-ai/gepa` · AFlow `github.com/FoundationAgents/AFlow` · DSPy `github.com/stanfordnlp/dspy` · ADAS `github.com/ShengranHu/ADAS` · Trace `github.com/microsoft/Trace` · EvoAgentX `github.com/EvoAgentX/EvoAgentX`
- TransCoder (in `github.com/facebookresearch/CodeGen`) · CodeXGLUE `github.com/microsoft/CodeXGLUE`
- Raindrop `raindrop.ai` · Workshop `github.com/raindrop-ai/workshop`
- Inspect AI `github.com/UKGovernmentBEIS/inspect_ai` · CUAD `github.com/TheAtticusProject/cuad` · ContractNLI `stanfordnlp.github.io/contract-nli` · LegalBench `github.com/HazyResearch/legalbench`
- AIOpsLab `github.com/microsoft/AIOpsLab` · ITBench `github.com/itbench-hub/ITBench` · Chaos Mesh `github.com/chaos-mesh/chaos-mesh` · Online Boutique `github.com/GoogleCloudPlatform/microservices-demo`
- Terminal-Bench `github.com/laude-institute/terminal-bench` · TB2 `github.com/laude-institute/terminal-bench-2` · docs `tbench.ai/docs`
- Reward-hacking refs: "One Token to Fool LLM-as-a-Judge" arXiv:2507.08794 · "Rubrics as Rewards" arXiv:2509.15557 · AISI autograder validation `aisi.gov.uk/blog/llm-judges-on-trial`
