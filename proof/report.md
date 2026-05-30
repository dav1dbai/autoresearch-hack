# Autoresearch Environments Report

## Goal

We want an outer-loop autoresearch system that improves a task-performing agent or system against a fixed environment:

- a task prompt or task instance
- a verifier, ideally deterministic and auditable
- a submission interface
- an execution budget of roughly 20 minutes per rollout

The project focus is the environment and outer loop, not over-specifying the inner loop. The key question is whether agents can realistically hill-climb performance by improving prompts, scaffolds, tool use, retrieval, and lightweight helper tooling without turning the benchmark into harness hacking.

## Core framing

The right abstraction is closer to an RL environment than a free-form agent demo:

- **Observation**: task prompt, local files, tools, context, optional retrieval surface
- **Action**: edit code/config/prompt, call tools, produce an artifact
- **Transition**: run task or evaluator
- **Reward**: verifier score, possibly multi-part
- **Episode budget**: fixed wall-clock, token, and execution limits

For this project, the environment should stay fixed while the outer loop is allowed to improve the agent stack:

- prompts
- routing and planning logic
- memory and trace use
- retrieval strategy
- helper tools
- local harness glue around the benchmark

That separation is what makes hill-climbing meaningful.

## Harness requirements

Each domain needs four concrete pieces:

1. **Task prompt**
   - what the agent is trying to do
   - hard constraints
   - allowed tools and files

2. **Verifier**
   - deterministic when possible
   - if model-based, constrained by structure and backed by deterministic checks
   - hidden evals when possible

3. **Tool surface**
   - local tools, MCP servers, retrieval endpoints, or domain-specific helpers
   - explicitly allowed or disallowed outside research

4. **Submission interface**
   - artifact format or final filesystem state
   - reward output format
   - logs/traces for analysis

## What the outer loop should optimize

The outer loop should not just optimize raw final score. That will overfit quickly on low-headroom tasks. It should optimize:

- task score on held-out tasks
- score improvement per rollout
- wall-clock efficiency
- token efficiency
- transfer to held-out tasks
- robustness across seeds / hidden instances
- exploit resistance

This matters especially for GPU kernels and other deterministic domains where absolute headroom may be limited.

## Reward hacking principles

Assume every environment will be hacked if the agent is capable enough.

Use these guardrails everywhere:

- keep verifier code immutable and outside the writable workspace
- hide eval cases or use held-out instances
- separate public train tasks from private eval tasks
- prohibit writes to tests and benchmark internals
- log full traces
- score generalization, not only seen-instance performance
- add penalties for excessive retries, eval count, or wall-clock
- use canary tasks or adversarial audit tasks

For judge-based domains:

- never rely on a single judge model
- validate structure and citations before any model grading
- use rebuttal or adversarial second-pass judging
- separate factual validity from severity or quality scoring

## Environment summary

### Strongest near-term environments

These are the three strongest environments for a hackathon:

1. **TerminalBench / Harbor**
2. **SRE / RCA**, especially RCAEval-style static diagnosis
3. **GPU kernels** on H100 using AutoKernel and/or KernelBench

### Secondary environments

These are viable but higher-risk or less mature for this project:

- multi-agent system optimization
- legal / loophole / misc domains

## 1. GPU kernels

### Recommendation

Use this only if an H100 is already available. If yes, it is a strong deterministic environment. If not, it is too operationally risky for a hackathon pillar.

### Existing harnesses

#### AutoKernel

AutoKernel already provides most of the local harness shape:

- `kernel.py` as the editable target
- `bench.py` as fixed benchmark plus correctness harness
- `reference.py` for the ground truth implementation
- `prepare.py` for one-time setup
- profile / extract / orchestrate utilities

This is very close to the desired autoresearch loop.

#### KernelBench

KernelBench provides:

- a benchmark corpus
- correctness and timing evaluation
- single-sample checking
- full-suite evaluation scripts
- hardware-specific baseline generation

KernelBench is a better benchmark substrate; AutoKernel is a better local edit/eval workflow.

### Headroom

This domain has real but uneven headroom.

Good cases:

- weak but correct starter kernels
- new fused kernels
- kernels with obvious memory or tiling problems
- hidden shapes that differ from the development distribution

Bad cases:

- already optimized kernels near hardware limits
- tasks where Triton is already close to cuBLAS/CUTLASS performance
- benchmarks with weak baselines already eliminated

Conclusion:

- the environment is useful
- but raw throughput is not always the right reward
- use normalized headroom closed, hidden-shape performance, and efficiency of search

### H100 feasibility

This is feasible on one H100 if tightly scoped:

- one GPU
- one kernel per task
- Triton-first
- precomputed baselines
- rollout budget of roughly 8-10 evals

Do not run profile/extract/model-level verification inside every rollout. Precompute tasks ahead of time.

### Infra requirements

- NVIDIA H100 machine
- CUDA and Triton working cleanly
- stable benchmarking setup
- pinned software environment
- GPU isolation or at least low contention

This does not run easily on commodity CPU-only infra.

### Outside research

Allowed outside research is possible but should be narrow:

- Triton docs
- CUDA tuning references
- curated kernel optimization notes

Open web access is not central here and is more likely to create noise than value.

### Off-the-shelf harness compatibility

You can wrap this with Codex, Claude Code, or another agent harness, but the benchmark itself does not natively care. Most gains come from:

- better proposal strategy
- better experiment prioritization
- better memory of what failed

This is not the best domain if the main story is "the outer loop improves agent tooling."

### Suggested harness

#### Prompt

Edit only `kernel.py`. Maximize performance on hidden evaluation shapes while preserving correctness against the reference implementation. Do not modify benchmarks, tests, or harness files.

#### Verifier

1. import / compile check
2. randomized correctness sweep
3. hidden-shape correctness sweep
4. timing on public shapes
5. timing on hidden shapes
6. anti-cheat checks

#### Tool surface

- local code editing
- local benchmark runner
- optional curated optimization docs

#### Submission

- modified `kernel.py`
- optional `notes.json` or trace bundle

### Main risks

- low headroom on hardened tasks
- noisy timing if GPU contention exists
- reward hacking against benchmark code
- long compile/eval loops if tasks are too heavy

### Bottom line

Use GPU kernels as a deterministic flagship only if the H100 setup is already solid. Scope tightly and avoid making raw speedup the only reward.

## 2. TerminalBench / Harbor

### Recommendation

This is the best all-around hackathon environment.

Why:

- high headroom
- cheap infra
- rich space for harness improvements
- compatible with off-the-shelf coding agents
- natural place to allow helper tools and outside research

### Existing harnesses

TerminalBench already provides task structure:

- English task prompt
- sandboxed environment
- hidden verifier/test script
- task packaging

Harbor provides the runtime harness:

- agent integrations
- task/runtime packaging
- private task sets
- custom tools and MCP servers
- resource control
- support for optimization workflows

### Headroom

This domain has strong headroom because:

- frontier agents still fail a meaningful fraction of tasks
- gains can come from planning, recovery, tool use, and environment understanding
- it is not physically bounded like GPU speed

This is also the best place for outer-loop improvements to show up clearly.

### Infra requirements

- Docker
- CPU machines are sufficient for many tasks
- no special GPU needed unless the task itself requires it
- reasonable disk for images and task data

This is the easiest of the three main environments to run on general infra.

### Outside research

This environment supports outside research naturally. Recommended policy:

- allow outside research only on tasks where it is meaningful
- prefer curated retrieval or pinned sources
- keep task-specific hidden answers off-limits

This is the domain where adding retrieval, docs search, and helper tools is most legitimate.

### Off-the-shelf harness compatibility

Very strong.

Harbor is the best place to run:

- Codex CLI
- Claude Code
- other installed terminal agents
- custom wrappers
- possibly orchestrators like Cook if wrapped as an installed agent

This makes it ideal for a benchmark where outer-loop gains come from harness improvements.

### Suggested harness

#### Prompt

Complete the task in the terminal sandbox within the time budget. You may modify working files but may not modify the verifier, hidden tests, or protected benchmark files.

#### Verifier

- existing hidden task verifier
- optionally extended with trace-based anti-cheat checks
- score is pass/fail or weighted pass score

#### Tool surface

- shell
- filesystem
- language runtimes needed by the task
- optional retrieval tool
- optional MCP tools
- optional Raindrop instrumentation

#### Submission

- final filesystem state
- command transcript
- optional structured result file

### Main risks

- reward hacking if tasks leak too much
- public benchmark contamination
- long-tail tasks that exceed the time budget

### Best practice

Do not use the whole public suite as-is. Create a private short-suite:

- 10-20 tasks
- each bounded to roughly 3-10 minute agent timeouts
- hidden tests
- tasks chosen to reward scaffolding improvements

### Bottom line

If the goal is to show that an outer loop can improve an agent via prompts, scaffolds, tools, and retrieval, this should be the main environment.

## 3. SRE / RCA

### Recommendation

This is the safest deterministic domain after TerminalBench and probably the best second environment for outer-loop work. Start with static RCA, not live remediation.

### Existing harnesses

#### RCAEval

RCAEval already provides:

- datasets
- case format
- reproducibility scripts
- baselines
- support for adding new methods

This is the most hackathon-friendly starting point.

#### OpenRCA

OpenRCA is more agent-shaped and realistic, but heavier operationally. It is better as a second-stage target if time permits.

### Headroom

There is meaningful headroom here because:

- diagnosis quality is still far from solved
- better retrieval and evidence ranking matter
- decomposition across logs, traces, and metrics can improve results

This is a good match for hill-climbing agent behavior through better tools.

### Infra requirements

#### RCAEval

- Python environment
- dataset download and storage
- CPU is usually enough for evaluator runs
- no live production stack required

This is very manageable.

#### OpenRCA

- heavier storage and memory requirements
- more operational complexity

This is likely too heavy as the first SRE harness.

### Outside research

Allowed, but carefully.

Good sources:

- architecture docs
- runbooks
- service dependency notes
- curated papers or observability docs

Bad sources:

- anything leaking benchmark labels
- benchmark-specific writeups
- public solutions tied to exact incidents

This domain needs provenance rules if outside retrieval is allowed.

### Off-the-shelf harness compatibility

Yes, but usually via a custom wrapper.

You can run Codex, Claude Code, or another agent over the task bundle, but unlike Harbor there is less turnkey integration. Still, tool-building is a legitimate path to improvement here:

- better trace summarization
- better suspect ranking
- log clustering helpers
- topology-aware reasoning tools

### Suggested harness

#### Prompt

Given this incident bundle, return a ranked list of likely root causes with evidence references and one safe next diagnostic action. Do not modify the environment.

#### Verifier

- exact or top-k root-cause scoring
- evidence reference validation
- optional explanation quality score

Keep remediation out of v0 unless a safe deterministic verifier exists.

#### Tool surface

- read-only access to logs, traces, metrics
- optional topology helper
- optional summarization or clustering tools
- optional pinned runbooks/docs

#### Submission

Structured JSON:

- ranked suspected root causes
- evidence references
- confidence
- optional explanation

### Main risks

- label leakage through public incident descriptions
- overfitting to datasets
- too much reliance on opaque judge models

### Bottom line

This is a strong environment for outer-loop improvements via retrieval, ranking, and helper-tool construction. Start with static RCAEval-style diagnosis.

## 4. Multi-agent system optimization

### Recommendation

This is conceptually important but hard to verify cleanly. It should probably be expressed through another benchmark rather than as an open-ended benchmark of "improve the agent system."

### Good substrate choices

- TerminalBench or Harbor private task suites
- code migration or repo tasks with hidden tests
- structured research benchmarks with fixed evaluation

### Why it is hard

- the benchmark can drift if you keep changing both agent and environment
- verification becomes unclear if the task is open-ended
- judge-only grading invites reward hacking

### When it works

It works when you optimize the agent scaffold against a fixed downstream task family:

- code porting
- bug fixing
- short research tasks with structured answers
- tool-use tasks

### Suggested harness

#### Prompt

You may edit the agent scaffold, tool policy, or prompt stack only. Maximize held-out task success rate under fixed time and token budgets.

#### Verifier

- downstream benchmark verifier
- hidden task set
- score aggregation over multiple tasks

#### Tool surface

- local code editing
- trace logs
- optional retrieval
- optional instrumentation like Raindrop

#### Submission

- new agent config or scaffold
- traces
- downstream benchmark results

### Bottom line

This is better treated as a meta-environment built on top of TerminalBench or another fixed task suite.

## 5. Legal / loopholes / misc domains

### Recommendation

This is viable only if narrowed aggressively. It is not a good first environment for a hackathon unless the task is highly structured.

### What works

- deontic reasoning tasks
- rule-grounded policy exploits
- structured fact patterns
- citation-backed answers

### What does not work well

- open-ended legal research
- free-form loophole discovery with a single LLM judge
- tasks with unclear ground truth

### Suggested harness

#### Prompt

Given this frozen ruleset and fact pattern, propose a strategy that remains textually within the rules while exploiting ambiguities or procedure. Cite all supporting rules.

#### Verifier

- rule citation validity check
- consistency check against fact pattern
- optional second-stage severity grading
- adversarial rebuttal model

### Main issue

The verifier is fragile unless much of the task is symbolic or rule-grounded.

### Bottom line

Interesting domain, but lower confidence for hackathon execution.

## Existing harnesses vs what still needs to be built

Most of the heavy lifting already exists in the benchmark repos.

### Reuse mostly unchanged

#### GPU

- AutoKernel local kernel harness
- KernelBench eval and timing scripts

#### Terminal

- Harbor runtime harness
- TerminalBench task format and verifiers

#### SRE

- RCAEval datasets and evaluation

### Thin adapter layer still needed

- common task manifest format
- common reward JSON output
- rollout timeout and resource policy
- hidden eval split management
- standard trace/log export
- Raindrop instrumentation
- optional retrieval service policy
- adapters for Codex / Claude Code / other agents

## Outside research policy

Outside research can be a legitimate source of improvement, but only if defined explicitly.

Recommended policy by environment:

- **GPU kernels**: curated only
- **TerminalBench**: curated or open depending on task
- **SRE/RCA**: curated only
- **legal**: curated only

Best pattern:

- a sidecar retrieval service or MCP server
- pinned paper/doc snapshots
- full provenance logging

This makes research a benchmarked capability rather than uncontrolled leakage.

## Off-the-shelf agents and harnesses

### Harbor / TerminalBench

Best compatibility.

These are the easiest environments for:

- Codex
- Claude Code
- other CLI coding agents
- custom wrappers
- harness experimentation

### SRE / RCA

Usable, but generally via custom wrappers rather than first-class integrations.

### GPU

Usable, but the benchmark mostly rewards artifact optimization, not broad harness sophistication.

## Recommended project plan

### Primary environment

**TerminalBench private short-suite**

Use this to demonstrate that the outer loop can improve agent performance through:

- better prompts
- better scaffolds
- better tool policy
- better retrieval
- helper-tool construction

### Secondary environment

**RCAEval-based static diagnosis**

Use this to show:

- structured deterministic verification
- benefits from retrieval and tool-building
- cleaner reward than free-form research tasks

### Deterministic flagship

**H100 GPU kernel environment**

Use this only if the GPU infra is already solid. Scope tightly and use it to demonstrate:

- deterministic reward
- low-level optimization
- normalized headroom metrics

### Deprioritize for hackathon

- fully open-ended multi-agent self-improvement without a fixed downstream benchmark
- free-form legal loophole search

## Final judgment

There is a realistic path for agents to hill-climb performance in these environments, but only if the environment stays fixed and the outer loop improves the agent stack rather than the benchmark itself.

Best fit for that thesis:

1. TerminalBench / Harbor
2. RCAEval-style SRE
3. H100 GPU kernels

Weakest fit:

- open-ended legal tasks
- unconstrained "improve the agent system" without a downstream verifier

If the goal is to ship something strong in a hackathon, the best strategy is:

- make TerminalBench the main environment
- make RCA the structured second environment
- add GPU kernels as a deterministic third environment if the H100 setup is ready

## Suggested next implementation docs

The next documents worth writing are:

1. a common environment manifest schema
2. a common reward output schema
3. a Harbor task spec for the TerminalBench short-suite
4. an RCA task JSON schema
5. an H100 kernel task directory template
6. an outside-research policy and provenance spec
