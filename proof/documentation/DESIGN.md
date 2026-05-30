# AR² — Autoresearch on Autoresearch

**Working title:** AR² (AR-squared). The thing we optimize is the *second derivative*: not reward, not the rate a research agent gains reward, but **how much better each successive version of the research agent is at gaining reward.**

Status: design for review. Do **not** start coding until this is signed off; then we parcel §10 to coding agents.

---

## 1. Thesis

Karpathy's `autoresearch` is an agent that edits `train.py`, runs a fixed 5-min eval, keeps/discards, and repeats — driving `val_bpb` down over experiments. That's an **inner loop**: reward `R` over iterations `t`, with `dR/dt > 0` when it works.

We climb the axis above it. Treat the **entire autoresearch repo as the artifact**. A meta-process rewrites that repo — rewriting the loop, the harness, and crucially **inventing its own fanout/parallelism** — producing AR v0 → v1 → … → vN. We measure each version by the reward its rollouts reach on **held-out** envs.

- **Inner axis:** `R` vs iteration `t` (one rollout on one env). Owned by Alex's envs + the current AR.
- **Outer axis (what we optimize):** held-out final reward `R*` vs **version** `N`.
- **Second derivative:** we want each version's *inner curve to lift* — `∂²R / ∂N∂t > 0`.

This is the **Darwin Gödel Machine** shape (empirically-validated self-code-modification + evolutionary archive), grounded on `karpathy/autoresearch` as v0 and Alex's verifiable envs as the referees.

---

## 2. The one invariant (read this twice)

> **The referee lives outside the mutable repo, and obs is injected by the harness — neither is reachable by the agent.**

If the agent owns the whole repo, it must **not** own the grader or the telemetry. Otherwise step 1 of reward hacking is *"delete the grader / silence the logs"* — which is literally what DGM did (it removed its own hallucination detector). So:

- `ar/` (mutable) contains the research **policy** and the self-improvement **policy**. The agent rewrites this freely.
- `harness/` (immutable) contains the **outer-loop driver, the selection rule, the sandbox runner, the referee invocation, the hack detector, and obs injection**. The agent never edits this.
- `envs/` (immutable to AR) contains the **Envs / evaluators**. Alex owns these.

The agent can improve *how it improves itself*. It cannot touch *how it is graded or watched.* That gap is the entire experiment: reward hacking is forced into the *interesting* forms (overfit, fabricated internal logs, env-quirk exploitation), all caught by the held-out gap + Raindrop traces.

---

## 3. Contracts (`harness/contracts.py` — single source of truth, immutable)

```python
from pathlib import Path
from typing import Callable, Literal, Protocol, runtime_checkable
from pydantic import BaseModel

Reward = float  # normalized to [0,1] by the Env; 1.0 = perfect. Lets us average across heterogeneous envs.

class Budget(BaseModel):
    wall_seconds: float
    usd: float | None = None
    tokens: int | None = None
    max_concurrency: int = 1          # cap on agent-spawned fanout; ENFORCED by harness, not the agent

class TaskSpec(BaseModel):
    env_id: str
    split: Literal["train", "heldout"]
    prompt: str                        # what to optimize, human-readable
    workdir: Path                      # the editable solution surface (e.g. a train.py). Referee is NOT here.
    payload: dict = {}                 # corpus refs / inputs

class Submission(BaseModel):
    workdir: Path                      # AR's final edited solution surface
    notes: str = ""

class StepResult(BaseModel):
    reward: Reward                     # normalized
    raw: dict = {}                     # raw metric, e.g. {"val_bpb": 0.979}
    feedback: str | None = None        # optional hint back to the inner loop
    done: bool = False

class Rollout(BaseModel):
    env_id: str
    split: Literal["train", "heldout"]
    rewards: list[Reward]              # per inner iteration  ->  dR/dt (the inner curve)
    final_reward: Reward
    cost: Budget
    trace_id: str                      # Raindrop tag
    hack_flags: list[str] = []

class Attempt(BaseModel):              # one node in the evolutionary archive
    version: int
    parent: int | None
    diff_summary: str
    train_reward: Reward
    heldout_reward: Reward             # the outer-curve y
    hack_flags: list[str]
    cost: Budget
    trace_id: str
    source_ref: str                    # git commit of this ar/ snapshot

class Archive(BaseModel):
    attempts: list[Attempt] = []
    def best(self) -> Attempt | None: ...
    def frontier(self) -> list[Attempt]: ...   # ancestors worth branching from (DGM tree)

@runtime_checkable
class Env(Protocol):
    """ALEX owns these. The referee. Evaluator runs OUT-OF-PROCESS and is read-only to AR."""
    id: str
    split: Literal["train", "heldout"]
    def reset(self) -> TaskSpec: ...
    def score(self, sub: Submission) -> StepResult: ...   # deterministic, pure
```

### AR entrypoint contract (the *only* thing `harness/` requires of `ar/`)

```python
# ar/entrypoint.py   — MUTABLE. This file (and everything in ar/) evolves across versions.
def solve(task: TaskSpec, budget: Budget,
          score: Callable[[Submission], StepResult],     # the isolated referee — callable, not editable
          spawn: Callable[..., list]) -> Submission:     # the capped fanout primitive
    """Do research: edit task.workdir to maximize score(). This is Karpathy's inner loop in v0;
       later versions may parallelize it via spawn(), add critics, change the search, etc."""

def improve(archive: Archive, budget: Budget,
            spawn: Callable[..., list]) -> Path:
    """Emit the SUCCESSOR ar/ folder (a new dir). v0 = invoke a coding agent over (this repo + archive).
       Later versions may rewrite this entirely — including how they fan out the search for improvements.
       Returns the path to the new ar/ snapshot."""
```

`solve` and `improve` both live in `ar/`, so both evolve. The harness only ever calls these two functions and `Env.score` — that's the whole surface.

### Harness functions (`harness/`, immutable)

```python
def score_repo(ar_dir: Path, envs: list[Env], budget: Budget) -> list[Rollout]:
    # For each env: fresh Modal sandbox (gVisor) -> inject obs (proxy base_url + Raindrop tags)
    # -> import ar.entrypoint.solve -> run with score=isolated-referee, spawn=capped-primitive
    # -> record inner curve from score() calls. Crash / timeout / cap-breach => reward 0.0.

def evaluate(ar_dir: Path, train: list[Env], heldout: list[Env], budget: Budget) -> Attempt:
    tr = score_repo(ar_dir, train,   budget)     # signal for improve() + the train reward
    ho = score_repo(ar_dir, heldout, budget)     # SELECTION + generalization (the outer y)
    a = Attempt(train_reward=mean(r.final_reward for r in tr),
                heldout_reward=mean(r.final_reward for r in ho), ...)
    a.hack_flags = detect_hacks(tr, ho)          # train≫heldout gap + Raindrop trace signals
    return a

def drive(ar0: Path, train, heldout, budget, K: int, M: int) -> Archive:   # the FIXED outer loop
    archive = Archive(); archive.add(evaluate(ar0, train, heldout, budget))
    for _ in range(K):
        parents = archive.frontier()
        # fan out M candidate successors in parallel on Modal:
        cands = parallel_map(lambda p: load(p.source_ref).improve(archive, budget, spawn),
                             parents, width=M)
        for c in cands:
            archive.add(evaluate(c, train, heldout, budget))
        record_curve(archive)        # outer point = best heldout_reward so far
    return archive
```

**Mutable/immutable line, precisely:** `solve` + `improve` (in `ar/`) evolve. `drive`, `evaluate`, `score_repo`, the selection rule, `detect_hacks`, the referee, and obs injection (in `harness/`) are fixed. Envs (in `envs/`) are fixed and external to `ar/`.

---

## 4. The loops

**Inner loop** (inside `ar/solve`, evolvable — v0 = Karpathy's): `edit workdir → score() → keep/discard → repeat until budget`. Produces the inner curve + best Submission. Later versions may fan this out (M parallel experiments via `spawn`, a critic agent, smarter search).

**Outer loop** (`harness/drive`, fixed): `evaluate(v0) → for each generation: improve() the frontier in parallel → evaluate candidates on train+held-out → archive → record curve`. Selection is on **held-out**. Output = the evolutionary archive + the **outer curve** (best held-out reward vs version), two-colored by hack flags.

---

## 5. The agents

| # | Agent | Kind | Runs in | Consumes | Produces |
|---|---|---|---|---|---|
| 1 | **Infra / Setup agent** | LLM coding agent, one-shot | local + Modal build | repo, dep list | base Modal **image** (Raindrop + telemetry proxy + AR deps) + sandbox image for candidate repos. Builds `infra/images.py`. |
| 2 | **Meta-research agent** (= `ar/improve` in v0) | LLM coding agent (Claude Code / Codex headless) | sandbox | the `ar/` repo + `Archive` (past diffs, rewards, hack flags) | a successor `ar/` folder (rewrites code, **sets up fanout**) |
| 3 | **Inner research agent(s)** (= `ar/solve`) | LLM agent(s), **count/topology evolves** | sandbox | a `TaskSpec` + `score` + `spawn` | a `Submission` (edited solution surface) |
| 4 | **Referee** | deterministic, **NOT an LLM** | isolated process | a `Submission` | a `StepResult` (the ground-truth reward) |
| 5 | **Hack detector** | deterministic + optional LLM summary | local | Raindrop `.db` + train/held-out gap | `hack_flags` + the two-colored curve annotations |
| 6 | **Orchestrator** | plain Python, **not an LLM** | local | envs, budget, archive | drives `drive()`, fans scoring out to Modal |

Notes:
- **Agents 2 and 3 are the only things that evolve.** Their *behavior* is whatever the current `ar/` defines. v0: agent 3 = one Karpathy-style loop; agent 2 = a single coding-agent edit. The interesting result is the meta-agent *adding agents* (fanout) to agent 3 over versions.
- **Agents 4, 5, 6 are deterministic and immutable.** No LLM judges anywhere in the reward path — integrity.
- Agent 1 is David's example ("one agent sets up the base container with Raindrop"): it owns reproducible images so every other agent runs identically.

---

## 6. Observability (baked in, two layers)

**Inner obs — Raindrop Workshop (`harness/telemetry.py`).** Every LLM/tool call by agents 2 & 3 is captured. We do **not** rely on each agent self-instrumenting — the harness injects, at the sandbox boundary, `OPENAI_BASE_URL` / `ANTHROPIC_BASE_URL` pointing at a **logging proxy**, plus `config_id` tags `{version, candidate, env_id, split}`. The proxy mirrors spans to Workshop's local `.db` (`~/.raindrop/raindrop_workshop.db`, UI :5899) and records tokens/latency; we compute cost = tokens × price. Because injection is at the sandbox env level, **AR can't disable its own telemetry** (another integrity edge).

**Outer obs — the meta dashboard (`obs/`).** The `Archive` (every `{version, parent, diff, train_reward, heldout_reward, hack_flags, cost, trace_id}`) plus: the **outer curve** (held-out reward vs version, two-colored), the **lineage tree** (DGM evolutionary tree), and cost-per-version. This is *our* science view.

**The bridge — hack detector.** Reads Raindrop spans (SQL over tool-call data: writes to evaluator/test paths, fabricated "passing" logs, suspiciously short trajectories, env-quirk exploits) **and** the `train_reward − heldout_reward` gap. A version whose train reward spikes but held-out doesn't = overfit/hack → red on the curve.

## 6.1 Telemetry collection & DB sync — the distributed-obs gap

**This is the piece that was missing.** Workshop's `.db` is **local** (`~/.raindrop/raindrop_workshop.db`, :5899), but telemetry is produced across **N remote Modal sandboxes** — it has no path home, and a single shared SQLite across many sandbox writers would lock. We bridge with a canonical store + an explicit sync, in two modes:

- **Pull (default, robust).** The harness injects an in-sandbox span shim as the agents' `base_url` (localhost *inside* the sandbox) — or points Codex/Claude OTEL at an in-sandbox collector — writing spans to `/work/trace.jsonl`. When `score_repo` finishes a rollout it reads `trace.jsonl` out via `sb.filesystem.read_text` alongside the `Submission`. **Telemetry rides home with the result — no call-home networking, no shared-writer DB locking.**
- **Push (optional, live demo).** A single `infra/collector.py` Modal endpoint all sandboxes stream to, for a real-time dashboard during long runs.

**`harness/db_sync.py`** makes the sync explicit: it merges per-run `trace.jsonl` artifacts into a **canonical SQLite** (`obs/traces.db`), computes cost (`tokens × price`), and **mirrors into the local Workshop `.db`** so the :5899 UI shows the entire distributed run. It runs incrementally each generation + once at the end; the hack detector and `obs/` query the canonical store. Every span carries `{version, candidate, env_id, split, trace_id}` so distributed traces stay separable after the merge.

---

## 7. Reward-hacking gallery (the demo's spine)

Expected, and what flags each:

| Hack | Detector |
|---|---|
| Overfit `improve` to train-env quirks | train ≫ held-out gap |
| Fabricate internal "it worked" logs | Raindrop: claimed success with no corresponding `score()` call |
| Exploit an env-specific shortcut | held-out (different env) fails to reproduce the gain |
| "Set up fanout" = request unbounded compute to win a cost metric | harness `max_concurrency` cap → cap-breach = reward 0 |
| Try to reach the grader | impossible by construction (referee external) — attempts show as sandbox FS errors in trace |

The headline slide = the **two-colored outer curve**: genuine slope (held-out rises) vs hacked slope (train-only), with a trace drill-down for each red point.

---

## 8. Repo layout (the deliverable)

```
autoresearch-meta/                 # our deliverable repo
├── ar/                            # ★ MUTABLE — the artifact that evolves (v0 = adapted karpathy/autoresearch)
│   ├── entrypoint.py              #   solve() + improve()  (the only harness-facing contract)
│   ├── program.md                 #   still here; now the WHOLE folder is fair game
│   └── ...                        #   inner-loop code, search, (later) fanout
├── harness/                       # ☒ IMMUTABLE — ours; the agent never edits this
│   ├── contracts.py               #   §3 — single source of truth
│   ├── score_repo.py              #   run an ar/ as a black box on envs, in a sandbox
│   ├── outer_loop.py              #   drive() + evaluate() + selection rule
│   ├── referee.py                 #   isolated out-of-process Env.score runner
│   ├── sandbox.py                 #   Modal sandbox wrappers (gVisor, caps, spawn primitive)
│   ├── telemetry.py               #   Raindrop proxy + tag injection + cost
│   ├── hack_detector.py           #   spans SQL + train/heldout gap -> hack_flags
│   └── archive.py                 #   evolutionary tree store (sqlite/jsonl)
├── envs/                          # ☒ IMMUTABLE to AR — Alex
│   ├── base.py                    #   Env protocol + nanochat reference adapter
│   ├── legal/  ...                #   Alex's legal env
│   └── gpu/    ...                #   Alex's GPU env
├── infra/                         # Agent #1 output — Modal images (base + sandbox), proxy
├── obs/                           # outer dashboard — curve + lineage tree
├── vendor/autoresearch/           # pristine karpathy v0 reference (read-only)
├── proof/documentation/           # design docs, decisions log, architecture plans
└── proof/documentation/DESIGN.md  # this file
```

---

## 9. Feasibility & scope for the 10-hour demo

The second derivative multiplies cost: `versions × candidates/gen × held-out envs × experiments/rollout × seconds/experiment`. To fit:

1. **Cheap envs sample the version axis densely** — legal / pure-LLM envs (seconds, CPU) give the curve + the hacks fast.
2. **nanochat-GPU is the flagship** — slashed inner budget (5min → ~30–60s, TinyStories tiny model; we want *relative* gains, not SOTA), proving it generalizes to "real research." This is where the Modal GPU credits + the work-account scoping matter.
3. **Parallelize candidate evaluation across Modal**; cap everything via `Budget`.
4. **Target:** a v0→~v3 outer curve with visible positive slope, two-colored, with **≥1 caught reward hack** + a trace drill-down. *Not* "we achieved RSI" — "the loop runs; here's the slope; here's exactly where it breaks."

---

## 10. Build plan (fan-out to coding agents — only after sign-off)

WP0 must land first; the rest parallelize against `contracts.py`.

| WP | Owner | Deliverable | Depends on |
|---|---|---|---|
| **WP0** | — | `harness/contracts.py` (§3) | — |
| WP1 | Alex | `envs/base.py` Env protocol + **nanochat adapter** as reference Env | WP0 |
| WP2 | agent | `harness/sandbox.py` — Modal sandbox + `spawn` primitive + caps | WP0 |
| WP3 | agent | `harness/score_repo.py` + `harness/referee.py` (black-box runner + isolated grader) | WP0,2 |
| WP4 | agent | `ar/entrypoint.py` v0 — port Karpathy loop to `solve`; seed `improve` (coding-agent edit) | WP0 |
| WP5 | agent | `harness/telemetry.py` — in-sandbox span shim + base_url/OTEL injection | WP0,2 |
| **WP5b** | agent | `harness/db_sync.py` — ★ merge per-run `trace.jsonl` → canonical SQLite + mirror to local Workshop | WP0,5 |
| WP6 | agent | `harness/hack_detector.py` — spans SQL + gap | WP0,5b |
| WP7 | agent | `harness/outer_loop.py` + `archive.py` — `drive`/`evaluate`/selection | WP0,3 |
| WP8 | Agent #1 | `infra/` — base + sandbox Modal images (Raindrop preinstalled) + `collector.py` (push obs) | WP0,5 |
| WP9 | agent | `obs/` — outer curve + lineage tree dashboard | WP0,7 |

Critical path: WP0 → WP3/WP7 → end-to-end on cheap env → add GPU flagship → obs polish.

---

## 11. Open decisions for review

1. **Reward normalization** — confirm Envs return `[0,1]` (relative-to-baseline) so held-out averaging across heterogeneous envs is meaningful.
2. **`improve` self-reference depth** — v0 has `improve` call a fixed coding agent. Confirm we *allow* later versions to rewrite `improve` itself (true second-order), with referee+driver fixed.
3. **Held-out hygiene** — the meta-agent sees held-out *scalar reward* for selection but **not** held-out env *contents* (else it overfits). Confirm.
4. **Env pool for the demo** — which cheap env(s) does Alex build first (legal?) + is nanochat-GPU the flagship?
5. **Mutation model** — Claude Code vs Codex for agent #2/#3 (we can run both and compare — itself a nice axis).
