# AR² — Decisions (SSOT)

**This file is the single source of truth for what to build, in what order, and how to verify it.**

| Doc | Role |
|-----|------|
| **`DECISIONS.md`** (this file) | Locked decisions + actionable work queue |
| `DESIGN.md` | Architecture narrative & agent roster (reference only; if conflict, this file wins) |
| `proof/` | All project docs (flat); see `proof/README.md` |

**How to use:** pick the next **BLOCKING** or **TODO** item, implement the **Action**, run **Acceptance**. Do not re-litigate items marked **DECIDED** unless you have new evidence.

**Status legend:** `BLOCKING` · `TODO` · `IN_PROGRESS` · `DONE` · `DEFER` (demo-only gap)

---

## Work queue (do in order)

| # | ID | Status | One-line action |
|---|-----|--------|-----------------|
| 0 | [D-00](#d-00-done-semantics-kill-the-inner-loop) | **DONE** | Fix `StepResult.done` + env returns so `solve()` runs ≥1 agent edit |
| 1 | [D-01](#d-01-nanochat-held-out--real-distribution-shift) | **TODO** | Different data seed/shard for nanochat `heldout` split |
| 2 | [D-02](#d-02-parent-sampling-not-greedy-frontier) | **DONE** | `Archive.sample_parents()` wired in `drive()` |
| 3 | [D-03](#d-03-persist-inner-curves-on-attempt) | **DONE** | `Attempt` stores rollouts; `evaluate()` stops dropping curves |
| 4 | [D-04](#d-04-metrics--dashboard-second-derivative-ui) | **PARTIAL** | `obs/metrics.py` + ΔS table in dashboard; layered hero TBD |
| 5 | [D-05](#d-05-wire-telemetry-by-default) | **PARTIAL** | `score_repo` defaults to `inject_for_rollout` |
| 6 | [D-06](#d-06-nanochat-val_bpb-not-from-agent-stdout) | **PARTIAL** | `envs/nanochat_eval.py` harness subprocess; stub path unchanged |
| 7 | [D-07](#d-07-improve-smoke-check-before-eval) | **DONE** | `smoke_check_version_snapshot()` in `drive()` before eval |
| 8 | [D-08](#d-08-eval-protocol--seeds) | **TODO** | `EvalProtocol` (K=8, seeds≥3) + seed loop in evaluate |
| 9 | [D-09](#d-09-outer-loop-cli) | **DONE** | `python -m harness` driver wiring env pools + persistence |
| 10 | [D-10](#d-10-dedupe-rollout-execution) | **PARTIAL** | `harness/rollout.py` + `harness/evaluate.py` landed; `ar_loader.py` extracted |
| 11 | [D-11](#d-11-modal-rollout-path) | **IN_PROGRESS** | Live Modal matmul rollout green (parallel agent) |
| 12 | [D-12](#d-12-env-spec-serialization) | **DEFER** | Formal `EnvSpec` on `BaseEnv` after Modal path stable |
| 13 | [D-13](#d-13-import-dag-mechanical-refactor) | **DONE** | Mechanical import-layer cleanup per `ARCHITECTURE_DAG.md` steps 1→6 |
| 14 | [D-14](#d-14-test-colocation) | **DONE** | Colocate unit tests with packages per `ARCHITECTURE_DAG.md` Part B T1→T6 |
| 15 | [D-15](#d-15-mutation-boundaries-version-snapshots) | **PARTIAL** | Snapshot + `AR2_REPO_ROOT`; D-15b snapshot runtime reload in `dynamic.py` |

---

## Decisions

### D-00: `done` semantics kill the inner loop

**Status:** DONE
**Landed:** Documented semantic on `StepResult.done` (`harness/contracts.py`); `done=False` in `envs/nanochat.py`, `envs/matmul.py` (both returns). Tests: `test_ar_entrypoint::test_env_done_false_runs_iterations` (unit), `test_integration::test_score_repo_inner_loop_accumulates_rewards` (rollout `len(rewards) >= 2`), corrected stale `test_envs` assertion. Full suite 149 passed.  
**Decision:** `StepResult.done` means **stop inner search** (target reached or terminal failure), **not** “this score call finished.” Envs with no natural terminal return `done=False`.

**Why:** `NanoChatEnv` and `MatmulEnv` return `done=True` on every score (`envs/nanochat.py:105`, `envs/matmul.py:238`). `solve()` breaks immediately after baseline (`ar/entrypoint.py:96-97`). **Zero agent edits ever run.**

**Action:**
1. Document semantic on `StepResult.done` in `harness/contracts.py`.
2. Set `done=False` in `envs/nanochat.py`, `envs/matmul.py` (unless truly terminal).
3. Add integration test: baseline + at least one agent iteration when env returns `done=False`.

**Acceptance:** `score_repo` rollout has `len(rewards) >= 2` on stub inner agent; `test_ar_entrypoint` covers env `done=False` path.

---

### D-01: Nanochat held-out ≠ real distribution shift

**Status:** TODO  
**Decision:** Train and held-out must differ by **data**, not label. Matmul already shifts by matrix shape; nanochat must too.

**Action:** Add `data_seed` (or `dataset_shard`) to `NanoChatEnv.__init__`; wire train vs held-out pools with different values in `envs/pools.py` (or future CLI).

**Acceptance:** Same `train.py`, different held-out seed → measurably different baseline reward; `_flag_overfit` can fire on synthetic train≫heldout gap in test.

---

### D-02: Parent sampling, not greedy frontier

**Status:** DONE
**Decision:** Replace top-k `Archive.frontier()` with weighted **`sample_parents(m)`** (D-02): sample non-hacked attempts with weight `∝ heldout_reward × 1/(1 + num_children)`. Keep unconditional `Archive.add`.

**Landed:** `Archive.sample_parents()` in `harness/contracts.py`; `drive()` uses `sample_parents(AR2_PARENT_K)`.

**Acceptance:** Unit test: regression with low held-out but few children has nonzero selection probability; greedy top-k alone would never pick it.

---

### D-03: Persist inner curves on Attempt

**Status:** DONE
**Decision:** Store full rollouts on each evaluation. Scalars (`train_reward`, `heldout_reward`) remain for selection; curves power S(N) / layered dashboard.

**Action:**
```python
# harness/contracts.py — Attempt
train_rollouts: list[Rollout] = Field(default_factory=list)
heldout_rollouts: list[Rollout] = Field(default_factory=list)
```
Add `harness/evaluate.py::attempt_from_rollouts(...)`; wire from `outer_loop.evaluate()`. Optional mirror: `obs/rollouts/v{N}_s{s}.jsonl`.

**Acceptance:** `archive.jsonl` round-trips with `train_rollouts[0].rewards`; dashboard/metrics can read without re-scoring.

---

### D-04: Metrics + dashboard (second derivative UI)

**Status:** TODO  
**Decision:** Hero = **layered inner curves** (qualitative). Headline stat = **ΔS(N)** (quantitative). See [Dashboard & metrics](#dashboard--metrics-decided). Selection stays on final held-out R*; ΔS is the scientific claim.

**Action:** Add `obs/metrics.py` (`inner_slope`, `S`, `delta_S`, seed CI). Rebuild `obs/dashboard.py` per [Dashboard](#dashboard--metrics-decided) below. `uv run python -m obs.dashboard` produces new panels.

**Acceptance:** Report shows layered curves + ΔS bars from persisted rollouts (not synthetic mock data).

---

### D-05: Wire telemetry by default

**Status:** TODO  
**Decision:** Telemetry injection is **on** for real runs, not opt-in. Span-based hack detectors must see data.

**Action:** `outer_loop.evaluate` / `drive` passes `inject=telemetry.inject` into `score_repo`. Use harness-controlled absolute path for `obs/traces.db` (not cwd-relative surprises).

**Acceptance:** After one eval, `obs/traces.db` has spans; `_flag_short_trajectories` can return non-empty on synthetic bad trace.

---

### D-06: Nanochat `val_bpb` not from agent stdout

**Status:** TODO  
**Decision:** Agent must not be able to `print("val_bpb: 0.001")` and win. Matmul pattern is correct (harness subprocess evaluator); nanochat must match.

**Action:** Run eval via harness-controlled script after `train.py`; parse checkpoint or harness-owned output file — not raw `train.py` stdout regex (`envs/nanochat.py`).

**Acceptance:** Test: malicious `train.py` that only prints fake `val_bpb` gets `reward=0.0`.

---

### D-07: Improve smoke-check before eval

**Status:** DONE
**Decision:** Broken `improve()` output must not burn a full eval budget as `reward=0.0` archive entry.

**Landed:** `harness/loop/snapshot.py::smoke_check_version_snapshot()`; `drive()` skips failed snapshots.

**Acceptance:** Corrupt snapshot never calls `score_repo`; archive does not gain a zero-reward version for import errors.

---

### D-08: Eval protocol + seeds

**Status:** PARTIAL
**Decision:** Separate **agent `Budget`** from harness **`EvalProtocol`**: `inner_max_iters=8`, `seeds≥3` for powered runs. Agents never see `EvalProtocol`.

**Landed:** `EvalProtocol` model in `contracts.py`; seed loop in `evaluate()` still TODO.

**Acceptance:** Three seeds produce three rollouts per env on Attempt; metrics CI whiskers non-degenerate in test fixture.

---

### D-09: Outer-loop CLI

**Status:** DONE
**Decision:** README command must work: one entrypoint wires pools, K/M gens, archive path, `.env`.

**Action:** Add `harness/__main__.py` (or `cli.py`): `uv run python -m harness --ar ar/ --backend local|modal`.

**Acceptance:** `uv run python -m harness --help` and dry-run stub completes with `obs/archive.jsonl` written.

---

### D-10: Dedupe rollout execution

**Status:** TODO  
**Decision:** Single implementation of tracked-score + solve invocation; local and Modal are thin wrappers.

**Action:** `harness/rollout.py::run_rollout_once`, `harness/ar_loader.py::load_ar` / `load_solve`; refactor `score_repo` + `modal_runner`.

**Acceptance:** All tests green; no duplicated `_load_solve` bodies.

**Note:** Safe to do in parallel with [D-11](#d-11-modal-rollout-path) if changes are additive (new modules first, thin wrappers last).

---

### D-11: Modal rollout path

**Status:** IN_PROGRESS (parallel agent)  
**Decision:** `AR2_BACKEND=modal` fans `(ar × env)` via `modal_runner.run_rollouts_parallel`; hackathon profile guard stays.

**Action:** Live matmul rollout on Modal; fix only `modal_runner`, `infra/images`, GPU runner hooks as needed.

**Acceptance:** One real Modal job returns valid `Rollout` with non-empty `rewards` after [D-00](#d-00-done-semantics-kill-the-inner-loop) fix.

**Do not:** change `Rollout.model_dump()` shape or `env_spec` keys mid-flight without syncing [D-10](#d-10-dedupe-rollout-execution).

---

### D-12: EnvSpec serialization

**Status:** DEFER until D-11 lands  
**Decision:** Replace `vars(env)` introspection with explicit `BaseEnv.to_spec()` / `from_spec()`.

**Action:** `EnvSpec` model in contracts; implement on `MatmulEnv`, `NanoChatEnv`.

**Acceptance:** Modal round-trip reconstructs env without private-field leakage.

---

## Dashboard & metrics (DECIDED)

Implementation spec (inline — HTML mockups removed):

### Data requirements
- Per version `N`, seed `s`: held-out curve `R(N,s,t)` for **t = 0..K-1**, **K = 8** (baseline + 7 edits).
- Persist on `Attempt.heldout_rollouts` (and/or `obs/rollouts/v{N}_s{s}.jsonl`).
- **Frozen eval panel** (≥2 OOD envs) for metrics only — never passed to `improve()`.
- Selection held-out (scalar in archive) ≠ frozen panel (unbiased measurement).

### Panels (render order)
| Order | Panel | Role |
|-------|--------|------|
| 0 | **Layered inner curves** | Hero — all versions overlaid; gray v0, indigo latest; hacked = red dashed behind |
| 1 | **ΔS(N) bars** | Headline stat — `S(N)−S(N−1)`, 95% CI, zero line |
| 2 | **S(N) line** | Inner slope with CI whiskers |
| 3 | **Outer R\*** | Secondary — final held-out + running best (clean) |
| 4 | **Train vs held-out gap** | Integrity dumbbells |
| 5 | **Lineage tree** | Archive parent → child |

### Metrics (`obs/metrics.py`)
- `inner_slope(rewards) -> float` — OLS slope vs t
- `S(attempt) -> float` — mean slope over held-out rollouts / seeds
- `delta_S(archive, N) -> float` — `S(N) − S(N−1)`
- Bootstrap or normal CI over seeds (95%)

### Styling
Indigo primary, slate neutrals, white cards, section chips (HERO / PRIMARY / SECONDARY / INTEGRITY / LINEAGE). Self-contained `obs/report.html`.

**Cut:** small-multiples grid — layered overlay is the canonical inner-curve view.

---

## Measurement protocol (DECIDED — for any claim beyond demo)

| Rule | Value |
|------|--------|
| Inner steps K | 8 scored calls (enforce via `EvalProtocol` or validate post-hoc) |
| Seeds per version | ≥ 5 (≥ 3 minimum for demo) |
| OOD envs on frozen panel | ≥ 2 genuinely different distributions |
| Primary endpoint | ΔS: `S(K) − S(0) > 0` with CI not crossing 0 |
| Secondary endpoint | Best clean final held-out R* monotonic (outer curve) |
| Null baselines | Frozen v0, random `improve`, human `program.md` at equal budget |
| Parent selection | [D-02](#d-02-parent-sampling-not-greedy-frontier) required before evidence run |

Demo framing if underpowered: **“loop runs, hacks detectable, fan visually lifts”** — not “statistically significant convergence.”

---

## Env pools (DECIDED)

| Pool | Purpose | Matmul (powered demo) | Nanochat (flagship) |
|------|---------|----------------------|---------------------|
| **train** | Inner improvement signal | 64³, 128³ train splits | train data seed A |
| **selection held-out** | Archive scalar for `improve()` digest | 96³ heldout | held-out seed B ≠ A |
| **frozen eval panel** | S(N), ΔS, layered curves only | e.g. 160³ + different target | separate shard; never in improve |

Wire via `envs/pools.py::default_pools()` — used by CLI ([D-09](#d-09-outer-loop-cli)).

---

## Parallel work split (DECIDED)

| Track | Owns | Must not break |
|-------|------|----------------|
| **Modal agent** | D-11, `modal_runner`, `infra/images`, live GPU | `Rollout` JSON shape, `AR2_BACKEND` API |
| **Contracts agent** | D-00–D-10, D-03 metrics, dashboard | Modal function signature until merge |
| **Merge rule** | Additive schema (`Field(default_factory=list)`); new modules before rewiring callers | — |

---

## Invariants — do not regress

| Invariant | Status |
|-----------|--------|
| Referee outside `ar/` (`harness/referee.py`, spawn child) | **OK** |
| No LLM in reward path | **OK** |
| `improve()` sees scalar archive only, not held-out env contents | **OK** |
| Unconditional `Archive.add` (no pruning) | **OK** |
| Reward normalized to [0,1] per env | **OK** (calibrate nanochat baseline operationally) |
| Obs + referee unreachable from `ar/` | **PARTIAL** — telemetry off ([D-05](#d-05-wire-telemetry-by-default)), nanochat forge ([D-06](#d-06-nanochat-val_bpb-not-from-agent-stdout)), `ar/` exec in harness process (documented risk below) |

### Known §2 gap (document, fix post-demo)
`ar/entrypoint.py` is `exec_module`'d in the **harness process** before sandbox isolation. A malicious mutation could patch harness at import time. Mitigation post-demo: load `ar/` only inside sandbox/container ([D-15 backlog]).

---

### D-13: Import DAG mechanical refactor

**Status:** DONE  
**Landed (2026-05-30):** Steps 1–6 from `ARCHITECTURE_DAG.md` — loader→runtime; GPU runner injected via CLI/`matmul_runner()`; loop off cloud (via `runtime.score.evaluate_rollouts`); `cloud/register.py`; modal `run_rollout` delegates to `run_rollout_once`; ar off tracing. Step 7 — split `gpu.py` into `local.py`, `modal_gpu.py`, `vast.py` with thin factory. 166 tests green; grep checklist clean.

---

### D-14: Test colocation

**Status:** DONE  
**Decision:** Unit tests live next to the code they exercise. Top-level `tests/` is **integration-only** (cross-package smoke). Env tests import `harness.contracts` only — no harness orchestration imports.

**Landed (2026-05-30):** Part B T1→T6 — colocated tests under `envs/`, `harness/**/`, `infra/**/`, `obs/`, `ar/`; `tests/` slimmed to integration + smoke; `testpaths` in `pyproject.toml`; shared Modal stubs in `harness/cloud/conftest.py`.

**Acceptance:** Part B checklist in `ARCHITECTURE_DAG.md` — `pytest envs/ -q` green with no forbidden imports in `envs/test_*.py`; `tests/test_envs.py` and `tests/test_e2e_matmul.py` deleted.

**Order:** After or in parallel with D-13 step 2 (env runner injection). Prefer T1 (env colocation) early — low conflict with Modal work.

---

### D-15: Mutation boundaries + version snapshots

**Status:** PARTIAL  
**Resolves:** DESIGN §1 (“whole repo evolves”) vs §2 (“harness immutable”) — **DECISIONS wins.**

**Three zones:**

| Zone | Paths | Who edits | Role |
|------|-------|-----------|------|
| **Integrity kernel** | `envs/`, `harness/contracts.py`, `harness/tracing/`, `harness/loop/outer.py` (host `drive`), `harness/cloud/`, `harness/backends/`, `infra/` | Humans / host CLI only | Referee, obs injection, transport, selection driver — agent never touches |
| **Mutable autoresearch** | `ar/` (solve + improve policy), `harness/runtime/` (rollout orchestration) | Meta-agent per generation | What AR² optimizes — copied into `versions/v_*/` each `improve()` |
| **Obs / science** | `obs/`, Raindrop OTLP | Harness writes | Metrics + dashboard; not part of candidate |

**Version snapshot layout** (return value of `improve()`, `Attempt.source_ref`):

```
versions/v_<id>/
  ar/                 ← entrypoint.py (solve, improve)
  harness/runtime/    ← score_repo, rollout, sandbox wiring (optional edits)
```

**Host vs candidate:** The running `python -m harness` process always uses the **host** `drive()` loop. Each candidate is evaluated by loading `ar/` from the snapshot; **reload of `harness/runtime/` from snapshot during eval** is follow-up (D-15b).

**Action (landed):** `improve()` copies `ar/` + `harness/runtime/` into `versions/v_*`; meta-agent `cwd` = snapshot root; prompt lists editable vs frozen zones only (no spawn/coaching).

**Acceptance:** After `improve()`, snapshot contains `ar/entrypoint.py` and `harness/runtime/score.py`; `load_ar(source_ref)` resolves `ar/` from snapshot root; tests green.

**D-15b (PARTIAL):** `harness/runtime/dynamic.py` reloads snapshot `harness/runtime/` in `evaluate_rollouts` / `run_rollout_once` when present.

---

## DEFER — demo tradeoffs (do not block 5–10 gen run)

| Topic | Decision |
|-------|----------|
| `infra/collector.py` Modal telemetry sink | **DEFER** — optional live push; use `obs/run_events.jsonl` + dashboard instead |
| Human baseline vs hand-tuned `program.md` | Nice-to-have comparison axis |
| Cross-env generalization (nanochat train → matmul held-out) | Needed for generalization claims, not for “loop works” demo |
| Pin model ID / temperature in `Attempt` | Log model string in `diff_summary` for now |
| Fake spans to proxy | Accept as demo “interesting hack”; real integrity needs [D-05](#d-05-wire-telemetry-by-default) |
| Large-scale evolution budget (~80 gen) | 5–10 gen demo; see measurement protocol |

---

## Changelog

| Date | Change |
|------|--------|
| 2026-05-30 | SSOT rewrite: actionable queue, merged dashboard spec, renumbered IDs, parallel-work rules |
| 2026-05-30 | D-13: `ARCHITECTURE_DAG.md` approved for mechanical import refactor |
| 2026-05-30 | Flattened `proof/` (removed nested `documentation/`, mockups HTML) |
| 2026-05-30 | Reconciled Alex `inner-loop/`: research notes → flat `proof/*.md`; removed KernelBench spike + sol-scraper |
