# AR² — cuda-oxide runs: honest findings

Post-hoc analysis of 6 cuda-oxide runs (3 `hero-*` + 3 `redo-*-k4`), read from `obs/runs/*/archive.jsonl` and per-run `ar2_workshop.db`. **Reward is a noisy single-rollout estimate; kernel throughput is the deterministic signal — trust the kernels, not the reward rankings.**

## TL;DR

- **The inner loop genuinely optimizes.** The agent took NVIDIA's *verbatim naive* GEMM (2135 GFLOPS) → **6554 GFLOPS (3.1×)** via discovered shared-memory tiling, in a clean monotonic climb that generalized 1024³→1536³. This is real.
- **The outer (meta) loop mostly made non-causal edits** — improve-prompt formatting / harness telemetry — so most headline reward "jumps" are variance, not a second derivative.
- **One genuine exception:** a *solve-side* meta-edit (dynamic stale-limit) that is logically sound and enabled the cleanest climb in any run.
- **Two root causes** the second derivative didn't fire: selection is on a single noisy rollout (it chases variance), and the seed `solve()` is bare (large unexploited headroom).

## What worked (real signal)

1. **Inner-loop optimization capability** — `redo-naive-gemm-k4 v0`: naive→tiled, **3.1×**, heldout curve `[0.178 → 0.520 → 0.539 → 0.546]` (monotonic), generalized from the 1024³ train problem to 1536³ heldout. Strongest evidence the system finds real GPU optimizations on a 3-week-old API.
2. **The one causal meta-edit** — `redo-reduction-k4 v1`: a *solve-side* dynamic stale-limit (tolerate ~5 non-improving inner steps instead of the default 2). Correct reasoning (`STALE_ITERS=2` gives up too fast); under it, `v3` produced the cleanest 5-step monotonic climb in any run: `[0.722 → 0.888 → 0.979 → 0.984 → 0.986]`. The only genuine outer→inner improvement story.
3. **Cleanest "it works" climbs** (honest inner-loop artifacts): `redo-reduction-k4 v3` heldout (above), `redo-naive-gemm-k4 v0` heldout (3.1×), `hero-reduction v0` heldout `[0.727 → 0.993 → 0.996 → 1.0]`.

## What didn't, and why

- **Most "best" attempts were non-causal.** `hero-gemm v9` (0.5209), `redo-gemm-k4 v9` (0.5165), `hero-naive_gemm v8`, `hero-reduction v8` all touched only the **improve-prompt formatting** or **harness telemetry** — never `solve()`. An edit that can't reach `solve()` cannot cause that version's own kernel score. They're lucky inner spikes that collapse the next iteration (v9: `[0.49, 0.52, 0.0, 0.0]`).
- **Variance dominates selection.** `naive_gemm v0` scored **0.176 (hero) vs 0.546 (redo)** — *same seed code*, pure agent luck. Single-rollout heldout means `drive` often breeds from a lucky sample, not a better strategy → gains don't compound.
- **The harness/seed `solve()` is bare** — single-edit hill-climb, fixed stale logic, no `spawn`/parallel probes. There's large, obvious headroom the meta-agent didn't take; it drifted to cosmetic edits, partly because noisy selection gave it no pressure toward substantive ones.
- **Reduction is near-solved at the starter** (`reduce_opt` ≈ 89% of H100 HBM peak), so there's little for the agent to find there.

## Known gap

**Best kernel artifacts are not persisted.** Rollout workdirs (`/tmp/cudaoxide_*/kernel.rs`) are discarded after scoring; `versions/` only snapshots `ar/` + `harness/runtime/`. So the 6554-GFLOPS tiled kernel's *score* exists but its *code* is gone. **Fix:** persist the best rollout's `kernel.rs` into the run dir.

## Most promising next experiment

Re-run **naive_gemm** with `redo-reduction-k4 v1`'s dynamic-stale-limit baked into the seed `solve()`, **average heldout over ~5 rollouts** (de-noise the selection signal), and **persist the winning `kernel.rs`**. If the inner agent beats 0.546 *and* the artifact is captured, that's "AR² improved its own loop — here's the faster kernel it found," and it survives scrutiny.

## Verdict

Real apparatus; real inner-loop optimization capability; the **second derivative is not yet demonstrated causally** (the one solve-side reduction edit aside). The two changes that would most likely change that: **(1) de-noise selection** (average over rollouts so the meta-loop selects on signal, not luck), and **(2) push the meta-agent toward structural `solve`-side edits** — the harness is bare enough that there's plenty to win, but it has to actually touch the search loop, not the prompt formatting.
