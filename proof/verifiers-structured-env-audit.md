# Prime Verifiers / Structured Environment Audit

Date: 2026-05-30

Scope: the domains and benchmark ideas in `report.md`, `harness-design.md`, and `better-report.md`.

Method: queried the public Prime Intellect Environments Hub API directly, because
the local `prime` CLI is configured with an unauthorized API key. Also checked
the public benchmark/project repositories named in the research notes.

## Executive Summary

Several of the proposed domains already have Prime Intellect Verifiers
environments. The highest-confidence direct reuse targets are:

1. GPU kernels: `popfido/kernelbench`, `primeintellect/kernelbench`,
   `sinatras/kernelbench-kguard`, plus related CUDA/Triton environments.
2. Terminal-Bench / Harbor: `primeintellect/terminal-bench-2`,
   `primeintellect/harbor`, `ibrahim/terminal-bench`, `popfido/terminalbench`,
   `pandelis/harbor-terminal-bench`.
3. Multi-agent / tool-agent-user: `primeintellect/tau2-bench`,
   `will/tau2-bench`, `prime/tau2-synth`, `prime-community/tau-bench-env`,
   plus `primeintellect/general-agent`.
4. SRE / RCA: `prime/openrca-env` is a direct Verifiers environment.
5. Legal warmups: `primeintellect/legalbench`, `srthkdev/legalbench`,
   `srthkdev/contract_nli_env`, `primeintellect/taxcalc-bench`,
   `primeintellect/contract-clause-review`, and
   `primeintellect/contract-review-clause-identification`.

The main things that do not appear to have direct Prime Verifiers coverage are
the exact free-form legal loophole task, CUAD span extraction by name, RCAEval by
name, ITBench by name, AIOpsLab by name, MLGym by name, and TransCoder by name.
Most of those still have structured upstream repos or datasets, so the remaining
work is mostly adapter work rather than inventing environments from scratch.

## Matrix

| Research domain | Prime Verifiers already exists? | Best direct PI envs found | Other structured envs/repos | Recommendation |
|---|---:|---|---|---|
| GPU kernels / KernelBench | Yes | `popfido/kernelbench` v0.1.3, `primeintellect/kernelbench` v0.1.6, `ppbhatt500/kernelbench` v0.1.3, `sinatras/kernelbench-kguard` v0.1.0 | `ScalingIntelligence/KernelBench`, `SakanaAI/robust-kbench`, BackendBench, PMPP, FlashInfer-Bench, TritonBench | Use existing PI KernelBench first. If reward hacking is a concern, inspect `kernelbench-kguard` and port checks into the canonical wrapper. |
| Other GPU/Triton kernels | Yes | `sinatras/pmpp` v1.1.3, `siro/backend-bench` v0.3.14, `primeintellect/backend-bench` v0.2.0, `tlait/flashinfer-bench-env` v0.2.0, `anushkad/tritonbench` v0.2.0, `primeintellect/gpu-puzzles` v0.1.0 | PMPP, BackendBench, FlashInfer-Bench, GPU Puzzles, KernelBook | Strong backup set if KernelBench is too narrow or infra-specific. |
| Terminal-Bench / Harbor | Yes | `primeintellect/terminal-bench-2` v0.2.1, `primeintellect/harbor` v0.1.5, `ibrahim/terminal-bench` v0.4.0, `popfido/terminalbench` v0.5.2, `pandelis/harbor-terminal-bench` v0.1.2 | `laude-institute/terminal-bench`, Harbor task format | Make this the main environment. There is no need to build a wrapper from scratch unless you need custom private tasks. |
| SWE / code agent tasks | Yes | `primeintellect/swe` v0.3.5, `primeintellect/swebench-pro` v0.1.0, `primeintellect/mini-swe-agent-plus` v0.2.25, `prime-community/mini-swe-agent-bench` v0.1.0 | SWE-bench, R2E-Gym, Harbor | Useful if the "Terminal-Bench short suite" becomes too broad; use hidden tests and Harbor-backed tasks. |
| SRE / static RCA | Partial yes | `prime/openrca-env` v0.1.7 | `microsoft/OpenRCA`, RCAEval if separately obtained | Use `prime/openrca-env` as the v0 SRE/RCA target. No clear PI hit for RCAEval by name. |
| SRE / live AIOps | No direct AIOpsLab hit | Related: `tonyteo/devops-troubleshoot` v0.1.0, `vinayak1998/devops-gym` v0.1.0, `prime/tau2-synth` includes `cloud_incident_response` synthetic domain | `microsoft/AIOpsLab`, `itbench-hub/ITBench`, Chaos Mesh, Online Boutique | Treat live SRE as adapter/build work. For a hackathon, use OpenRCA or synthetic cloud incident response before AIOpsLab. |
| ITBench | No clear PI hit | None found by `itbench` search | `itbench-hub/ITBench` | Structured upstream exists, but expect adapter work. |
| Multi-agent / tau-bench | Yes | `primeintellect/tau2-bench` v0.2.3, `will/tau2-bench` v0.2.0, `prime/tau2-synth` v0.2.0, `prime-community/tau-bench-env` v0.1.0, `vibrantlabsai/tau2_infinity` v0.2.1 | `sierra-research/tau2-bench` | Best existing substrate for the multi-agent/tool-user part. This is better than inventing a new multi-agent verifier. |
| Multi-agent self-improvement / general agent | Yes, but less canonical | `primeintellect/general-agent` v0.1.4, `tonyteo/autonomous-skill-evolution` v1.0.0, `stochi0/autoresearch` v0.1.0, `stochi0/rubric-discovery` v0.2.0 | GEPA, AFlow, DSPy/MIPROv2, ADAS | Use these as references, but still score against a deterministic downstream task family. |
| Code porting / TransCoder | No exact TransCoder hit | Related deterministic code envs: `primeintellect/livecodebench` v0.2.7, `primeintellect/code-env` v0.3.2, HumanEval and MBPP envs | TransCoder-test/ST in `facebookresearch/CodeGen`, HumanEval, MBPP | If the desired story is porting, build an adapter. If any test-based coding task is acceptable, reuse LiveCodeBench, MBPP, or HumanEval. |
| MBPP / HumanEval | Yes | `jashvira/MBPP` v0.1.2, `ohadrubin/mbpp_baseline` v0.1.2, `pmahdavi/humaneval` v0.1.1, `pmahdavi/humanevalplus` v0.1.1, `stelioszach/code-humaneval*` v0.1.4 | MBPP, HumanEval, HumanEval+ | Good deterministic inner tasks for the outer-loop/multi-agent optimizer. |
| PaperBench / research reproduction | Yes | `stalkermustang/paperbench-env` v0.1.3 | OpenAI PaperBench | Exists, but likely too long/heavy for 20-minute rollouts unless heavily sliced. |
| MLGym | No exact PI hit | None found by `MLGym` search | `facebookresearch/MLGym` | Structured upstream exists, but likely adapter work and task slicing. |
| LegalBench | Yes | `primeintellect/legalbench` v0.1.1, `srthkdev/legalbench` v0.1.1, `prime-community/legalbench` v0.1.1, `felix/legalbench` v0.0.2 | `HazyResearch/legalbench` | Use as legal warm-up / deterministic-ish classification/extraction tasks. |
| ContractNLI | Yes | `srthkdev/contract_nli_env` v0.1.0 | Stanford ContractNLI | Good contract reasoning warm-up. |
| CUAD | No direct CUAD hit | Related: `primeintellect/contract-clause-review` v0.1.1, `primeintellect/contract-review-clause-identification` v0.1.1 | CUAD dataset from The Atticus Project | The PI contract-clause envs may be sufficient. If CUAD span-F1 specifically matters, build or fork an adapter. |
| Tax / SARA-style legal arithmetic | Yes for taxcalc | `primeintellect/taxcalc-bench` v0.1.0, `nguyen599/taxcalc-bench` v0.1.0, `ascl1u/taxcalcbench-rlm` v0.1.0 | TaxCalcBench, SARA if separately used | Use TaxCalcBench as the deterministic legal/math warm-up unless SARA is required by name. |
| Free-form legal loophole discovery | Not directly | Related: `narcolepticchicken/dgcl-reward-hacking` v0.1.0, citation-laundering envs, contract-clause envs, LegalBench | Inspect AI, CUAD, ContractNLI, LegalBench | Novel build. Reuse PI legal envs for gates/calibration, but the actual loophole verifier still needs custom design. |

## Most Actionable Build Path

1. Start with `primeintellect/terminal-bench-2` or `primeintellect/harbor`.
   This best matches the desired "agent improves through scaffolds/tools/retrieval"
   story and already has the task/harness shape.
2. Add `popfido/kernelbench` or `primeintellect/kernelbench` as the clean
   deterministic GPU flagship. Evaluate `sinatras/kernelbench-kguard` before
   rolling your own anti-hacking layer.
3. Use `prime/openrca-env` for SRE/RCA. Defer AIOpsLab/ITBench unless live ops
   is essential.
4. For multi-agent optimization, use `primeintellect/tau2-bench` or `will/tau2-bench`
   as the structured tool-agent-user substrate. For pure deterministic inner
   tasks, use MBPP/HumanEval/LiveCodeBench rather than TransCoder unless porting
   is core to the demo.
5. For legal, use `primeintellect/legalbench`, `srthkdev/contract_nli_env`, and
   `primeintellect/taxcalc-bench` as warm-ups. Treat free-form loophole discovery
   as new verifier work with deterministic citation/span gates.

## Hub Queries Used

Public API root:

```text
https://api.primeintellect.ai/api/v1/environmentshub/
```

Example searches:

```text
?limit=20&offset=0&search=kernelbench
?limit=20&offset=0&search=terminal-bench
?limit=20&offset=0&search=openrca
?limit=20&offset=0&search=legalbench
?limit=20&offset=0&search=contract
?limit=20&offset=0&search=tax
?limit=20&offset=0&search=tau
?limit=20&offset=0&search=paperbench
?limit=20&offset=0&search=MLGym
?limit=20&offset=0&search=transcoder
?limit=20&offset=0&search=itbench
```

## Caveats

- Environment names on the Hub are not unique. Prefer owner-qualified IDs.
- Star count is not a quality signal for new environments; inspect code before
  training against any reward.
- Several "related" environments may be thin wrappers or demos. Before using one
  in an outer loop, run `prime env info owner/name` or pull the package and check
  whether it has train/eval splits, hidden tests, and immutable verifier logic.
- The local `prime` CLI should be re-authenticated if you want `prime env info`,
  `prime env pull`, or `prime eval run` to work without direct API calls.
