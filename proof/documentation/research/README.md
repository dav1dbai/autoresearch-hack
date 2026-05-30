# Hackathon research notes (Alex)

Early design exploration for verifiable envs and the double auto-research loop.
**Not executable code** — the landed implementation lives in `envs/`, `harness/`, and `ar/`.

| File | Contents |
|------|----------|
| [harness-design.md](harness-design.md) | Double-loop harness design, five domain sketches, anti-reward-hacking principles |
| [better-report.md](better-report.md) | Env shortlist (KernelBench, Terminal-Bench, verifiers audit) |
| [verifiers-structured-env-audit.md](verifiers-structured-env-audit.md) | Prime Intellect / verifiers env inventory |
| [report.md](report.md) | Longer hackathon research report |

**Reconciled 2026-05-30:** Removed `inner-loop/domains/` (standalone KernelBench spike + sol-scraper data).
That code duplicated `envs/matmul.py` + `harness/` and was never wired to `harness.contracts`.
