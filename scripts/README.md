# Scripts

Shell helpers for local and Modal e2e runs. All assume repo root as cwd.

| Script | Purpose |
|--------|---------|
| `run_k3_e2e.sh` | Delegates to Modal K=3 outer loop |
| `run_k3_modal_e2e.sh` | K=3 on Modal with live dashboard refresh |
| `modal_smoke.sh` | Quick Modal GPU smoke (K=0) |
| `modal_outer_loop_smoke.sh` | Short outer loop on Modal |
| `local_meta_outer_loop.sh` | Local stub-only loop (MATMUL_STUB) |
| `analyze_run.py` | Post-run archive summary + report |
| `reload_run_to_workshop.py` | Reload `obs/runs/<run_id>` spans into the active Workshop |
| `sanity_check_transcripts.py` | Preflight agent.log scan |
| `noop_agent.sh` / `noop_mutate.sh` | No-op Codex substitutes for CI |
| `raindrop_codex_exec.sh` | Codex + Raindrop Workshop MCP (default agent cmd) |
| `workshop_codex_smoke.sh` | Local proof: Codex + Raindrop → live_events in Workshop |
| `workshop_ngrok.sh` | Tunnel Workshop to Modal via ngrok (`--host-header=localhost:5899`) |
| `modal_workshop_ngrok.sh` | Sourced by Modal e2e scripts (uses `workshop_ngrok.sh`) |
| `clean_stale_runs.sh` | Reset Workshop DB + wipe obs/ + archive old logs |

Set `AR2_FRESH=1` on e2e scripts to reset that run's archive/db/version outputs.
Set `AR2_RUN_ID=<name>` to isolate parallel outer loops under `obs/runs/<name>/`
and `versions/<name>/`:

```bash
AR2_RUN_ID=matmul-a AR2_FRESH=1 AR2_K=3 ./scripts/run_k3_modal_e2e.sh
AR2_RUN_ID=matmul-b AR2_FRESH=1 AR2_K=3 ./scripts/run_k3_modal_e2e.sh
```

The master run index is written to `obs/runs/index.html`. To repopulate Workshop
from a prior run:

```bash
python scripts/reload_run_to_workshop.py matmul-a
```
