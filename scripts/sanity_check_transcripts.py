#!/usr/bin/env python3
"""Preflight: scan inner/outer agent transcripts before a real Modal run."""
from __future__ import annotations

import json
import re
import sys
from pathlib import Path

_REPO = Path(__file__).resolve().parents[1]
_FAIL_PATTERNS = (
    (re.compile(r"context window", re.I), "Codex context window overflow"),
    (re.compile(r"tokens used\s*\n\s*0\s*$", re.M), "Codex reported 0 tokens (no work done)"),
    (re.compile(r"Auth\(TokenRefreshFailed", re.I), "MCP OAuth refresh failed (Slack/etc.)"),
    (re.compile(r"AuthRequired", re.I), "MCP auth required — may bloat/fail headless runs"),
)
_MIN_LOG_BYTES = 200  # header-only logs are suspicious for a real agent run


def _scan_log(path: Path, label: str) -> list[str]:
    issues: list[str] = []
    try:
        text = path.read_text(encoding="utf-8", errors="replace")
        size = path.stat().st_size
    except OSError:
        return [f"{label}: log missing (tmpdir cleaned up)"]
    if size < _MIN_LOG_BYTES:
        issues.append(f"{label}: log too short ({size} B) — agent likely did nothing")
    for pat, msg in _FAIL_PATTERNS:
        if pat.search(text):
            issues.append(f"{label}: {msg}")
    return issues


def _find_inner_logs() -> list[Path]:
    tmp = Path("/var/folders")
    if not tmp.exists():
        return []
    return sorted(
        p for p in tmp.glob("**/matmul_*/agent.log")
        if p.stat().st_mtime > (__import__("time").time() - 86400)
    )[-12:]


def _check_archive(archive: Path) -> list[str]:
    issues: list[str] = []
    if not archive.exists():
        return ["obs/archive.jsonl missing — no completed evaluate() yet"]
    attempts = []
    for line in archive.read_text().splitlines():
        if line.strip():
            attempts.append(json.loads(line))
    if not attempts:
        return ["obs/archive.jsonl empty"]
    rewards = {(a["train_reward"], a["heldout_reward"]) for a in attempts}
    if len(rewards) == 1 and len(attempts) > 1:
        issues.append(
            "All attempts share identical train/heldout rewards — "
            "likely MATMUL_STUB=1 (no speed signal) or agents made no edits"
        )
    for a in attempts:
        for r in a.get("train_rollouts", []) + a.get("heldout_rollouts", []):
            rw = r.get("rewards") or []
            if len(rw) >= 2 and len(set(rw)) == 1:
                issues.append(
                    f"v{a['version']} {r.get('env_id')}: flat inner curve {rw[:3]} "
                    "(stub scoring or no kernel edits)"
                )
                break
    return issues


def main() -> int:
    issues: list[str] = []

    stub = __import__("os").environ.get("MATMUL_STUB", "0") == "1"
    if stub:
        issues.append("MATMUL_STUB=1 — rewards ignore kernel speed; not useful for optimization")

    backend = __import__("os").environ.get("AR2_BACKEND", "local")
    if backend != "modal":
        issues.append(f"AR2_BACKEND={backend!r} — Modal containers will not start")

    outer_logs = sorted((_REPO / "versions").glob("v_*/agent.log")) if (_REPO / "versions").is_dir() else []
    if not outer_logs:
        issues.append("No versions/v_*/agent.log — no improve() runs yet")
    for p in outer_logs:
        issues.extend(_scan_log(p, f"outer {p.parent.name}"))

    inner_logs = _find_inner_logs()
    if not inner_logs:
        issues.append("No recent matmul_*/agent.log under /var/folders — inner solve logs missing")
    else:
        for p in inner_logs[-6:]:
            issues.extend(_scan_log(p, f"inner {p.parent.name}"))

    issues.extend(_check_archive(_REPO / "obs" / "archive.jsonl"))

    print("=" * 60)
    print("AR² agent transcript preflight")
    print("=" * 60)
    print(f"Outer logs:  {len(outer_logs)}")
    print(f"Inner logs:  {len(inner_logs)} (recent)")
    print()

    if issues:
        print("BLOCKERS / WARNINGS:")
        for i, msg in enumerate(issues, 1):
            print(f"  {i}. {msg}")
        print()
        print("Fix before Modal K=3:")
        print("  • Use MATMUL_STUB=0 + --gpu --gpu-smoke (or full GPU pools)")
        print("  • AR2_BACKEND=modal AR2_GPU_BACKEND=modal")
        print("  • Trim Codex MCP (Slack OAuth errors + context blow-up in improve logs)")
        print("  • Re-run K=1 locally with stub OFF first; re-check transcripts")
        return 1

    print("OK — no obvious transcript blockers.")
    print("Next: AR2_K=3 ./scripts/run_k3_modal_e2e.sh")
    return 0


if __name__ == "__main__":
    sys.exit(main())
