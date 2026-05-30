"""TEMPLATE for new envs (legal, GPU): implement reset()+score() with a deterministic,
out-of-process, read-only evaluator. Normalize reward to [0,1].

NanoChatEnv adapts Karpathy's autoresearch task as the AR² reference environment.
The evaluator (evaluate_bpb in ar/prepare.py) is read-only and never copied into the
agent's editable workdir — it runs isolated via subprocess.

Integrity boundary:
  - agent editable surface: workdir/train.py   (copy of ar/train.py)
  - evaluator:              ar/prepare.py       (read-only, out of workdir, out of agent reach)
"""
from __future__ import annotations

import os
import re
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path
from typing import Literal

from harness.contracts import Env, Split, Submission, StepResult, TaskSpec
from envs.base import BaseEnv

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

# Source files (read-only; never placed in agent workdir)
_AR_DIR = Path(__file__).parent.parent / "ar"
_TRAIN_SRC = _AR_DIR / "train.py"

# Normalization: a reasonable CPU-mode/stub baseline bpb. Real GPU runs will
# calibrate this against the actual v0 baseline. reward = clip(1 - bpb/BASELINE, 0, 1)
# so reward=0 at or above baseline, reward→1 as bpb→0.
_BASELINE_BPB: float = float(os.environ.get("NANOCHAT_BASELINE_BPB", "1.0"))

# Env var: set NANOCHAT_STUB=1 to skip real training (unit-test / CPU mode).
_STUB_MODE: bool = os.environ.get("NANOCHAT_STUB", "0") == "1"

# Synthetic stub output bpb (injected instead of real run). Read at call time so tests can override.
_STUB_BPB_DEFAULT: float = 0.8

# ---------------------------------------------------------------------------
# Env
# ---------------------------------------------------------------------------

class NanoChatEnv(BaseEnv):
    """Env that wraps Karpathy's nanochat autoresearch task.

    - reset()  copies train.py into an isolated tmpdir; the agent edits only that dir.
    - score()  runs train.py out-of-process (or stubs it), greps val_bpb from the log,
               and normalizes to [0, 1].

    The evaluator (ar/prepare.py::evaluate_bpb) is NOT in the workdir — it is imported
    by train.py from its fixed location and is never writable by the agent.
    """

    id: str
    split: Split

    def __init__(
        self,
        split: Split = "train",
        stub: bool | None = None,
        baseline_bpb: float = _BASELINE_BPB,
    ) -> None:
        self.id = f"nanochat-{split}"
        self.split = split
        self._stub = _STUB_MODE if stub is None else stub
        self._baseline_bpb = baseline_bpb
        self._workdir: Path | None = None

    # ------------------------------------------------------------------
    # Env protocol
    # ------------------------------------------------------------------

    def reset(self) -> TaskSpec:
        """Prepare an isolated workdir with a copy of train.py as the editable surface."""
        tmp = Path(tempfile.mkdtemp(prefix="nanochat_"))
        dst = tmp / "train.py"
        shutil.copy2(_TRAIN_SRC, dst)
        self._workdir = tmp
        return TaskSpec(
            env_id=self.id,
            split=self.split,
            prompt=(
                "Minimize val_bpb (validation bits-per-byte, lower is better). "
                "You may freely edit train.py. Do not modify prepare.py or any file "
                "outside your workdir. Run: python train.py > run.log 2>&1"
            ),
            workdir=tmp,
            payload={"editable_file": "train.py"},
        )

    def score(self, sub: Submission) -> StepResult:
        """Run train.py from sub.workdir out-of-process; parse val_bpb; normalize."""
        val_bpb = self._run(sub.workdir)
        reward = max(0.0, min(1.0, 1.0 - val_bpb / self._baseline_bpb))
        return StepResult(
            reward=reward,
            raw={"val_bpb": val_bpb},
            feedback=f"val_bpb={val_bpb:.6f}  (reward={reward:.4f})",
            done=False,  # no natural terminal; let solve() keep iterating (D-00)
        )

    # ------------------------------------------------------------------
    # Internal
    # ------------------------------------------------------------------

    def _run(self, workdir: Path) -> float:
        if self._stub:
            return float(os.environ.get("NANOCHAT_STUB_BPB", str(_STUB_BPB_DEFAULT)))

        log_path = workdir / "run.log"
        result = subprocess.run(
            [sys.executable, str(workdir / "train.py")],
            cwd=str(workdir),
            stdout=log_path.open("w"),
            stderr=subprocess.STDOUT,
            timeout=600,  # hard wall; real budget is enforced inside train.py (TIME_BUDGET=300)
        )
        log_text = log_path.read_text()
        return _parse_bpb(log_text)


def _parse_bpb(log: str) -> float:
    """Extract val_bpb from train.py output. Raises ValueError on parse failure."""
    m = re.search(r"^val_bpb:\s*([\d.]+)", log, re.MULTILINE)
    if not m:
        raise ValueError(f"val_bpb not found in log. Tail:\n{log[-500:]}")
    return float(m.group(1))
