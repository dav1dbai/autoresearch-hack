"""Shared helpers for the Qwen3.6-27B inference bench harness."""

from __future__ import annotations

import json
import os
import subprocess
import time
from pathlib import Path
from typing import Any

import yaml

BENCH_ROOT = Path(__file__).resolve().parent
RESULTS_DIR = BENCH_ROOT / "results"


def load_yaml(name: str) -> dict[str, Any]:
    return yaml.safe_load((BENCH_ROOT / name).read_text())


def gpu_name_slug() -> str:
    try:
        import torch

        return torch.cuda.get_device_name(0).replace(" ", "_")
    except Exception:
        return os.environ.get("GPU_NAME", "unknown").replace(" ", "_")


def run_cmd(cmd: list[str] | str, *, cwd: Path | None = None, timeout: int = 600) -> tuple[int, str]:
    if isinstance(cmd, str):
        cmd = ["bash", "-lc", cmd]
    proc = subprocess.run(
        cmd,
        cwd=cwd or BENCH_ROOT,
        capture_output=True,
        text=True,
        timeout=timeout,
    )
    out = (proc.stdout or "") + (proc.stderr or "")
    return proc.returncode, out.strip()


def wait_http(url: str, timeout_s: float = 300.0, interval_s: float = 2.0) -> tuple[bool, str]:
    import urllib.error
    import urllib.request

    deadline = time.time() + timeout_s
    last_err = ""
    while time.time() < deadline:
        try:
            with urllib.request.urlopen(url, timeout=5) as resp:
                if 200 <= resp.status < 500:
                    return True, f"HTTP {resp.status}"
        except urllib.error.HTTPError as ex:
            if ex.code < 500:
                return True, f"HTTP {ex.code}"
            last_err = str(ex)
        except Exception as ex:
            last_err = str(ex)
        time.sleep(interval_s)
    return False, last_err or "timeout"


def write_json(path: Path, data: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, indent=2) + "\n")
