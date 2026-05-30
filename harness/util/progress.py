"""Structured progress lines to stderr (unbuffered, safe for Modal drivers)."""
from __future__ import annotations

import sys
from datetime import datetime


def progress(msg: str) -> None:
    ts = datetime.now().strftime("%H:%M:%S")
    print(f"[ar2 {ts}] {msg}", file=sys.stderr, flush=True)
