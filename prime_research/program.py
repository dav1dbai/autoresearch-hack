from __future__ import annotations

from pathlib import Path


DEFAULT_PROGRAM = Path(__file__).parent / "programs" / "kernel_legal_v0.md"


def load_program(path: str | Path | None = None) -> str:
    program_path = Path(path) if path else DEFAULT_PROGRAM
    return program_path.read_text()

