from __future__ import annotations

import argparse
import json
from pathlib import Path

from .config import DEFAULT_CONFIG, select_tasks
from .inner_loop import run_inner_loop
from .program import DEFAULT_PROGRAM, load_program


def main() -> None:
    parser = argparse.ArgumentParser(description="Run kernel/legal iterative autoresearch tasks.")
    parser.add_argument("--config", default=str(DEFAULT_CONFIG))
    parser.add_argument("--task", action="append", help="Task name from config. Repeatable.")
    parser.add_argument("--run-root", default="runs/prime-research")
    parser.add_argument("--agent-cmd", default=None)
    parser.add_argument("--program", default=str(DEFAULT_PROGRAM))
    parser.add_argument("--keep", action="store_true", help="Do not clean prior task workspace.")
    args = parser.parse_args()

    results = []
    program_text = load_program(args.program)
    for task in select_tasks(args.config, args.task):
        result = run_inner_loop(
            task,
            run_root=Path(args.run_root),
            agent_cmd=args.agent_cmd,
            program_text=program_text,
            clean=not args.keep,
        )
        results.append(result.model_dump())
        print(json.dumps(result.model_dump(), indent=2))

    out = Path(args.run_root) / "summary.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(results, indent=2, default=str))


if __name__ == "__main__":
    main()
