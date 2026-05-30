"""Run a hidden pytest suite against a submitted module; emit a [0,1] reward.

The tests live only in the verifier sandbox (injected after the agent stops),
so the agent can neither read nor edit them. Reward is the fraction of tests
that pass; a suite that fails to collect (e.g. submission doesn't import) is 0.

Usage:
    python run_pytest.py --tests tests --reward-mode fraction
The submission file is staged into this directory by the executor before this
runs, so `import <module>` works from here.
"""

from __future__ import annotations

import argparse
import json
import re
import subprocess
import sys


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--tests", default="tests")
    parser.add_argument("--reward-mode", choices=["fraction", "all_or_nothing"], default="fraction")
    args = parser.parse_args()

    proc = subprocess.run(
        [sys.executable, "-m", "pytest", args.tests, "-q", "--tb=short", "-p", "no:cacheprovider"],
        capture_output=True,
        text=True,
    )
    out = proc.stdout + "\n" + proc.stderr

    def count(label: str) -> int:
        m = re.search(rf"(\d+) {label}", out)
        return int(m.group(1)) if m else 0

    passed = count("passed")
    failed = count("failed") + count("error") + count("errors")
    total = passed + failed

    if total == 0:
        reward = 0.0
    elif args.reward_mode == "all_or_nothing":
        reward = 1.0 if failed == 0 else 0.0
    else:
        reward = round(passed / total, 4)

    print(json.dumps({
        "reward": reward,
        "passed": passed,
        "failed": failed,
        "total": total,
        "stdout_tail": out[-2000:],
    }))


if __name__ == "__main__":
    main()
