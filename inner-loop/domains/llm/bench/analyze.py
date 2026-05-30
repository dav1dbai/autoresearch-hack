"""Merge bench JSON artifacts into summary tables."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path


def load_results(paths: list[Path]) -> list[dict]:
    rows: list[dict] = []
    for path in paths:
        data = json.loads(path.read_text())
        if isinstance(data, list):
            rows.extend(data)
        elif "runs" in data:
            rows.extend(data["runs"])
        else:
            rows.append(data)
    return rows


def to_csv(rows: list[dict], out: Path) -> None:
    fields = [
        "engine",
        "track",
        "workload",
        "concurrency",
        "output_tok_s",
        "ttft_p50",
        "ttft_p95",
        "itl_p50",
        "e2e_p50",
        "n_ok",
        "n_error",
    ]
    with out.open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        for r in rows:
            m = r.get("metrics") or r
            w.writerow(
                {
                    "engine": r.get("engine", ""),
                    "track": r.get("track", ""),
                    "workload": r.get("workload", ""),
                    "concurrency": r.get("concurrency", ""),
                    "output_tok_s": m.get("output_tok_s", ""),
                    "ttft_p50": (m.get("ttft_ms") or {}).get("p50", ""),
                    "ttft_p95": (m.get("ttft_ms") or {}).get("p95", ""),
                    "itl_p50": (m.get("itl_ms") or {}).get("p50", ""),
                    "e2e_p50": (m.get("e2e_ms") or {}).get("p50", ""),
                    "n_ok": m.get("n_ok", ""),
                    "n_error": m.get("n_error", ""),
                }
            )


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("inputs", nargs="+", type=Path)
    p.add_argument("-o", "--output", type=Path, default=Path("summary.csv"))
    args = p.parse_args()
    rows = load_results(args.inputs)
    to_csv(rows, args.output)
    print(f"Wrote {len(rows)} rows to {args.output}")


if __name__ == "__main__":
    main()
