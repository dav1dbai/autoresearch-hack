"""Score grounded legal loophole findings without a reference solution.

The verifier intentionally uses deterministic gates first:
- every quoted clause must be present in the source document
- every finding must have the required structured fields
- duplicated quote/description pairs are discounted

This is not legal advice. It is a benchmark scorer for grounded ambiguity/risk
analysis artifacts.
"""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path
from typing import Any


REQUIRED = {
    "quoted_clause_text",
    "location",
    "loophole_description",
    "severity_1to5",
    "exploitation_scenario",
}


def _load_findings(path: Path) -> list[dict[str, Any]]:
    data = json.loads(path.read_text())
    if not isinstance(data, list):
        raise ValueError("findings JSON must be a list")
    findings: list[dict[str, Any]] = []
    for item in data:
        if not isinstance(item, dict):
            continue
        findings.append(item)
    return findings


def _words(text: str) -> set[str]:
    return set(re.findall(r"[a-zA-Z][a-zA-Z0-9_-]{3,}", text.lower()))


def _compact(text: str) -> str:
    return re.sub(r"\s+", " ", text).strip()


def _contains_quote(contract: str, quote: str) -> bool:
    return bool(quote) and _compact(quote) in _compact(contract)


def _specificity(text: str) -> float:
    has_amount = bool(re.search(r"\$|percent|%|\b\d{1,3}(,\d{3})*\b", text, re.I))
    has_date = bool(re.search(r"\b(day|days|month|months|year|years|notice|deadline|term)\b", text, re.I))
    has_party = bool(re.search(r"\b(company|customer|vendor|supplier|party|buyer|seller|contractor)\b", text, re.I))
    return (has_amount + has_date + has_party) / 3


def score_findings(contract: str, findings: list[dict[str, Any]]) -> dict[str, Any]:
    contract_words = _words(contract)
    seen: set[tuple[str, str]] = set()
    scored = []

    for idx, finding in enumerate(findings):
        missing = sorted(REQUIRED - set(finding))
        quote = str(finding.get("quoted_clause_text", "")).strip()
        desc = str(finding.get("loophole_description", "")).strip()
        scenario = str(finding.get("exploitation_scenario", "")).strip()
        location = str(finding.get("location", "")).strip()

        try:
            severity = float(finding.get("severity_1to5", 0))
        except (TypeError, ValueError):
            severity = 0

        quote_present = _contains_quote(contract, quote)
        duplicate_key = (quote[:160], desc[:160])
        duplicate = duplicate_key in seen
        seen.add(duplicate_key)

        desc_words = _words(desc)
        scenario_words = _words(scenario)
        grounded_terms = len((desc_words | scenario_words) & contract_words)
        novelty_terms = len((desc_words | scenario_words) - contract_words)

        grounded = 1.0 if quote_present else 0.0
        severity_score = min(max(severity, 0.0), 5.0) / 5.0
        detail_score = min((len(desc_words) + len(scenario_words)) / 55.0, 1.0)
        specificity_score = _specificity(scenario)
        term_score = min(grounded_terms / 10.0, 1.0)
        novelty_score = min(novelty_terms / 18.0, 1.0)

        if duplicate:
            duplicate_penalty = 0.35
        else:
            duplicate_penalty = 1.0

        if not quote_present or missing:
            reward = 0.0
        else:
            reward = duplicate_penalty * (
                0.25 * grounded
                + 0.20 * detail_score
                + 0.20 * specificity_score
                + 0.15 * severity_score
                + 0.10 * term_score
                + 0.10 * novelty_score
            )

        scored.append(
            {
                "index": idx,
                "reward": round(reward, 4),
                "quote_present": quote_present,
                "duplicate": duplicate,
                "missing": missing,
                "severity": severity,
            }
        )

    total = sum(item["reward"] for item in scored)
    # Precision-heavy cap: more than 6 findings must be exceptional, not padded.
    count_penalty = 1.0 if len(scored) <= 6 else 6 / len(scored)
    reward = min(total / 3.0, 1.0) * count_penalty
    unsupported = sum(1 for item in scored if not item["quote_present"] or item["missing"])
    if unsupported:
        reward *= max(0.0, 1.0 - 0.15 * unsupported)

    return {
        "reward": round(reward, 4),
        "score": round(reward, 4),
        "n_findings": len(scored),
        "unsupported": unsupported,
        "findings": scored,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--contract", required=True)
    parser.add_argument("--findings", required=True)
    args = parser.parse_args()

    contract = Path(args.contract).read_text()
    findings = _load_findings(Path(args.findings))
    print(json.dumps(score_findings(contract, findings), indent=2))


if __name__ == "__main__":
    main()
