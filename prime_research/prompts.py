from __future__ import annotations

from pathlib import Path

from .contracts import PreparedTask


COMMON = """You are running an iterative autoresearch loop on one open-ended task.

Your job is not to answer from a reference solution. There is no target transcript
to imitate. Improve the artifact in ./submission by repeatedly proposing a change,
running the verifier command, reading the scalar reward, and keeping improvements.

Rules:
- Edit only files under ./submission unless creating your own notes under ./notes.
- Do not modify verifier files, task files, score commands, or protected inputs.
- Treat the verifier as the only source of reward.
- After each verifier run, append one JSON line to ./experiments.jsonl with:
  {"iter": <int>, "reward": <float>, "summary": "...", "score": {...}}
- Keep a short running research log in ./notes/research.md.
- Stop when the budget is exhausted or further changes are unlikely to help.
"""


def kernel_prompt(prepared: PreparedTask) -> str:
    artifact = prepared.artifact_path.relative_to(prepared.workdir)
    task_file = Path("task.py")
    return f"""{COMMON}

Domain: GPU KernelBench-style kernel optimization.

Visible task file: ./{task_file}
Submission artifact: ./{artifact}
Verifier command:
  {' '.join(prepared.score_cmd)}

Goal:
- Implement `ModelNew` as a drop-in replacement for the reference `Model`.
- Correctness is a hard gate. Speed reward only matters after all correctness
  checks pass.
- Optimize hidden-style randomized inputs and avoid brittle hardcoding.

Research loop:
1. Inspect task.py and the current submission.
2. Make one focused implementation or optimization change.
3. Run the verifier command.
4. If correctness fails, fix correctness before optimizing speed.
5. If correctness passes, improve speedup while preserving correctness.

Anti-hacking constraints:
- Do not edit task.py or verifier files.
- Do not fall back to the reference model for core computation.
- Do not cache by input pointer or assume a single fixed input identity.
- Do not hide work on non-default streams to beat the timer.
"""


def legal_prompt(prepared: PreparedTask) -> str:
    artifact = prepared.artifact_path.relative_to(prepared.workdir)
    return f"""{COMMON}

Domain: legal loophole / ambiguity discovery.

Visible source document: ./contract.md
Submission artifact: ./{artifact}
Verifier command:
  {' '.join(prepared.score_cmd)}

Goal:
- Produce high-quality, grounded legal-risk findings in JSON.
- Each finding must quote exact text from the source document. Unsupported claims
  get zeroed before quality scoring.
- This is open-ended: the reward comes from grounded novelty, specificity,
  exploitability, and severity, not matching a reference answer.

Required JSON shape:
[
  {{
    "quoted_clause_text": "exact source text",
    "location": "section or paragraph",
    "loophole_description": "specific ambiguity or exploit",
    "severity_1to5": 1,
    "exploitation_scenario": "concrete parties, dates or amounts, and outcome"
  }}
]

Research loop:
1. Read the contract and current findings.
2. Add, remove, or sharpen one finding.
3. Run the verifier command.
4. Keep changes that improve grounded score and remove unsupported findings.

Anti-hacking constraints:
- Do not invent clause text.
- Do not pad with repetitive findings.
- Do not use generic legal advice; every finding must attach to source text.
"""

