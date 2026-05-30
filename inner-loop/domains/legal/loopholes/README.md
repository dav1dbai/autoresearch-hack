# Legal Loophole Domain

This is an open-ended verifier for grounded legal-risk findings. There is no
reference answer for the agent to reconstruct. The score rewards structured,
specific, source-grounded loophole or ambiguity findings and zeroes unsupported
claims before applying quality heuristics.

Submission format: `submission/findings.json`.

Run locally:

```bash
python inner-loop/domains/legal/loopholes/verify_findings.py \
  --contract inner-loop/domains/legal/loopholes/tasks/demo_contract.md \
  --findings runs/prime-research/legal-demo-loopholes/work/submission/findings.json
```

