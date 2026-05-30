# Kernel / Legal Autoresearch Program v0

Run a tight experiment loop. Keep the best artifact by verifier reward, not by
plausibility.

For every iteration:

1. State one hypothesis for why reward can improve.
2. Make the smallest artifact change that tests that hypothesis.
3. Run the verifier command exactly as written in `program.md`.
4. Record the reward and the important score fields in `experiments.jsonl`.
5. If reward improves, keep the change and explain why it likely worked.
6. If reward regresses or correctness/grounding fails, revert or repair before
   trying unrelated ideas.

Kernel-specific priorities:

- First make `ModelNew` correct for the task.
- Then remove obvious PyTorch fallback overhead, improve memory locality, and use
  Triton where it is simpler than custom CUDA.
- Prefer robust shape/general-input handling over brittle constants.

Legal-specific priorities:

- First satisfy the quote-grounding gate.
- Then improve specificity: named parties, timing, money/thresholds, and a clear
  exploit path.
- Remove generic or duplicative findings even if they sound plausible.

