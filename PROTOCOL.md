# Can routing rescue the recurrent Qwen math adapter?

Frozen before this joined-score audit. The four frozen Qwen3-1.7B prediction archives are from earlier experiments. Their separate aggregate scores and the 64 GSM8K and 24 generated linear-algebra evaluation cases were already seen. The new result is the within-question complementarity and a compute-allocation upper bound, not a fresh generalisation estimate or a new inference run.

## Question

The three-refinement tail has lower word-problem accuracy than stock Qwen with its existing worked-solution prompt. Does this reflect a bad common depth choice or failures shared by the archived adapter variants? If knowing the answer key cannot route around the failures, any input-only router restricted to these fixed candidates and this same decoding cannot. A larger search or different prompts could change the candidate set.

Primary endpoint: on the 64 archived GSM8K tasks, count how many questions could be answered by an oracle choosing among the one-pass 1,200-microstep adapter, the three-refinement 1,200-microstep adapter, and the one-pass 3,600-microstep training-compute control. Compare the exact union to the original stock Qwen count and report their difference. The arithmetic set of 24 generated algebra tasks is a separate comparison, never pooled into the headline. The existing stock arm uses a worked-solution prompt and the adapters emit programs. This is a portfolio ceiling on this fixture, not an architecture-only comparison.

## Analysis and controls

- Match by case ID, source and gold target across pinned archives. Recompute correctness with exact rational arithmetic and require agreement with the archived field. Keep failures and missing answers in the denominator.
- Alongside the unlimited correct-choice bound, require every question to run the one-pass adapter, give the oracle a second candidate on at most k problems, and let it retain the first answer. Compute the exact accuracy envelope for (a) the three-refinement action, (b) the 3,600-microstep one-pass control action, (c) either of those actions and (d) the native Qwen fallback. Charge one extra candidate per escalated question. This counts candidate calls and does not assert equal FLOPs, wall time, or peak memory across arms. Verify the formula against exhaustive subset search on a small fixture.
- For the equal-microstep pair, report fixes and breaks if all questions use three refinements, with an exact two-sided McNemar test as a paired finite-fixture check. Treat the test and all algebra results as descriptive because samples and the underlying task family were previously inspected and there is one training seed.
- Check proof validity separately. Report how many three-refinement GSM8K answers pass exact program verification while still missing the reference answer. Verification validates the emitted program's deduction, conditional on the program. It does not certify that the program encodes the question.
- Release only a row-indexed correctness/proof audit table and aggregate results. Publish no question text, model weights, credentials or generated answers. The replay from score archives verifies their hashes. The public replay from the compact table verifies the exact arithmetic and all summary metrics, independent of local model code.

Terminology correction after the first run: the archived run counters are microsteps with gradient accumulation, not optimizer steps. Endpoints and arms are unchanged.
