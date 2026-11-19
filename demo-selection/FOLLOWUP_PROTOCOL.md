# Follow-up on demonstration selection and answer-position pooling

Finalised 19 November 2026 after a development-only diagnostic; source and manifest hashes are recorded before follow-up inference. This is a separate follow-up; it does not replace the original frozen study or its outcomes.

## Why a new experiment is needed

The original study's development and pool traces strongly preferred the A label. Its singleton success relation therefore supplied no informative test of semantic transfer or nontransitivity. Scoring an answer label immediately after the assistant header can reflect response-opening preference as well as task content. This follow-up changes the readout to an explicit assistant-side answer prefix and measures the remaining position effect with cyclic option permutations.

[FEEDER, ICML 2025](https://proceedings.mlr.press/v267/jin25i.html) motivates model-conditioned preselection. The [full paper](https://arxiv.org/html/2506.06033v1) distinguishes preselection of a training pool from downstream query-specific retrieval. The present greedy direct-coverage core remains a FEEDER-inspired alternative, not a reproduction of its tree. An improvement from the new readout must not be attributed to demonstration selection.

## Freeze order and permitted development

1. Finish the original frozen confirmation without changing it. Preserve all original traces and failed gates.
2. Use only separately identified development prompts to inspect tokenization and verify the explicit assistant prefix. Record these calls and all failures. Development is a diagnostic check, not a source of confirmation accuracy. Any further choice among prefixes must precede the final protocol/source freeze.
3. Freeze this final protocol, source, fresh 32-group manifest, checkpoint hashes, tokenizer, random seeds, selectors, inference settings, pooling rule and gates before running the follow-up pool or confirmation. Fresh confirmation outcomes may not be inspected during development.
4. Recompute the 240 distinct singleton demo/target comparisons with the new readout. Save a new trace; do not alter the old table. Deterministically recompute the eight-item model core and freeze its IDs and trace hash before the first fresh confirmation call. These calls are training-pool preprocessing, not test inference.
5. Run the frozen 32-group confirmation once. Do not change prompts, cores, gates or tie handling in response to its outcomes. An implementation correction requires a documented new freeze and new confirmation cohort if any outcomes have been inspected.

The generation seed, bootstrap seed, manifest and core receipts are part of the final freeze. A file labelled draft is not itself a preregistration receipt.

## Fresh task and independent groups

Generate 32 new coefficient vectors and questions using the clear signed-operation grammar. Exclude every coefficient vector and question text already used in the pool, development, original confirmation and the earlier counterfactual/exemplar studies where applicable. The implementation excludes exact vectors from the initial 92 unique programs and both earlier 64-case generator seeds; exclusion is exact-vector, not coefficient-multiset. Check and publish the disjointness results. If an older task family uses permutations of a coefficient multiset, state whether exclusion is exact-vector or multiset-level; do not silently describe the weaker exclusion as entirely novel semantic support.

Each new coefficient vector defines one independent semantic group, with one wording and four distinct candidate vectors. Exactly one candidate is the intended program; the other three are declared perturbations. Four domains, two held-out wording templates and four base correct-answer positions form a balanced 4 by 2 by 4 grid. This gives 32 groups and balances domain, wording and original answer position. Randomize coefficient and clause-order assignments within that design using the frozen seed. The deterministic grammar parser must recover every intended vector before inference.

The task remains bounded four-choice program recognition. It is neither unrestricted program generation nor a human reasoning benchmark. Four cyclic rotations of a question's candidates are repeated measurements of one semantic group, not four independent examples.

## Readout and pooling

Use the same installed local Qwen checkpoint and no-thinking chat template. All follow-up results use the standard public model wrapper on a complete prompt, batch one, rather than the original chunked/last-head shortcut. In the development diagnostic, the two paths had matching choices but a maximum four-label discrepancy of .75 logit on one representative prompt. That result is retained; numerical identity is not claimed. Full-forward scoring removes the shortcut from the follow-up readout definition. Render the user task normally, then append the explicit assistant-side prefix `Answer:` after the assistant generation header. Score the four leading-space continuations ` A`, ` B`, ` C`, ` D`. Record their tokenizer IDs and verify each is a single token under the frozen tokenizer. If this assumption fails, stop and document a new scoring definition before confirmation; do not mix single-token and sequence-likelihood arms.

For each prompt, convert the four label logits to float32, then normalize their conditional softmax in Python double precision for reproducible replay. This is normalization over allowed answers, not an estimate of total unrestricted-generation probability or calibrated correctness. A raw fixed-order prediction is the argmax of these probabilities.

Rotate only the target candidate list by offsets 0, 1, 2 and 3. Keep the target question, demonstration identities, demonstration order and demonstration answer labels unchanged. The same semantic candidate consequently occupies each answer position once. Map every rotation's four probabilities back to exact candidate-vector identity, then average its probabilities across the four rotations. Use mean probabilities, not mean logits, mean log probabilities or majority voting.

If the largest and second-largest pooled probabilities differ by at most `1e-6`, abstain and count that group as incorrect in the primary accuracy. Report abstention count and coverage separately. Do not break pooled ties by original candidate position, hidden correctness or access to the gold vector. In particular, an always-A readout must not become a claim of semantic recovery through arbitrary tie breaking.

Original single-order accuracy uses rotation zero and is secondary. It must not be substituted for the pooled primary after inspection. Record option-label frequencies, per-rotation accuracy, semantic agreement across rotations and pooled margins for diagnosis.

## Context selectors and controlled inputs

Retain all eight context arms from the original study:

- Zero demonstrations.
- Uniform two-example selection with seed 11.
- Uniform two-example selection with seed 29.
- Uniform two-example selection with seed 47.
- Full sixteen-example-pool TF-IDF cosine top-two retrieval.
- Full-pool maximum marginal relevance, relevance weight `.65`, two examples.
- Fixed text-diverse eight-example core with the same top-two retrieval.
- Recomputed model-conditioned eight-example core with the same top-two retrieval.

Selectors see only the visible query question text, never its intended vector, correct option or candidate correctness. Labelled pool outputs are permitted in demonstration rendering and offline model-conditioned preselection. Keep the existing deterministic selection rules and tie breaks, and freeze any replacement rules explicitly before follow-up inference. Each arm selects its context once per semantic group and reuses it across all four rotations. All nonzero arms use two demonstrations, a common prompt ceiling and the same readout. No question truncation or arbitrary token padding is allowed.

The singleton table uses the new assistant-prefix readout at the original pool candidate order. It is a declared training-pool selection diagnostic and is not assumed position-invariant. Record how many target labels and semantic target vectors drive its successful edges. If it is degenerate again, retain that result and the deterministic resulting core; do not search other cores using confirmation outcomes. Any successor using a rotationally pooled preselection table is a separately budgeted experiment.

The direct-coverage core still consists of eight pool examples. An example covers its own labelled pool item by identity for the coreset objective, not as a measured inference success. It does not rely on transitivity. If all successful edges are explained by a constant-label readout, or there are no nontrivial eligible chains, make no claim about sufficiency transitivity. Raw counts may be reported descriptively with the degenerate readout explained.

## Primary outcomes and frozen gates

Primary accuracy is exact intended-vector selection from the four-rotation mean-probability pool, counting an abstention as incorrect. Report one binary outcome per independent semantic group and all 32 groups in every arm.

The primary selection contrast is model-core pooled accuracy minus full-pool lexical pooled accuracy. A positive selection-efficacy claim requires all three conditions:

1. Mean accuracy gain is at least `.10` (ten percentage points).
2. The paired 95% percentile cluster-bootstrap lower bound is strictly above zero.
3. Total realized confirmation input tokens for model-core, including all four rotations, are at most `1.10` times the corresponding lexical total.

Use 20,000 paired bootstrap resamples of the 32 semantic groups with the frozen analysis seed. These intervals condition on this checkpoint, fixed demonstration pool, frozen preselection table and synthetic generator. Report integer successes, denominators, differences, intervals and gate components; do not suppress failed contrasts.

Report model-core against each of the seven other arms. A claim of superiority to all baselines requires at least `.05` gain against every comparator and every paired interval's lower bound above zero using Bonferroni-adjusted two-sided percentile intervals with per-comparison alpha `.05/7`. Passing only the lexical gate does not establish superiority over MMR, a random arm or the text-diverse core.

The secondary readout contrast is lexical pooled accuracy minus lexical rotation-zero accuracy, using the same paired group bootstrap. This measures a quality–cost trade-off. It is not a demonstration-selection effect. Report all per-arm pooled and rotation-zero accuracies and by-domain/template breakdowns as descriptive diagnostics. There is no post hoc switch of the primary endpoint to whichever diagnostic improves.

Chance-level reference accuracy is 25% for an independent uniform guess over four candidates. Report it as a reference, not as proof that any particular tied or deterministic heuristic must achieve exactly 25% on this finite cohort. Do not treat the four rotations as independent Bernoulli trials or combine old and fresh confirmation cohorts into a newly claimed held-out estimate.

## Compute, RAM and reproducibility

Confirmation requires `32 × 4 × 8 = 1,024` scored prefills. Recomputing the singleton table adds 240; diagnostic warmup/development calls are separately counted. State actual calls and tokens, including failed calls and offline selection. If the available runtime budget requires fewer arms or cases, amend and refreeze before confirmation; never silently omit a strong comparator after seeing outcomes.

The pooled readout evaluates four candidate orders, so its nominal query computation is approximately four times the single-order readout at a fixed context. Record actual input tokens and elapsed time. Do not claim a free accuracy improvement, end-to-end efficiency gain, or inference savings merely because the retrieval pool shrinks from sixteen to eight. Preselection cost and its potential amortization are separate from fixed-shot inference cost.

Use one local model process, no downloads, batch one, full-forward prompts limited to 1,500 tokens, a 128 MiB allocator-cache limit and the existing 2,560 MiB MLX allocation guideline with explicit checks. Record peak active MLX allocation and process maximum RSS separately; neither is a hard operating-system limit. Abort and retain failure evidence if the configured guideline is crossed. Save incremental progress and never overwrite completed traces.

Publish checkpoint/config/tokenizer hashes, source and protocol receipts, manifest and core receipts, selected demonstration IDs, per-rotation four-label logits/probabilities, token counts and timings. Model-independent replay must reconstruct candidate identities, pooling and abstention, selection decisions, disjointness, all exact accuracy counts, paired intervals and both efficacy gates. No proposed research direction, outcome of the earlier zero-accuracy generation experiment or original all-A result is counted as a fresh success here.
