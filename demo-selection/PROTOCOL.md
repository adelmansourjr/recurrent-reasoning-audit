# Demonstration preselection and direct sufficiency checks

Frozen before model inference on 19 November 2026. The question is whether an eight-example, model-conditioned preselection pool improves bounded program recognition over strong selectors using the entire sixteen-example pool.

## Connection to the paper

[FEEDER, ICML 2025](https://proceedings.mlr.press/v267/jin25i.html) motivates model-conditioned preselection. [Appendix A3](https://arxiv.org/html/2506.06033v1) assumes transitive sufficiency when merging a tree. Here a complete directed singleton table measures that relation in a small, executable domain. Greedy direct-coverage preselection is an inspired alternative, not a reproduction of FEEDER's tree or a general refutation of its empirical findings. The paper itself provides a supportive case study for transitivity; testing broader validity is the follow-up question.

## Task and splits

The new generator uses clear signed-operation clauses and four candidate coefficient vectors. Its sixteen labelled pool programs, twelve development programs and sixty-four confirmation programs are disjoint coefficient vectors. Confirmation has two previously unused wording templates per program, giving 128 cases in 64 independent semantic groups. All splits are generated and hashed before any new model result. This is a bounded four-choice semantic-discrimination task. It is not free-form program generation, and its accuracy must never be compared with the earlier zero-accuracy generation experiment as an inference improvement.

Domain labels are shared across splits. Wordings and coefficient vectors differ; operations and grammar are synthetic and known. This is compositional interpolation with held-out wording, not an unknown real-world task. Numeric source values are archived but not shown to the model; intended coefficient vectors determine correctness. The candidate list always contains the correct vector and three distinct perturbations. Answer positions are counterbalanced independently within every domain in the pool and confirmation cohorts, so domain alone does not predict the answer label. Development positions are shuffled independently. A pre-inference audit caught and removed an initial domain/label shortcut before any model result was collected. A declared grammar parser is a symbolic ceiling and receives no learned inference budget.

## Selectors and common inference

All arms use the same frozen local Qwen3-1.7B 4-bit checkpoint, chat template, no-thinking setting and next-token A/B/C/D logit scoring. Constrained recognition removes generation-format compliance as a confound; all predictions map to a candidate program. Last-position head scoring is checked against the standard wrapper. A longer warmup checks chunked-cache versus full-forward label scores and predictions. BF16 matrix-kernel shapes can change raw logits slightly: verification requires matching label argmax and maximum four-label discrepancy below .5 logit, while recording the measured differences. An initial .01 all-vocabulary tolerance failed at .1875 during warmup, before any pool, development or confirmation result; it was replaced by this declared label-level numerical check and refrozen. This is warmup agreement, not a proof of numerical identity for every input. Each prompt has at most 1,500 tokens. No arbitrary padding or truncation is used.

- Zero demonstrations.
- Three independently seeded uniform two-example selectors, seeds 11, 29 and 47.
- Full-pool TF-IDF cosine top-two retrieval.
- Full-pool maximum marginal relevance, relevance weight .65, two examples.
- Fixed text-diverse eight-example coreset, followed by identical TF-IDF top-two retrieval.
- Model-conditioned eight-example coreset, followed by identical TF-IDF top-two retrieval.

The complete 240-call table evaluates every distinct singleton demo/target pair in the pool. Direct successful targets define each example's coverage; a retained example covers its own labelled record by identity, not by a measured inference claim. The proposed core greedily adds the largest number of newly covered pool targets, breaking ties by total direct coverage and fixed ID; it always contains eight examples. It does not assume transitivity. No test gold vector, candidate correctness, hidden reference answer or program parser is available to selectors. Query retrieval uses the visible question text only. Gold pool labels are allowed for model-conditioned preselection and demonstration rendering. Development results are reported without tuning any source, prompt, selector or gate; final confirmation is a fresh, frozen measurement.

All nonzero arms use exactly two examples and the same maximum token budget. Realised input tokens are reported per arm; primary improvement is accepted only if the model-core arm uses at most 10% extra total tokens relative to the lexical comparator. Selection search-size reduction from sixteen to eight is not an end-to-end compute claim. All 240 preprocessing calls and their tokens are separately charged. No inference saving from fewer prompt examples is claimed because shot count is fixed.

## Outcomes and fixed gates

Primary accuracy is selection of the exact intended coefficient vector across all confirmation cases. Primary contrast is model-core minus full-pool lexical retrieval. A positive efficacy claim requires at least ten percentage points gain, the paired 95% cluster-bootstrap lower bound above zero, and the realised token criterion. Two paraphrases of one program are averaged as one cluster. Twenty thousand resamples and seed 53081 are fixed. The bootstrap is conditional on this deterministic checkpoint, selection table and synthetic generator.

All seven comparisons are reported. A stronger claim of superiority over every baseline requires at least five points gain and simultaneous 95% intervals above zero using Bonferroni-adjusted cluster-bootstrap percentiles. No such claim is made merely because one weak baseline loses. By-template results, true outcome counts, selector timings and all arms are retained.

The paper-specific diagnostic enumerates every distinct triple with A→B and B→C but not A→C in the observed singleton table. These dependent triples are descriptive counts, not independent trials. For the first twelve witnesses in fixed lexicographic order, rerun their three edges after each of two cyclic rotations of target candidate positions. Report how many configurations retain the violation. This diagnostic is explicitly selected after inspection of the pool table, uses no confirmation labels and is not a new held-out accuracy claim. No pooled transitivity rate or universal counterexample beyond the measured model/prompt setup is inferred.

## RAM and reproducibility

Use the installed checkpoint only. One model process at a time, batch one, 128-token prefill chunks, 128 MiB allocator cache and a 2,560 MiB MLX allocation guideline, plus an explicit peak-allocation abort check after each prefill chunk and scoring call. The guideline is not an operating-system hard limit; transient allocation can precede the check. Record process maximum RSS and MLX peak allocation; the latter is not total system RAM. No downloads or additional model loads. Save progress every eight target cases or one singleton-demo row. Completed traces are not overwritten; failures remain visible. Model-independent replay reconstructs data and selector choices, verifies hashes, exact outcomes, clustered intervals and gates.
