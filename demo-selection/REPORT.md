# Demonstration preselection and answer-position audit

Adel Mansour · 19 November 2026

This study tests whether model-conditioned demonstration preselection improves exact recognition of small signed programs. It also checks whether conditional success relations reflect informative examples or a biased answer readout. It follows Mengyue Yang and colleagues’ [FEEDER paper, ICML 2025](https://proceedings.mlr.press/v267/jin25i.html). The code is an inspired direct-coverage alternative, not a reproduction of FEEDER’s tree or a claim of a general reasoning breakthrough.

## Result that holds up

The predeclared demonstration-selection improvement gate **did not pass**. On 32 fresh program groups, pooled model-core accuracy was 81.25% versus 84.38% for full-pool lexical retrieval: -3.12 pp, 95% paired group-bootstrap interval [-15.62 pp, +6.25 pp]. There is no confirmed improvement from the proposed preselection method.

The first frozen readout selected A on every pool, development and confirmation prompt. All eight methods therefore scored 32/128 on its balanced confirmation cohort. The original result is retained, rather than replaced by the follow-up. A four-case development diagnostic isolated an explicit answer-prefix/label-token effect. The independently frozen follow-up uses the standard model wrapper and counterbalanced option pooling. Its results are not compared with the original generation study as a reasoning improvement.

## Fresh confirmation with strong baselines

Each method uses the same local Qwen3-1.7B four-bit checkpoint. The fresh cohort contains 32 disjoint coefficient vectors, balanced over four domains, two wordings and four base answer positions. Each target is scored at four cyclic candidate orders. Probabilities over the four allowed labels are mapped back to candidate identity and averaged; ties within 1e-6 abstain and count as wrong. Thus rotations are repeated measurements, not 128 independent cases.

| Method | Pooled exact choices /32 | Pooled accuracy | Fixed-order accuracy | Input tokens over four orders |
|---|---:|---:|---:|---:|
| Zero-shot | 21/32 | 65.62% | 43.75% | 19,984 |
| Random 11 | 25/32 | 78.12% | 59.38% | 50,612 |
| Random 29 | 24/32 | 75.00% | 56.25% | 50,556 |
| Random 47 | 25/32 | 78.12% | 53.12% | 50,616 |
| Full-pool lexical | 27/32 | 84.38% | 53.12% | 50,840 |
| Full-pool MMR | 26/32 | 81.25% | 50.00% | 50,588 |
| Text-diverse core | 24/32 | 75.00% | 62.50% | 50,464 |
| Model-conditioned core | 26/32 | 81.25% | 59.38% | 50,752 |

The primary contrast was fixed as model-core minus full-pool lexical. It required at least ten percentage points gain, a strictly positive paired 95% lower confidence bound and at most 10% extra realised input tokens. The observed token ratio was 0.9983. Passing a single comparison would not establish superiority over every baseline. The simultaneous seven-comparator gate **did not pass**.

| Model-core minus comparator | Difference | Paired 95% interval | Simultaneous 95% interval |
|---|---:|---|---|
| Zero-shot | +15.62 pp | [-3.12 pp, +34.38 pp] | [-9.38 pp, +40.62 pp] |
| Random 11 | +3.12 pp | [-9.38 pp, +15.62 pp] | [-15.62 pp, +21.88 pp] |
| Random 29 | +6.25 pp | [-9.38 pp, +21.88 pp] | [-12.50 pp, +28.12 pp] |
| Random 47 | +3.12 pp | [-6.25 pp, +12.50 pp] | [-12.50 pp, +18.75 pp] |
| Full-pool lexical | -3.12 pp | [-15.62 pp, +6.25 pp] | [-18.75 pp, +12.50 pp] |
| Full-pool MMR | +0.00 pp | [+0.00 pp, +0.00 pp] | [+0.00 pp, +0.00 pp] |
| Text-diverse core | +6.25 pp | [+0.00 pp, +15.62 pp] | [+0.00 pp, +18.75 pp] |

Some paired outcome vectors are identical on this finite cohort, producing a zero-width empirical-bootstrap interval. That does not establish population equivalence. A supplementary [exact empirical-bootstrap sensitivity check](bootstrap_sensitivity.json) computes the discrete resampling distribution without Monte Carlo. It shifts the primary lower endpoint from -15.625 to -12.5 points and the secondary lower endpoint from +9.375 to +12.5 points. The frozen 20,000-resample intervals remain the primary values; neither substantive verdict changes.

The secondary lexical pooling-minus-fixed-order difference was +31.25 pp, interval [+9.38 pp, +50.00 pp]. Pooling costs four scored orders rather than one. This is a readout quality–cost comparison, not a selection effect or free accuracy gain. A hand-coded parser of this declared synthetic grammar answers 32/32; it is a task ceiling, not a general natural-language solution.

## Conditional success and sufficiency checks

The corrected singleton table had 132 correct edges out of 240 and 192 failures of transitive inference among 894 distinct two-edge paths. These paths share cases and are not independent observations. The relation here is operational conditional answer correctness; it is not FEEDER’s complete causal sufficiency/necessity definition. In particular, a target may already be answered correctly without its demonstration.

The original all-A readout had 60 correct singleton edges, 168 two-edge paths and zero distinct-triple violations, although changing demonstrations never changed a predicted choice. Logits did change. [An exact negative control](negative_control.py) shows why transitivity alone can be vacuous: for a demo-insensitive predictor, conditional correctness depends only on the target; the reflexive completion of that relation is transitive at any accuracy. The checker exhausts all 511 binary correctness assignments up to eight targets. This is a small diagnostic argument, with no priority claim and no refutation of FEEDER.

The paper’s [Appendix A3 and A10.1](https://arxiv.org/html/2506.06033v1) motivate testing transitivity; its geographical example demonstrates one favourable instance. The present bounded program task examines a different setting. Any explored counterexample is conditional on this checkpoint, prompt, decoder and training pool.

### Exploratory witness check

The separately frozen [witness protocol](WITNESS_PROTOCOL.md) additionally requires zero-shot failures on both intermediate and final target before counting a canonical chain. Exact outcomes and option-position robustness are recorded in [witness results](witness_followup_results.json). This diagnostic was selected after inspecting the earlier pool relation; it does not supply an independent estimate of a population violation rate.

20 canonical triples passed the zero-shot filter. The first 12 were tested at two changed option orders, producing 24 dependent configurations. Only 1 configuration retained the pattern with both zero-shot controls wrong, and 0 triples survived both changed orders. No stable option-position-controlled witness is claimed. The diagnostic added 64 zero-shot and 46 deduplicated edge calls.

## Cost, memory and verification

The original study scored 240 singleton, 96 development and 1,024 confirmation prompts. The development readout diagnostic scored 32 prompts and two full-wrapper comparisons. The follow-up scored 240 new singleton and 1,024 fresh confirmation prompts. Checkpoint loading and warmup checks are additional work; phase timing excludes them. Offline preselection tokens and elapsed time remain in the raw traces. Halving the retrieval pool does not halve prompt inference, because every nonzero context still contains two demonstrations. No end-to-end efficiency gain is claimed.

The maximum observed MLX allocation across these recorded phases and the diagnostic was 1.51 GiB. Model runs were sequential, batch one, with a 128 MiB allocator cache and a 2.5 GiB allocation guideline plus observed-allocation abort checks. MLX allocation and process RSS are distinct; neither is total system RAM. Existing local weights were used without downloads.

The first warmup’s overly strict numerical-identity tolerance failed before any experiment outcome. That failure and the corrected freeze are preserved. Representative development comparisons had the same decisions but up to .75 logit differences between chunked and full-forward scoring, so the follow-up uses the standard full wrapper. It does not claim the two implementations are numerically identical.

The frozen follow-up CLI’s direct Python-object comparison has a JSON integer-key/string-key mismatch in raw position counts. The separate verify_followup.py wrapper normalises JSON keys before exact comparison. Frozen inference source and all reported outcomes are unchanged. Use that documented wrapper for replay.

The independent auditors verify hashes, exact grammar, split disjointness, domain/label counterbalance, gold-independent selection, raw-logit argmax, cyclic pooling, abstentions and reported counts. They include 1,007 independent pooling fixtures. Bootstrap intervals are conditional on the fixed checkpoint, training pool, coreset and synthetic generator. No additional training seeds, natural benchmark replication or human comparison were run.

## Reproduce without loading a model

```sh
python demo-selection/study.py analyse --verify demo-selection/results.json
python demo-selection/independent_audit.py
python demo-selection/negative_control.py
python demo-selection/verify_followup.py
python demo-selection/followup_audit.py
python demo-selection/witness_followup.py analyse --verify demo-selection/witness_followup_results.json
python demo-selection/bootstrap_sensitivity.py
```

Inference instructions and exact guards are in [the first protocol](PROTOCOL.md) and [the follow-up protocol](FOLLOWUP_PROTOCOL.md). Completed traces are never overwritten. Synthetic questions, candidate vectors, raw four-label logits, selected demonstration IDs, hashes and costs are public; no model weights or personal data are included.
