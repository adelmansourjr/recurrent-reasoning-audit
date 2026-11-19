# Baseline conditioned singleton chain diagnostic

This is a separate exploratory diagnostic specified after the answer-prefixed singleton table became available. It uses only the sixteen-item labelled training pool and does not select cases or methods using fresh confirmation outcomes. It is not a population hypothesis test, a reproduction of the FEEDER tree or a refutation of that paper. The observed relation is exact correctness after one demonstration under the declared model, target options and readout.

## Freeze and execution order

Run `witness_followup.py freeze` once after the answer-prefixed preselection table is complete and before any diagnostic baseline or witness inference. The receipt records this protocol and helper source hashes, original and follow-up freeze hashes, the completed follow-up preselection trace hash, and the fixed pool manifest hash. The original frozen study and follow-up confirmation files are never edited by this helper. A later change requires an explicitly separate diagnostic and must retain existing artifacts.

Run this diagnostic only after the primary confirmation model process finishes. Reuse `selection_followup.PrefixedScorer`: the pinned local checkpoint, standard full-forward wrapper, explicit assistant `Answer:` prefix and single-token leading-space answer labels. Use batch one, the same 1,500-token prompt ceiling, 128 MiB allocator-cache guideline and explicit observed 2,560 MiB peak-allocation abort. These are allocation checks, not an operating-system RAM guarantee. No model or dataset downloads are permitted. Save every raw readout, input-token count, elapsed time and process/MLX memory metadata.

## Baseline and deterministic witness selection

First score all sixteen pool questions without demonstrations at all four target-option cyclic rotations: exactly 64 calls. Demonstrations are absent; only the target candidate order changes. The four measurements are repeated presentations of a target, not independent examples.

Read the already completed canonical singleton matrix, whose entry `matrix[a][b]` means that showing pool example a yielded a correct choice on target b. Enumerate all distinct triples `(a,b,c)` in ascending lexicographic order for which a-to-b and b-to-c are correct, while a-to-c is incorrect. Do not include assumed self-edges.

A triple is eligible only when canonical zero-shot predictions on both b and c are incorrect. Thus each positive edge changes the outcome relative to the canonical no-demonstration baseline; an already solved target cannot supply a vacuous positive edge. Preserve the count and complete ordered list of eligible triples, then take the first twelve or all if fewer exist. Do not rank by margin, ease, apparent importance or desired outcome. If no eligible triples exist, report that and perform no rotated edge calls.

Before scoring any rotated edge, save a selection receipt containing the freeze, canonical raw-triple count, complete eligible list, chosen prefix, hash of all zero-shot rows and the exact deduplicated edge-query list.

## Robustness checks

For every selected triple, score its three directed edges at target-option rotations one and two. Keep target wording, candidate-vector identities, demonstration identity, demonstration order, demonstration candidate ordering and demonstration answer label fixed. Change only the target option order. Deduplicate queries sharing `(demo,target,rotation)` and reuse that single recorded readout transparently. The maximum extra calls are twelve times three edges times two rotations, or 72; deduplication may reduce this.

For each triple and rotation, report the correctness of all three edges and both already measured zero-shot controls separately. A retained nontransitive pattern requires a-to-b and b-to-c correct with a-to-c incorrect. **Strict survival** additionally requires zero-shot b and c both remain incorrect at that same rotation. Report both pattern retention and strict survival, every selected configuration including failures, and how many triples strictly survive both rotations. Never count a baseline-correct rotated target as a strict successful witness.

## Reporting and limits

Report canonical raw and baseline-conditioned eligible counts, selected triples, unique scored edges, all raw logits, option labels and exact target correctness; provide zero-shot accuracy and label counts by rotation, all costs, memory metadata and source hashes. Offline replay must reconstruct lexicographic selection, deduplication, label-to-vector correctness and every retained/failed configuration. No existing trace or result is overwritten. Progress remains available after failure.

Selected triples share examples and edges; rotations share target semantics. Counts are descriptive and no binomial interval, independence assumption, population transitivity estimate or selection-efficacy claim is justified. Even a surviving witness applies only to this checkpoint, bounded recognition task, prompt and operational success relation. It can motivate direct verification of inferred sufficiency links; it does not establish that FEEDER is generally incorrect, nor that a replacement improves held-out accuracy. A null or fragile result is retained without changing witness-selection rules.

Commands, executed sequentially after other model work ends:

```sh
python witness_followup.py freeze
python witness_followup.py run /absolute/path/to/existing/Qwen3-1.7B-4bit
python witness_followup.py analyse
python witness_followup.py analyse --verify witness_followup_results.json
```
