"""A model-free negative control for a conditional-correctness relation.

If a choice predictor f ignores demonstration identity, let c[j] indicate that
f(x_j) is correct. On distinct vertices R(i,j)=c[j]. Therefore, on a distinct
triple, R(a,b) and R(b,c) imply c[c], hence R(a,c). This says nothing about
accuracy: any binary vector c, including all failures, has this property.

The reflexive completion S(i,j)=(i==j) or c[j] is transitive on all triples.
If either adjacent pair repeats a vertex the implication is immediate;
otherwise the second edge establishes c[c], or the conclusion is reflexive.
An off-diagonal graph with false self-edges is NOT necessarily transitive on
all triples: two successful targets can make a two-cycle with no self-edge.

This is a finite diagnostic of the study's operational relation, not a claim
about FEEDER's complete causal sufficiency/necessity definitions, a refutation
of its experiments, or a novelty claim. Observed choice invariance is limited
to the saved prompts and is weaker than invariance of the four score logits.
"""
import argparse
from collections import Counter, defaultdict
from itertools import product
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parent


def exhaustive_check(max_vertices=8):
    assignments = distinct_triples = reflexive_triples = 0
    vacuous_assignments = 0
    for n in range(max_vertices + 1):
        for correct in product((False, True), repeat=n):
            assignments += 1
            if not any(correct):
                vacuous_assignments += 1
            for a, b, c in product(range(n), repeat=3):
                if len({a, b, c}) == 3:
                    distinct_triples += 1
                    assert not (correct[b] and correct[c]) or correct[c]
                ab = a == b or correct[b]
                bc = b == c or correct[c]
                ac = a == c or correct[c]
                reflexive_triples += 1
                assert not (ab and bc) or ac
    # A counterexample to calling the false-diagonal graph fully transitive.
    off_diagonal = [[False, True], [True, False]]
    assert off_diagonal[0][1] and off_diagonal[1][0] and not off_diagonal[0][0]
    return {
        "max_vertices": max_vertices,
        "binary_correctness_assignments": assignments,
        "distinct_triples_checked": distinct_triples,
        "reflexive_completion_triples_checked": reflexive_triples,
        "all_false_assignments_included": vacuous_assignments,
        "distinct_triple_violations": 0,
        "reflexive_completion_violations": 0,
        "false_diagonal_two_cycle_counterexample_checked": True,
    }


def valid_prediction(row):
    logits = row["label_logits"]
    assert len(logits) == 4 and all(x == x and abs(x) != float("inf") for x in logits)
    expected = max(range(4), key=lambda i: (logits[i], -i))
    assert row["prediction"] == expected
    return expected


def load(path):
    return json.loads(path.read_text())


def pool_control(directory, manifest):
    path = directory / "preselect_trace.json"
    if not path.exists():
        return {"available": False}
    trace = load(path)
    pool = manifest["pool"]
    n = len(pool)
    rows = trace["rows"]
    assert len(rows) == n * (n - 1)
    assert {(r["demo"], r["target"]) for r in rows} == {
        (i, j) for i in range(n) for j in range(n) if i != j
    }
    targets = defaultdict(list)
    histogram = Counter()
    matrix = [[False] * n for _ in range(n)]
    for row in rows:
        prediction = valid_prediction(row)
        histogram[prediction] += 1
        targets[row["target"]].append(row)
        gold = pool[row["target"]]["options"].index(pool[row["target"]]["coefs"])
        matrix[row["demo"]][row["target"]] = prediction == gold
    assert trace["matrix"] == matrix
    invariant = [j for j in range(n) if len({r["prediction"] for r in targets[j]}) == 1]
    column_constant = [j for j in range(n) if len({matrix[i][j] for i in range(n) if i != j}) == 1]
    raw_logit_spreads = []
    for j in range(n):
        raw_logit_spreads.append(max(max(r["label_logits"][label] for r in targets[j])
                                    - min(r["label_logits"][label] for r in targets[j])
                                    for label in range(4)))
    paths = violations = 0
    for a, b, c in product(range(n), repeat=3):
        if len({a, b, c}) != 3:
            continue
        if matrix[a][b] and matrix[b][c]:
            paths += 1
            violations += not matrix[a][c]
    if len(invariant) == n:
        assert len(column_constant) == n and violations == 0
    return {
        "available": True,
        "singleton_calls": len(rows),
        "choice_histogram_A_B_C_D": [histogram[i] for i in range(4)],
        "targets_with_choice_invariant_to_observed_demo": len(invariant),
        "targets": n,
        "all_observed_choices_demo_invariant": len(invariant) == n,
        "targets_with_constant_off_diagonal_correctness_column": len(column_constant),
        "correctness_relation_direct_edges": sum(map(sum, matrix)),
        "distinct_two_edge_paths": paths,
        "distinct_triple_violations": violations,
        "maximum_observed_within_target_logit_spread": max(raw_logit_spreads, default=0),
        "pool_zero_shot_measured": False,
        "scope": "Invariance across the observed 15 singleton demonstrations per pool target; no zero-shot pool comparison was collected.",
    }


def phase_control(directory, manifest, phase):
    path = directory / f"{phase}_trace.json"
    if not path.exists():
        return {"available": False}
    trace = load(path)
    cases = manifest[phase]
    lookup = {r["id"]: r for r in cases}
    rows = trace["rows"]
    arms = sorted({r["arm"] for r in rows})
    assert "zero" in arms
    by_case = defaultdict(dict)
    histograms = {arm: Counter() for arm in arms}
    for row in rows:
        prediction = valid_prediction(row)
        assert row["id"] in lookup and row["arm"] not in by_case[row["id"]]
        by_case[row["id"]][row["arm"]] = row
        histograms[row["arm"]][prediction] += 1
    assert set(by_case) == set(lookup)
    assert all(set(data) == set(arms) for data in by_case.values())
    assert len(rows) == len(cases) * len(arms)
    comparisons = {}
    for arm in arms:
        if arm == "zero":
            continue
        changed = improved = harmed = same_choice = 0
        max_score_delta = 0.0
        for case_id, data in by_case.items():
            target = lookup[case_id]
            gold = target["options"].index(target["coefs"])
            zero, other = data["zero"], data[arm]
            changed += zero["prediction"] != other["prediction"]
            same_choice += zero["prediction"] == other["prediction"]
            improved += zero["prediction"] != gold and other["prediction"] == gold
            harmed += zero["prediction"] == gold and other["prediction"] != gold
            max_score_delta = max(max_score_delta, max(abs(x - y) for x, y in
                                  zip(zero["label_logits"], other["label_logits"], strict=True)))
        comparisons[arm] = {"changed_choices": changed, "unchanged_choices": same_choice,
                            "improved_exact_choices": improved, "harmed_exact_choices": harmed,
                            "max_abs_four_logit_difference_vs_zero": max_score_delta}
    exact_counts = {}
    for arm in arms:
        exact_counts[arm] = sum(data[arm]["prediction"] == lookup[case_id]["options"].index(
            lookup[case_id]["coefs"]) for case_id, data in by_case.items())
    return {"available": True, "cases": len(cases), "arms": arms,
            "choice_histograms_A_B_C_D": {a: [histograms[a][i] for i in range(4)] for a in arms},
            "correct_counts": exact_counts, "comparisons_to_zero": comparisons,
            "all_few_shot_choices_equal_zero": all(c["changed_choices"] == 0 for c in comparisons.values())}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--directory", type=Path, default=ROOT)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    manifest = load(args.directory / "manifest.json")
    result = {"theorem_check": exhaustive_check(),
              "pool": pool_control(args.directory, manifest),
              "phases": {phase: phase_control(args.directory, manifest, phase)
                         for phase in ("dev", "test")},
              "interpretation": "A transitive observed correctness relation and unchanged accuracy under a smaller demonstration pool can be vacuous when predictions ignore demonstrations. This diagnostic is not FEEDER's full causal sufficiency/necessity relation and does not refute the paper. Argmax invariance is not logit invariance."}
    text = json.dumps(result, indent=2) + "\n"
    if args.output:
        args.output.write_text(text)
    print(text, end="")


if __name__ == "__main__":
    main()
