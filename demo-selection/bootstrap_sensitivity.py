"""Supplementary exact empirical-bootstrap sensitivity, without model inference.

The frozen study's primary intervals remain its predeclared 20,000-draw Monte
Carlo percentile intervals. This separate audit integrates that empirical
resampling distribution exactly using integer polynomial coefficients. It
does not replace the registered estimator or select a new reporting endpoint.
"""
from collections import Counter, defaultdict
from hashlib import sha256
import json
import math
from pathlib import Path

ROOT = Path(__file__).resolve().parent


def read(name):
    return json.loads((ROOT / name).read_text())


def pooled_prediction(rows):
    """Independent mapping via explicit original candidate identities."""
    original = ("first", "second", "third", "fourth")
    totals = {identity: 0.0 for identity in original}
    for row in rows:
        rotation = row["rotation"]
        ordering = original[rotation:] + original[:rotation]
        logits = row["label_logits"]
        shifted = [math.exp(value - max(logits)) for value in logits]
        denominator = sum(shifted)
        for identity, weight in zip(ordering, shifted, strict=True):
            totals[identity] += weight / denominator / 4
    ordered = sorted(original, key=totals.get, reverse=True)
    if totals[ordered[0]] - totals[ordered[1]] <= 1e-6:
        return None
    return original.index(ordered[0])


def exact_resampling(differences):
    """Integer convolution of (-1,0,1) outcome frequencies, raised to power n."""
    n = len(differences)
    assert n == 32 and set(differences) <= {-1, 0, 1}
    counts = Counter(differences)
    coefficients = {0: 1}
    for _ in range(n):
        next_coefficients = defaultdict(int)
        for existing_sum, multiplicity in coefficients.items():
            for outcome, frequency in counts.items():
                next_coefficients[existing_sum + outcome] += multiplicity * frequency
        coefficients = dict(next_coefficients)
    denominator = n ** n
    assert sum(coefficients.values()) == denominator

    def quantile(numerator, level_denominator):
        cumulative = 0
        for score_sum, multiplicity in sorted(coefficients.items()):
            before = cumulative
            cumulative += multiplicity
            # Exact rational comparison, including discrete CDF boundaries.
            if cumulative * level_denominator >= numerator * denominator:
                return {"value": score_sum / n, "sum": score_sum,
                        "level_fraction": [numerator, level_denominator],
                        "cdf_immediately_before": before / denominator,
                        "cdf_at_value": cumulative / denominator}
        raise AssertionError("Quantile not found")

    q95 = [quantile(1, 40), quantile(39, 40)]
    q7 = [quantile(1, 280), quantile(279, 280)]
    return {"groups": n, "paired_difference_counts": {str(k): counts[k] for k in (-1, 0, 1)},
            "mean_difference": sum(differences) / n,
            "exact_95_interval": [item["value"] for item in q95],
            "exact_7_comparison_bonferroni_interval": [item["value"] for item in q7],
            "quantile_details_95": q95, "quantile_details_7_comparison": q7,
            "integer_probability_denominator": str(denominator),
            "integration": "Integer polynomial convolution; no Monte Carlo sampling."}


def analyse():
    cases = read("followup_manifest.json")
    trace = read("followup_test_trace.json")
    registered = read("followup_results.json")
    assert len(cases) == 32
    ids = [case["id"] for case in cases]
    gold = {case["id"]: case["options"].index(case["coefs"]) for case in cases}
    grouped = defaultdict(list)
    for row in trace["rows"]:
        assert row["prediction"] == max(range(4), key=lambda i: (row["label_logits"][i], -i))
        grouped[row["id"], row["arm"]].append(row)
    assert len(grouped) == 32 * 8 and len(trace["rows"]) == 1024
    arms = list(registered["accuracy"])
    outcomes = {arm: {} for arm in arms}
    fixed = {arm: {} for arm in arms}
    for (case_id, arm), rows in grouped.items():
        assert len(rows) == 4 and {row["rotation"] for row in rows} == {0, 1, 2, 3}
        outcomes[arm][case_id] = int(pooled_prediction(rows) == gold[case_id])
        fixed[arm][case_id] = int(next(row for row in rows if row["rotation"] == 0)["prediction"] == gold[case_id])
    comparisons = {}
    for arm in arms:
        assert sum(outcomes[arm].values()) / 32 == registered["accuracy"][arm]
        if arm == "model_core":
            continue
        differences = [outcomes["model_core"][i] - outcomes[arm][i] for i in ids]
        result = exact_resampling(differences)
        original = registered["comparisons"][arm]
        assert result["mean_difference"] == original["difference"]
        result["registered_mc_95_interval"] = original["paired_group_95_interval"]
        result["registered_mc_simultaneous_interval"] = original["simultaneous_95_interval"]
        result["difference_exact_minus_registered_95_endpoints"] = [
            a - b for a, b in zip(result["exact_95_interval"], original["paired_group_95_interval"])]
        comparisons[arm] = result
    secondary_differences = [outcomes["lexical"][i] - fixed["lexical"][i] for i in ids]
    secondary = exact_resampling(secondary_differences)
    secondary["registered_mc_95_interval"] = registered["secondary_pooling_vs_fixed_lexical"]["paired_group_95_interval"]
    assert secondary["mean_difference"] == registered["secondary_pooling_vs_fixed_lexical"]["difference"]
    primary = comparisons["lexical"]
    ratio = registered["primary_gate"]["token_ratio"]
    exact_primary_gate = primary["mean_difference"] >= .10 and primary["exact_95_interval"][0] > 0 and ratio <= 1.10
    assert exact_primary_gate is False and registered["primary_gate"]["passed"] is False
    assert secondary["exact_95_interval"][0] > 0 and secondary["registered_mc_95_interval"][0] > 0
    inputs = ("followup_manifest.json", "followup_test_trace.json", "followup_results.json")
    return {"scope": "Supplementary sensitivity audit. Registered Monte Carlo intervals remain the primary reported analysis.",
            "source_sha256": sha256(Path(__file__).read_bytes()).hexdigest(),
            "input_sha256": {name: sha256((ROOT / name).read_bytes()).hexdigest() for name in inputs},
            "resampling_units": "32 independent semantic groups; rotations are repeated measurements.",
            "registered_monte_carlo_draws": 20000,
            "finite_accuracy_grid_step": 1 / 32,
            "model_core_minus_baselines": comparisons,
            "secondary_lexical_pooling_minus_fixed": secondary,
            "claim_verdict": {"registered_primary_gate_passed": False,
                               "exact_bootstrap_primary_gate_passed": False,
                               "secondary_interval_positive_under_both": True,
                               "secondary_cost": "Four scored option orders versus one; not a compute-matched selection improvement."},
            "interpretation": "On this discrete 1/32 accuracy grid, Monte Carlo percentile estimates can cross a neighboring CDF step. The selection-improvement gate still fails; the secondary pooling contrast remains positive under both calculations. Neither observation establishes a general reasoning breakthrough."}


if __name__ == "__main__":
    result = analyse()
    (ROOT / "bootstrap_sensitivity.json").write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps({"primary": result["model_core_minus_baselines"]["lexical"],
                      "secondary": result["secondary_lexical_pooling_minus_fixed"],
                      "verdict": result["claim_verdict"]}, indent=2))
