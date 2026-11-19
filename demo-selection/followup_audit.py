"""Independent model-free checks of the separately frozen follow-up."""
from collections import Counter, defaultdict
from itertools import product
import importlib.util
import json
import math
from pathlib import Path
import random
import sys

sys.dont_write_bytecode = True
ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))
import selection_followup as followup
import study
from independent_audit import independent_parse, independent_core, check_prediction


def read(path):
    return json.loads(path.read_text())


def independent_pool(rows):
    """Map through explicit symbolic candidate identities, not modular arithmetic."""
    original = ("candidate-one", "candidate-two", "candidate-three", "candidate-four")
    values = defaultdict(float)
    for row in rows:
        rotation = row["rotation"]
        rotated = original[rotation:] + original[:rotation]
        logits = row["label_logits"]
        shift = max(logits)
        weights = [math.exp(x - shift) for x in logits]
        denominator = sum(weights)
        for identity, weight in zip(rotated, weights, strict=True):
            values[identity] += weight / denominator / 4
    probabilities = [values[identity] for identity in original]
    descending = sorted(probabilities, reverse=True)
    margin = descending[0] - descending[1]
    prediction = None if margin <= 1e-6 else probabilities.index(descending[0])
    return probabilities, prediction, margin


def pooling_fixtures():
    tested = 0
    # Constant answer-position preference must pool to abstention, even if A is gold.
    for logits in ([20, -20, -20, -20], [0, 0, 0, 0], [8, 2, -3, 1]):
        rows = [{"rotation": r, "label_logits": list(logits)} for r in range(4)]
        result = followup.pool(rows)
        assert result["prediction"] is None and result["margin"] <= 1e-6
        tested += 1
    # A consistent semantic candidate occupies each answer position in turn.
    for correct in range(4):
        rows = []
        for rotation in range(4):
            identities = list(range(4))
            rotated = identities[rotation:] + identities[:rotation]
            logits = [12.0 if identity == correct else -12.0 for identity in rotated]
            rows.append({"rotation": rotation, "label_logits": logits})
        result = followup.pool(rows)
        assert result["prediction"] == correct
        tested += 1
    rng = random.Random(188013)
    for _ in range(1000):
        rows = [{"rotation": r, "label_logits": [rng.uniform(-30, 30) for _ in range(4)]}
                for r in range(4)]
        expected, choice, margin = independent_pool(rows)
        actual = followup.pool(rows)
        assert max(abs(a - b) for a, b in zip(expected, actual["probabilities"])) < 1e-14
        assert actual["prediction"] == choice
        assert abs(actual["margin"] - margin) < 1e-14
        rng.shuffle(rows)
        shuffled = followup.pool(rows)
        assert max(abs(a - b) for a, b in zip(expected, shuffled["probabilities"])) < 1e-14
        assert shuffled["prediction"] == choice
        tested += 1
    return tested


def main():
    receipt = followup.check()
    cases = read(ROOT / "followup_manifest.json")
    initial = read(ROOT / "manifest.json")
    assert len(cases) == len({r["group"] for r in cases}) == len({r["id"] for r in cases}) == 32
    old_vectors = {tuple(r["coefs"]) for rows in initial.values() for r in rows}
    assert len(old_vectors) == 92
    spec = importlib.util.spec_from_file_location("earlier_counterfactual_audit",
             ROOT.parent / "counterfactual" / "experiment.py")
    earlier = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(earlier)
    for seed in (earlier.LAYOUT_SEED, 426001):
        old_vectors.update(tuple(r["coefs"]) for r in earlier.tasks(seed))
    new_vectors = {tuple(r["coefs"]) for r in cases}
    assert len(new_vectors) == 32 and not new_vectors & old_vectors
    old_texts = {study.question(r) for rows in initial.values() for r in rows}
    old_texts.update(earlier.question(r, False) for seed in (earlier.LAYOUT_SEED, 426001)
                     for r in earlier.tasks(seed))
    assert not {study.question(r) for r in cases} & old_texts
    grid = Counter()
    for row in cases:
        assert independent_parse(study.question(row)) == row["coefs"]
        assert len({tuple(v) for v in row["options"]}) == 4
        assert row["options"].count(row["coefs"]) == 1
        grid[row["domain"], row["template"], row["options"].index(row["coefs"])] += 1
        # The intended candidate occurs in each label position once under rotation.
        original = row["options"]
        rotated_gold = []
        for rotation in range(4):
            visible, label, options = study.display(row, rotation)
            assert options == original[rotation:] + original[:rotation]
            assert options[label] == row["coefs"]
            rotated_gold.append(label)
        assert sorted(rotated_gold) == list(range(4))
    assert set(grid) == set(product(range(4), (2, 3), range(4)))
    assert set(grid.values()) == {1}
    source = (ROOT / "selection_followup.py").read_text()
    assert "full=self.model(mx.array([ids]))" in source
    assert "self.base.encode('Answer:'" in source
    assert "base.encode(' '+x" in source

    bank = initial["pool"]
    selector = study.Selector(bank, list(range(8)))
    for case in cases:
        view = study.public_view(case)
        assert set(view) == {"id", "text"}
        poisoned = {**view, "coefs": [999] * 4, "gold": "bad", "options": ["bad"]}
        for arm in study.ARMS:
            chosen = selector.select(view, arm)
            assert chosen == selector.select(poisoned, arm)
            assert len(chosen) == (0 if arm == "zero" else 2)
            assert len(set(chosen)) == len(chosen)
    fixtures = pooling_fixtures()
    replayed = {}
    pre_path = ROOT / "followup_preselect_trace.json"
    if pre_path.exists():
        pre = read(pre_path)
        assert pre["freeze"] == receipt
        rows = pre["rows"]
        assert len(rows) == 240
        assert {(r["demo"], r["target"]) for r in rows} == {
            (i, j) for i in range(16) for j in range(16) if i != j}
        direct = [[False] * 16 for _ in range(16)]
        for result in rows:
            check_prediction(result)
            target = bank[result["target"]]
            direct[result["demo"]][result["target"]] = (
                result["prediction"] == target["options"].index(target["coefs"]))
        assert direct == pre["matrix"]
        assert pre["core"] == independent_core(direct)
        core_receipt = read(ROOT / "followup_core_receipt.json")
        assert core_receipt["before_confirmation"] is True
        assert core_receipt["followup_freeze"] == receipt
        assert core_receipt["preselect_trace_sha256"] == study.digest(pre_path)
        assert core_receipt["core"] == pre["core"]
        replayed["singleton_rows"] = 240
        test_path = ROOT / "followup_test_trace.json"
        if test_path.exists():
            trace = read(test_path)
            assert trace["freeze"] == receipt
            assert trace["preselect_sha256"] == study.digest(pre_path)
            assert trace["readout"] == pre["readout"]
            assert len(set(trace["readout"]["label_token_ids"])) == 4
            assert trace["readout"]["assistant_prefix"] == "Answer:"
            lookup = {r["id"]: r for r in cases}
            expected = {(r["id"], a, rot) for r in cases for a in study.ARMS for rot in range(4)}
            observed = {(r["id"], r["arm"], r["rotation"]) for r in trace["rows"]}
            assert len(trace["rows"]) == len(observed) == 1024 and observed == expected
            selected = study.Selector(bank, pre["core"])
            grouped = defaultdict(list)
            for row in trace["rows"]:
                check_prediction(row)
                case = lookup[row["id"]]
                assert row["selected"] == selected.select(study.public_view(case), row["arm"])
                grouped[row["id"], row["arm"]].append(row)
            counts, abstentions = Counter(), Counter()
            for (case_id, arm), rows in grouped.items():
                assert {r["rotation"] for r in rows} == {0, 1, 2, 3}
                probabilities, prediction, margin = independent_pool(rows)
                target = lookup[case_id]
                gold = target["options"].index(target["coefs"])
                counts[arm] += prediction == gold
                abstentions[arm] += prediction is None
                assert followup.pool(rows)["prediction"] == prediction
            replayed["confirmation_rows"] = 1024
            replayed["correct_groups_of_32"] = {a: counts[a] for a in study.ARMS}
            replayed["abstentions"] = {a: abstentions[a] for a in study.ARMS}
    print(json.dumps({"audit": "PASS", "no_model_loaded": True,
                      "fresh_disjoint_vectors": 32, "counterbalanced_grid_cells": 32,
                      "pooling_fixtures": fixtures, "replayed": replayed}, indent=2))


if __name__ == "__main__":
    main()
