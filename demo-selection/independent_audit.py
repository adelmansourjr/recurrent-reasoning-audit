"""Model-free independent checks for the frozen demonstration study.

Run from any directory with Python 3.11+. No model/data downloads or inference.
This auditor is separate from the frozen generator and scoring implementation.
"""
from collections import Counter, defaultdict
from copy import deepcopy
import importlib.util
import json
from pathlib import Path
import re
import sys

sys.dont_write_bytecode = True
ROOT = Path(__file__).resolve().parent


def independent_parse(text):
    """Parse the declared synthetic language without the study's parser."""
    found = {}
    for clause in text.split("\n")[1].split("."):
        clause = clause.strip()
        if not clause:
            continue
        letters = re.findall(r"\b[A-D]\b", clause)
        assert len(letters) == 1, clause
        letter = letters[0]
        assert letter not in found, clause
        words = clause.lower()
        if any(s in words for s in ("ignore", "does not count", "no contribution", "no effect")):
            coefficient = 0
        else:
            sign = -1 if any(s in words for s in ("subtract", "decrease", "reduce")) else 1
            assert any(s in words for s in ("add", "increase", "subtract", "decrease", "reduce"))
            magnitude = 3 if "three times" in words else 2 if "twice" in words else 1
            assert any(s in words for s in ("three times", "twice", "once"))
            coefficient = sign * magnitude
        found[letter] = coefficient
    assert set(found) == set("ABCD")
    return [found[x] for x in "ABCD"]


def counterbalance(rows, gold_label, name):
    counts = Counter((row["domain"], gold_label(row)) for row in rows)
    for domain in sorted({row["domain"] for row in rows}):
        values = [counts[domain, label] for label in range(4)]
        assert max(values) - min(values) <= 1, (
            "Domain leaks answer position", name, domain, values
        )
    return {str(domain): [counts[domain, label] for label in range(4)]
            for domain in range(4)}


def independent_core(matrix):
    selected, covered = [], set()
    for _ in range(8):
        candidates = []
        for index, row in enumerate(matrix):
            if index in selected:
                continue
            coverage = {index} | {target for target, good in enumerate(row) if good}
            candidates.append((len(coverage - covered), len(coverage), -index, index, coverage))
        _, _, _, winner, coverage = max(candidates, key=lambda x: x[:3])
        selected.append(winner)
        covered |= coverage
    return sorted(selected)


def trace_rows(path):
    return json.loads(path.read_text())


def check_prediction(row):
    logits = row["label_logits"]
    assert len(logits) == 4
    assert all(isinstance(x, (int, float)) and x == x and abs(x) != float("inf") for x in logits)
    assert row["prediction"] == max(range(4), key=lambda k: (logits[k], -k))
    assert 0 < row["input_tokens"] <= 1500
    assert row["elapsed_ns"] >= 0


def main():
    spec = importlib.util.spec_from_file_location("frozen_demo_study", ROOT / "study.py")
    study = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(study)
    receipt = study.check_freeze()
    manifest = json.loads((ROOT / "manifest.json").read_text())
    assert manifest == study.generate()
    vectors = {name: {tuple(row["coefs"]) for row in rows} for name, rows in manifest.items()}
    assert len(vectors["pool"]) == 16 and len(vectors["dev"]) == 12 and len(vectors["test"]) == 64
    assert not vectors["pool"] & vectors["dev"]
    assert not vectors["pool"] & vectors["test"]
    assert not vectors["dev"] & vectors["test"]
    pool = manifest["pool"]
    balances = {}
    for name, rows in manifest.items():
        assert len({row["id"] for row in rows}) == len(rows)
        for row in rows:
            assert independent_parse(study.question(row)) == row["coefs"]
            assert len({tuple(x) for x in row["options"]}) == 4
            assert row["options"].count(row["coefs"]) == 1
            assert all(len(v) == 4 and all(-3 <= x <= 3 for x in v) for v in row["options"])
        balances[name] = counterbalance(rows, lambda r: r["options"].index(r["coefs"]), name)
    for template in (2, 3):
        rows = [r for r in manifest["test"] if r["template"] == template]
        balances[f"test_template_{template}"] = counterbalance(
            rows, lambda r: r["options"].index(r["coefs"]), f"test_template_{template}"
        )
    grouped = defaultdict(list)
    for row in manifest["test"]:
        grouped[row["group"]].append(row)
    assert len(grouped) == 64 and all(len(rows) == 2 for rows in grouped.values())
    assert all(len({tuple(r["coefs"]) for r in rows}) == 1 for rows in grouped.values())
    assert {r["template"] for r in pool} == {0}
    assert {r["template"] for r in manifest["dev"]} == {1}
    assert {r["template"] for r in manifest["test"]} == {2, 3}

    matrix = [[i != j and (i * 7 + j * 3) % 5 < 2 for j in range(16)] for i in range(16)]
    assert study.edges_and_core(matrix) == independent_core(matrix)
    selector = study.Selector(pool, independent_core(matrix))
    for row in manifest["test"]:
        public = study.public_view(row)
        assert set(public) == {"id", "text"}
        contaminated = {**public, "coefs": [999] * 4, "options": ["poison"], "gold": -100}
        for arm in study.ARMS:
            chosen = selector.select(public, arm)
            assert chosen == selector.select(contaminated, arm), ("Hidden-label dependence", arm)
            assert len(chosen) == (0 if arm == "zero" else 2)
            assert len(chosen) == len(set(chosen))
            assert all(0 <= index < 16 for index in chosen)
            if arm not in ("zero", "random_11", "random_29", "random_47"):
                changed_id = {**public, "id": "id-must-not-affect-text-retrieval"}
                assert chosen == selector.select(changed_id, arm)

    verified = {}
    pre_path = ROOT / "preselect_trace.json"
    if pre_path.exists():
        pre = trace_rows(pre_path)
        assert pre["freeze"] == receipt
        assert len(pre["rows"]) == 240
        pairs = {(r["demo"], r["target"]) for r in pre["rows"]}
        assert pairs == {(i, j) for i in range(16) for j in range(16) if i != j}
        direct = [[False] * 16 for _ in range(16)]
        for row in pre["rows"]:
            check_prediction(row)
            direct[row["demo"]][row["target"]] = (
                row["prediction"] == pool[row["target"]]["options"].index(pool[row["target"]]["coefs"])
            )
        assert direct == pre["matrix"]
        assert pre["core"] == independent_core(direct)
        verified["preselection_rows"] = 240
        selector = study.Selector(pool, pre["core"])
        for phase in ("dev", "test"):
            path = ROOT / f"{phase}_trace.json"
            if not path.exists():
                continue
            trace = trace_rows(path)
            assert trace["freeze"] == receipt
            assert trace["preselect_sha256"] == study.digest(pre_path)
            rows = manifest[phase]
            lookup = {r["id"]: r for r in rows}
            keys = {(r["id"], r["arm"]) for r in trace["rows"]}
            assert keys == {(r["id"], arm) for r in rows for arm in study.ARMS}
            assert len(trace["rows"]) == len(keys)
            counts = Counter()
            for result in trace["rows"]:
                check_prediction(result)
                target = lookup[result["id"]]
                assert result["selected"] == selector.select(study.public_view(target), result["arm"])
                counts[result["arm"]] += result["prediction"] == target["options"].index(target["coefs"])
            verified[phase] = {arm: {"correct": counts[arm], "cases": len(rows)} for arm in study.ARMS}
    print(json.dumps({"audit": "PASS", "no_model_loaded": True,
                      "domain_answer_counts": balances, "replayed": verified}, indent=2))


if __name__ == "__main__":
    main()
