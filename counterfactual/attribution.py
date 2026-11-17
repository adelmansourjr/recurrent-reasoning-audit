"""Matched intervention on the example in Qwen's structured reasoning prompt."""

import argparse
from collections import Counter
import json
from math import comb
from pathlib import Path

from experiment import (
    CONFIG_DIGEST, ROOT, TOKENIZER_DIGEST, WEIGHT_DIGEST, decode, extract_batch,
    fingerprint, paired_interval, prompt, question, tasks, write_json,
)

SEED = 426001
ARMS = ("example_a", "example_b", "no_example")
EXAMPLE_A = (
    "An account lists A=4, B=5, C=6, D=7. Credit twice A and D. Debit B. "
    "C is a memo that does not count. The net change is 2*A - B + D.\n"
    "Program: [2,-1,0,1]\n\n"
)
EXAMPLE_B = (
    "An account lists A=4, B=5, C=6, D=7. Debit twice A and D. Credit B. "
    "C is a memo that does not count. The net change is -2*A + B - D.\n"
    "Program: [-2,1,0,-1]\n\n"
)
VECTORS = {"example_a": [2, -1, 0, 1], "example_b": [-2, 1, 0, -1]}
EXAMPLES = dict(zip(ARMS, (EXAMPLE_A, EXAMPLE_B, ""), strict=True))


def cohort():
    quota = Counter()
    chosen = []
    for case in tasks(SEED):
        group = (case["family"], case["alias"])
        if quota[group] < 4:
            chosen.append(case)
            quota[group] += 1
    assert set(quota.values()) == {4}
    assert not ({question(case, False) for case in chosen}
                & {question(case, False) for case in tasks()})
    assert all(case["coefs"] not in VECTORS.values() for case in chosen)
    return chosen


def run(checkpoint, trace):
    import mlx.core as mx
    from mlx_lm import load

    names = ("config.json", "model.safetensors", "tokenizer.json")
    hashes = [fingerprint(checkpoint / name) for name in names]
    if hashes != [CONFIG_DIGEST, WEIGHT_DIGEST, TOKENIZER_DIGEST]:
        raise ValueError("Checkpoint does not match the frozen artifact")
    rows = cohort()
    model, tokenizer = load(str(checkpoint))
    mx.set_cache_limit(128 * 1024 * 1024)
    warm = {"family": 0, "values": [7, 13, 19, 23], "coefs": [1, -2, 0, 3],
            "order": [1, 3, 0, 2], "variant": 1}
    extract_batch(model, tokenizer, [prompt(warm, "structured")], 40, "Program: [")
    mx.reset_peak_memory()
    costs = {key: Counter() for key in ARMS}
    data = {}
    for start in range(0, len(rows), 8):
        batch = rows[start:start + 8]
        order = ARMS if start % 16 == 0 else tuple(reversed(ARMS))
        for arm in order:
            prompts = [
                prompt(case, "structured").replace(EXAMPLE_A, EXAMPLES[arm], 1) for case in batch
            ]
            assert all(p.count(EXAMPLES[arm]) == 1 for p in prompts if EXAMPLES[arm])
            if arm == "no_example":
                assert all(EXAMPLE_A not in p for p in prompts)
            texts, metrics = extract_batch(model, tokenizer, prompts, 40, "Program: [")
            costs[arm].update(metrics)
            for case, response in zip(batch, texts, strict=True):
                data.setdefault(case["id"], {})[arm] = decode(response, "structured")
        print(f"Finished {start + len(batch)} / {len(rows)} attribution questions", flush=True)
    payload = {"attribution_sha256": fingerprint(ROOT / "ATTRIBUTION.md"),
               "checkpoint_sha256": dict(zip(names, hashes, strict=True)),
               "rows": data, "costs": costs, "mlx_peak_active_bytes": mx.get_peak_memory()}
    write_json(trace, payload)


def analyse(trace):
    assert trace["attribution_sha256"] == fingerprint(ROOT / "ATTRIBUTION.md")
    rows = cohort()
    records = trace["rows"]
    assert set(records) == {case["id"] for case in rows}
    metrics = {}
    effects = {}
    for arm in ARMS:
        candidates = [records[case["id"]][arm] for case in rows]
        metrics[arm] = {"valid": sum(v is not None for v in candidates),
                        "grounded": sum(v == c["coefs"] for v, c in zip(candidates, rows, strict=True)),
                        **{f"copied_{side}": sum(v == pattern for v in candidates)
                           for side, pattern in VECTORS.items()}}
    for arm, other in (("example_a", "example_b"), ("example_b", "example_a")):
        match = {
            key: [int(records[case["id"]][key] == VECTORS[arm]) for case in rows]
            for key in (arm, other)
        }
        wins = sum(a > b for a, b in zip(match[arm], match[other], strict=True))
        losses = sum(a < b for a, b in zip(match[arm], match[other], strict=True))
        n = wins + losses
        p_value = min(1.0, 2 * sum(comb(n, i) for i in range(min(wins, losses) + 1)) / 2**n)
        difference = (wins - losses) / len(rows)
        interval = paired_interval(match, arm, other, rows)
        effects[arm] = {"paired_difference": difference, "paired_95_interval": interval,
                        "paired_copy_wins": wins, "paired_copy_losses": losses,
                        "exact_sign_p": p_value,
                        "gate_met": (sum(match[arm]) / len(rows) >= .5
                                     and difference >= .30 and interval[0] > 0)}
    return {"attribution_sha256": trace["attribution_sha256"], "cases": len(rows),
            "metrics": metrics, "paired_copying_effects": effects, "costs": trace["costs"],
            "mlx_peak_active_bytes": trace["mlx_peak_active_bytes"]}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    score = commands.add_parser("score")
    score.add_argument("checkpoint", type=Path)
    score.add_argument("trace", type=Path)
    analysis = commands.add_parser("analyse")
    analysis.add_argument("trace", type=Path)
    target = analysis.add_mutually_exclusive_group(required=True)
    target.add_argument("--output", type=Path)
    target.add_argument("--verify", type=Path)
    args = parser.parse_args()
    if args.command == "score":
        run(args.checkpoint, args.trace)
        return
    result = analyse(json.loads(args.trace.read_text()))
    if args.output:
        write_json(args.output, result)
        print(json.dumps(result, indent=2))
    else:
        assert result == json.loads(args.verify.read_text())
        print("Verified the matched example intervention.")


if __name__ == "__main__":
    main()
