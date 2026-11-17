"""One-probe grounding audit for bounded linear question programs."""

import argparse
from collections import Counter
from hashlib import file_digest, sha256
import importlib.metadata
from itertools import product
import json
from pathlib import Path
import random
import re
import time


ROOT = Path(__file__).resolve().parent
COEFFICIENT_LIMIT = 3
PROBE = (1, 7, 49, 343)
BASIS = tuple(tuple(int(i == j) for i in range(4)) for j in range(4))
LETTERS = "ABCD"
ARMS = ("worked", "structured", "masked", "review")
LAYOUT_SEED = 711873
PROBE_SEED = 218452
BOOTSTRAP_SEED = 559912
CONFIG_DIGEST = "507a6701220524eb8b283425bf0856a9ae4f21f4052e563896ddd668994b1dc7"
WEIGHT_DIGEST = "0e86d9677e519323849eac1bc272caae88567a481ff188c431f70be543d9995f"
TOKENIZER_DIGEST = "aeb13307a71acd8fe81861d94ad54ab689df773318809eed3cbe794b4492dae4"
EXAMPLE_VECTOR = [2, -1, 0, 1]
VECTOR = re.compile(r"\[\s*([+-]?\d+)\s*,\s*([+-]?\d+)\s*,\s*([+-]?\d+)\s*,\s*([+-]?\d+)\s*\]")
FAMILIES = (
    ("The ledger labels four source counts in a library transfer.",
     "the team shelves", "the team withdraws", "is a survey figure unrelated to stock",
     "What is the signed net change in books on the shelves?"),
    ("A contest judge tracks the net points change from four score sheets.",
     "the judge awards", "the judge deducts", "is from a practice heat that does not count",
     "What is the signed net change in the contestant's points?"),
    ("An electrical storage controller logs four reference energy amounts.",
     "the battery receives", "the battery supplies to the grid", "is only a meter calibration reference",
     "What is the signed net change in energy stored in the battery?"),
    ("A theatre ticket office records four reservation counts.",
     "the office adds to the held seats", "the office releases from the held seats",
     "belongs to a separate show", "What is the signed net change in seats held for this show?"),
)
COEFFICIENT_PATTERNS = (
    (-1, 1, 2, 3), (-3, -2, 1, 0), (-2, 2, 0, 1), (-1, -1, 2, 3),
    (-2, -1, 2, 0), (3, -1, 0, -2), (2, 1, 0, -3), (-3, -1, 2, 0),
)


def dot(coefs, values):
    return sum(c * x for c, x in zip(coefs, values, strict=True))


def tasks(seed=LAYOUT_SEED):
    rng = random.Random(seed)
    rows = []
    for family in range(len(FAMILIES)):
        for index in range(16):
            coeffs = rng.sample(COEFFICIENT_PATTERNS[(index + 3 * family) % 8], 4)
            entries = rng.sample(range(7, 69), 4)
            if index >= 8:
                pair = rng.choice([(i, j) for i in range(4) for j in range(4)
                                   if coeffs[i] > 0 > coeffs[j]])
                entries[pair[1]] = entries[pair[0]]
            row = {"id": f"{family}-{index:02}", "family": family, "alias": index >= 8,
                   "coefs": coeffs, "values": entries, "order": rng.sample(list(range(4)), 4),
                   "variant": index % 2}
            rows.append(row)
    rng.shuffle(rows)
    return rows


def question(case, masked):
    setup, plus, minus, distractor, ask = FAMILIES[case["family"]]
    amounts = ", ".join(f"{letter}={'?' if masked else value}"
                        for letter, value in zip(LETTERS, case["values"]))
    terms = []
    for index in case["order"]:
        coef = case["coefs"][index]
        letter = LETTERS[index]
        if coef == 0:
            terms.append(f"{letter} {distractor}.")
            continue
        verb = plus if coef > 0 else minus
        times = ("", "the amount", "twice the amount", "three times the amount")[abs(coef)]
        if case["variant"]:
            terms.append(f"For source {letter}, {verb} with a quantity equal to {times} of {letter}.")
        else:
            terms.append(f"In the case of {letter}, {verb} {times} labelled {letter}.")
    return "\n".join((setup, f"Source counts are {amounts}.", " ".join(terms), ask))


def prompt(case, arm, previous=None, recovery=False):
    if arm == "worked":
        reminder = ("\nUse a one-line calculation and finish with a line Answer: <integer>."
                    if recovery else "")
        return (
            "Solve the signed word problem. Show a short calculation, then write Answer: "
            "followed by a single signed integer.\n\n" + question(case, False) + reminder
        )
    examples = (
        "An account lists A=4, B=5, C=6, D=7. Credit twice A and D. Debit B. "
        "C is a memo that does not count. The net change is 2*A - B + D.\n"
        "Program: [2,-1,0,1]\n\n"
    )
    instruction = (
        "Translate the signed net change into four integer coefficients in order A, B, C, D. "
        "Use 0 for a source not counted. Do not calculate the final number. "
        "Output a single line Program: [a,b,c,d].\n\n"
    )
    body = instruction + examples + question(case, arm == "masked")
    if arm == "review":
        body += (
            f"\n\nFirst attempt: {previous if previous is not None else 'unreadable'}. "
            "Independently review the sign and multiplicity of every source, including excluded entries. "
            "Return the verified or corrected Program: [a,b,c,d] only."
        )
    return body


def decode(response, arm):
    if arm == "worked":
        answers = re.findall(r"(?:answer\s*:\s*|\\boxed\{)\s*\*{0,2}([+−-]?[\d,]+)",
                             response, flags=re.IGNORECASE)
        if not answers:
            matches = re.findall(r"(?m)^\s*([+−-]?\d+)\s*[.!]?\s*$", response)
            answers = matches[-1:]
        if answers:
            return int(answers[-1].replace("−", "-").replace(",", ""))
        return None
    canonical = response.replace("−", "-")
    tagged = re.findall(r"program\s*:\s*(\[[^\]\n]+\])", canonical, flags=re.IGNORECASE)
    hits = VECTOR.findall(tagged[-1] if tagged else canonical)
    if len(hits) != 1:
        return None
    values = tuple(map(int, hits[0]))
    if all(-COEFFICIENT_LIMIT <= item <= COEFFICIENT_LIMIT for item in values):
        return list(values)
    return None


def certify():
    codes = list(product(range(-COEFFICIENT_LIMIT, COEFFICIENT_LIMIT + 1), repeat=4))
    rng = random.Random(PROBE_SEED)
    controls = [tuple(rng.randint(1, 343) for _ in range(4)) for _ in range(32)]

    def collisions(probe):
        counts = Counter(dot(code, probe) for code in codes)
        return {"unique_programs": sum(n == 1 for n in counts.values()),
                "colliding_pairs": sum(n * (n - 1) // 2 for n in counts.values())}

    signature = collisions(PROBE)
    assert signature["unique_programs"] == len(codes)
    assert min(dot(code, PROBE) for code in codes) == -1200
    assert max(dot(code, PROBE) for code in codes) == 1200
    aliases = [collisions(probe) for probe in controls]
    # With larger allowed coefficients, the radix guarantee can fail.
    assert dot((4, 0, 0, 0), PROBE) == dot((-3, 1, 0, 0), PROBE)
    return {"programs": len(codes), "powers_of_seven": signature,
            "one_random_probe": {"replicates": len(controls),
                                 "min_unique_programs": min(a["unique_programs"] for a in aliases),
                                 "mean_unique_programs": sum(a["unique_programs"] for a in aliases) / len(controls),
                                 "max_unique_programs": max(a["unique_programs"] for a in aliases),
                                 "mean_colliding_pairs": sum(a["colliding_pairs"] for a in aliases) / len(controls)}}


def fingerprint(path):
    with path.open("rb") as handle:
        return file_digest(handle, "sha256").hexdigest()


def protocol_digest():
    return sha256(ROOT.joinpath("PROTOCOL.md").read_bytes()).hexdigest()


def extract_batch(model, tokenizer, texts, budget, prefix=""):
    import mlx.core as mx
    from mlx_lm import batch_generate

    items = [tokenizer.apply_chat_template([{"role": "user", "content": text}], tokenize=True,
                                           add_generation_prompt=True, enable_thinking=False)
             for text in texts]
    if prefix:
        prefix_tokens = tokenizer.encode(prefix, add_special_tokens=False)
        items = [item + prefix_tokens for item in items]
    mx.synchronize()
    start = time.perf_counter_ns()
    response = batch_generate(model, tokenizer, items, max_tokens=budget,
                              completion_batch_size=8, prefill_batch_size=8)
    mx.synchronize()
    metric = {"prompt_tokens": response.stats.prompt_tokens,
              "generated_tokens": response.stats.generation_tokens,
              "elapsed_ns": time.perf_counter_ns() - start}
    return [prefix + value for value in response.texts], metric


def run(checkpoint, output, recovery):
    import mlx.core as mx
    from mlx_lm import load

    fingerprints = {name: fingerprint(checkpoint / name)
                    for name in ("config.json", "model.safetensors", "tokenizer.json")}
    if (fingerprints["config.json"] != CONFIG_DIGEST
            or fingerprints["model.safetensors"] != WEIGHT_DIGEST
            or fingerprints["tokenizer.json"] != TOKENIZER_DIGEST):
        raise ValueError("The supplied checkpoint does not match the frozen Qwen weights")
    mx.set_cache_limit(128 * 1024 * 1024)
    model, tokenizer = load(str(checkpoint))
    warmup = {"family": 0, "values": [7, 13, 19, 23], "coefs": [1, -2, 0, 3],
              "order": [1, 3, 0, 2], "variant": 1}
    extract_batch(model, tokenizer, [prompt(warmup, arm) for arm in ARMS], 28)
    mx.reset_peak_memory()
    rows = tasks()
    batches = []
    for start in range(0, len(rows), 8):
        batch = rows[start:start + 8]
        order = ("worked", "structured", "masked", "review") if start % 16 == 0 else (
            "masked", "structured", "review", "worked"
        )
        for arm in order:
            prompts = [prompt(case, arm, case.get("structured"), recovery) for case in batch]
            budget = (224 if recovery else 80) if arm == "worked" else 40
            prefix = "Program: [" if recovery and arm != "worked" else ""
            texts, metric = extract_batch(model, tokenizer, prompts, budget, prefix)
            for case, value in zip(batch, texts, strict=True):
                case[arm] = decode(value, arm)
            batches.append({"method": arm, "case_ids": [case["id"] for case in batch], **metric})
        print(f"Finished {start + len(batch)} / {len(rows)} paired questions", flush=True)
    trace = {"protocol_sha256": protocol_digest(), "checkpoint_sha256": fingerprints,
             "software": {n: importlib.metadata.version(n) for n in ("mlx", "mlx-lm")},
             "mlx_peak_active_bytes": mx.get_peak_memory(),
             "rows": [{n: case[n] for n in ("id", *ARMS)} for case in rows], "batches": batches}
    if recovery:
        trace["recovery_sha256"] = fingerprint(ROOT / "RECOVERY.md")
    write_json(output, trace)


def variable_probe(case_id):
    rng = random.Random(f"{PROBE_SEED}:{case_id}")
    return tuple(rng.randint(1, 343) for _ in range(4))


def paired_interval(correct, arm, other, cases):
    rng = random.Random(BOOTSTRAP_SEED)
    diffs = [a - b for a, b in zip(correct[arm], correct[other], strict=True)]
    strata = [[delta for case, delta in zip(cases, diffs, strict=True)
               if case["family"] == family and case["alias"] == alias]
              for family, alias in product(range(len(FAMILIES)), (False, True))]
    trials = sorted(sum(sum(rng.choices(group, k=len(group))) for group in strata) / len(cases)
                    for _ in range(5000))
    return [trials[124], trials[4874]]


def analyse(trace):
    if trace["protocol_sha256"] != protocol_digest():
        raise ValueError("Protocol hash does not match the scoring archive")
    if "recovery_sha256" in trace and trace["recovery_sha256"] != fingerprint(ROOT / "RECOVERY.md"):
        raise ValueError("Follow-up protocol hash does not match the scoring archive")
    cases = tasks()
    parsed = trace["rows"]
    assert [case["id"] for case in cases] == [row["id"] for row in parsed]
    stats = {arm: Counter() for arm in ARMS}
    correct = {arm: [] for arm in ARMS if arm != "worked"}
    forced_alias = Counter()
    verified_programs = 0
    for case, row in zip(cases, parsed, strict=True):
        gold, natural = case["coefs"], case["values"]
        random_probe = variable_probe(case["id"])
        alias_group = "stress" if case["alias"] else "ordinary"
        if case["alias"]:
            a, b = next((i, j) for i in range(4) for j in range(i)
                        if natural[i] == natural[j])
            mutant = gold.copy()
            mutant[a], mutant[b] = mutant[b], mutant[a]
            assert mutant != gold
            forced_alias["count"] += 1
            for test, needle in (("accepted_original", natural), ("accepted_random", random_probe),
                                 ("accepted_radix", PROBE)):
                forced_alias[test] += int(dot(mutant, needle) == dot(gold, needle))

        native = row["worked"]
        stats["worked"]["executable"] += native is not None
        if native == dot(gold, natural):
            stats["worked"]["answer_correct"] += 1
            stats["worked"][f"{alias_group}_answer_correct"] += 1
        for arm in correct:
            candidate = row[arm]
            if arm == "review" and candidate is None:
                candidate = row["structured"]
            same = candidate == gold
            correct[arm].append(int(same))
            stats[arm]["grounded"] += same
            stats[arm][f"{alias_group}_grounded"] += same
            stats[arm]["executable"] += candidate is not None
            stats[arm]["copied_shared_example"] += candidate == EXAMPLE_VECTOR
            if candidate is None:
                continue
            answers = {}
            for name, values in (("original", natural), ("random", random_probe), ("radix", PROBE)):
                answers[name] = dot(candidate, values) == dot(gold, values)
                stats[arm][f"wrong_accepted_{name}"] += not same and answers[name]
            basis_agrees = all(dot(gold, unit) == dot(candidate, unit) for unit in BASIS)
            assert same == answers["radix"] == basis_agrees
            verified_programs += 1
            stats[arm]["answer_correct"] += answers["original"]
            stats[arm][f"{alias_group}_answer_correct"] += answers["original"]

    cost = {}
    for arm in ARMS:
        batches = [batch for batch in trace["batches"] if batch["method"] == arm]
        assert sorted(case_id for batch in batches for case_id in batch["case_ids"]) == sorted(
            case["id"] for case in cases
        )
        if arm == "review":
            batches += [batch for batch in trace["batches"] if batch["method"] == "structured"]
        cost[arm] = {name: sum(batch[name] for batch in batches)
                     for name in ("prompt_tokens", "generated_tokens", "elapsed_ns")}
    comparisons = {}
    for other in ("structured", "review"):
        compar = {"gain": (sum(correct["masked"]) - sum(correct[other])) / len(cases),
                  "paired_95_interval": paired_interval(correct, "masked", other, cases),
                  "masked_to_baseline_tokens": (
                      cost["masked"]["prompt_tokens"] + cost["masked"]["generated_tokens"]
                  ) / (cost[other]["prompt_tokens"] + cost[other]["generated_tokens"])}
        compar["practical_gate"] = (
            compar["gain"] >= .10 and compar["paired_95_interval"][0] > 0
            and compar["masked_to_baseline_tokens"] <= 1.10
        )
        comparisons[other] = compar
    result = {
        "cases": len(cases), "protocol_sha256": protocol_digest(),
        "discrete_certificate": certify(), "forced_alias_mutant": dict(forced_alias),
        "matched_coefficient_oracle_and_four_probes": verified_programs,
        "model": {arm: dict(counts) for arm, counts in stats.items()}, "model_cost": cost,
        "comparisons": comparisons, "mlx_peak_active_bytes": trace["mlx_peak_active_bytes"],
    }
    if "recovery_sha256" in trace:
        result["recovery_sha256"] = trace["recovery_sha256"]
    return result


def write_json(path, value):
    path.write_text(json.dumps(value, indent=2) + "\n")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    commands.add_parser("demo")
    score = commands.add_parser("score")
    score.add_argument("checkpoint", type=Path)
    score.add_argument("trace", type=Path)
    score.add_argument("--recovery", action="store_true")
    analysis = commands.add_parser("analyse")
    analysis.add_argument("trace", type=Path)
    target = analysis.add_mutually_exclusive_group(required=True)
    target.add_argument("--output", type=Path)
    target.add_argument("--verify", type=Path)
    args = parser.parse_args()
    if args.command == "demo":
        original, wrong = [1, -1, 2, 0], [-1, 1, 2, 0]
        nominal = (18, 18, 20, 11)
        assert dot(original, nominal) == dot(wrong, nominal) == 40
        assert dot(original, PROBE) == 92 and dot(wrong, PROBE) == 104
        print(json.dumps(certify(), indent=2))
        print("Same answer on the original problem (40), different answers on the probe (92 vs 104).")
    elif args.command == "score":
        run(args.checkpoint, args.trace, args.recovery)
    else:
        result = analyse(json.loads(args.trace.read_text()))
        if args.output:
            write_json(args.output, result)
            print(json.dumps(result, indent=2))
        else:
            assert result == json.loads(args.verify.read_text())
            print("Verified exact grounding, paired results, adversarial aliases and cost accounting.")


if __name__ == "__main__":
    main()
