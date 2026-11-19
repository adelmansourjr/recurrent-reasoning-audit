"""Development-only answer-prefix, label-token and option-order diagnostic.

Run only after the frozen study process has finished. Uses one local model.
This is an exploratory diagnostic, not fresh confirmation of efficacy.
"""
import argparse
from collections import Counter
import json
from pathlib import Path
import time

from study import (
    ROOT, LETTERS, Scorer, Selector, check_freeze, digest, display,
    make_prompt, public_view, write,
)

LIMIT = 2560 * 1024**2


def prediction(values):
    return max(range(4), key=lambda i: (values[i], -i))


def guard(mx):
    if mx.get_peak_memory() > LIMIT:
        raise MemoryError("Readout diagnostic exceeded 2,560 MiB observed MLX allocation")


def ids_for(scorer, prompt, prefix):
    ids = scorer.tokenizer.apply_chat_template(
        [{"role": "user", "content": prompt}], tokenize=True,
        add_generation_prompt=True, enable_thinking=False,
    )
    if prefix:
        ids += scorer.tokenizer.encode(prefix, add_special_tokens=False)
    if len(ids) > 1500:
        raise ValueError("Diagnostic prompt ceiling exceeded")
    return ids


def prefixed_score(scorer, prompt, label_ids):
    from mlx_lm.models.cache import make_prompt_cache
    mx = scorer.mx
    ids = ids_for(scorer, prompt, "Answer:")
    cache = make_prompt_cache(scorer.model)
    start = time.perf_counter_ns()
    for offset in range(0, len(ids), 128):
        h = scorer.model.model(mx.array([ids[offset:offset + 128]]), cache=cache)
        mx.eval(h, [layer.state for layer in cache])
        guard(mx)
    h = h[:, -1:, :]
    logits = (scorer.model.model.embed_tokens.as_linear(h)
              if scorer.model.args.tie_word_embeddings else scorer.model.lm_head(h))
    selected = logits[0, 0, label_ids].astype(mx.float32)
    mx.eval(selected)
    values = selected.tolist()
    guard(mx)
    result = {"prediction": prediction(values), "label_logits": values,
              "input_tokens": len(ids), "elapsed_ns": time.perf_counter_ns() - start}
    del selected, logits, h, cache
    mx.clear_cache()
    scorer.count += 1
    return result


def wrapper_check(scorer, prompt, prefix, label_ids, chunk_result):
    mx = scorer.mx
    ids = ids_for(scorer, prompt, prefix)
    start = time.perf_counter_ns()
    # Deliberately use the unoptimised public wrapper as an independent path.
    full = scorer.model(mx.array([ids]))
    values = full[0, -1, label_ids].astype(mx.float32)
    mx.eval(values)
    actual = values.tolist()
    guard(mx)
    expected = chunk_result["label_logits"]
    result = {"full_wrapper_label_logits": actual,
              "chunk_label_logits": expected,
              "max_abs_label_delta": max(abs(a - b) for a, b in zip(actual, expected)),
              "full_prediction": prediction(actual),
              "chunk_prediction": chunk_result["prediction"],
              "same_argmax": prediction(actual) == chunk_result["prediction"],
              "label_delta_below_half_logit": max(abs(a-b) for a,b in zip(actual,expected)) < .5,
              "input_tokens": len(ids), "elapsed_ns": time.perf_counter_ns() - start}
    del full, values
    mx.clear_cache()
    scorer.count += 1
    return result


def run(checkpoint, output):
    output = Path(output)
    progress = output.with_name(output.stem + "_progress.json")
    plan_path = output.with_name(output.stem + "_plan.json")
    if any(p.exists() for p in [output, progress, plan_path]):
        raise FileExistsError("Choose a fresh output; existing diagnostic artifacts are preserved")
    freeze = check_freeze()
    manifest = json.loads((ROOT / "manifest.json").read_text())
    # First four development cases are fixed before diagnostic inference and
    # use every candidate position through rotations. Never touch test rows.
    targets = manifest["dev"][:4]
    assert len(targets) == 4
    selector = Selector(manifest["pool"])
    plan = {"scope": "Exploratory development-only diagnostic; no efficacy claim",
            "reason": "Original pool and development readouts always selected A",
            "source_sha256": digest(__file__), "study_freeze": freeze,
            "before_diagnostic_inference": True,
            "ids": [r["id"] for r in targets], "rotations": [0, 1, 2, 3],
            "selector": "lexical; same selected demonstrations across target rotations",
            "conditions": {"bare": "Original Scorer.score, bare A-D tokens",
                           "answer_prefix_space_labels": "Append assistant Answer:; score leading-space A-D tokens"},
            "wrapper_checks": [[targets[0]["id"], 0, "bare"],
                               [targets[-1]["id"], 0, "answer_prefix_space_labels"]],
            "memory_observed_abort_bytes": LIMIT}
    write(plan_path, plan)
    scorer = Scorer(checkpoint)
    spaced = [scorer.tokenizer.encode(" " + label, add_special_tokens=False) for label in LETTERS]
    if not all(len(ids) == 1 for ids in spaced):
        write(progress, {"plan": plan, "leading_space_label_encodings": spaced,
                         "error": "Leading-space labels are not each one token", "meta": scorer.meta()})
        raise ValueError("Leading-space labels are not each one token; preserve diagnostic failure")
    spaced = [ids[0] for ids in spaced]
    rows, checks = [], []
    for index, target in enumerate(targets):
        selected = selector.select(public_view(target), "lexical")
        for rotation in range(4):
            prompt = make_prompt(target, manifest["pool"], selected, rotation)
            _, correct, _ = display(target, rotation)
            for condition in ["bare", "answer_prefix_space_labels"]:
                result = (scorer.score(prompt) if condition == "bare"
                          else prefixed_score(scorer, prompt, spaced))
                row = {"id": target["id"], "group": target["group"],
                       "rotation": rotation, "condition": condition,
                       "selected": selected, "correct_label": correct,
                       "correct": result["prediction"] == correct,
                       "prompt": prompt, **result}
                rows.append(row)
                if (index, rotation, condition) in [(0, 0, "bare"), (3, 0, "answer_prefix_space_labels")]:
                    check = wrapper_check(scorer, prompt,
                                          "" if condition == "bare" else "Answer:",
                                          scorer.labels if condition == "bare" else spaced, result)
                    checks.append({"id": target["id"], "rotation": rotation,
                                   "condition": condition, **check})
                write(progress, {"plan": plan, "label_ids": {"bare": scorer.labels, "leading_space": spaced},
                                 "rows": rows, "wrapper_checks": checks, "meta": scorer.meta()})
        print("Readout development diagnostic", index + 1, "/ 4", flush=True)
    summaries = {}
    for condition in ["bare", "answer_prefix_space_labels"]:
        current = [r for r in rows if r["condition"] == condition]
        pooled = []
        for target in targets:
            records = [r for r in current if r["id"] == target["id"]]
            # Rotation r moves original candidate j to displayed slot j-r.
            means = [sum(r["label_logits"][(candidate-r["rotation"]) % 4]
                         for r in records) / 4 for candidate in range(4)]
            correct = display(target)[1]
            pooled.append({"id": target["id"], "mean_logits_by_original_candidate": means,
                           "prediction": prediction(means), "correct_label": correct,
                           "correct": prediction(means) == correct})
        summaries[condition] = {"individual_presentations": len(current),
                                "correct_presentations": sum(r["correct"] for r in current),
                                "label_counts": dict(Counter(LETTERS[r["prediction"]] for r in current)),
                                "four_rotation_correct_cases": sum(r["correct"] for r in pooled),
                                "pooled": pooled}
    write(output, {"plan": plan, "plan_sha256": digest(plan_path),
                   "label_ids": {"bare": scorer.labels, "leading_space": spaced},
                   "rows": rows, "wrapper_checks": checks, "summary": summaries,
                   "meta": scorer.meta(),
                   "interpretation": "Four selected development cases only; prefix and label tokenisation change together. No fresh efficacy claim or isolated causal attribution."})
    print(json.dumps({"summary": summaries, "wrapper_checks": checks,
                      "meta": scorer.meta()}, indent=2), flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("checkpoint", type=Path)
    parser.add_argument("output", type=Path)
    args = parser.parse_args()
    run(args.checkpoint, args.output)


if __name__ == "__main__":
    main()
