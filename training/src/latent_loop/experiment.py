"""Offline preparation, real Qwen LoRA training, and diagnostic evaluation.

Run with PYTHONPATH=src and the existing jev-like MLX Python environment.
Prefixes cache the frozen first 27 layers, never the trainable final layer.
"""
from __future__ import annotations

import argparse
from collections import Counter
import hashlib
import importlib.metadata
import json
import math
import os
from pathlib import Path
import random
import time

os.environ.update(HF_HUB_OFFLINE="1", TRANSFORMERS_OFFLINE="1", TOKENIZERS_PARALLELISM="false")

from latent_loop.data import LABELS, canonical_json, clef_record, generate, render

DEFAULT_MODEL = os.environ.get("QWEN_MODEL_DIRECTORY", "/path/to/Qwen3-1.7B-4bit")


def write_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2) + "\n")


def sha256(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for block in iter(lambda: f.read(8 * 1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def load_model(path, seed=17, dim=128, cached_final_only=False):
    import mlx.core as mx
    from mlx_lm import load
    from mlx.utils import tree_flatten
    from latent_loop.model import QwenLoop
    mx.random.seed(seed)
    mx.set_cache_limit(128 * 1024 * 1024)
    checkpoint = Path(path).resolve(strict=True)
    config = json.loads((checkpoint / "config.json").read_text())
    if config.get("model_type") != "qwen3":
        raise ValueError("This implementation validates the local Qwen3 architecture only")
    backbone, tok = load(str(checkpoint), lazy=True)
    if cached_final_only:
        if len(backbone.model.layers)!=28:raise ValueError('Cached final-block architecture mismatch')
        for i in range(27):backbone.model.layers[i]=None
        backbone.model.embed_tokens=None
    for _, value in tree_flatten(backbone.parameters()):
        mx.eval(value)
    backbone.freeze()
    model = QwenLoop(backbone, dim=dim)
    return model, tok


def tokenize(case, tokenizer, max_tokens):
    ids, boundaries = [], []
    # Deliberate segmented encoding: this exact sequence is used by every arm.
    for line in render(case):
        part = tokenizer.encode(line, add_special_tokens=False)
        if not part:
            raise ValueError("Empty fact line")
        ids.extend(part)
        boundaries.append(len(ids) - 1)
    if len(ids) > max_tokens:
        raise ValueError(f"{case['id']}: {len(ids)} tokens exceeds {max_tokens}; no truncation allowed")
    return ids, boundaries


def prepare(args):
    import mlx.core as mx
    import numpy as np
    root = Path(args.out)
    if (root / "manifest.json").exists():
        raise ValueError("Cache already exists; use a new --out directory")
    root.mkdir(parents=True, exist_ok=True)
    model, tokenizer = load_model(args.model)
    data = generate(args.seed)
    start = time.perf_counter()
    manifest = {"model": str(Path(args.model).resolve()), "seed": args.seed,
                "versions": {n: importlib.metadata.version(n) for n in ("mlx", "mlx-lm", "numpy", "transformers")},
                "checkpoint_sha256": sha256(Path(args.model) / "model.safetensors"),
                "config_sha256": sha256(Path(args.model) / "config.json"),
                "tokenizer_sha256": sha256(Path(args.model) / "tokenizer.json"),
                "max_tokens": args.max_tokens, "cases": len(data),
                "cache_boundary": "before_final_qwen_block", "native_predictions": []}
    option_ids = [tokenizer.encode(x, add_special_tokens=False) for x in ("A", "B", "C")]
    if not all(len(x) == 1 for x in option_ids):
        raise ValueError("A/B/C must be single tokens")
    # Native baseline uses an instruction format, independently of sidecar inputs.
    for i, case in enumerate(data):
        ids, boundaries = tokenize(case, tokenizer, args.max_tokens)
        h = model.encode_prefix(mx.array([ids]))
        mx.eval(h)
        np.savez(root / f"{case['id']}.npz", prefix=np.array(h.astype(mx.float32)),
                 bfloat16=np.array(h.dtype == mx.bfloat16),
                 boundaries=np.array(boundaries, dtype=np.int32), ids=np.array(ids, dtype=np.int32))
        if case["split"] in ("val", "test", "ood"):
            user = "".join(render(case)) + "\nReturn only A for ready, B for blocked, C for inspect."
            prompt = tokenizer.apply_chat_template([{"role": "user", "content": user}],
                       tokenize=True, add_generation_prompt=True, enable_thinking=False)
            if len(prompt) > args.max_tokens:
                raise ValueError("Native baseline exceeds token budget")
            hidden = model.backbone.model(mx.array([prompt]))[:, -1:, :]
            logits = model.backbone.model.embed_tokens.as_linear(hidden)[0, 0].astype(mx.float32)
            allowed = logits[mx.array([x[0] for x in option_ids])]
            probabilities = mx.softmax(allowed)
            mass = mx.exp(mx.logsumexp(allowed) - mx.logsumexp(logits))
            mx.eval(probabilities, mass)
            manifest["native_predictions"].append(dict(id=case["id"], split=case["split"],
                       label=case["label"], probs=np.array(probabilities).tolist(), allowed_mass=float(mass.item())))
        if (i + 1) % 12 == 0:
            print(f"prepare {i+1}/{len(data)} elapsed={time.perf_counter()-start:.1f}s", flush=True)
        del h
        mx.clear_cache()
    write_json(root / "cases.json", data)
    manifest["cases_sha256"] = sha256(root / "cases.json")
    manifest["cache_sha256"] = {f"{c['id']}.npz": sha256(root / f"{c['id']}.npz") for c in data}
    manifest.update(seconds=time.perf_counter()-start, mlx_peak_gb=mx.get_peak_memory()/1e9)
    write_json(root / "manifest.json", manifest)
    write_json(root / "clef_records.json", [{"id": c["id"], "split": c["split"], "record": clef_record(c)}
               for c in data if c["split"] in ("test", "ood")])
    print(json.dumps({k:v for k,v in manifest.items() if k not in ("native_predictions", "cache_sha256")}), flush=True)


def attach_lora(model):
    from mlx_lm.tuner.utils import linear_to_lora_layers
    from mlx.utils import tree_flatten
    config = {"rank": 8, "scale": 2.0, "dropout": 0.0,
              "keys": ["self_attn.q_proj", "self_attn.v_proj"]}
    linear_to_lora_layers(model.backbone, 1, config)
    trainable = tree_flatten(model.trainable_parameters())
    backbone_trainable = [k for k, _ in trainable if k.startswith("backbone.")]
    if len(backbone_trainable) != 4 or not all("layers.27." in k and "lora_" in k for k in backbone_trainable):
        raise ValueError(f"Unexpected backbone trainables: {backbone_trainable}")
    return {"config": config, "names": [k for k, _ in trainable],
            "parameters": sum(v.size for _, v in trainable)}


def read_cache(root, cases):
    import mlx.core as mx
    import numpy as np
    result = {}
    for case in cases:
        with np.load(root / f"{case['id']}.npz") as data:
            dtype = mx.bfloat16 if bool(data["bfloat16"]) else mx.float16
            result[case["id"]] = (mx.array(data["prefix"], dtype=dtype), data["boundaries"].tolist())
    return result


def metrics(rows):
    import numpy as np
    if not rows:
        return {"n": 0}
    p = np.array([x["probs"] for x in rows], dtype=np.float64)
    y = np.array([x["label"] for x in rows])
    correct = p.argmax(-1) == y
    n = len(y)
    acc = float(correct.mean())
    denom = 1 + 1.96**2 / n
    center = (acc + 1.96**2 / (2*n)) / denom
    radius = 1.96 * math.sqrt(acc*(1-acc)/n + 1.96**2/(4*n*n)) / denom
    onehot = np.eye(3)[y]
    conf = p.max(-1)
    ece = 0.0
    for lower in np.arange(0, 1, .1):
        mask = (conf >= lower) & (conf < lower + .1 + (1e-8 if lower > .89 else 0))
        if mask.any():
            ece += mask.mean() * abs(correct[mask].mean() - conf[mask].mean())
    recalls = {LABELS[k]: float(correct[y==k].mean()) if (y==k).any() else None for k in range(3)}
    return dict(n=n, accuracy=acc, accuracy_wilson95=[center-radius, center+radius],
                multiclass_brier_sum=float(((p-onehot)**2).sum(-1).mean()),
                nll=float(-np.log(np.clip(p[np.arange(n), y], 1e-9, 1)).mean()),
                ece10=float(ece), recall=recalls)


def evaluate(model, cases, cache, loops, intervention=None):
    import mlx.core as mx
    import numpy as np
    rows = []
    start = time.perf_counter()
    for case in cases:
        prefix, boundaries = cache[case["id"]]
        logits, trace = model(prefix, boundaries, loops=loops, intervention=intervention)
        probs = mx.softmax(logits, axis=-1)
        mx.eval(probs, trace)
        # Trajectories are diagnostics, not human-readable explanations.
        norms = [float(mx.linalg.norm(t).item()) for t in trace]
        deltas = [float(mx.linalg.norm(trace[i]-trace[i-1]).item()) for i in range(1, len(trace))]
        rows.append(dict(id=case["id"], label=case["label"], probs=np.array(probs[0]).tolist(),
                         loops=loops, state_norms=norms, state_deltas=deltas))
    return {"metrics": metrics(rows), "seconds_including_final_qwen_layer": time.perf_counter()-start,
            "predictions": rows}


def train(args):
    import mlx.core as mx
    import mlx.nn as nn
    import mlx.optimizers as optim
    from mlx.utils import tree_flatten
    root, out = Path(args.cache), Path(args.out)
    if (out / "config.json").exists():
        raise ValueError("Run already exists; choose another --out")
    manifest = json.loads((root / "manifest.json").read_text())
    if sha256(root / "cases.json") != manifest["cases_sha256"]:
        raise ValueError("Dataset cache provenance mismatch")
    for name, expected in manifest["cache_sha256"].items():
        if sha256(root / name) != expected:
            raise ValueError(f"Prefix cache mismatch: {name}")
    # Enforce cache/backbone identity, rather than silently changing encoders.
    if sha256(Path(args.model) / "model.safetensors") != manifest["checkpoint_sha256"]:
        raise ValueError("Checkpoint differs from cached frozen prefix")
    for filename, key in (("config.json", "config_sha256"), ("tokenizer.json", "tokenizer_sha256")):
        if sha256(Path(args.model) / filename) != manifest[key]:
            raise ValueError(f"{filename} differs from cache")
    cases = json.loads((root / "cases.json").read_text())
    cache = read_cache(root, cases)
    model, _ = load_model(args.model, args.seed, args.dim)
    lora = attach_lora(model)
    out.mkdir(parents=True, exist_ok=True)
    write_json(out / "config.json", dict(model=args.model, seed=args.seed, loops=args.loops,
        steps=args.steps, lr=args.lr, dim=args.dim, slots=4, lora=lora,
        manifest_sha256=sha256(root / "manifest.json"),
        checkpoint_sha256=manifest["checkpoint_sha256"],
        scope="synthetic pilot; final-layer quantized-base LoRA plus recurrent sidecar"))
    train_cases = [c for c in cases if c["split"] == "train"]
    val_cases = [c for c in cases if c["split"] == "val"]
    rng = random.Random(args.seed)
    optimizer = optim.Adam(learning_rate=args.lr)

    def loss_fn(net, prefix, boundaries, target):
        logits, _ = net(prefix, boundaries, loops=args.loops)
        return nn.losses.cross_entropy(logits, target, reduction="mean")

    value_grad = nn.value_and_grad(model, loss_fn)
    start = time.perf_counter()
    log = []
    best_loss = float("inf")
    for step in range(1, args.steps + 1):
        case = rng.choice(train_cases)
        prefix, boundaries = cache[case["id"]]
        loss, grads = value_grad(model, prefix, boundaries, mx.array([case["label"]]))
        optimizer.update(model, grads)
        mx.eval(model.trainable_parameters(), optimizer.state, loss)
        if not math.isfinite(float(loss.item())):
            raise FloatingPointError("Non-finite training loss")
        if step % args.eval_every == 0 or step == args.steps:
            val = evaluate(model, val_cases, cache, args.loops)
            record = dict(step=step, loss=float(loss.item()), val=val["metrics"],
                          elapsed=time.perf_counter()-start)
            log.append(record)
            print(json.dumps(record), flush=True)
            if val["metrics"]["nll"] < best_loss:
                best_loss = val["metrics"]["nll"]
                mx.save_safetensors(str(out / "trainable.safetensors"), dict(tree_flatten(model.trainable_parameters())))
                write_json(out / "selection.json", {"step": step, "val_nll": best_loss})
        mx.clear_cache()
    model.load_weights(str(out / "trainable.safetensors"), strict=False)
    model.eval()
    results = {"scope": "pilot; no superiority claim from a single seed", "splits": {}}
    for split in ("test", "ood"):
        subset = [c for c in cases if c["split"] == split]
        results["splits"][split] = {f"depth_{depth}": evaluate(model, subset, cache, depth)
                                  for depth in sorted(set([0, 1, args.loops, 8]))}
        for intervention in ("reset", "zero"):
            results["splits"][split][intervention] = evaluate(model, subset, cache, args.loops, intervention)
        results["splits"][split]["native_qwen_restricted_ABC"] = {
            "metrics": metrics([r for r in manifest["native_predictions"] if r["split"] == split])}
    results.update(training_seconds=time.perf_counter()-start, mlx_peak_gb=mx.get_peak_memory()/1e9)
    write_json(out / "training.json", log)
    write_json(out / "evaluation.json", results)
    print(json.dumps({s:{k:v["metrics"]["accuracy"] for k,v in v.items()} for s,v in results["splits"].items()}), flush=True)


def smoke(args):
    import mlx.core as mx
    import mlx.nn as nn
    import mlx.optimizers as optim
    from mlx.utils import tree_flatten
    model, tokenizer = load_model(args.model)
    ids, boundaries = tokenize(generate()[0], tokenizer, args.max_tokens)
    prefix = model.encode_prefix(mx.array([ids]))
    concepts = model.concepts(prefix, boundaries)
    full = model.backbone.model(mx.array([ids]))[:, boundaries, :]
    difference = mx.max(mx.abs(full-concepts))
    mx.eval(difference)
    assert difference.item() < .01, difference.item()
    lora = attach_lora(model)
    initial = {k: v + 0 for k,v in tree_flatten(model.trainable_parameters()) if "lora_b" in k}
    def loss(net):
        logits, _ = net(prefix, boundaries, loops=4)
        return nn.losses.cross_entropy(logits, mx.array([0]), reduction="mean")
    value, grads = nn.value_and_grad(model, loss)(model)
    grads_flat = dict(tree_flatten(grads))
    grad_norms = {k:float(mx.linalg.norm(v).item()) for k,v in grads_flat.items() if "lora_b" in k}
    assert all(math.isfinite(v) and v > 0 for v in grad_norms.values()), grad_norms
    optimizer = optim.Adam(learning_rate=1e-4)
    optimizer.update(model, grads)
    mx.eval(model.trainable_parameters(), optimizer.state)
    updated = dict(tree_flatten(model.trainable_parameters()))
    assert all(float(mx.max(mx.abs(updated[k]-v)).item()) > 0 for k,v in initial.items())
    logits1, _ = model(prefix, boundaries, loops=1)
    logits4, trace = model(prefix, boundaries, loops=4)
    difference_loops = float(mx.max(mx.abs(logits1-logits4)).item())
    assert difference_loops > 1e-7
    result = dict(status="passed", device=str(mx.default_device()), tokens=len(ids),
                  cached_vs_full_max_error=float(difference.item()), lora_b_grad_norms=grad_norms,
                  depth1_vs_depth4_logit_delta=difference_loops, trace_steps=len(trace),
                  trainable_parameters=lora["parameters"], peak_gb=mx.get_peak_memory()/1e9)
    write_json(Path(args.out) / "smoke.json", result)
    print(json.dumps(result, indent=2))


def clef(args):
    # Explicit local entry point; retain Clef's trained joint schema head.
    import sys
    sys.path.insert(0, str(Path(args.clef_project).resolve()))
    from run import DecisionRunner
    runner = DecisionRunner()
    cases = json.loads((Path(args.cache) / "cases.json").read_text())
    subset = [c for c in cases if c["split"] in ("test", "ood")]
    rows = []
    for c in subset[:args.limit]:
        result = runner.decide(clef_record(c), max_tokens=args.max_tokens)
        probs = result["answers"]["readiness"]["probabilities"]
        rows.append(dict(id=c["id"], split=c["split"], label=c["label"],
                         probs=[probs[k] for k in LABELS], runtime=result["metrics"]))
        print(f"clef {len(rows)}/{min(args.limit,len(subset))}", flush=True)
    write_json(Path(args.out) / "clef.json", {"predictions": rows,
               "metrics": {s:metrics([r for r in rows if r["split"]==s]) for s in ("test","ood")}})


def decide(args):
    import mlx.core as mx
    import numpy as np
    if not args.input:
        raise ValueError("decide needs --input containing memory (list of lines) and query")
    root = Path(args.run)
    config = json.loads((root / "config.json").read_text())
    if sha256(Path(args.model) / "model.safetensors") != config["checkpoint_sha256"]:
        raise ValueError("Run/checkpoint mismatch")
    case = json.loads(Path(args.input).read_text())
    if not isinstance(case.get("memory"), list) or not all(isinstance(x, str) for x in case["memory"]):
        raise ValueError("memory must be a list of strings")
    if not isinstance(case.get("query"), str) or not case["query"].strip():
        raise ValueError("query must be a nonempty string")
    case = {"id": "interactive", "memory": [x.rstrip()+"\n" for x in case["memory"]], "query": case["query"]}
    model, tokenizer = load_model(args.model, config["seed"], config["dim"])
    attach_lora(model)
    model.load_weights(str(root / "trainable.safetensors"), strict=False)
    model.eval()
    ids, boundaries = tokenize(case, tokenizer, args.max_tokens)
    start = time.perf_counter()
    logits, trace = model(model.encode_prefix(mx.array([ids])), boundaries, loops=config["loops"])
    probs = mx.softmax(logits, axis=-1)
    mx.eval(probs)
    values = np.array(probs[0]).tolist()
    response = dict(model="experimental-qwen-latent-loop", decision=LABELS[max(range(3), key=lambda i:values[i])],
                    probabilities=dict(zip(LABELS, values)), loops=config["loops"],
                    input_tokens=len(ids), end_to_end_seconds=time.perf_counter()-start,
                    scope="Uncalibrated synthetic research prototype")
    print(json.dumps(response, indent=2))


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("command", choices=["prepare", "smoke", "train", "clef", "decide"])
    p.add_argument("--model", default=DEFAULT_MODEL)
    p.add_argument("--cache", default="results/cache")
    p.add_argument("--out", default="results/smoke")
    p.add_argument("--seed", type=int, default=17)
    p.add_argument("--max-tokens", type=int, default=512)
    p.add_argument("--dim", type=int, default=128)
    p.add_argument("--loops", type=int, default=4)
    p.add_argument("--steps", type=int, default=240)
    p.add_argument("--eval-every", type=int, default=60)
    p.add_argument("--lr", type=float, default=1e-4)
    p.add_argument("--clef-project", default="/Users/adelmansour/Desktop/jev like")
    p.add_argument("--limit", type=int, default=96)
    p.add_argument("--input")
    p.add_argument("--run", default="results/loop4-seed17-2400")
    args = p.parse_args()
    if args.command == "train" and (args.loops < 1 or args.steps < 1 or args.eval_every < 1):
        p.error("Training loops, steps, and eval-every must be positive")
    {"prepare":prepare, "smoke":smoke, "train":train, "clef":clef, "decide":decide}[args.command](args)


if __name__ == "__main__":
    main()
