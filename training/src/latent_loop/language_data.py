"""Human-written language benchmark. Logic annotations are targets, never inputs."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path
import random
import re

LABELS = ("False", "Unknown", "True")


def story_key(premises):
    # Group parapunctuation/whitespace changes conservatively when splitting.
    normalized = sorted(re.sub(r"\W+", " ", p.lower()).strip() for p in premises)
    return hashlib.sha256(json.dumps(normalized).encode()).hexdigest()


def predicates(formula):
    bound = set(re.findall(r"[∀∃]\s*([A-Za-z_][A-Za-z_0-9]*)", formula))
    return set(re.findall(r"([A-Za-z_][A-Za-z_0-9]*)\s*\(", formula)) - bound


def load_folio(root="data/folio-v0", seed=9127):
    root = Path(root)
    train = [json.loads(x) for x in (root / "train.jsonl").read_text().splitlines()]
    test = [json.loads(x) for x in (root / "validation.jsonl").read_text().splitlines()]
    train_keys = {story_key(x["premises"]) for x in train}
    test_keys = {story_key(x["premises"]) for x in test}
    if train_keys & test_keys:
        raise ValueError("Train/test premise-family overlap")
    keys = sorted(train_keys)
    random.Random(seed).shuffle(keys)
    dev = set(keys[:round(.15 * len(keys))])
    cases = []
    for source, rows in (("train", train), ("external", test)):
        for i, row in enumerate(rows):
            key = story_key(row["premises"])
            label = row["label"].replace("Uncertain", "Unknown")
            concepts = [sorted(predicates(f)) for f in row["premises-FOL"]]
            edges = [[int(bool(set(a) & set(b))) for b in concepts] for a in concepts]
            link_valid = len(concepts) == len(row['premises'])
            cases.append(dict(id=f"{source}-{i:04d}", story=key,
                split="dev" if source == "train" and key in dev else source,
                premises=row["premises"], conclusion=row["conclusion"],
                label=LABELS.index(label), concept_targets=concepts, edge_targets=edges if link_valid else [],link_valid=link_valid))
    return cases


def encode_text(tokenizer, premises, conclusion, query_only=False):
    """No predicate vocabulary, formula, label, or gold adjacency enters this function."""
    ids = tokenizer.encode(
        "Reason only from the stated premises. A conclusion may be True, False, or Unknown.\n",
        add_special_tokens=False)
    spans = []
    for i, p in enumerate(premises if not query_only else []):
        start = len(ids)
        ids += tokenizer.encode(f"Premise {i+1}: {p.strip()}\n", add_special_tokens=False)
        spans.append([start, len(ids)])
    start = len(ids)
    ids += tokenizer.encode(f"Conclusion: {conclusion.strip()}\nAnswer:", add_special_tokens=False)
    spans.append([start, len(ids)])
    return ids, spans
