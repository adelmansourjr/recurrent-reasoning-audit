"""Procedural prerequisite reasoning, with exact three-valued logic labels.

Only rendered observations reach the neural model. Oracle state stays in the
dataset for evaluation; no answer, depth or missing-fact indicator is an input.
This is a static decision benchmark, not yet a closed-loop POMDP environment.
"""
from __future__ import annotations

import hashlib
import json
import random
from collections import Counter

LABELS = ("ready", "blocked", "inspect")
RULE = ("A service is ready only when every prerequisite is ready. A blocked "
        "prerequisite blocks it. If none is blocked but evidence is missing, "
        "inspect before acting. For repeated observations use the largest time.\n")


def oracle(deps, observations, query):
    latest = {}
    for name, time, value in observations:
        if name not in latest or time > latest[name][0]:
            latest[name] = time, value
    visiting = set()

    def visit(name):
        if name in visiting:
            raise ValueError("Cyclic prerequisites")
        if not deps.get(name):
            return latest.get(name, (0, "inspect"))[1]
        visiting.add(name)
        states = [visit(dep) for dep in deps[name]]
        visiting.remove(name)
        return "blocked" if "blocked" in states else "inspect" if "inspect" in states else "ready"

    return visit(query)


def render(case):
    # Explicit allowlist prevents oracle labels/metadata from entering prompts.
    lines = [RULE]
    lines.extend(case["memory"])
    lines.append(f"Decide readiness of {case['query']}. Options: ready, blocked, inspect.\n")
    return lines


def _case(rng, depth, namespace, paraphrase):
    names = rng.sample([f"{namespace}{i}" for i in range(64)], depth + 5)
    # Two chains merge at query. Two unrelated leaf observations are distractors.
    deps = {names[i]: [names[i + 1]] for i in range(depth)}
    deps[names[0]].append(names[depth + 1])
    leaves = [names[depth], names[depth + 1]]
    observations = []
    for name in leaves + names[depth + 2:]:
        value = rng.choice(LABELS)
        if value != "inspect":
            observations.append((name, 9, value))
        # Old evidence may contradict new evidence, including a now-unknown leaf.
        # Do not add old evidence for an unknown leaf: that would make it known.
        if value != "inspect" and rng.random() < .5:
            observations.append((name, 2, "blocked" if value == "ready" else "ready"))
    rng.shuffle(observations)
    memory = []
    edges = list(deps.items())
    rng.shuffle(edges)
    for name, required in edges:
        joined = " and ".join(required)
        memory.append((f"{name} requires {joined}.\n" if not paraphrase else
                       f"Prerequisites for {name}: {joined}.\n"))
    for name, time, value in observations:
        memory.append((f"At time {time}, {name} is {value}.\n" if not paraphrase else
                       f"Observation t={time}: {name} has status {value}.\n"))
    # Interleave event presentation; timestamps, not list order, define recency.
    rng.shuffle(memory)
    query = names[0]
    return dict(memory=memory, query=query, deps=deps,
                observations=observations, depth=depth,
                label=LABELS.index(oracle(deps, observations, query)))


def generate(seed=17, counts=None):
    counts = counts or {"train": 96, "val": 24, "test": 48, "ood": 48}
    if any(n < 3 or n % 3 for n in counts.values()):
        raise ValueError("Each split must have a positive multiple of three cases")
    data = []
    hashes = set()
    for si, (split, count) in enumerate(counts.items()):
        rng = random.Random(seed + 1009 * si)
        used = Counter()
        target = count // 3
        while sum(used.values()) < count:
            case = _case(rng, rng.choice([4, 5] if split == "ood" else [1, 2, 3]),
                         "X" if split == "ood" else "S", split == "ood")
            digest = hashlib.sha256("".join(render(case)).encode()).hexdigest()
            if used[case["label"]] >= target or digest in hashes:
                continue
            case.update(id=f"{split}-{sum(used.values()):04d}", split=split,
                        input_sha256=digest)
            data.append(case)
            hashes.add(digest)
            used[case["label"]] += 1
    return data


def clef_record(case):
    return {"state": "".join(render(case)), "questions": {"readiness": {
        "type": "choice", "instructions": "Decide readiness using only the supplied rules and observations.",
        "criteria": {"ready": "Every required leaf is ready.",
                     "blocked": "At least one required leaf is blocked.",
                     "inspect": "No known blocker, but at least one required leaf has no observation."}}}}


def canonical_json(value):
    return json.dumps(value, sort_keys=True, separators=(",", ":"))
