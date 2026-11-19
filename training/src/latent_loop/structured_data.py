"""Version 2: query-matched worlds and explicit, observation-only graph inputs.

State order is blocked=0, unknown=1, ready=2, so conjunction is min.
Every world contains all three query answers with IDENTICAL memory content.
Graphs are supplied structured inputs; no natural-language relation parser is
claimed. No world truth or target intermediate state enters model inputs.
"""
from __future__ import annotations

import hashlib
import json
import random
import re

STATES = ("blocked", "unknown", "ready")
PHRASES = {
    "train": (("blocked", "unavailable", "failing"),
              ("unknown", "not measured", "undetermined"),
              ("ready", "operational", "working")),
    "val": (("blocked", "failing", "unavailable"),
            ("not measured", "unknown", "undetermined"),
            ("working", "ready", "operational")),
    "test": (("blocked", "unavailable", "failing"),
             ("unknown", "not measured", "undetermined"),
             ("ready", "operational", "working")),
    "language": (("out of service", "not functioning"),
                 ("not yet assessed", "no reliable measurement"),
                 ("functioning normally", "available for use")),
}


def local_record(state, rng, split="train"):
    """Grounding requires resolving shuffled time-stamped semantic reports."""
    phrases = PHRASES[split]
    old = rng.randrange(3)
    old_phrase, new_phrase = rng.choice(phrases[old]), rng.choice(phrases[state])
    early, late = rng.sample(range(1, 10), 2)
    early, late = min(early, late), max(early, late)
    if split == "language":
        lines = [f"Report timestamp {early}: {old_phrase}.",
                 f"Report timestamp {late}: {new_phrase}."]
    else:
        forms = ["At time {t}, status is {p}.", "Time {t}: the service is {p}.",
                 "Status report at {t}: {p}."]
        lines = [rng.choice(forms).format(t=early,p=old_phrase),
                 rng.choice(forms).format(t=late,p=new_phrase)]
    rng.shuffle(lines)
    return "Use the newest timestamp. Map the service status to blocked, unknown or ready.\n" + "\n".join(lines) + "\nCurrent status:"


def grounding_dataset(seed=731):
    result = []
    seen = set()
    counts = {"train": 240, "val": 60, "test": 120, "language": 120}
    for si, (split,n) in enumerate(counts.items()):
        rng = random.Random(seed+1009*si)
        for label in range(3):
            for _ in range(n//3):
                while True:
                    text = local_record(label, rng, split)
                    digest = hashlib.sha256(text.encode()).hexdigest()
                    if digest not in seen:
                        break
                seen.add(digest)
                result.append(dict(id=f"{split}-{len(result):05d}",split=split,text=text,
                                   label=label,input_sha256=digest))
    return result


def compiled_grounding_dataset(seed=8773):
    from latent_loop.memory_compile import newest_report
    result=[]
    seen=set()
    counts={"train":360,"val":90,"test":180,"language":180}
    for si,(split,n) in enumerate(counts.items()):
        rng=random.Random(seed+1009*si)
        for label in range(3):
            for _ in range(n//3):
                while True:
                    raw=local_record(label,rng,split)
                    # Disjoint ranges plus explicit hash checks prevent selected-
                    # observation duplication between train/validation/test.
                    numbers=sorted({int(x) for x in re.findall(r'\b\d+\b',raw)})
                    replacements={old:si*10000+rng.randrange(100,9000) for old in numbers}
                    ordered=sorted(replacements.values())
                    replacements=dict(zip(numbers,ordered))
                    raw=re.sub(r'\b\d+\b',lambda m:str(replacements[int(m.group())]),raw)
                    encoded=newest_report(raw)
                    digest=hashlib.sha256(encoded.encode()).hexdigest()
                    if digest not in seen:
                        break
                seen.add(digest)
                result.append(dict(id=f"{split}-{len(result):05d}",split=split,text=raw,
                    encoded_text=encoded,label=label,input_sha256=digest))
    return result


def trace_states(deps, leaf_states, steps):
    """Exact synchronous operator, used ONLY for labels/reference evaluation."""
    current = [leaf_states[i] if not deps[i] else 1 for i in range(len(deps))]
    trace = [current]
    for _ in range(steps):
        current = [min(current[j] for j in required) if required else leaf_states[i]
                   for i, required in enumerate(deps)]
        trace.append(current)
    return trace


def oracle(deps, leaf_states):
    memo, visiting = {}, set()
    def visit(i):
        if i in visiting:
            raise ValueError("cycle")
        if i in memo:
            return memo[i]
        visiting.add(i)
        value = min(visit(j) for j in deps[i]) if deps[i] else leaf_states[i]
        memo[i] = value
        visiting.remove(i)
        return value
    return [visit(i) for i in range(len(deps))]


def required_leaves(deps, query):
    visited, leaves = set(), set()
    def visit(i):
        if i in visited:
            return
        visited.add(i)
        if not deps[i]:
            leaves.add(i)
        for j in deps[i]:
            visit(j)
    visit(query)
    return sorted(leaves)


def make_world(rng, depth, fanout, world_id, split):
    deps, values, roots = [], [], []
    for desired in range(3):
        base = len(deps)
        # Long chain plus shared terminal dependencies: a DAG, not a tree.
        for i in range(depth):
            children = [base+i+1] if i+1 < depth else [base+depth]
            extra = list(range(base+depth,base+depth+fanout))
            rng.shuffle(extra)
            children = sorted(set(children+extra[:rng.randrange(fanout+1)]))
            deps.append(children)
            values.append(1)  # ignored for internal nodes
        for _ in range(fanout):
            deps.append([])
            values.append(2)
        root = base
        roots.append(root)
        reachable = required_leaves(deps,root)
        # Fix the root's state while varying which reachable leaf carries it.
        if desired != 2:
            values[rng.choice(reachable)] = desired
    n=len(deps)
    order=list(range(n))
    rng.shuffle(order)
    old_to_new={old:new for new,old in enumerate(order)}
    deps_new=[[old_to_new[j] for j in deps[i]] for i in order]
    values_new=[values[i] for i in order]
    roots_new=[old_to_new[i] for i in roots]
    names=rng.sample([f"N{i:04d}" for i in range(10000)],n)
    actual=oracle(deps_new,values_new)
    assert [actual[i] for i in roots_new]==[0,1,2]
    identity={"deps":deps_new,"leaf_states":values_new,"roots":roots_new}
    fingerprint=hashlib.sha256(json.dumps(identity,sort_keys=True).encode()).hexdigest()
    return dict(id=world_id,split=split,deps=deps_new,leaf_states=values_new,
                roots=roots_new,names=names,depth=depth,fanout=fanout,
                labels=actual,world_sha256=fingerprint)


def worlds(seed=1127, counts=None):
    counts=counts or {"train":256,"val":64,"test":200,"depth12":200,"depth24":200,"fanout8":200}
    result=[]
    seen=set()
    for si,(split,count) in enumerate(counts.items()):
        rng=random.Random(seed+100003*si)
        for i in range(count):
            depth=12 if split=="depth12" else 24 if split=="depth24" else 48 if split=="depth48" else rng.randint(1,4)
            fanout=8 if split=="fanout8" else rng.randint(1,3)
            while True:
                w=make_world(rng,depth,fanout,f"{split}-{i:05d}",split)
                if w["world_sha256"] not in seen:
                    break
            seen.add(w["world_sha256"])
            result.append(w)
    return result


def render_world(world,query):
    """An inspection/export format; all three queries share the same memories."""
    lines=[]
    for i,required in enumerate(world["deps"]):
        if required:
            lines.append(f"{world['names'][i]} requires "+" and ".join(world['names'][j] for j in required)+".")
        else:
            lines.append(f"{world['names'][i]} is {STATES[world['leaf_states'][i]]}.")
    return "\n".join(lines)+f"\nQuery: {world['names'][query]}"
