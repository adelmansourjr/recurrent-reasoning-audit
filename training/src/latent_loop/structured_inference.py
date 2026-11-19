"""Runnable v3 memory compiler + Qwen LoRA grounder + learned recurrent core.

Dependencies come from a documented record grammar or explicit typed inputs.
No readiness solver/oracle is called by this inference path.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import re
import time

from latent_loop.memory_compile import newest_report


def compile_case(case):
    deps,reports={},{}
    for raw in case["memory"]:
        line=raw.strip()
        match=re.fullmatch(r"(\S+) requires (.+)\.",line)
        if not match:
            match=re.fullmatch(r"Prerequisites for (\S+): (.+)\.",line)
        if match:
            name,children=match.groups()
            values=children.split(" and ")
            if name in deps and deps[name]!=values:
                raise ValueError("Conflicting dependency declarations")
            deps[name]=values
            continue
        match=re.fullmatch(r"At time (\d+), (\S+) is (.+)\.",line)
        if not match:
            match=re.fullmatch(r"Observation t=(\d+): (\S+) has status (.+)\.",line)
        if not match:
            raise ValueError(f"Unsupported memory syntax: {line}")
        timestamp,name,phrase=match.groups()
        reports.setdefault(name,[]).append(f"Time {timestamp}: the service is {phrase}.")
    query=case["query"]
    names=sorted(set(deps)|set(reports)|{query}|{j for children in deps.values() for j in children})
    if len(names)>256:
        raise ValueError("This prototype supports at most 256 concepts")
    index={name:i for i,name in enumerate(names)}
    edges=[[index[j] for j in deps.get(name,[])] for name in names]
    if any(len(set(children))>9 for children in edges):
        raise ValueError("This verified local operator supports at most nine prerequisites per concept")
    visited,active=set(),set()
    def check(i):
        if i in active:
            raise ValueError("Cyclic dependency graph")
        if i in visited:
            return
        active.add(i)
        for j in edges[i]:
            check(j)
        active.remove(i)
        visited.add(i)
    for i in range(len(names)):
        check(i)
    encoded={index[name]:newest_report("\n".join(lines)) for name,lines in reports.items()
             if not edges[index[name]]}
    return dict(names=names,deps=edges,query=index[query],reports=encoded)


class Runtime:
    def __init__(self,seed=17,root="results/v3",core_root="results/v4"):
        import mlx.core as mx
        from latent_loop.experiment import load_model,attach_lora,DEFAULT_MODEL,sha256
        from latent_loop.structured_model import Grounder,BeliefGraph
        from mlx.utils import tree_flatten
        start=time.perf_counter()
        manifest=json.loads((Path(root)/"cache/manifest.json").read_text())
        if sha256(Path(DEFAULT_MODEL)/"model.safetensors")!=manifest["checkpoint_sha256"] or sha256(Path(DEFAULT_MODEL)/"tokenizer.json")!=manifest["tokenizer_sha256"]:
            raise ValueError("Qwen checkpoint/tokenizer does not match the training manifest")
        wrapper,self.tokenizer=load_model(DEFAULT_MODEL,seed)
        attach_lora(wrapper)
        mx.random.seed(seed+7919)
        self.grounder=Grounder(wrapper.backbone)
        weights=mx.load(str(Path(root)/f"grounder-{seed}/trainable.safetensors"))
        expected=dict(tree_flatten(self.grounder.trainable_parameters()))
        if weights.keys()!=expected.keys() or any(weights[k].shape!=expected[k].shape for k in expected):
            raise ValueError("Incomplete or incompatible grounding adapter/head")
        self.grounder.load_weights(list(weights.items()),strict=False)
        self.processor=BeliefGraph()
        self.processor.load_weights(str(Path(core_root)/f"core-{seed}/trainable.safetensors"))
        self.encoder=wrapper
        self.cache={}
        self.load_seconds=time.perf_counter()-start

    def ground(self,text):
        import mlx.core as mx
        import numpy as np
        if text not in self.cache:
            ids=self.tokenizer.encode(text,add_special_tokens=False)
            prefix=self.encoder.encode_prefix(mx.array([ids]))
            probs=mx.softmax(self.grounder(prefix),axis=-1)
            mx.eval(probs)
            self.cache[text]=np.array(probs[0]).tolist()
        return self.cache[text]

    def decide(self,case,diagnostics=None):
        import mlx.core as mx
        import numpy as np
        from latent_loop.information import optimal_information_policy
        names=("blocked","unknown","ready")
        start=time.perf_counter()
        graph=compile_case(case)
        n=len(graph["names"])
        p=np.tile([0.,1.,0.],(n,1)).astype(np.float32)
        adjacency=np.zeros((n,n),dtype=np.float32)
        leaves=np.array([not x for x in graph["deps"]])
        for i,children in enumerate(graph["deps"]):
            adjacency[i,children if children else [i]]=1
            if leaves[i] and i in graph["reports"]:
                p[i]=self.ground(graph["reports"][i])
        # N is a sufficient structural bound for any N-vertex acyclic graph.
        final,_=self.processor(mx.array(p[None]),mx.array(adjacency[None]),mx.array(leaves[None]),n)
        mx.eval(final)
        states=np.array(final[0])
        probs=states[graph["query"]]
        state=names[int(np.argmax(probs))]
        result=dict(decision=state,probabilities=dict(zip(names,probs.tolist())),loops=n,
            leaf_beliefs={graph["names"][i]:dict(zip(names,p[i].tolist())) for i in range(n) if leaves[i]},
            scope="Supported memory grammar; supplied relations; uncalibrated learned categorical states")
        if state=="unknown" and diagnostics:
            reachable,frontier=set(),[]
            def visit(i):
                if i in reachable:
                    return
                reachable.add(i)
                if leaves[i] and int(np.argmax(p[i]))==1:
                    frontier.append(graph["names"][i])
                for j in graph["deps"][i]:
                    visit(j)
            visit(graph["query"])
            if len(frontier)>12:
                result["action"]={"type":"inspect","missing":frontier,"reason":"Exact planning budget is limited to 12 unknown facts"}
            elif any(name not in diagnostics for name in frontier):
                result["action"]={"type":"inspect","missing":frontier,"reason":"Priors/costs not supplied for every unknown leaf"}
            elif frontier:
                priors=[diagnostics[x]["ready_prior"] for x in frontier]
                costs=[diagnostics[x]["cost"] for x in frontier]
                policy=optimal_information_policy(priors,costs)
                value,action=policy((1<<len(frontier))-1)
                result["action"]={"type":"observe","concept":frontier[action],"expected_utility":value} if isinstance(action,int) else {
                    "type":"act","decision":action,"expected_utility":value}
        else:
            result["action"]={"type":"act" if state!="unknown" else "inspect","decision":state}
        result["seconds_after_loading"]=time.perf_counter()-start
        return result


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("--input",default="configs/example.json")
    p.add_argument("--seed",type=int,default=17)
    p.add_argument("--out",default="results/v4/example.json")
    p.add_argument("--original-benchmark",action="store_true")
    args=p.parse_args()
    from latent_loop.experiment import write_json
    runtime=Runtime(args.seed)
    if args.original_benchmark:
        cases=json.loads(Path("results/cache/cases.json").read_text())
        original=("ready","blocked","inspect")
        rows=[]
        for case in cases:
            if case["split"] not in ("test","ood"):
                continue
            prediction=runtime.decide({"memory":case["memory"],"query":case["query"]})
            actual="unknown" if original[case["label"]]=="inspect" else original[case["label"]]
            rows.append(dict(id=case["id"],split=case["split"],correct=prediction["decision"]==actual,
                             label=actual,prediction=prediction))
        result=dict(scope="Retrospective comparison on earlier known benchmark; not a new confirmatory test",
            seed=args.seed,splits={s:dict(n=sum(r["split"]==s for r in rows),
                accuracy=sum(r["correct"] for r in rows if r["split"]==s)/sum(r["split"]==s for r in rows)) for s in ["test","ood"]},
            predictions=rows,unique_neural_groundings=len(runtime.cache))
    else:
        case=json.loads(Path(args.input).read_text())
        result=runtime.decide(case,case.get("diagnostics"))
        result["load_seconds"]=runtime.load_seconds
    write_json(args.out,result)
    print(json.dumps({k:v for k,v in result.items() if k!="predictions"},indent=2))


if __name__=="__main__":
    main()
