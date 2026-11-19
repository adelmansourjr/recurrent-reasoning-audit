"""Version-2 experiment stages. Every stage writes separate reviewable outputs."""
from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import random
import time

os.environ.update(HF_HUB_OFFLINE="1",TRANSFORMERS_OFFLINE="1",TOKENIZERS_PARALLELISM="false")

from latent_loop.experiment import DEFAULT_MODEL,load_model,sha256,write_json
from latent_loop.structured_data import grounding_dataset,compiled_grounding_dataset,worlds,trace_states,STATES


def prepare(args):
    import mlx.core as mx
    import numpy as np
    out=Path(args.root)/"cache"
    if (out/"manifest.json").exists():
        raise ValueError("Existing cache; preserve it")
    out.mkdir(parents=True,exist_ok=True)
    net,tok=load_model(DEFAULT_MODEL)
    cases=compiled_grounding_dataset() if args.compiled else grounding_dataset()
    start=time.perf_counter()
    for i,c in enumerate(cases):
        ids=tok.encode(c.get("encoded_text",c["text"]),add_special_tokens=False)
        h=net.encode_prefix(mx.array([ids]))
        mx.eval(h)
        np.savez(out/f"{c['id']}.npz",prefix=np.array(h.astype(mx.float32)),bfloat16=np.array(h.dtype==mx.bfloat16))
        if (i+1)%60==0:
            print(f"grounding cache {i+1}/{len(cases)} {time.perf_counter()-start:.1f}s",flush=True)
        del h
        mx.clear_cache()
    write_json(out/"grounding.json",cases)
    graph_data=worlds(seed=77171,counts={"test":300,"depth12":300,"depth24":300,"depth48":200,"fanout8":300}) if args.compiled else worlds()
    write_json(out/"worlds.json",graph_data)
    write_json(out/"manifest.json",dict(model=DEFAULT_MODEL,checkpoint_sha256=sha256(Path(DEFAULT_MODEL)/"model.safetensors"),
        tokenizer_sha256=sha256(Path(DEFAULT_MODEL)/"tokenizer.json"),
        grounding_sha256=sha256(out/"grounding.json"),worlds_sha256=sha256(out/"worlds.json"),
        cache_sha256={p.name:sha256(p) for p in out.glob("*.npz")},seconds=time.perf_counter()-start,
        boundary="before final Qwen block",data_seed=8773 if args.compiled else 731,
        world_seed=77171 if args.compiled else 1127,compiled=args.compiled))


def read_grounding(root):
    import mlx.core as mx
    import numpy as np
    cases=json.loads((root/"cache/grounding.json").read_text())
    cache={}
    for c in cases:
        with np.load(root/"cache"/f"{c['id']}.npz") as z:
            cache[c["id"]]=mx.array(z["prefix"],dtype=mx.bfloat16 if bool(z["bfloat16"]) else mx.float16)
    return cases,cache


def grounding_eval(model,cases,cache):
    import mlx.core as mx
    import numpy as np
    rows=[]
    for c in cases:
        probs=mx.softmax(model(cache[c["id"]]),axis=-1)
        mx.eval(probs)
        rows.append(dict(id=c["id"],label=c["label"],split=c["split"],probs=np.array(probs[0]).tolist()))
    acc=sum(max(range(3),key=lambda k:r["probs"][k])==r["label"] for r in rows)/len(rows)
    nll=-sum(np.log(max(r["probs"][r["label"]],1e-9)) for r in rows)/len(rows)
    return dict(accuracy=acc,nll=float(nll),n=len(rows),predictions=rows)


def train_grounder(args):
    import mlx.core as mx
    import mlx.nn as nn
    import mlx.optimizers as optim
    from mlx.utils import tree_flatten
    from latent_loop.experiment import attach_lora
    from latent_loop.structured_model import Grounder
    root=Path(args.root)
    if args.head=="mlp" and args.lora:
        raise ValueError("The MLP capacity control freezes Qwen; specify --no-lora")
    suffix=f"{args.seed}" if args.lora else f"mlp-{args.seed}" if args.head=="mlp" else f"frozen-{args.seed}"
    out=root/f"grounder-{suffix}"
    if (out/"evaluation.json").exists():
        print("grounder already complete",flush=True)
        return
    manifest=json.loads((root/"cache/manifest.json").read_text())
    if sha256(Path(DEFAULT_MODEL)/"model.safetensors")!=manifest["checkpoint_sha256"]:
        raise ValueError("Qwen checkpoint changed")
    cases,cache=read_grounding(root)
    wrapper,_=load_model(DEFAULT_MODEL,args.seed)
    if args.lora:
        attach_lora(wrapper)
    mx.random.seed(args.seed+7919)
    model=Grounder(wrapper.backbone,head=args.head)
    # Frozen-prefix features are reused identically by frozen and LoRA controls.
    count=sum(v.size for _,v in tree_flatten(model.trainable_parameters()))
    train=[c for c in cases if c["split"]=="train"]
    val=[c for c in cases if c["split"]=="val"]
    rng=random.Random(args.seed)
    optimizer=optim.Adam(learning_rate=2e-4)
    def loss(net,h,y):
        return nn.losses.cross_entropy(net(h),y,reduction="mean")
    vg=nn.value_and_grad(model,loss)
    best=float("inf")
    log=[]
    out.mkdir(parents=True,exist_ok=True)
    write_json(out/"config.json",dict(seed=args.seed,lora=args.lora,head=args.head,steps=args.steps,trainable_parameters=count,
        cache_manifest_sha256=sha256(root/"cache/manifest.json")))
    start=time.perf_counter()
    for step in range(1,args.steps+1):
        c=rng.choice(train)
        value,grads=vg(model,cache[c["id"]],mx.array([c["label"]]))
        optimizer.update(model,grads)
        mx.eval(value,model.trainable_parameters(),optimizer.state)
        if step%200==0 or step==args.steps:
            result=grounding_eval(model,val,cache)
            log.append(dict(step=step,loss=float(value.item()),val_accuracy=result["accuracy"],val_nll=result["nll"]))
            print(f"grounder {args.seed} {log[-1]}",flush=True)
            if result["nll"]<best:
                best=result["nll"]
                mx.save_safetensors(str(out/"trainable.safetensors"),dict(tree_flatten(model.trainable_parameters())))
                write_json(out/"selection.json",dict(step=step,val_nll=best))
        mx.clear_cache()
    model.load_weights(str(out/"trainable.safetensors"),strict=False)
    evaluations={s:grounding_eval(model,[c for c in cases if c["split"]==s],cache) for s in ["val","test","language"]}
    write_json(out/"evaluation.json",dict(splits=evaluations,seconds=time.perf_counter()-start,peak_gb=mx.get_peak_memory()/1e9))
    write_json(out/"training.json",log)
    print(json.dumps({s:r["accuracy"] for s,r in evaluations.items()}),flush=True)


def graph_arrays(items,steps=4,noise=0.0,rng=None,belief_bank=None):
    import numpy as np
    rng=rng or np.random.default_rng(17)
    b,n=len(items),max(len(w["deps"]) for w in items)
    initial=np.zeros((b,n,3),dtype=np.float32)
    initial[:,:,1]=1
    adj=np.zeros((b,n,n),dtype=np.float32)
    leaves=np.ones((b,n),dtype=bool)
    valid=np.zeros((b,n),dtype=np.float32)
    roots=np.zeros((b,3),dtype=np.int32)
    labels=np.zeros((b,3),dtype=np.int32)
    target=np.ones((b,steps,n),dtype=np.int32)
    for bi,w in enumerate(items):
        size=len(w["deps"])
        trace=trace_states(w["deps"],w["leaf_states"],steps)
        target[bi,:,:size]=np.array(trace[1:])
        roots[bi]=w["roots"]
        labels[bi]=[w["labels"][q] for q in w["roots"]]
        for i,required in enumerate(w["deps"]):
            if required:
                adj[bi,i,required]=1
                leaves[bi,i]=False
                valid[bi,i]=1
            else:
                adj[bi,i,i]=1
                state=w["leaf_states"][i]
                if belief_bank is None:
                    initial[bi,i]=np.eye(3,dtype=np.float32)[state]
                else:
                    rows=belief_bank[state]
                    initial[bi,i]=rows[int(rng.integers(len(rows)))]
        for i in range(size,n):
            adj[bi,i,i]=1
    if noise:
        eps=rng.uniform(0,noise,(b,n,1)).astype(np.float32)
        initial=(1-eps)*initial+eps*rng.dirichlet([1,1,1],(b,n)).astype(np.float32)
        # Only leaf anchors are perturbed; internal nodes start truly unknown.
        initial=np.where(leaves[:,:,None],initial,np.array([0,1,0],dtype=np.float32))
    return initial,adj,leaves,valid,roots,labels,target


def core_eval(model,items,steps,belief_bank=None,reset=False,permutation=False,seed=771):
    import mlx.core as mx
    import numpy as np
    rng=np.random.default_rng(seed)
    rows=[]
    start=time.perf_counter()
    for offset in range(0,len(items),24):
        batch=items[offset:offset+24]
        initial,adj,leaves,valid,roots,labels,_=graph_arrays(batch,1,rng=rng,belief_bank=belief_bank)
        if permutation:
            # All input tensors and query pointers are renamed consistently.
            for i in range(len(batch)):
                perm=rng.permutation(len(batch[i]["deps"]))
                inv=np.argsort(perm)
                m=len(perm)
                initial[i,:m]=initial[i,perm].copy()
                adj[i,:m,:m]=adj[i,:m,:m][perm][:,perm].copy()
                leaves[i,:m]=leaves[i,perm].copy()
                roots[i]=inv[roots[i]]
        final,_=model(mx.array(initial),mx.array(adj),mx.array(leaves),steps,reset=reset)
        query=final[mx.arange(len(batch))[:,None],mx.array(roots)]
        mx.eval(query)
        values=np.array(query)
        for i,w in enumerate(batch):
            rows.append(dict(id=w["id"],labels=labels[i].tolist(),probs=values[i].tolist(),
                correct=(values[i].argmax(-1)==labels[i]).astype(int).tolist()))
    correct=np.array([r["correct"] for r in rows])
    nll=-np.mean([np.log(max(p[y],1e-9)) for r in rows for p,y in zip(r["probs"],r["labels"])])
    return dict(worlds=len(rows),queries=len(rows)*3,accuracy=float(correct.mean()),
        all_three_correct=float(correct.all(-1).mean()),nll=float(nll),
        seconds=time.perf_counter()-start,predictions=rows)


def train_core(args):
    import mlx.core as mx
    import mlx.nn as nn
    import mlx.optimizers as optim
    import numpy as np
    from mlx.utils import tree_flatten
    from latent_loop.structured_model import BeliefGraph
    root=Path(args.root)
    name=f"core-{args.objective}-{args.seed}"
    out=root/name
    if (out/"evaluation.json").exists():
        print("core already complete",flush=True)
        return
    mx.random.seed(args.seed)
    mx.set_cache_limit(128*1024*1024)
    rng=np.random.default_rng(args.seed)
    dataset=json.loads((root/"cache/worlds.json").read_text())
    train=[w for w in dataset if w["split"]=="train"]
    val=[w for w in dataset if w["split"]=="val"]
    model=BeliefGraph()
    optimizer=optim.Adam(learning_rate=1e-3)
    def loss_fn(net,initial,adj,leaves,valid,roots,labels,target):
        final,trace=net(initial,adj,leaves,4)
        if args.objective=="process":
            all_p=mx.stack(trace,axis=1)
            loss=-mx.take_along_axis(mx.log(mx.maximum(all_p,1e-8)),target[:,:,:,None],axis=-1).squeeze(-1)
            return mx.sum(loss*valid[:,None,:])/(4*mx.sum(valid))
        query=final[mx.arange(final.shape[0])[:,None],roots]
        return -mx.mean(mx.take_along_axis(mx.log(mx.maximum(query,1e-8)),labels[:,:,None],axis=-1))
    vg=nn.value_and_grad(model,loss_fn)
    best=float("inf")
    out.mkdir(parents=True,exist_ok=True)
    write_json(out/"config.json",dict(seed=args.seed,objective=args.objective,steps=args.steps,training_depth=4,
        dim=48,trainable_parameters=sum(v.size for _,v in tree_flatten(model.trainable_parameters())),
        worlds_sha256=sha256(root/"cache/worlds.json"),inputs="supplied dependency graph and initial leaf beliefs"))
    start=time.perf_counter()
    log=[]
    for step in range(1,args.steps+1):
        selected=[train[i] for i in rng.integers(0,len(train),24)]
        arrays=graph_arrays(selected,4,noise=.03,rng=rng)
        value,grads=vg(model,*[mx.array(x) for x in arrays])
        optimizer.update(model,grads)
        mx.eval(value,model.trainable_parameters(),optimizer.state)
        if step%200==0 or step==args.steps:
            result=core_eval(model,val,4)
            log.append(dict(step=step,loss=float(value.item()),val_accuracy=result["accuracy"],val_nll=result["nll"]))
            print(f"{name} {log[-1]}",flush=True)
            if result["nll"]<best:
                best=result["nll"]
                mx.save_safetensors(str(out/"trainable.safetensors"),dict(tree_flatten(model.trainable_parameters())))
                write_json(out/"selection.json",dict(step=step,val_nll=best))
    model.load_weights(str(out/"trainable.safetensors"))
    evaluation={}
    for split in ["test","depth12","depth24","fanout8"]:
        subset=[w for w in dataset if w["split"]==split]
        needed=32 if split in ("depth12","depth24") else 4
        evaluation[split]={f"depth{d}":core_eval(model,subset,d) for d in sorted(set([1,4,needed]))}
        evaluation[split]["reset"]=core_eval(model,subset,needed,reset=True)
    permutation=core_eval(model,[w for w in dataset if w["split"]=="test"],4,permutation=True)
    base=evaluation["test"]["depth4"]
    delta=max(abs(a-b) for x,y in zip(base["predictions"],permutation["predictions"])
              for aa,bb in zip(x["probs"],y["probs"]) for a,b in zip(aa,bb))
    write_json(out/"evaluation.json",dict(splits=evaluation,permutation_max_probability_error=delta,
        seconds=time.perf_counter()-start,peak_gb=mx.get_peak_memory()/1e9))
    write_json(out/"training.json",log)
    print(json.dumps({s:{k:v["accuracy"] for k,v in value.items()} for s,value in evaluation.items()}),flush=True)


def pipeline(args):
    import mlx.core as mx
    import numpy as np
    from latent_loop.structured_model import BeliefGraph
    root=Path(args.root)
    suffix=f"{args.seed}" if args.lora else f"frozen-{args.seed}"
    out=root/f"pipeline-{suffix}"
    out.mkdir(parents=True,exist_ok=True)
    grounder=f"grounder-{args.seed}" if args.lora else f"grounder-frozen-{args.seed}"
    ground=json.loads((root/grounder/"evaluation.json").read_text())
    model=BeliefGraph()
    core_root=Path(args.core_root) if args.core_root else root
    model.load_weights(str(core_root/f"core-process-{args.seed}/trainable.safetensors"))
    dataset=json.loads((root/"cache/worlds.json").read_text())
    results={}
    for language in ["test","language"]:
        predictions=ground["splits"][language]["predictions"]
        bank={k:[p["probs"] for p in predictions if p["label"]==k] for k in range(3)}
        for split in sorted({w["split"] for w in dataset} - {"train","val"}):
            subset=[w for w in dataset if w["split"]==split]
            needed=64 if split=="depth48" else 32 if split in ("depth12","depth24") else 4
            results[f"{split}_{language}"]=core_eval(model,subset,needed,belief_bank=bank)
    write_json(out/"evaluation.json",dict(splits=results,seed=args.seed,
        grounder=grounder,core_checkpoint_sha256=sha256(core_root/f"core-process-{args.seed}/trainable.safetensors"),
        manifest_sha256=sha256(root/"cache/manifest.json"),
        scope="Soft Qwen posterior anchors, learned graph update; typed graph structure supplied; no oracle states as model input"))
    print(json.dumps({s:v["accuracy"] for s,v in results.items()}),flush=True)


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("stage",choices=["prepare","grounder","core","pipeline"])
    p.add_argument("--root",default="results/v2")
    p.add_argument("--seed",type=int,default=17)
    p.add_argument("--steps",type=int,default=1600)
    p.add_argument("--lora",action=argparse.BooleanOptionalAction,default=True)
    p.add_argument("--objective",choices=["process","final"],default="process")
    p.add_argument("--compiled",action="store_true")
    p.add_argument("--core-root")
    p.add_argument("--head",choices=["linear","mlp"],default="linear")
    args=p.parse_args()
    {"prepare":prepare,"grounder":train_grounder,"core":train_core,"pipeline":pipeline}[args.stage](args)


if __name__=="__main__":
    main()
