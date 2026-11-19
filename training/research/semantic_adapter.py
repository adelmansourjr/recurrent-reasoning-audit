"""Train an actual generative NL->FOL Qwen LoRA; keep inference checking separate.

This is a semantic-parser comparator, not evidence of continuous latent deduction.
"""
from __future__ import annotations
import argparse
import json
from pathlib import Path
import random
import time
import mlx.core as mx
import mlx.nn as nn
import mlx.optimizers as optim
import numpy as np
from mlx.utils import tree_flatten,tree_map
from mlx_lm import load,stream_generate
from mlx_lm.models.base import create_attention_mask
from mlx_lm.tuner.utils import linear_to_lora_layers
from latent_loop.experiment import DEFAULT_MODEL,write_json,sha256
from latent_loop.language_data import load_folio,story_key
from latent_loop.logic import parse,solve,canonicalize

INSTRUCTION=('Translate each numbered sentence into first-order logic. Use a consistent open vocabulary of predicate and entity identifiers across all sentences. '
    'Use only identifiers, parentheses, commas, and the operators ∀ ∃ ¬ ∧ ∨ ⊕ → ↔ =. Quantify variables x,y,z. '
    'Return exactly a JSON list of formula strings in sentence order. Do not explain or execute anything.')
CANONICAL_INSTRUCTION=('Translate each numbered sentence into first-order logic. Represent concepts with anonymous predicate identifiers p0,p1,p2,... in order of first use, '
    'and named individuals with constants c0,c1,c2,... in order of first use. Reuse an identifier for the same concept or individual across sentences. '
    'Explicitly quantify variables x,y,z. Use only predicates, equality, parentheses, commas and ∀ ∃ ¬ ∧ ∨ ⊕ → ↔. '
    'Preserve argument order and quantifier scope. Return exactly a JSON list of formula strings, one per sentence, in order. No prose.')


def prompt(tok,sentences,instruction=INSTRUCTION):
    text='\n'.join(f'{i+1}. {s.strip()}' for i,s in enumerate(sentences))
    return tok.apply_chat_template([dict(role='system',content=instruction),dict(role='user',content=text)],
        tokenize=True,add_generation_prompt=True,enable_thinking=False)


def unwrap(text):
    text=text.strip()
    if text.startswith('```json'):text=text[7:].lstrip()
    elif text.startswith('```'):text=text[3:].lstrip()
    if text.endswith('```'):text=text[:-3].rstrip()
    return text


def generate_program(model,tok,ids):
    text=''
    iterator=stream_generate(model,tok,prompt=ids,max_tokens=384)
    try:
        for response in iterator:
            text+=response.text
            try:
                if isinstance(json.loads(unwrap(text)),list):break
            except json.JSONDecodeError:pass
    finally:iterator.close()
    return text


def model_load(adapter=False,cached=False):
    mx.set_cache_limit(128*1024*1024);mx.random.seed(17)
    model,tok=load(DEFAULT_MODEL,lazy=True)
    if cached:
        for i in range(26):model.model.layers[i]=None
    for _,v in tree_flatten(model.parameters()):mx.eval(v)
    model.freeze()
    if adapter:linear_to_lora_layers(model,2,dict(rank=16,scale=2.,dropout=0.,keys=['self_attn.q_proj','self_attn.v_proj']))
    return model,tok


class ParserTrain(nn.Module):
    def __init__(self,backbone):super().__init__();self.backbone=backbone
    def __call__(self,h,start,targets):
        mask=create_attention_mask(h,None)
        for layer in self.backbone.model.layers[-2:]:h=layer(h,mask,None)
        h=self.backbone.model.norm(h)[:,start:,:]
        logits=self.backbone.model.embed_tokens.as_linear(h).astype(mx.float32)
        return nn.losses.cross_entropy(logits,targets,reduction='mean')


def prepare(args):
    root=Path(args.root)/'cache';root.mkdir(parents=True,exist_ok=True)
    if (root/'manifest.json').exists():raise ValueError('Preserve completed cache')
    assignments={r['story']:r['split'] for r in load_folio()};seen=set();rows=[];excluded=[]
    for line in Path('data/folio-v0/train.jsonl').read_text().splitlines():
        r=json.loads(line);key=story_key(r['premises'])
        if key in seen:continue
        seen.add(key)
        try:
            if len(r['premises'])!=len(r['premises-FOL']):raise ValueError('Premise/formula count mismatch')
            [parse(f) for f in r['premises-FOL']]
            formulas,mapping=canonicalize(r['premises-FOL']) if args.canonical else (r['premises-FOL'],None)
        except ValueError as e:excluded.append(dict(story=key,error=str(e)));continue
        rows.append(dict(id=f"parse-{len(rows):04d}",story=key,split=assignments[key],
            sentences=r['premises'],formulas=formulas,canonical_mapping=mapping))
    model,tok=model_load();start=time.perf_counter()
    for i,r in enumerate(rows):
        x=prompt(tok,r['sentences'],CANONICAL_INSTRUCTION if args.canonical else INSTRUCTION);target=tok.encode(json.dumps(r['formulas'],ensure_ascii=False),add_special_tokens=False)
        ids=x+target
        if len(ids)>1024:raise ValueError('No truncation allowed')
        r['start']=len(x)-1;r['targets']=target+[tok.eos_token_id];r['tokens']=len(ids)
        h=model.model.embed_tokens(mx.array([ids]));mask=create_attention_mask(h,None)
        for layer in model.model.layers[:-2]:h=layer(h,mask,None)
        mx.eval(h)
        np.savez(root/f"{r['id']}.npz",prefix=np.array(h.astype(mx.float16)))
        del h;mx.clear_cache()
        if (i+1)%40==0:print('semantic cache',i+1,len(rows),time.perf_counter()-start,flush=True)
    write_json(root/'cases.json',rows)
    write_json(root/'manifest.json',dict(checkpoint_sha256=sha256(Path(DEFAULT_MODEL)/'model.safetensors'),
        tokenizer_sha256=sha256(Path(DEFAULT_MODEL)/'tokenizer.json'),cases_sha256=sha256(root/'cases.json'),
        splits={s:sum(r['split']==s for r in rows) for s in ('train','dev')},excluded=excluded,
        filter='Only sentence/formula count alignment and safe parser syntax on training release; no external labels or examples for fitting',
        instruction=CANONICAL_INSTRUCTION if args.canonical else INSTRUCTION,canonical=args.canonical,
        cache_boundary='Before Qwen block 26; blocks 26 and 27 have trainable q/v rank16 LoRA',
        seconds=time.perf_counter()-start))


def train(args):
    root=Path(args.root)/'cache';out=Path(args.root)/'parser-17';out.mkdir(parents=True,exist_ok=True)
    if (out/'training-complete.json').exists():print('parser already trained');return
    rows=json.loads((root/'cases.json').read_text())
    manifest=json.loads((root/'manifest.json').read_text())
    if sha256(Path(DEFAULT_MODEL)/'model.safetensors')!=manifest['checkpoint_sha256']:raise ValueError('Source model changed')
    model,tok=model_load(adapter=True,cached=True);net=ParserTrain(model)
    training=[r for r in rows if r['split']=='train'];dev=[r for r in rows if r['split']=='dev']
    rng=random.Random(17);cache={}
    def inputs(r):
        if r['id'] not in cache:
            with np.load(root/f"{r['id']}.npz") as z:cache[r['id']]=mx.array(z['prefix'],dtype=mx.bfloat16)
        return cache[r['id']],r['start'],mx.array([r['targets']])
    opt=optim.AdamW(learning_rate=1e-4,weight_decay=.01)
    vg=nn.value_and_grad(net,lambda m,*x:m(*x));best=float('inf');history=[];acc=None
    names=[k for k,v in tree_flatten(net.trainable_parameters())]
    assert len(names)==8 and all(('layers.26.' in k or 'layers.27.' in k) and 'lora_' in k for k in names)
    write_json(out/'config.json',dict(rank=16,scale=2.,blocks=[26,27],seed=17,steps=args.steps,accumulation=4,
        learning_rate=1e-4,trainable_parameters=sum(v.size for k,v in tree_flatten(net.trainable_parameters())),
        manifest_sha256=sha256(root/'manifest.json'),trainable_names=names))
    start=time.perf_counter()
    # Include the unmodified model in development checkpoint selection.
    for step in range(args.steps+1):
        if step:
            r=rng.choice(training);value,grads=vg(net,*inputs(r))
            acc=grads if acc is None else tree_map(lambda a,b:a+b,acc,grads)
            if step%4==0:
                grads,norm=optim.clip_grad_norm(tree_map(lambda g:g/4,acc),1.)
                if step in (4,8):
                    norms={k:float(mx.linalg.norm(v).item()) for k,v in tree_flatten(grads)}
                    write_json(out/f'gradient-check-{step}.json',norms)
                    if step==8 and not all(np.isfinite(v) and v>0 for v in norms.values()):raise ValueError('Nonfinite/zero adapter gradient')
                opt.update(net,grads);acc=None;mx.eval(net.trainable_parameters(),opt.state,norm)
            mx.eval(value,acc)
        if step%200==0:
            losses=[float(net(*inputs(r)).item()) for r in dev]
            nll=sum(losses)/len(losses)
            history.append(dict(step=step,dev_mean_example_token_nll=nll,elapsed=time.perf_counter()-start))
            print('semantic parser',history[-1],flush=True)
            if nll<best:
                best=nll;mx.save_safetensors(str(out/'trainable.safetensors'),dict(tree_flatten(net.trainable_parameters())))
                write_json(out/'selection.json',history[-1])
            write_json(out/'training.json',history)
        if step%10==0:mx.clear_cache()
    write_json(out/'training-complete.json',dict(seconds=time.perf_counter()-start,best_dev_nll=best,peak_gb=mx.get_peak_memory()/1e9))
    selected=mx.load(str(out/'trainable.safetensors'))
    mx.save_safetensors(str(out/'adapters.safetensors'),{k.removeprefix('backbone.'):v for k,v in selected.items()})
    write_json(out/'adapter_config.json',dict(model=DEFAULT_MODEL,fine_tune_type='lora',num_layers=2,
        lora_parameters=dict(rank=16,scale=2.,dropout=0.,keys=['self_attn.q_proj','self_attn.v_proj'])))


def evaluate(args):
    root=Path(args.root)
    instruction=json.loads((root/'cache/manifest.json').read_text())['instruction']
    model,tok=model_load(adapter=args.adapter)
    if args.adapter:
        weights=mx.load(str(root/'parser-17/trainable.safetensors'))
        weights={k.removeprefix('backbone.'):v for k,v in weights.items()}
        expected=dict(tree_flatten(model.trainable_parameters()))
        if set(weights)!=set(expected) or any(weights[k].shape!=expected[k].shape for k in expected):raise ValueError('Semantic adapter keys/shapes differ')
        model.load_weights(list(weights.items()),strict=False)
    # Fixed first 64 official examples, before any parser external evaluation.
    rows=[json.loads(x) for x in Path('data/folio-v0/validation.jsonl').read_text().splitlines()][:64]
    outputs=[];start=time.perf_counter()
    for i,r in enumerate(rows):
        sentences=r['premises']+[r['conclusion']]
        ids=prompt(tok,sentences,instruction)
        text=generate_program(model,tok,ids)
        result=dict(id=f'external-{i:04d}',gold_label=r['label'].replace('Uncertain','Unknown'),text=text)
        try:
            formulas=json.loads(unwrap(text))
            if not isinstance(formulas,list) or len(formulas)!=len(sentences) or not all(isinstance(f,str) for f in formulas):
                raise ValueError('Output must contain exactly one formula string per sentence')
            result['logic']=solve(formulas[:-1],formulas[-1],timeout_ms=1000)
            result['label']=result['logic']['label']
        except (ValueError,TypeError,RecursionError) as e:result.update(label=None,error=str(e))
        result['correct']=result['label']==result['gold_label'];outputs.append(result)
        write_json(root/('parser-evaluation.json' if args.adapter else 'base-parser-evaluation.json'),
            dict(predictions=outputs,complete=False,subset='Fixed first 64 official FOLIO v0 validation examples; no reranking or fallback'))
        print('parser evaluation',args.adapter,i+1,result['label'],result['gold_label'],'elapsed',time.perf_counter()-start,flush=True)
        mx.clear_cache()
    write_json(root/('parser-evaluation.json' if args.adapter else 'base-parser-evaluation.json'),
        dict(predictions=outputs,complete=True,n=len(outputs),accuracy=sum(r['correct'] for r in outputs)/len(outputs),
            decision_coverage=sum(r['label'] is not None for r in outputs)/len(outputs),seconds=time.perf_counter()-start,
            subset='Fixed first 64 official FOLIO v0 validation examples; no reranking or fallback',
            limitation='Solver correctness is conditional on language translation; quantifier scope and predicate consistency can be mistranslated'))


def predict(args):
    from latent_loop.language_inference import EvidenceMemory
    root=Path(args.root)
    instruction=json.loads((root/'cache/manifest.json').read_text())['instruction']
    request=json.loads(Path(args.input).read_text());memory=EvidenceMemory(request.get('memory'))
    premises=[r['text'] for r in memory.current()]+request.get('premises',[])
    if not premises:raise ValueError('No premises supplied')
    model,tok=model_load(adapter=args.adapter)
    if args.adapter:
        weights={k.removeprefix('backbone.'):v for k,v in mx.load(str(root/'parser-17/trainable.safetensors')).items()}
        expected=dict(tree_flatten(model.trainable_parameters()))
        if set(weights)!=set(expected) or any(weights[k].shape!=expected[k].shape for k in expected):raise ValueError('Adapter mismatch')
        model.load_weights(list(weights.items()),strict=False)
    ids=prompt(tok,premises+[request['conclusion']],instruction)
    if len(ids)>1024:raise ValueError('Input exceeds tested context; no truncation')
    text=generate_program(model,tok,ids)
    result=dict(premises=premises,conclusion=request['conclusion'],raw_translation=text,memory_provenance=memory.current(),
        grounding_verified=False,adapter=args.adapter)
    try:
        formulas=json.loads(unwrap(text))
        if not isinstance(formulas,list) or len(formulas)!=len(premises)+1 or not all(isinstance(x,str) for x in formulas):raise ValueError('Wrong formula output count/type')
        result['formulas']=formulas;result['logic']=solve(formulas[:-1],formulas[-1],find_gaps=True)
        label=result['logic']['label'];result['label']=label
        wording={'True':'follows','False':'is contradicted','Unknown':'is undetermined'}
        result['text']=f"Under the model's inferred logical representation, the conclusion {wording[label]}." if label else 'The inferred representation could not yield a consistent, completed solver decision.'
        gaps=result['logic'].get('gap_candidates',{}).get('candidates',[])
        if gaps:
            result['suggested_observation']=gaps[0]
            result['text']+=f" A candidate missing observation is whether {gaps[0]['observation']} holds; its conditional consequences are listed in the result."
    except (ValueError,TypeError,RecursionError) as e:result.update(label=None,error=str(e),text='The language-to-logic translation could not be checked.')
    write_json(args.out or root/'language-example.json',result);print(json.dumps(result,indent=2))


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('stage',choices=['prepare','train','evaluate','predict'])
    p.add_argument('--steps',type=int,default=1200);p.add_argument('--adapter',action=argparse.BooleanOptionalAction,default=True)
    p.add_argument('--input',default='configs/language-example.json');p.add_argument('--out')
    p.add_argument('--root',default='results/v6');p.add_argument('--canonical',action='store_true')
    a=p.parse_args()
    if a.steps%200:raise ValueError('steps must be multiple of 200')
    globals()[a.stage](a)
