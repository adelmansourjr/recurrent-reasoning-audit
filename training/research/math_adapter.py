"""Train/evaluate a Qwen mathematical-program LoRA with an exact proof kernel.

Uses only training GSM8K calculator annotations and synthetic linear equations
for fitting. Test prompts contain questions only, never answers or derivations.
"""
from __future__ import annotations
import argparse
import ast
import hashlib
import json
from pathlib import Path
import random
import re
import time
from fractions import Fraction
from latent_loop.mathematics import execute,expression,rational,display,explain

ROOT=Path('results/v8/math')
INSTRUCTION=('Solve the question by constructing a typed mathematical derivation. Return only a JSON list of steps. '
    'Arithmetic step: {"op":"calculate","expression":"48/2"}. Later expressions may use previous results s0,s1,... . '
    'Equation step: {"op":"solve_linear","left":"3*x+2","right":"17"}. The variable must be x. '
    'Expressions allow rational numbers, parentheses, + - * / and bounded integer ** powers. '
    'The final step computes the requested answer. Do not put answers in extra fields or repeat a remembered answer. '
    'If the question cannot be represented faithfully with these operations, return [].')


def prompt(tok,question):
    return tok.apply_chat_template([dict(role='system',content=INSTRUCTION),dict(role='user',content=question)],
        tokenize=True,add_generation_prompt=True,enable_thinking=False)


def digest(text):return hashlib.sha256(' '.join(text.lower().split()).encode()).hexdigest()


def annotation_program(answer):
    """Use published train-only operations; recompute every annotated value."""
    program=[];values={}
    for expr,claimed in re.findall(r'<<([^<>]*?)=([^<>]*?)>>',answer):
        expr=expr.strip().replace(',','').replace('^','**')
        _,value,_=expression(expr)
        if value!=rational(claimed.replace(',','').strip()): raise ValueError('Inexact published annotation')
        # Replace matching literal intermediate values with explicit DAG links.
        tree=ast.parse(expr,mode='eval')
        class Ref(ast.NodeTransformer):
            def visit_Constant(self,node):
                if type(node.value) in (int,float):
                    v=rational(ast.get_source_segment(expr,node))
                    if v in values:return ast.copy_location(ast.Name(id=values[v],ctx=ast.Load()),node)
                return node
        expr=ast.unparse(Ref().visit(tree))
        program.append(dict(op='calculate',expression=expr));values[value]=f's{len(program)-1}'
    result=execute(program)
    expected=rational(answer.split('####')[-1].strip().replace(',',''))
    if rational(result['answer'])!=expected:raise ValueError('Final annotation does not compute final answer')
    return program,display(expected)


def algebra_cases(split,n):
    rng=random.Random({'train':413,'dev':419,'test':431}[split]);rows=[]
    for i in range(n):
        a,b,c,d=[rng.randint(2,29) for _ in range(4)]
        if split=='test':a+=100;c+=40
        # Test includes novel language and both-sides compositions, with fresh numbers.
        if split=='test':
            variants=[(f'Determine the value that makes {a}*(x+{b}) = {c}*x+{d} true.',f'{a}*(x+{b})',f'{c}*x+{d}'),
                (f'A balance satisfies (x-{b})/{a} + {c} = {d}. Find x.',f'(x-{b})/{a}+{c}',str(d)),
                (f'Find the unknown in {a}*x + {b} = {c}*x - {d}.',f'{a}*x+{b}',f'{c}*x-{d}'),
                (f'Which rational x satisfies {a}*(2*x-{b})/{c} = {d}?',f'{a}*(2*x-{b})/{c}',str(d))]
        else:
            variants=[(f'Solve {a}*x+{b}={d} for x.',f'{a}*x+{b}',str(d)),
                (f'Find x: {a}*(x-{b})={c}.',f'{a}*(x-{b})',str(c)),
                (f'If x/{a}+{b}={d}, what is x?',f'x/{a}+{b}',str(d)),
                (f'{a} times a number minus {b} equals {c}. What is the number?',f'{a}*x-{b}',str(c))]
        question,left,right=variants[i%4];program=[dict(op='solve_linear',left=left,right=right)]
        answer=execute(program)['answer']
        rows.append(dict(question=question,program=program,answer=answer,source='generated-linear',split=split,
            id=f'algebra-{split}-{i:04d}',family=i%4))
    return rows


def dataset():
    rows=[];excluded=[]
    for i,line in enumerate(Path('data/gsm8k/train.jsonl').read_text().splitlines()):
        r=json.loads(line)
        try:program,answer=annotation_program(r['answer'])
        except (ValueError,SyntaxError,ZeroDivisionError,OverflowError):excluded.append(i);continue
        rows.append(dict(question=r['question'],program=program,answer=answer,id=f'gsm-train-{i:04d}',source='GSM8K-train'))
    rows.sort(key=lambda r:digest(r['question']))
    # Reserve development questions before fitting; no test filtering by ability.
    rows=rows[:400]
    for i,r in enumerate(rows):r['split']='dev' if i<80 else 'train'
    rows+=algebra_cases('train',120)+algebra_cases('dev',24)
    for r in rows:r['id']=r['id'].replace('gsm-train','gsm-source')
    return rows,excluded


def prepare(args):
    import mlx.core as mx
    import numpy as np
    import semantic_adapter as sem
    from mlx_lm.models.base import create_attention_mask
    from latent_loop.experiment import DEFAULT_MODEL,write_json,sha256
    root=Path(args.root)/'cache';root.mkdir(parents=True,exist_ok=True)
    if (root/'manifest.json').exists():raise ValueError('Preserve existing cache')
    rows,excluded=dataset();model,tok=sem.model_load();start=time.perf_counter()
    for i,r in enumerate(rows):
        x=prompt(tok,r['question']);target=tok.encode(json.dumps(r['program']),add_special_tokens=False)
        ids=x+target
        if len(ids)>1024:raise ValueError('No question or proof truncation')
        r['start']=len(x)-1;r['targets']=target+[tok.eos_token_id];r['tokens']=len(ids)
        h=model.model.embed_tokens(mx.array([ids]));mask=create_attention_mask(h,None)
        for layer in model.model.layers[:-2]:h=layer(h,mask,None)
        mx.eval(h);np.savez(root/f"{r['id']}.npz",prefix=np.array(h.astype(mx.float16)))
        del h;mx.clear_cache()
        if (i+1)%80==0:print('math cache',i+1,len(rows),time.perf_counter()-start,flush=True)
    write_json(root/'cases.json',rows)
    write_json(root/'manifest.json',dict(checkpoint_sha256=sha256(Path(DEFAULT_MODEL)/'model.safetensors'),
        tokenizer_sha256=sha256(Path(DEFAULT_MODEL)/'tokenizer.json'),cases_sha256=sha256(root/'cases.json'),
        source_sha256=sha256('data/gsm8k/train.jsonl'),instruction=INSTRUCTION,
        splits={s:sum(r['split']==s for r in rows) for s in ('train','dev')},
        excluded_train_annotations=excluded,training_only_filter='Published annotations safely executable and exact final answer; no test filtering',
        cache_boundary='Before Qwen block 26; last two q/v rank16 LoRA',seconds=time.perf_counter()-start))


def load(args):
    import mlx.core as mx
    from mlx.utils import tree_flatten
    import semantic_adapter as sem
    from latent_loop.experiment import DEFAULT_MODEL,sha256
    root=Path(args.root);manifest_path=root/'cache/manifest.json'
    manifest=json.loads(manifest_path.read_text())
    for filename,key in [('model.safetensors','checkpoint_sha256'),('tokenizer.json','tokenizer_sha256')]:
        if sha256(Path(DEFAULT_MODEL)/filename)!=manifest[key]:raise ValueError('Source model/tokenizer changed')
    if args.adapter:
        cfg=json.loads((root/'parser-17/config.json').read_text())
        if cfg['manifest_sha256']!=sha256(manifest_path):raise ValueError('Math cache manifest changed')
    model,tok=sem.model_load(adapter=args.adapter)
    if args.adapter:
        values=mx.load(str(Path(args.root)/'parser-17/trainable.safetensors'))
        values={k.removeprefix('backbone.'):v for k,v in values.items()}
        expected=dict(tree_flatten(model.trainable_parameters()))
        if set(values)!=set(expected) or any(values[k].shape!=expected[k].shape for k in expected):raise ValueError('Adapter keys/shapes differ')
        model.load_weights(list(values.items()),strict=False)
        if cfg.get('latent_steps',1)>1:
            from latent_loop.recurrent_math import RecurrentQwen
            model=RecurrentQwen(model,cfg['latent_steps'])
    return model,tok


def infer(model,tok,question):
    import semantic_adapter as sem
    from mlx_lm import stream_generate
    import math
    ids=prompt(tok,question)
    if len(ids)>1024:raise ValueError('Question exceeds 1024 tokens')
    text='';logprobs=[]
    iterator=stream_generate(model,tok,prompt=ids,max_tokens=384)
    try:
        for response in iterator:
            text+=response.text
            logprobs.append(float(response.logprobs[response.token].item()))
            try:
                if isinstance(json.loads(sem.unwrap(text)),list):break
            except json.JSONDecodeError:pass
    finally:iterator.close()
    mean_logprob=sum(logprobs)/len(logprobs) if logprobs else -100.
    result=dict(question=question,generated=text,grounding_verified=False,
        mean_generated_token_logprob=mean_logprob,token_geometric_probability=math.exp(mean_logprob),
        token_confidence_is_answer_probability=False)
    try:
        program=json.loads(sem.unwrap(text));result['program']=program
        if program==[]:
            result.update(status='unsupported_or_uncertain',answer=None,text="I am not knowledgeable enough to justify this using my supported mathematical operations.")
        else:
            result.update(execute(program),status='checked_conditional_derivation')
            result['text']='Under this interpretation of the question: '+explain(result)
    except (ValueError,TypeError,KeyError,RecursionError,ZeroDivisionError,OverflowError) as exc:
        result.update(status='invalid_derivation',answer=None,error=str(exc),text='I could not construct a valid checked derivation. I need to review the interpretation or use a different method.')
    return result


def evaluate(args):
    from latent_loop.experiment import write_json
    root=Path(args.root);model,tok=load(args)
    rows=[]
    for i,line in enumerate(Path('data/gsm8k/test.jsonl').read_text().splitlines()[:64]):
        r=json.loads(line);rows.append(dict(id=f'gsm-test-{i:04d}',source='GSM8K-test',question=r['question'],
            answer=r['answer'].split('####')[-1].strip().replace(',','')))
    rows+=algebra_cases('test',24)
    filename=root/('evaluation.json' if args.adapter else 'base-evaluation.json');outputs=[];start=time.perf_counter()
    for i,r in enumerate(rows):
        result=infer(model,tok,r['question']);result.update(id=r['id'],source=r['source'],gold_answer=r['answer'])
        result['correct']=result['answer'] is not None and rational(result['answer'])==rational(r['answer'])
        outputs.append(result)
        summary={s:dict(n=len(g:=[x for x in outputs if x['source']==s]),accuracy=sum(x['correct'] for x in g)/len(g),
            checked_coverage=sum(x['status']=='checked_conditional_derivation' for x in g)/len(g)) for s in {x['source'] for x in outputs}}
        write_json(filename,dict(predictions=outputs,splits=summary,complete=len(outputs)==len(rows),seconds=time.perf_counter()-start,
            warning='Derivation validity is conditional on interpretation. Public test may occur in Qwen pretraining; no decontamination claim. Fixed first64 GSM8K test plus24 novel generated linear cases.'))
        if (i+1)%16==0:print('math evaluation',i+1,len(rows),summary,flush=True)


def predict(args):
    from latent_loop.experiment import write_json
    model,tok=load(args);request=json.loads(Path(args.input).read_text())
    result=infer(model,tok,request['question'])
    result['confidence_decision']='uncalibrated_review_interpretation'
    calibration_path=Path(args.root)/'correctness-calibration.json'
    if calibration_path.exists() and result['answer'] is not None:
        from latent_loop.calibration import correctness_probability
        from latent_loop.experiment import sha256
        calibration=json.loads(calibration_path.read_text())
        if not args.adapter or calibration['weights_sha256']!=sha256(Path(args.root)/'parser-17/trainable.safetensors'):
            raise ValueError('Correctness calibration does not match this adapter')
        result['answer_confidence']=correctness_probability(result['mean_generated_token_logprob'],calibration)
        result['confidence_calibration']=calibration
        result['confidence_decision']='express_conditional_answer' if result['answer_confidence']>=.8 else 'review_interpretation'
        if result['confidence_decision']=='review_interpretation':
            result['candidate_derivation_text']=result['text']
            result['text']='I am not confident enough in my interpretation to present a settled answer. The candidate program and checked arithmetic are available for review.'
    write_json(args.out,result);print(json.dumps(result,indent=2))


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('stage',choices=['prepare','train','evaluate','predict'])
    p.add_argument('--root',default=str(ROOT));p.add_argument('--steps',type=int,default=1200)
    p.add_argument('--adapter',action=argparse.BooleanOptionalAction,default=True)
    p.add_argument('--input',default='configs/math-example.json');p.add_argument('--out',default=str(ROOT/'example.json'))
    a=p.parse_args()
    if a.stage=='train':
        import semantic_adapter as sem
        sem.train(a)
    else:globals()[a.stage](a)
