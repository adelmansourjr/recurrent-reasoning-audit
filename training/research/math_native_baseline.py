"""Exploratory conventional stock-Qwen math comparator, no tool/program schema."""
import json
from pathlib import Path
import re
import time
from mlx_lm import stream_generate
import semantic_adapter as sem
from math_adapter import algebra_cases
from latent_loop.mathematics import rational
from latent_loop.experiment import write_json

model,tok=sem.model_load();rows=[]
for i,line in enumerate(Path('data/gsm8k/test.jsonl').read_text().splitlines()[:64]):
    r=json.loads(line);rows.append(dict(id=f'gsm-test-{i:04d}',source='GSM8K-test',question=r['question'],answer=r['answer'].split('####')[-1].strip().replace(',','')))
rows+=algebra_cases('test',24)
outputs=[];start=time.perf_counter()
instruction='Solve the mathematics question carefully. Give a concise calculation or algebraic derivation. End with exactly #### followed by the final numeric answer. A fraction such as 3/2 is allowed. Do not use JSON.'
for i,r in enumerate(rows):
    ids=tok.apply_chat_template([dict(role='system',content=instruction),dict(role='user',content=r['question'])],tokenize=True,add_generation_prompt=True,enable_thinking=False)
    text=''
    for response in stream_generate(model,tok,prompt=ids,max_tokens=384):text+=response.text
    matches=re.findall(r'####\s*\$?\s*(-?\d[\d,]*(?:\.\d+)?(?:\s*/\s*-?\d[\d,]*)?)',text)
    result=dict(id=r['id'],source=r['source'],text=text,answer=matches[-1].replace(',','').replace(' ','') if matches else None,gold_answer=r['answer'])
    result['correct']=result['answer'] is not None and rational(result['answer'])==rational(r['answer'])
    outputs.append(result)
    splits={s:dict(n=len(g:=[x for x in outputs if x['source']==s]),accuracy=sum(x['correct'] for x in g)/len(g),answer_format_coverage=sum(x['answer'] is not None for x in g)/len(g)) for s in {x['source'] for x in outputs}}
    write_json('results/v8/math/native-baseline.json',dict(predictions=outputs,splits=splits,complete=len(outputs)==len(rows),seconds=time.perf_counter()-start,instruction=instruction,
        selection='Exploratory comparator added after observing stock typed-program format failures; no test answers used in prompt. Same384-token budget, thinking disabled. Final-answer scoring, not a checked proof comparison.'))
    if (i+1)%16==0:print('native math',i+1,len(rows),splits,flush=True)
