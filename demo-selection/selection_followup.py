"""Separate answer-prefixed, option-counterbalanced confirmation study."""
import argparse,json,math,random,time,importlib.util
from collections import Counter
from pathlib import Path
import study

ROOT=Path(__file__).resolve().parent
SEED=7239107
def new_cases():
 previous=json.loads((ROOT/'manifest.json').read_text());used={tuple(r['coefs']) for rows in previous.values() for r in rows};rng=random.Random(SEED)
 spec=importlib.util.spec_from_file_location('prior_counterfactual',ROOT.parent/'counterfactual/experiment.py');prior=importlib.util.module_from_spec(spec);spec.loader.exec_module(prior)
 used.update(tuple(r['coefs']) for seed in (prior.LAYOUT_SEED,426001) for r in prior.tasks(seed))
 positions={}
 for domain in range(4):
  for template in (2,3):
   p=list(range(4));rng.shuffle(p);positions[(domain,template)]=p
 rows=[]
 for i in range(32):
  domain=i%4;template=2+(i//4)%2;v=study.vector(rng,used)
  row={'id':f'followup-{i:03}','group':f'followup-{i:03}','domain':domain,'template':template,'coefs':v,'order':rng.sample(list(range(4)),4),'values':rng.sample(range(5,70),4)}
  row['options']=study.options(v,positions[(domain,template)][i//8],rng);rows.append(row)
 assert len({tuple(r['coefs']) for r in rows})==32
 for domain in range(4):
  for template in (2,3):
   counts=Counter(study.display(r)[1] for r in rows if r['domain']==domain and r['template']==template)
   assert set(counts.values())=={1} and len(counts)==4
 return rows
class PrefixTokenizer:
 def __init__(self,base):self.base=base
 def apply_chat_template(self,*args,**kwargs):
  return self.base.apply_chat_template(*args,**kwargs)+self.base.encode('Answer:',add_special_tokens=False)
 def encode(self,*args,**kwargs):return self.base.encode(*args,**kwargs)
class PrefixedScorer(study.Scorer):
 def __init__(self,checkpoint):
  super().__init__(checkpoint);base=self.tokenizer
  labels=[base.encode(' '+x,add_special_tokens=False) for x in study.LETTERS]
  assert all(len(x)==1 for x in labels)
  self.labels=[x[0] for x in labels];self.tokenizer=PrefixTokenizer(base)
 def score(self,prompt):
  mx=self.mx
  ids=self.tokenizer.apply_chat_template([{'role':'user','content':prompt}],tokenize=True,add_generation_prompt=True,enable_thinking=False)
  assert len(ids)<=1500
  start=time.perf_counter_ns();full=self.model(mx.array([ids]));four=full[0,-1,self.labels].astype(mx.float32);mx.eval(four)
  values=four.tolist()
  if mx.get_peak_memory()>2560*1024**2:raise MemoryError('Followup exceeded observed 2,560MiB guard')
  result={'prediction':max(range(4),key=lambda i:(values[i],-i)),'label_logits':values,'input_tokens':len(ids),'elapsed_ns':time.perf_counter_ns()-start}
  del full,four;mx.clear_cache();self.count+=1
  if self.count%25==0:print('Followup calls',self.count,'peak MiB',round(mx.get_peak_memory()/1024**2,1),flush=True)
  return result
 def meta(self):
  return {**super().meta(),'followup_readout':'standard full-forward model wrapper; explicit assistant Answer: prefix; leading-space labels'}
def scorer(checkpoint):return PrefixedScorer(checkpoint)
def softmax(v):
 m=max(v);x=[math.exp(a-m) for a in v];return [a/sum(x) for a in x]
def pool(records):
 total=[0.]*4
 for row in records:
  probs=softmax(row['label_logits']);rot=row['rotation']
  for index,p in enumerate(probs):total[(index+rot)%4]+=p/4
 order=sorted(range(4),key=lambda i:(-total[i],i))
 return {'probabilities':total,'prediction':None if total[order[0]]-total[order[1]]<=1e-6 else order[0],'margin':total[order[0]]-total[order[1]]}
def freeze():
 study.check_freeze();study.write(ROOT/'followup_manifest.json',new_cases())
 receipt={'source_sha256':study.digest(__file__),'protocol_sha256':study.digest(ROOT/'FOLLOWUP_PROTOCOL.md'),'manifest_sha256':study.digest(ROOT/'followup_manifest.json'),'initial_freeze_sha256':study.digest(ROOT/'freeze.json'),'historical_generator_sha256':study.digest(ROOT.parent/'counterfactual/experiment.py'),'seed':SEED}
 study.write(ROOT/'followup_freeze.json',receipt);print(study.canonical(receipt))
def check():
 r=json.loads((ROOT/'followup_freeze.json').read_text())
 assert r['source_sha256']==study.digest(__file__)
 assert r['protocol_sha256']==study.digest(ROOT/'FOLLOWUP_PROTOCOL.md')
 assert r['manifest_sha256']==study.digest(ROOT/'followup_manifest.json')
 assert r['initial_freeze_sha256']==study.digest(ROOT/'freeze.json')
 assert r['historical_generator_sha256']==study.digest(ROOT.parent/'counterfactual/experiment.py')
 assert json.loads((ROOT/'followup_manifest.json').read_text())==new_cases()
 study.check_freeze();return r
def run(checkpoint,phase):
 receipt=check();path=ROOT/f'followup_{phase}_trace.json'
 if path.exists():raise ValueError('Refuse to overwrite completed trace')
 s=scorer(checkpoint);initial=json.loads((ROOT/'manifest.json').read_text());bank=initial['pool'];rows=[]
 if phase=='preselect':
  matrix=[[False]*16 for _ in range(16)]
  for i in range(16):
   for j in range(16):
    if i==j:continue
    result=s.score(study.make_prompt(bank[j],bank,[i]));result.update(demo=i,target=j);rows.append(result)
    matrix[i][j]=result['prediction']==study.display(bank[j])[1]
   study.write(ROOT/'followup_preselect_progress.json',{'rows':rows,'completed_demos':i+1,'meta':s.meta()})
  core=study.edges_and_core(matrix)
  study.write(path,{'freeze':receipt,'rows':rows,'matrix':matrix,'core':core,'meta':s.meta(),'readout':{'assistant_prefix':'Answer:','label_token_ids':s.labels}})
  study.write(ROOT/'followup_core_receipt.json',{'followup_freeze':receipt,'core':core,'preselect_trace_sha256':study.digest(path),'before_confirmation':True})
  print('Followup core',core,flush=True);return
 pre=json.loads((ROOT/'followup_preselect_trace.json').read_text());assert pre['freeze']==receipt
 core_receipt=json.loads((ROOT/'followup_core_receipt.json').read_text());assert core_receipt['preselect_trace_sha256']==study.digest(ROOT/'followup_preselect_trace.json') and core_receipt['core']==pre['core'] and core_receipt['followup_freeze']==receipt
 select=study.Selector(bank,pre['core']);cases=json.loads((ROOT/'followup_manifest.json').read_text())
 for k,case in enumerate(cases):
  arms=study.ARMS[k%8:]+study.ARMS[:k%8]
  for arm in arms:
   selected=select.select(study.public_view(case),arm)
   for rot in range(4):
    result=s.score(study.make_prompt(case,bank,selected,rot));result.update(id=case['id'],arm=arm,rotation=rot,selected=selected);rows.append(result)
  if (k+1)%4==0:
   study.write(ROOT/'followup_test_progress.json',{'completed_cases':k+1,'rows':rows,'meta':s.meta()});print('Followup',k+1,'/32',flush=True)
 study.write(path,{'freeze':receipt,'preselect_sha256':study.digest(ROOT/'followup_preselect_trace.json'),'rows':rows,'meta':s.meta(),'readout':{'assistant_prefix':'Answer:','label_token_ids':s.labels}})
def analyse():
 receipt=check();cases=json.loads((ROOT/'followup_manifest.json').read_text());byid={r['id']:r for r in cases};initial=json.loads((ROOT/'manifest.json').read_text())
 pre=json.loads((ROOT/'followup_preselect_trace.json').read_text());trace=json.loads((ROOT/'followup_test_trace.json').read_text())
 assert pre['freeze']==receipt and trace['freeze']==receipt and trace['preselect_sha256']==study.digest(ROOT/'followup_preselect_trace.json')
 assert len(pre['rows'])==240 and pre['core']==study.edges_and_core(pre['matrix'])
 seen=set();selector=study.Selector(initial['pool'],pre['core']);outcomes={a:{} for a in study.ARMS};raw={a:Counter() for a in study.ARMS};costs={a:Counter() for a in study.ARMS};ties=Counter();fixed={a:{} for a in study.ARMS};pooled_rows=[]
 for row in trace['rows']:
  key=(row['id'],row['arm'],row['rotation']);assert key not in seen;seen.add(key)
  assert row['prediction']==max(range(4),key=lambda i:(row['label_logits'][i],-i))
  assert row['selected']==selector.select(study.public_view(byid[row['id']]),row['arm'])
  raw[row['arm']][row['prediction']]+=1;costs[row['arm']].update({'input_tokens':row['input_tokens'],'elapsed_ns':row['elapsed_ns']})
 assert len(seen)==32*8*4
 for case in cases:
  gold=study.display(case)[1]
  for arm in study.ARMS:
   records=[r for r in trace['rows'] if r['id']==case['id'] and r['arm']==arm];assert {r['rotation'] for r in records}=={0,1,2,3}
   result=pool(records);ties[arm]+=result['prediction'] is None;outcomes[arm][case['id']]=int(result['prediction']==gold)
   fixed[arm][case['id']]=int(next(r for r in records if r['rotation']==0)['prediction']==gold)
   pooled_rows.append({'id':case['id'],'arm':arm,**result,'gold':gold})
 out={'freeze':receipt,'task':'Four-choice program recognition; explicit assistant Answer: prefix; mean probability over four option rotations','cases':32,'groups':32,'accuracy':{a:sum(v.values())/32 for a,v in outcomes.items()},'fixed_order_accuracy':{a:sum(v.values())/32 for a,v in fixed.items()},'ties_counted_wrong':dict(ties),'raw_position_counts':{a:dict(v) for a,v in raw.items()},'input_tokens':{a:v['input_tokens'] for a,v in costs.items()},'inference_elapsed_ns':{a:v['elapsed_ns'] for a,v in costs.items()},'comparisons':{},'preprocessing':{'calls':240,'input_tokens':sum(r['input_tokens'] for r in pre['rows']),'meta':pre['meta']},'memory':trace['meta'],'pooled_predictions':pooled_rows,'symbolic_grammar_ceiling':{'correct':sum(study.symbolic_parse(study.question(r))==r['coefs'] for r in cases),'cases':32}}
 ids=[r['id'] for r in cases]
 for arm in study.ARMS:
  if arm=='model_core':continue
  diffs=[outcomes['model_core'][i]-outcomes[arm][i] for i in ids]
  out['comparisons'][arm]={'difference':sum(diffs)/32,'paired_group_95_interval':study.bootstrap(diffs),'simultaneous_95_interval':study.bootstrap(diffs,.05/7)}
 out['by_domain']={str(d):{a:sum(outcomes[a][r['id']] for r in cases if r['domain']==d)/sum(r['domain']==d for r in cases) for a in study.ARMS} for d in range(4)}
 out['by_template']={str(t):{a:sum(outcomes[a][r['id']] for r in cases if r['template']==t)/sum(r['template']==t for r in cases) for a in study.ARMS} for t in (2,3)}
 out['per_rotation_accuracy']={a:{str(rot):sum(r['prediction']==study.display(byid[r['id']],rot)[1] for r in trace['rows'] if r['arm']==a and r['rotation']==rot)/32 for rot in range(4)} for a in study.ARMS}
 out['rotation_semantic_agreement']={a:sum(len({(r['prediction']+r['rotation'])%4 for r in trace['rows'] if r['id']==case['id'] and r['arm']==a})==1 for case in cases)/32 for a in study.ARMS}
 primary=out['comparisons']['lexical'];ratio=out['input_tokens']['model_core']/out['input_tokens']['lexical']
 out['primary_gate']={'contrast':'pooled model_core minus pooled full-pool lexical','minimum_gain':.10,'token_ratio':ratio,'passed':primary['difference']>=.10 and primary['paired_group_95_interval'][0]>0 and ratio<=1.1}
 out['all_baselines_gate']=all(c['difference']>=.05 and c['simultaneous_95_interval'][0]>0 for c in out['comparisons'].values())
 diffs=[outcomes['lexical'][i]-fixed['lexical'][i] for i in ids]
 out['secondary_pooling_vs_fixed_lexical']={'difference':sum(diffs)/32,'paired_group_95_interval':study.bootstrap(diffs),'extra_model_calls':'Four rotations versus one, not compute-matched; no efficiency claim'}
 return out
def main():
 ap=argparse.ArgumentParser(description=__doc__);sp=ap.add_subparsers(dest='command',required=True);sp.add_parser('freeze');sp.add_parser('check')
 runp=sp.add_parser('run');runp.add_argument('checkpoint',type=Path);runp.add_argument('phase',choices=['preselect','test'])
 a=sp.add_parser('analyse');a.add_argument('--verify',type=Path);args=ap.parse_args()
 if args.command=='freeze':freeze()
 elif args.command=='check':print(study.canonical(check()))
 elif args.command=='run':run(args.checkpoint,args.phase)
 else:
  out=analyse()
  if args.verify:assert out==json.loads(args.verify.read_text());print('Verified fresh followup outcomes and claim gates.')
  else:study.write(ROOT/'followup_results.json',out);print(json.dumps({k:v for k,v in out.items() if k!='pooled_predictions'},indent=2))
if __name__=='__main__':main()
