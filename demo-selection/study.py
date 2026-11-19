"""Frozen local-model demonstration preselection and sufficiency audit.

Synthetic bounded program recognition, not unrestricted program generation.
Model-free commands require Python 3.11+. Inference uses an existing MLX Qwen.
"""
import argparse
from collections import Counter
from hashlib import sha256,file_digest
from itertools import combinations
import json,math,random,re,time
from pathlib import Path

ROOT=Path(__file__).resolve().parent
LETTERS='ABCD'
ARMS=('zero','random_11','random_29','random_47','lexical','mmr','diverse_core','model_core')
WORDS={1:'once',2:'twice',3:'three times'}
DOMAINS=('account balance','warehouse stock','contest score','battery charge')
CHECKPOINT_HASHES={'config.json':'507a6701220524eb8b283425bf0856a9ae4f21f4052e563896ddd668994b1dc7','model.safetensors':'0e86d9677e519323849eac1bc272caae88567a481ff188c431f70be543d9995f','tokenizer.json':'aeb13307a71acd8fe81861d94ad54ab689df773318809eed3cbe794b4492dae4'}

def digest(path):
 with Path(path).open('rb') as h:return file_digest(h,'sha256').hexdigest()
def canonical(x):return json.dumps(x,sort_keys=True,separators=(',',':'),ensure_ascii=True)
def write(path,obj):
 path=Path(path);path.parent.mkdir(parents=True,exist_ok=True)
 temp=path.with_suffix(path.suffix+'.tmp');temp.write_text(json.dumps(obj,indent=2,ensure_ascii=True)+'\n');temp.replace(path)
def vector(rng,used):
 while True:
  v=tuple(rng.randint(-3,3) for _ in range(4))
  if v not in used and min(v)<0<max(v) and len(set(v))>=3:
   used.add(v);return list(v)
def options(v,position,rng):
 candidates=[v.copy()]
 nz=[i for i,c in enumerate(v) if c]
 x=v.copy();i=rng.choice(nz);x[i]=-x[i];candidates.append(x)
 for i,j in combinations(range(4),2):
  x=v.copy();x[i],x[j]=x[j],x[i]
  if x not in candidates:candidates.append(x);break
 while len(candidates)<4:
  x=v.copy();i=rng.randrange(4);x[i]=rng.choice([c for c in range(-3,4) if c!=x[i]])
  if x not in candidates:candidates.append(x)
 wrong=candidates[1:];rng.shuffle(wrong);wrong.insert(position,v.copy());return wrong
def clause(letter,c,t):
 if c==0:return (f'Ignore source {letter}.',f'Source {letter} does not count.',f'{letter} makes no contribution.',f'Source {letter} has no effect.')[t]
 op='add' if c>0 else 'subtract';word=WORDS[abs(c)]
 return (f'For source {letter}, {op} its amount {word}.',
         f'{op.capitalize()} source {letter} {word}.',
         f'{letter} contributes an {"increase" if c>0 else "decrease"} of its amount {word}.',
         f'{"Increase" if c>0 else "Reduce"} the total by source {letter} {word}.')[t]
def question(row):
 terms=[clause(LETTERS[i],row['coefs'][i],row['template']) for i in row['order']]
 return f"Track the signed change in {DOMAINS[row['domain']]}.\n"+' '.join(terms)+'\nWhich coefficient vector in A, B, C, D order represents this change?'
def display(row,rotation=0):
 choices=row['options'][rotation:]+row['options'][:rotation]
 text=question(row)+'\n'+'\n'.join(f'{a}: {v}' for a,v in zip(LETTERS,choices))
 return text,choices.index(row['coefs']),choices
def public_view(row):return {'id':row['id'],'text':question(row)}
def generate():
 rng=random.Random(991763);used=set();splits={}
 for name,n,templates in [('pool',16,[0]),('dev',12,[1]),('test',64,[2,3])]:
  rows=[]
  position_grid={}
  for variant in range(len(templates)):
   for domain in range(4):
    positions=[j%4 for j in range((n+3-domain)//4)]
    rng.shuffle(positions);position_grid[(variant,domain)]=positions
  for i in range(n):
   v=vector(rng,used);vals=rng.sample(range(5,70),4)
   if i%2:vals[1]=vals[0]
   for variant,t in enumerate(templates):
    row={'id':f'{name}-{i:03}-{variant}','group':f'{name}-{i:03}','coefs':v,'values':vals,'domain':i%4,'template':t,'order':rng.sample(list(range(4)),4)}
    row['options']=options(v,position_grid[(variant,i%4)][i//4],rng);rows.append(row)
  splits[name]=rows
 for name in ['pool','test']:
  for domain in range(4):
   counts=Counter(display(r)[1] for r in splits[name] if r['domain']==domain)
   assert len(counts)==4 and len(set(counts.values()))==1
 assert not({question(r) for r in splits['pool']} & {question(r) for r in splits['test']})
 assert len({tuple(r['coefs']) for r in splits['pool']+splits['dev']+splits['test']})==92
 return splits
def symbolic_parse(text):
 result={}
 patterns=[(r'For source ([ABCD]), (add|subtract) its amount (once|twice|three times)\.',1),
           (r'(Add|Subtract) source ([ABCD]) (once|twice|three times)\.',2),
           (r'([ABCD]) contributes an (increase|decrease) of its amount (once|twice|three times)\.',1),
           (r'(Increase|Reduce) the total by source ([ABCD]) (once|twice|three times)\.',2)]
 for pat,order in patterns:
  for a,b,c in re.findall(pat,text):
   letter,op=(a,b) if order==1 else (b,a)
   result[letter]=(1 if op.lower() in ['add','increase'] else -1)*{'once':1,'twice':2,'three times':3}[c]
 for pat in [r'Ignore source ([ABCD])\.',r'Source ([ABCD]) does not count\.',r'([ABCD]) makes no contribution\.',r'Source ([ABCD]) has no effect\.']:
  for letter in re.findall(pat,text):result[letter]=0
 if set(result)!=set(LETTERS):raise ValueError('Grammar parser failed')
 return [result[x] for x in LETTERS]
def tokens(text):return re.findall(r'[a-z]+|\d+',text.lower())
def features(text,idf):
 counts=Counter(tokens(text));v={k:(1+math.log(n))*idf.get(k,1.) for k,n in counts.items()};norm=math.sqrt(sum(x*x for x in v.values()))
 return {k:x/norm for k,x in v.items()} if norm else {}
def dot(a,b):return sum(x*b.get(k,0.) for k,x in a.items())
class Selector:
 def __init__(self,pool,core=None):
  # Labels may be rendered in demonstrations; retrieval sees question text only.
  self.pool=pool;self.idf={t:math.log((len(pool)+1)/(1+sum(t in set(tokens(question(r))) for r in pool)))+1 for t in set().union(*(set(tokens(question(r))) for r in pool))}
  self.v=[features(question(r),self.idf) for r in pool];self.core=core if core is not None else list(range(8))
  self.diverse=self.diversify()
 def diversify(self):
  selected=[0]
  while len(selected)<8:
   selected.append(max((i for i in range(len(self.pool)) if i not in selected),key=lambda i:(min(1-dot(self.v[i],self.v[j]) for j in selected),-i)))
  return sorted(selected)
 def select(self,query,arm):
  if arm=='zero':return []
  if arm.startswith('random_'):
   rng=random.Random(arm+':'+query['id']);return sorted(rng.sample(range(len(self.pool)),2))
  v=features(query['text'],self.idf);scores=[dot(v,u) for u in self.v]
  candidates=self.core if arm=='model_core' else self.diverse if arm=='diverse_core' else list(range(len(self.pool)))
  first=max(candidates,key=lambda i:(scores[i],-i))
  if arm=='mmr':second=max((i for i in candidates if i!=first),key=lambda i:(.65*scores[i]-.35*dot(self.v[i],self.v[first]),-i))
  else:second=max((i for i in candidates if i!=first),key=lambda i:(scores[i],-i))
  return sorted([first,second])
def make_prompt(row,pool,selected,rotation=0):
 instruction='Select the exact signed coefficient vector. Include multiplicity and sign; ignore excluded sources. Choose A, B, C or D.\n\n'
 demos=[]
 for i in selected:
  text,gold,_=display(pool[i]);demos.append(text+'\nAnswer: '+LETTERS[gold])
 text,_,_=display(row,rotation)
 return instruction+('\n\n'.join(demos)+'\n\n' if demos else '')+text+'\nAnswer:'
def freeze():
 manifest=generate()
 for rows in manifest.values():
  for r in rows:assert symbolic_parse(question(r))==r['coefs']
 write(ROOT/'manifest.json',manifest)
 receipt={'protocol_sha256':digest(ROOT/'PROTOCOL.md'),'source_sha256':digest(__file__),'manifest_sha256':digest(ROOT/'manifest.json'),'test_generated_before_model_run':True,'seed':991763}
 write(ROOT/'freeze.json',receipt);print(canonical(receipt))
def check_freeze():
 receipt=json.loads((ROOT/'freeze.json').read_text())
 assert receipt['protocol_sha256']==digest(ROOT/'PROTOCOL.md')
 assert receipt['source_sha256']==digest(__file__)
 assert receipt['manifest_sha256']==digest(ROOT/'manifest.json')
 assert json.loads((ROOT/'manifest.json').read_text())==generate()
 return receipt

class Scorer:
 def __init__(self,checkpoint):
  import mlx.core as mx
  from mlx_lm import load
  self.mx=mx
  for name,expected in CHECKPOINT_HASHES.items():assert digest(Path(checkpoint)/name)==expected,name
  mx.set_cache_limit(128*1024**2);mx.set_memory_limit(2560*1024**2)
  self.model,self.tokenizer=load(str(checkpoint));mx.eval(self.model.parameters())
  self.labels=[self.tokenizer.encode(x,add_special_tokens=False) for x in LETTERS];assert all(len(x)==1 for x in self.labels)
  self.labels=[x[0] for x in self.labels];self.count=0
  # Validate last-position head optimisation against the public wrapper.
  ids=self.tokenizer.apply_chat_template([{'role':'user','content':'Choose A or B. Answer:'}],tokenize=True,add_generation_prompt=True,enable_thinking=False)
  x=mx.array([ids]);full=self.model(x)[:,-1,:];hidden=self.model.model(x)[:,-1:,:]
  fast=self.model.model.embed_tokens.as_linear(hidden) if self.model.args.tie_word_embeddings else self.model.lm_head(hidden)
  error=mx.max(mx.abs(full-fast[:,0,:])).item()
  af=full[0,self.labels].astype(mx.float32).tolist();bf=fast[0,0,self.labels].astype(mx.float32).tolist()
  assert max(range(4),key=lambda i:af[i])==max(range(4),key=lambda i:bf[i]),'Last-position head prediction mismatch'
  assert max(abs(a-b) for a,b in zip(af,bf))<.5,'Label-score head discrepancy exceeds BF16 tolerance'
  self.head_check_max_abs=error;del full,fast,hidden,x;mx.clear_cache()
  from mlx_lm.models.cache import make_prompt_cache
  long_ids=ids*9;x=mx.array([long_ids]);full=self.model(x)[:,-1,self.labels].astype(mx.float32);mx.eval(full)
  cache=make_prompt_cache(self.model)
  for start in range(0,len(long_ids),64):
   hidden=self.model.model(mx.array([long_ids[start:start+64]]),cache=cache);mx.eval(hidden,[c.state for c in cache])
  last=hidden[:,-1:,:];chunk=self.model.model.embed_tokens.as_linear(last) if self.model.args.tie_word_embeddings else self.model.lm_head(last)
  chunk=chunk[0,0,self.labels].astype(mx.float32);mx.eval(chunk)
  a=full[0].tolist();b=chunk.tolist();self.chunk_check_max_abs=max(abs(v-w) for v,w in zip(a,b))
  assert max(range(4),key=lambda i:a[i])==max(range(4),key=lambda i:b[i]),'Chunked-cache prediction mismatch'
  assert self.chunk_check_max_abs<.5,self.chunk_check_max_abs
  del full,chunk,hidden,last,cache,x;mx.clear_cache();mx.reset_peak_memory()
 def score(self,prompt):
  mx=self.mx
  from mlx_lm.models.cache import make_prompt_cache
  ids=self.tokenizer.apply_chat_template([{'role':'user','content':prompt}],tokenize=True,add_generation_prompt=True,enable_thinking=False)
  if len(ids)>1500:raise ValueError('Frozen prompt ceiling exceeded')
  cache=make_prompt_cache(self.model);start=time.perf_counter_ns()
  for s in range(0,len(ids),128):
   h=self.model.model(mx.array([ids[s:s+128]]),cache=cache)
   mx.eval(h,[c.state for c in cache])
   if mx.get_peak_memory()>2560*1024**2:raise MemoryError('Observed MLX allocation crossed the frozen2,560MiB abort threshold')
  h=h[:,-1:,:]
  logits=self.model.model.embed_tokens.as_linear(h) if self.model.args.tie_word_embeddings else self.model.lm_head(h)
  four=logits[0,0,self.labels].astype(mx.float32);mx.eval(four)
  vals=four.tolist();prediction=max(range(4),key=lambda i:(vals[i],-i))
  if mx.get_peak_memory()>2560*1024**2:raise MemoryError('Observed MLX allocation crossed the frozen2,560MiB abort threshold')
  elapsed=time.perf_counter_ns()-start
  result={'prediction':prediction,'label_logits':vals,'input_tokens':len(ids),'elapsed_ns':elapsed}
  del logits,h,four,cache;mx.clear_cache();self.count+=1
  if self.count%25==0:print('Model calls',self.count,'peak active MiB',round(mx.get_peak_memory()/1024**2,1),flush=True)
  return result
 def meta(self):
  import importlib.metadata,resource
  return {'checkpoint_sha256':CHECKPOINT_HASHES,'mlx_peak_active_bytes':self.mx.get_peak_memory(),'process_maxrss_bytes':resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,'model_calls':self.count,'head_check_max_abs':self.head_check_max_abs,'chunk_check_max_abs':self.chunk_check_max_abs,'software':{x:importlib.metadata.version(x) for x in ['mlx','mlx-lm']}}

def edges_and_core(matrix):
 n=len(matrix);masks=[sum((1<<j) for j in range(n) if i==j or matrix[i][j]) for i in range(n)]
 selected=[];covered=0
 while len(selected)<8:
  i=max((i for i in range(n) if i not in selected),key=lambda i:(((masks[i]&~covered).bit_count()),masks[i].bit_count(),-i))
  selected.append(i);covered|=masks[i]
 return sorted(selected)
def run(checkpoint,phase):
 receipt=check_freeze();manifest=json.loads((ROOT/'manifest.json').read_text());pool=manifest['pool'];scorer=Scorer(checkpoint)
 trace_path=ROOT/(phase+'_trace.json')
 if trace_path.exists():raise ValueError('Refuse to overwrite completed model trace')
 if phase=='preselect':
  rows=[];matrix=[[False]*16 for _ in range(16)]
  for i in range(16):
   for j in range(16):
    if i==j:continue
    result=scorer.score(make_prompt(pool[j],pool,[i]));result.update(demo=i,target=j);rows.append(result)
    matrix[i][j]=result['prediction']==display(pool[j])[1]
   write(ROOT/'preselect_progress.json',{'completed_demos':i+1,'rows':rows,'meta':scorer.meta()})
  core=edges_and_core(matrix);write(trace_path,{'freeze':receipt,'rows':rows,'matrix':matrix,'core':core,'meta':scorer.meta()})
  print('Fixed model-conditioned core',core,flush=True);return
 pre=json.loads((ROOT/'preselect_trace.json').read_text());assert pre['freeze']==receipt
 selector=Selector(pool,pre['core']);rows=[]
 if phase in ['dev','test']:
  targets=manifest[phase]
  for k,target in enumerate(targets):
   arm_order=ARMS[k%len(ARMS):]+ARMS[:k%len(ARMS)]
   for arm in arm_order:
    start=time.perf_counter_ns();selected=selector.select(public_view(target),arm);select_ns=time.perf_counter_ns()-start
    result=scorer.score(make_prompt(target,pool,selected));result.update(id=target['id'],arm=arm,selected=selected,selector_ns=select_ns);rows.append(result)
   if (k+1)%8==0:
    write(ROOT/(phase+'_progress.json'),{'completed_cases':k+1,'rows':rows,'meta':scorer.meta()});print(phase,k+1,'/',len(targets),flush=True)
  write(trace_path,{'freeze':receipt,'preselect_sha256':digest(ROOT/'preselect_trace.json'),'rows':rows,'meta':scorer.meta()});return
 if phase=='witness':
  matrix=pre['matrix'];triples=[(a,b,c) for a in range(16) for b in range(16) for c in range(16) if len({a,b,c})==3 and matrix[a][b] and matrix[b][c] and not matrix[a][c]][:12]
  for triple in triples:
   for rotation in (1,2):
    for i,j in [(triple[0],triple[1]),(triple[1],triple[2]),(triple[0],triple[2])]:
     result=scorer.score(make_prompt(pool[j],pool,[i],rotation));result.update(triple=list(triple),rotation=rotation,demo=i,target=j);rows.append(result)
  write(trace_path,{'freeze':receipt,'preselect_sha256':digest(ROOT/'preselect_trace.json'),'triples':[list(x) for x in triples],'rows':rows,'meta':scorer.meta()});return
 raise ValueError(phase)

def bootstrap(differences,alpha=.05,seed=53081):
 rng=random.Random(seed);n=len(differences);values=sorted(sum(rng.choice(differences) for _ in range(n))/n for _ in range(20000))
 return [values[int(alpha/2*len(values))],values[min(len(values)-1,int((1-alpha/2)*len(values)))]]
def analyse():
 receipt=check_freeze();manifest=json.loads((ROOT/'manifest.json').read_text());pre=json.loads((ROOT/'preselect_trace.json').read_text());assert pre['freeze']==receipt
 matrix=pre['matrix'];assert len(pre['rows'])==240
 for row in pre['rows']:assert matrix[row['demo']][row['target']]==(row['prediction']==display(manifest['pool'][row['target']])[1])
 assert pre['core']==edges_and_core(matrix)
 transitive_paths=[];violations=[]
 for a in range(16):
  for b in range(16):
   for c in range(16):
    if len({a,b,c})==3 and matrix[a][b] and matrix[b][c]:
     transitive_paths.append((a,b,c))
     if not matrix[a][c]:violations.append((a,b,c))
 out={'freeze':receipt,'pool_size':16,'core_size':len(pre['core']),'core':pre['core'],'sufficiency':{'direct_edges':sum(map(sum,matrix)),'two_edge_paths':len(transitive_paths),'violations':len(violations),'first_witnesses':[list(x) for x in violations[:12]]},'phases':{}}
 for phase in ['dev','test']:
  path=ROOT/(phase+'_trace.json')
  if not path.exists():continue
  trace=json.loads(path.read_text());assert trace['freeze']==receipt and trace['preselect_sha256']==digest(ROOT/'preselect_trace.json')
  targets=manifest[phase];lookup={x['id']:x for x in targets};seen=set();scores={a:{} for a in ARMS};token_counts={a:[] for a in ARMS};timing={a:[] for a in ARMS}
  for row in trace['rows']:
   key=(row['id'],row['arm']);assert key not in seen;seen.add(key);target=lookup[row['id']]
   assert row['selected']==Selector(manifest['pool'],pre['core']).select(public_view(target),row['arm'])
   scores[row['arm']][row['id']]=int(row['prediction']==display(target)[1]);token_counts[row['arm']].append(row['input_tokens']);timing[row['arm']].append(row['elapsed_ns'])
  assert len(seen)==len(targets)*len(ARMS)
  groups=sorted({r['group'] for r in targets});bygroup={a:[sum(scores[a][r['id']] for r in targets if r['group']==g)/sum(r['group']==g for r in targets) for g in groups] for a in ARMS}
  phase_result={'cases':len(targets),'groups':len(groups),'accuracy':{a:sum(scores[a].values())/len(targets) for a in ARMS},'input_tokens':{a:sum(v) for a,v in token_counts.items()},'max_prompt_tokens':max(max(v) for v in token_counts.values()),'inference_elapsed_ns':{a:sum(v) for a,v in timing.items()},'comparisons':{},'memory':trace['meta']}
  for arm in ARMS:
   if arm=='model_core':continue
   diff=[x-y for x,y in zip(bygroup['model_core'],bygroup[arm])]
   phase_result['comparisons'][arm]={'difference':sum(diff)/len(diff),'paired_cluster_95_interval':bootstrap(diff),'simultaneous_95_interval':bootstrap(diff,.05/7)}
  if phase=='test':
   primary=phase_result['comparisons']['lexical'];token_ratio=phase_result['input_tokens']['model_core']/phase_result['input_tokens']['lexical']
   phase_result['primary_gate']={'contrast':'model_core vs full-pool lexical','minimum_gain':.10,'max_token_ratio':1.10,'observed_token_ratio':token_ratio,'passed':primary['difference']>=.10 and primary['paired_cluster_95_interval'][0]>0 and token_ratio<=1.10}
   phase_result['all_baselines_gate']=all(v['difference']>=.05 and v['simultaneous_95_interval'][0]>0 for v in phase_result['comparisons'].values())
   phase_result['by_template']={str(t):{a:sum(scores[a][r['id']] for r in targets if r['template']==t)/sum(r['template']==t for r in targets) for a in ARMS} for t in [2,3]}
  out['phases'][phase]=phase_result
 witness=ROOT/'witness_trace.json'
 if witness.exists():
  trace=json.loads(witness.read_text());assert trace['freeze']==receipt
  survivors=[]
  for triple in trace['triples']:
   for rotation in (1,2):
    edges={}
    for row in trace['rows']:
     if row['triple']==triple and row['rotation']==rotation:edges[(row['demo'],row['target'])]=row['prediction']==display(manifest['pool'][row['target']],rotation)[1]
    a,b,c=triple
    if edges[(a,b)] and edges[(b,c)] and not edges[(a,c)]:survivors.append({'triple':triple,'rotation':rotation})
  out['sufficiency']['option_rotation_checks']={'triples':len(trace['triples']),'rotation_configurations':2*len(trace['triples']),'surviving_violations':len(survivors),'survivors':survivors}
 out['symbolic_grammar_ceiling']={'correct':sum(symbolic_parse(question(r))==r['coefs'] for r in manifest['test']),'cases':len(manifest['test'])}
 return out
def main():
 ap=argparse.ArgumentParser(description=__doc__);sp=ap.add_subparsers(dest='command',required=True);sp.add_parser('freeze')
 runp=sp.add_parser('run');runp.add_argument('checkpoint',type=Path);runp.add_argument('phase',choices=['preselect','dev','test','witness'])
 ana=sp.add_parser('analyse');ana.add_argument('--verify',type=Path);sp.add_parser('check')
 args=ap.parse_args()
 if args.command=='freeze':freeze()
 elif args.command=='run':run(args.checkpoint,args.phase)
 elif args.command=='check':print(canonical(check_freeze()))
 else:
  result=analyse()
  if args.verify:assert result==json.loads(args.verify.read_text());print('Verified manifests, selections, outcomes, clustered intervals and claim gates.')
  else:write(ROOT/'results.json',result);print(json.dumps(result,indent=2))
if __name__=='__main__':main()
