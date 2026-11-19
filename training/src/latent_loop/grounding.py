"""Commit source-backed representations before a conclusion is available.

The translator remains fallible, but a query cannot mutate a committed fact.
"""
from __future__ import annotations
import hashlib
import json
import copy
from pathlib import Path
from latent_loop.logic import solve


def memory_key(records,model_fingerprint):
    payload=json.dumps(dict(records=records,model=model_fingerprint),sort_keys=True,ensure_ascii=False)
    return hashlib.sha256(payload.encode()).hexdigest()


def representation_key(formulas):
    return hashlib.sha256(json.dumps(formulas,ensure_ascii=False).encode()).hexdigest()


def validate_snapshot(snapshot):
    if memory_key(snapshot['records'],snapshot['model_fingerprint'])!=snapshot['key']:
        raise ValueError('Committed source records changed without a new memory revision')
    if representation_key(snapshot['formulas'])!=snapshot['representation_sha256']:
        raise ValueError('Committed formulas changed without a new representation')


def commit_memory(records,translator,model_fingerprint,cache_dir=None):
    if not records or any(not isinstance(r,dict) or not all(isinstance(r.get(k),str) and r[k].strip() for k in ('id','text','source')) for r in records):
        raise ValueError('Nonempty source-backed memory records required')
    key=memory_key(records,model_fingerprint)
    path=Path(cache_dir)/f'{key}.json' if cache_dir else None
    if path and path.exists():
        snapshot=json.loads(path.read_text())
        if snapshot['key']!=key or snapshot['records']!=records:raise ValueError('Committed memory fingerprint mismatch')
        validate_snapshot(snapshot)
        return snapshot
    # The translator receives only premise texts, never a candidate conclusion.
    formulas=translator([r['text'] for r in records])
    if not isinstance(formulas,list) or len(formulas)!=len(records) or not all(isinstance(f,str) for f in formulas):
        raise ValueError('Memory translation must give exactly one formula per source')
    checked=solve(formulas,formulas[0])
    snapshot=dict(key=key,model_fingerprint=model_fingerprint,records=copy.deepcopy(records),formulas=list(formulas),
        representation_sha256=representation_key(formulas),
        theory=checked,grounding_verified=False,query_independent_commitment=True)
    if path:path.parent.mkdir(parents=True,exist_ok=True);path.write_text(json.dumps(snapshot,indent=2))
    return snapshot


def query_memory(snapshot,conclusion,translator,find_gaps=True):
    validate_snapshot(snapshot)
    if snapshot['theory']['status']=='inconsistent_premises':raise ValueError('Committed facts are inconsistent; review the sources/grounding')
    # A deep copy prevents a query translator from changing stored memory.
    context=json.loads(json.dumps(snapshot))
    formulas=translator(conclusion,context)
    if not isinstance(formulas,list) or len(formulas)!=1 or not isinstance(formulas[0],str):raise ValueError('Exactly one query formula required')
    query=formulas[0];result=solve(snapshot['formulas'],query,find_gaps=find_gaps)
    return dict(label=result['label'],logic=result,formulas=snapshot['formulas']+[query],
        committed_memory_key=snapshot['key'],query_independent_commitment=True,grounding_verified=False)
