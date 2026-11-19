"""Finite logically coherent concept beliefs and nonmyopic observation decisions.

Priors and logical constraints are supplied, not learned or assumed calibrated.
This is a bounded propositional belief planner, not an open-world FOL model.
"""
from __future__ import annotations
from functools import lru_cache
import math
from latent_loop.logic import parse,concept_graph


def atoms(tree):
    op=tree[0]
    if op=='atom':return {tree}
    if op=='not':return atoms(tree[1])
    if op in ('and','or','xor','implies','iff'):return atoms(tree[1])|atoms(tree[2])
    raise ValueError('Belief constraints must be ground Boolean formulas; no quantifiers or equality')


def holds(tree,bits,index):
    op=tree[0]
    if op=='atom':return bool(bits&(1<<index[tree]))
    if op=='not':return not holds(tree[1],bits,index)
    a=holds(tree[1],bits,index);b=holds(tree[2],bits,index)
    return {'and':a and b,'or':a or b,'xor':a!=b,'implies':not a or b,'iff':a==b}[op]


class ConceptBelief:
    def __init__(self,priors,constraints,query):
        if not isinstance(priors,dict) or not 1<=len(priors)<=12:raise ValueError('Supply1 to12 atom priors')
        entries=[]
        for text,p in priors.items():
            tree=parse(text)
            if tree[0]!='atom' or any(t in ('x','y','z','u','v','w') for t in tree[2]):raise ValueError('Priors need named ground atoms')
            if type(p) not in (int,float) or not math.isfinite(p) or not 0<=p<=1:raise ValueError('Invalid atom prior')
            entries.append((text,tree,float(p)))
        entries.sort();self.names=[x[0] for x in entries];self.index={x[1]:i for i,x in enumerate(entries)}
        if len(self.index)!=len(entries):raise ValueError('Duplicate atom with different formatting')
        self.by_name={name:i for i,name in enumerate(self.names)}
        self.constraints=[parse(f) for f in constraints];self.query=parse(query)
        required=set().union(*(atoms(t) for t in self.constraints+[self.query]))
        if required-set(self.index):raise ValueError('Every ground atom needs an explicit prior')
        arities={}
        for _,tree,_ in entries:
            if tree[1] in arities and arities[tree[1]]!=len(tree[2]):raise ValueError('Inconsistent predicate arity')
            arities[tree[1]]=len(tree[2])
        logworlds=[]
        for bits in range(1<<len(entries)):
            if not all(holds(t,bits,self.index) for t in self.constraints):continue
            logmass=0.;possible=True
            for i,(_,_,p) in enumerate(entries):
                chance=p if bits&(1<<i) else 1-p
                if not chance:possible=False;break
                logmass+=math.log(chance)
            if possible:logworlds.append((bits,logmass))
        if not logworlds:raise ValueError('Constraints and supplied priors have no positive-mass world')
        maximum=max(x[1] for x in logworlds);normalizer=sum(math.exp(logp-maximum) for _,logp in logworlds)
        self.worlds=[(bits,math.exp(logp-maximum)/normalizer) for bits,logp in logworlds]
        self.graph=concept_graph(self.constraints+[self.query],query_index=len(self.constraints))
    def _observed(self,observations):
        mask=value=0
        for name,status in (observations or {}).items():
            if name not in self.by_name or type(status) is not bool:raise ValueError('Observation needs an existing atom and Boolean value')
            bit=1<<self.by_name[name];mask|=bit
            if status:value|=bit
        return mask,value
    def posterior(self,observations=None):
        mask,value=self._observed(observations)
        worlds=[(bits,p) for bits,p in self.worlds if bits&mask==value];mass=sum(p for _,p in worlds)
        if not mass:raise ValueError('Observation contradicts the supplied belief; revise the model')
        marginals={name:sum(p for bits,p in worlds if bits&(1<<i))/mass for i,name in enumerate(self.names)}
        return dict(query_probability=sum(p for bits,p in worlds if holds(self.query,bits,self.index))/mass,
            marginals=marginals,possible_worlds=len(worlds),observations=observations or {},
            probabilities_calibrated=False,belief='Product Bernoulli prior conditioned on supplied logical constraints and observations')
    def plan(self,costs,observations=None,review_reward=.8,horizon=None):
        if len(costs)>8 or any(name not in self.by_name or not math.isfinite(c) or c<0 for name,c in costs.items()):
            raise ValueError('At most8 named observations with finite nonnegative costs')
        if not 0<=review_reward<=1:raise ValueError('Review reward must be in [0,1]')
        horizon=len(costs) if horizon is None else horizon
        if type(horizon) is not int or not 0<=horizon<=8:raise ValueError('Horizon must be0 to8')
        initial=self._observed(observations);calls=0
        @lru_cache(None)
        def value(mask,known,remaining):
            nonlocal calls
            calls+=1;worlds=[(w,p) for w,p in self.worlds if w&mask==known];mass=sum(p for _,p in worlds)
            if not mass:raise ValueError('No feasible posterior')
            probability=sum(p for w,p in worlds if holds(self.query,w,self.index))/mass
            scores={'answer_true':probability,'answer_false':1-probability,'review':review_reward}
            if remaining:
                for name,cost in sorted(costs.items()):
                    bit=1<<self.by_name[name]
                    if mask&bit:continue
                    yes=sum(p for w,p in worlds if w&bit)/mass
                    if yes<1e-15 or yes>1-1e-15:continue
                    gain=-cost+yes*value(mask|bit,known|bit,remaining-1)[0]+(1-yes)*value(mask|bit,known,remaining-1)[0]
                    scores['observe:'+name]=gain
            action=max(scores,key=scores.get)
            return scores[action],action,scores
        optimum,action,scores=value(*initial,horizon)
        return dict(action=action,expected_utility=optimum,action_values=scores,planning_states=calls,
            horizon=horizon,review_reward=review_reward,posterior=self.posterior(observations),
            concept_graph=self.graph,assumptions='Supplied ground Boolean theory, supplied prior odds, deterministic truthful observations, unit correct-answer reward and zero wrong-answer reward; costs are supplied. Double-precision finite enumeration; no learned or calibrated world belief is claimed.')
