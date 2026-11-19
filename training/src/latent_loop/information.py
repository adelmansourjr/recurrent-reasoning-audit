"""Exact decision-theoretic controller, not a learned neural policy.

Unknown required leaves are independent Bernoulli readiness variables with
specified priors. Observations reveal them perfectly. Reward is 1 for a correct
terminal binary action minus observation costs. Known blockers terminate.
"""
from functools import lru_cache
import math


def optimal_information_policy(priors, costs):
    priors,costs=tuple(priors),tuple(costs)
    if len(priors)>16:
        raise ValueError("Exact subset planning supports at most 16 unknown facts")
    if len(priors)!=len(costs) or any(not math.isfinite(p) or not 0<p<1 for p in priors) or any(not math.isfinite(c) or c<0 for c in costs):
        raise ValueError("Require matching priors in (0,1) and nonnegative costs")
    @lru_cache(None)
    def value(mask):
        indices=[j for j in range(len(priors)) if mask & (1<<j)]
        q=math.prod(priors[j] for j in indices)
        candidates=[(q,"ready"),(1-q,"blocked")]
        for j in indices:
            after=value(mask^(1<<j))[0]
            candidates.append((-costs[j]+1-priors[j]+priors[j]*after,j))
        return max(candidates,key=lambda x:x[0])
    return value


def one_step_policy(priors,costs,mask):
    indices=[j for j in range(len(priors)) if mask & (1<<j)]
    q=math.prod(priors[j] for j in indices)
    candidates=[(q,"ready"),(1-q,"blocked")]
    for j in indices:
        next_q=q/priors[j]
        candidates.append((-costs[j]+1-priors[j]+priors[j]*max(next_q,1-next_q),j))
    return max(candidates,key=lambda x:x[0])


def rollout(priors,costs,hidden,policy):
    mask=(1<<len(priors))-1
    spent=0.0
    queries=[]
    while True:
        _,action=policy(mask)
        if isinstance(action,str):
            truth="ready" if all(hidden) else "blocked"
            return dict(utility=float(action==truth)-spent,correct=action==truth,queries=queries)
        if not mask & (1<<action):
            raise ValueError("Querying a fact already observed")
        queries.append(action)
        spent+=costs[action]
        mask^=1<<action
        if not hidden[action]:
            return dict(utility=1-spent,correct=True,queries=queries)
