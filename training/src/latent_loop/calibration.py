"""Empirical scalar-temperature calibration, separate from proof validity."""
from __future__ import annotations
import math


def rescale(probabilities,temperature=1.):
    if not math.isfinite(temperature) or temperature<=0:raise ValueError('Temperature must be positive and finite')
    if len(probabilities)<2 or any(not math.isfinite(x) or x<0 for x in probabilities) or sum(probabilities)<=0:
        raise ValueError('Expected nonnegative finite probabilities with positive mass')
    values=[math.log(max(p,1e-12))/temperature for p in probabilities]
    maximum=max(values);values=[math.exp(v-maximum) for v in values];total=sum(values)
    return [v/total for v in values]


def fit_temperature(rows):
    """Minimize convex NLL in inverse temperature, bounded to [0.01,100]."""
    if not rows:raise ValueError('No calibration data')
    for row in rows:
        rescale(row['probs'])
        if type(row['label']) is not int or not 0<=row['label']<len(row['probs']):raise ValueError('Bad class label')
    logits=[[math.log(max(p,1e-12)) for p in r['probs']] for r in rows]
    def derivative(beta):
        total=0.
        for row,z in zip(rows,logits):
            maximum=max(z);w=[math.exp(beta*(v-maximum)) for v in z];mass=sum(w)
            total+=sum(p*v for p,v in zip(w,z))/mass-z[row['label']]
        return total/len(rows)
    lo,hi=.01,100.
    if derivative(lo)>=0:beta=lo
    elif derivative(hi)<=0:beta=hi
    else:
        for _ in range(90):
            mid=(lo+hi)/2
            if derivative(mid)<0:lo=mid
            else:hi=mid
        beta=(lo+hi)/2
    return dict(temperature=1/beta,inverse_temperature=beta,boundary=beta in (.01,100.),
        fit_n=len(rows),method='positive scalar temperature; NLL minimization',
        warning='A fitted temperature is empirical postprocessing, not a guarantee of accuracy or calibration under distribution shift')


def metrics(rows,temperature=1.,binary_true=False,threshold=.8):
    bins=[dict(n=0,correct=0,confidence_sum=0.) for _ in range(10)]
    nll=brier=correct=selected=selected_correct=0
    for row in rows:
        p=rescale(row['probs'],temperature);label=row['label']
        if binary_true:p=[p[0]+p[1],p[2]]
        prediction=max(range(len(p)),key=p.__getitem__);confidence=p[prediction];ok=prediction==label
        nll-=math.log(max(p[label],1e-12));brier+=sum((v-int(k==label))**2 for k,v in enumerate(p));correct+=ok
        bucket=bins[min(9,int(confidence*10))];bucket['n']+=1;bucket['correct']+=ok;bucket['confidence_sum']+=confidence
        if confidence>=threshold:selected+=1;selected_correct+=ok
    n=len(rows)
    for b in bins:
        b['accuracy']=b['correct']/b['n'] if b['n'] else None
        b['mean_confidence']=b['confidence_sum']/b['n'] if b['n'] else None
    return dict(n=n,accuracy=correct/n,nll=nll/n,brier=brier/n,
        ece=sum(abs(b['correct']-b['confidence_sum']) for b in bins)/n,reliability_bins=bins,
        threshold=threshold,answer_coverage=selected/n,answered_n=selected,
        answered_accuracy=selected_correct/selected if selected else None,
        selective_risk=1-selected_correct/selected if selected else None)


def decide(probabilities,review_cost=.2):
    """Bayes action under unit error loss and fixed review cost; no risk guarantee."""
    if not 0<=review_cost<=1:raise ValueError('Review cost must be in [0,1]')
    p=rescale(probabilities);index=max(range(len(p)),key=p.__getitem__)
    risk=1-p[index]
    return dict(action='answer' if risk<=review_cost else 'review_or_ask',class_index=index,
        empirical_confidence=p[index],expected_error_loss=risk,review_cost=review_cost,
        assumption='Probabilities must match correctness likelihoods in the current domain; this is not established for arbitrary input')


def sigmoid(value):
    if value>=0:return 1/(1+math.exp(-value))
    e=math.exp(value);return e/(1+e)


def fit_correctness(rows):
    """Two-parameter logistic correctness model, using generated token logprob.

    A slope penalty and two balanced intercept pseudo-observations stabilize
    small or separable samples. No test outcomes are inputs to fitting.
    """
    if len(rows)<10:raise ValueError('At least ten calibration outputs required')
    mean=sum(r['score'] for r in rows)/len(rows)
    scale=max(.01,math.sqrt(sum((r['score']-mean)**2 for r in rows)/len(rows)))
    data=[((r['score']-mean)/scale,float(r['correct'])) for r in rows]+[(0.,0.),(0.,1.)]
    if any(not math.isfinite(x) or y not in (0.,1.) for x,y in data):raise ValueError('Invalid correctness observations')
    def objective(a,b):
        total=.5*a*a
        for x,y in data:
            z=a*x+b;total+=max(z,0)+math.log1p(math.exp(-abs(z)))-y*z
        return total
    a=b=0.
    for _ in range(80):
        ga=a;gb=0.;haa=1.;hab=0.;hbb=0.
        for x,y in data:
            p=sigmoid(a*x+b);w=p*(1-p)
            ga+=(p-y)*x;gb+=p-y;haa+=w*x*x;hab+=w*x;hbb+=w
        determinant=haa*hbb-hab*hab
        da=(hbb*ga-hab*gb)/determinant;db=(haa*gb-hab*ga)/determinant
        if abs(da)+abs(db)<1e-9:break
        rate=1.;before=objective(a,b)
        while rate>1e-8 and objective(a-rate*da,b-rate*db)>before:rate*=.5
        a-=rate*da;b-=rate*db
    return dict(method='regularized logistic answer-correctness calibration',mean=mean,scale=scale,slope=a,intercept=b,
        fit_n=len(rows),slope_l2=1.,balanced_intercept_pseudo_observations=2,
        score='Mean generated token log probability, not itself a probability of answer correctness')


def correctness_probability(score,calibration):
    if not math.isfinite(score):raise ValueError('Nonfinite confidence score')
    return sigmoid(calibration['slope']*(score-calibration['mean'])/calibration['scale']+calibration['intercept'])
