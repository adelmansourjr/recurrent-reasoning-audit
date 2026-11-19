"""Safe first-order formula parser and optional Z3 inference, never Python eval.

Formula verification is conditional on faithful language grounding. Solver timeout
and inconsistent premises are distinct from logically undetermined conclusions.
"""
from __future__ import annotations
import re

NAME=r'(?:[^\W\d]|_)(?:\w|-(?=\w)|[.’](?=\w))*\+*'
IDENT=rf'(?:{NAME}|[0-9]+)'
TOKEN=re.compile(rf'\s*(<->|->|!=|[∀∃¬∧∨⊕→↔⟷≠=(),.:~&|]|{IDENT})')
PREC={'↔':1,'⟷':1,'<->':1,'→':2,'->':2,'⊕':3,'∨':4,'|':4,'∧':5,'&':5}
OPS={'↔':'iff','⟷':'iff','<->':'iff','→':'implies','->':'implies','⊕':'xor','∨':'or','|':'or','∧':'and','&':'and'}


def parse(formula):
    if len(formula)>12000:raise ValueError('Formula too long')
    tokens=[];pos=0
    while pos<len(formula):
        if not formula[pos:].strip():break
        m=TOKEN.match(formula,pos)
        if not m:raise ValueError(f'Unsupported formula syntax at {pos}: {formula[pos:pos+20]!r}')
        tokens.append(m[1]);pos=m.end()
    if len(tokens)>2000:raise ValueError('Formula too complex')
    i=0
    def take():
        nonlocal i
        if i>=len(tokens):raise ValueError('Unexpected end of formula')
        t=tokens[i];i+=1;return t
    def peek():return tokens[i] if i<len(tokens) else None
    def expect(t):
        found=take()
        if found!=t:raise ValueError(f'Expected {t!r}, found {found!r}')
    def name():
        t=take()
        if not re.fullmatch(IDENT,t):raise ValueError('Expected identifier')
        return t
    def expr(minimum=0,depth=0):
        if depth>100:raise ValueError('Formula nesting too deep')
        t=take()
        if t in ('∀','∃','forall','exists'):
            var=name()
            if peek() in ('.',':'):take()
            left=('all' if t in ('∀','forall') else 'exists',var,expr(0,depth+1))
        elif t in ('¬','~','not'):
            left=('not',expr(6,depth+1))
        elif t=='(':
            left=expr(0,depth+1);expect(')')
        elif re.fullmatch(IDENT,t):
            if peek()=='(':
                take();args=[]
                if peek()!=')':
                    args.append(name())
                    while peek()==',':take();args.append(name())
                expect(')');left=('atom',t,tuple(args))
            elif peek() in ('=','!=','≠'):
                op=take();left=('eq',t,name())
                if op!='=':left=('not',left)
            else:left=('atom',t,())
        else:raise ValueError(f'Unexpected token {t!r}')
        while peek() in PREC and PREC[peek()]>=minimum:
            op=take();p=PREC[op]
            right=expr(p if OPS[op]=='implies' else p+1,depth+1)
            left=(OPS[op],left,right)
        return left
    result=expr()
    if i!=len(tokens):raise ValueError(f'Trailing tokens: {tokens[i:]}')
    return result


def concept_graph(asts,query_index=None):
    """Export ordered arguments and scoped variables, without flattening logic."""
    nodes={};edges=[];roots=[];counter=0
    def symbol(name,env):
        if name in env:return env[name]
        key='entity:'+name
        nodes.setdefault(key,dict(id=key,kind='entity',name=name))
        return key
    def visit(a,env,fi):
        nonlocal counter
        key=f'formula{fi}:node{counter}';counter+=1;op=a[0]
        nodes[key]=dict(id=key,kind=op,formula=fi,formula_role='query' if fi==query_index else 'premise')
        def edge(child,role):edges.append(dict(source=key,target=child,role=role))
        if op in ('all','exists'):
            binder=key+':variable'
            nodes[binder]=dict(id=binder,kind='bound_variable',name=a[1],binder=key)
            edge(binder,'binds');edge(visit(a[2],dict(env,**{a[1]:binder}),fi),'body')
        elif op=='atom':
            concept=f'predicate:{a[1]}/{len(a[2])}'
            nodes.setdefault(concept,dict(id=concept,kind='predicate',name=a[1],arity=len(a[2])))
            edge(concept,'instantiates')
            for i,t in enumerate(a[2]):edge(symbol(t,env),f'argument_{i}')
        elif op=='eq':
            edge(symbol(a[1],env),'left_argument');edge(symbol(a[2],env),'right_argument')
        elif op=='not':edge(visit(a[1],env,fi),'body')
        else:
            edge(visit(a[1],env,fi),'left');edge(visit(a[2],env,fi),'right')
        return key
    for i,a in enumerate(asts):roots.append(visit(a,{},i))
    return dict(nodes=list(nodes.values()),edges=edges,formula_roots=roots,query_formula_index=query_index,
        meaning='Graph of inferred logical forms: predicate identity, ordered argument roles, quantifier binding, Boolean scope; not verified natural-language grounding')


def canonicalize(formulas):
    """Capture-avoiding symbol renaming and explicit parentheses, never new facts."""
    asts=[parse(f) for f in formulas];predicates={};constants={};arities={};bound_names=set()
    def binders(a):
        if a[0] in ('all','exists'):bound_names.add(a[1]);binders(a[2])
        elif a[0] not in ('atom','eq'):
            for child in a[1:]:binders(child)
    for a in asts:binders(a)
    def term(t,env):
        if t in env:return t
        if t in ('x','y','z','u','v','w'):raise ValueError(f'Unbound variable {t} in annotation')
        if t not in constants:
            i=len(constants)
            while f'c{i}' in bound_names or f'c{i}' in constants.values():i+=1
            constants[t]=f'c{i}'
        return constants[t]
    def render(a,env):
        op=a[0]
        if op in ('all','exists'):
            quant='∀' if op=='all' else '∃'
            return f'{quant}{a[1]} ({render(a[2],env|{a[1]})})'
        if op=='atom':
            _,name,args=a
            if name in arities and arities[name]!=len(args):raise ValueError('Inconsistent annotation predicate arity')
            arities[name]=len(args)
            if name not in predicates:predicates[name]=f'p{len(predicates)}'
            return predicates[name]+'('+', '.join(term(t,env) for t in args)+')' if args else predicates[name]
        if op=='eq':return f'{term(a[1],env)} = {term(a[2],env)}'
        if op=='not':return f'¬({render(a[1],env)})'
        sign={'and':'∧','or':'∨','xor':'⊕','iff':'↔','implies':'→'}[op]
        return f'({render(a[1],env)} {sign} {render(a[2],env)})'
    result=[render(a,set()) for a in asts]
    return result,dict(predicates=predicates,constants=constants,arities=arities)


def solve(premises,conclusion,timeout_ms=1500,find_gaps=False):
    import z3
    if not 1<=len(premises)<=128:raise ValueError('Provide 1..128 premises')
    asts=[parse(p) for p in premises];qast=parse(conclusion)
    entity=z3.DeclareSort('Entity');constants={};symbols={};arities={}
    def term(t,env):
        if t in env:return env[t]
        if t in ('x','y','z','u','v','w'):raise ValueError(f'Unbound variable {t}')
        if t not in constants:constants[t]=z3.Const('entity:'+t,entity)
        return constants[t]
    def build(a,env):
        op=a[0]
        if op in ('all','exists'):
            var=z3.Const('bound:'+a[1],entity)
            body=build(a[2],dict(env,**{a[1]:var}))
            return z3.ForAll([var],body) if op=='all' else z3.Exists([var],body)
        if op=='atom':
            _,name,args=a;n=len(args)
            if name in arities and arities[name]!=n:raise ValueError(f'Inconsistent arity for {name}')
            arities[name]=n
            if name not in symbols:
                symbols[name]=z3.Function('predicate:'+name,*([entity]*n),z3.BoolSort()) if n else z3.Bool('predicate:'+name)
            return symbols[name](*[term(t,env) for t in args]) if n else symbols[name]
        if op=='eq':return term(a[1],env)==term(a[2],env)
        if op=='not':return z3.Not(build(a[1],env))
        b,c=build(a[1],env),build(a[2],env)
        return {'and':z3.And,'or':z3.Or,'xor':z3.Xor,'iff':lambda x,y:x==y,'implies':z3.Implies}[op](b,c)
    facts=[build(a,{}) for a in asts];q=build(qast,{})
    solver=z3.Solver();solver.set(timeout=timeout_ms)
    for i,f in enumerate(facts):solver.assert_and_track(f,z3.Bool(f'premise_{i}'))
    base=solver.check()
    result=dict(premises=premises,conclusion=conclusion,base_status=str(base),
        concepts=[dict(predicate=k,arity=v) for k,v in sorted(arities.items())],
        entities=sorted(constants),grounding_verified=False,concept_graph=concept_graph(asts+[qast],query_index=len(asts)),
        scope='First-order uninterpreted predicates, equality, nonempty domain, no unique-name assumption')
    if base!=z3.sat:
        result.update(label=None,status='inconsistent_premises' if base==z3.unsat else 'solver_unknown',
            reason=solver.reason_unknown() if base==z3.unknown else 'Premises have no model')
        return result
    outcomes={};cores={}
    for tag,assumption in (('not_conclusion',z3.Not(q)),('conclusion',q)):
        solver.push();solver.assert_and_track(assumption,z3.Bool('query_assumption'))
        status=solver.check();outcomes[tag]=str(status)
        if status==z3.unsat:cores[tag]=[str(x) for x in solver.unsat_core()]
        solver.pop()
    if outcomes['not_conclusion']=='unsat':label='True'
    elif outcomes['conclusion']=='unsat':label='False'
    elif all(s=='sat' for s in outcomes.values()):label='Unknown'
    else:label=None
    result.update(label=label,status='decided' if label else 'solver_unknown',
        checks=outcomes,unsat_cores=cores,
        inference_verified=label in ('True','False'),
        note='An unsat core identifies sufficient formulas; it is not a verified translation or a natural-language proof.')
    if find_gaps and label=='Unknown':
        result['gap_candidates']=observation_gaps(facts,q,symbols,constants,arities,min(timeout_ms,150))
    return result


def observation_gaps(facts,q,symbols,constants,arities,timeout_ms=150,limit=16):
    """Check conditional utility of named ground facts; no invented outcome priors."""
    import itertools
    import z3
    candidates=[];examined=0
    qq=z3.simplify(q);nq=z3.simplify(z3.Not(q))
    for name in sorted(symbols):
        arity=arities[name]
        if arity>2:continue
        for args in itertools.product(sorted(constants),repeat=arity):
            atom=symbols[name](*[constants[t] for t in args]) if arity else symbols[name]
            if z3.eq(z3.simplify(atom),qq) or z3.eq(z3.simplify(atom),nq):continue
            if examined>=limit:break
            examined+=1
            labels=[];feasible=[]
            for outcome in (True,False):
                s=z3.Solver();s.set(timeout=timeout_ms);s.add(*facts)
                s.add(atom if outcome else z3.Not(atom));base=s.check();feasible.append(base==z3.sat)
                if base!=z3.sat:labels.append(None);continue
                checks=[]
                for assumption in (z3.Not(q),q):
                    s.push();s.add(assumption);checks.append(s.check());s.pop()
                if checks[0]==z3.unsat:labels.append('True')
                elif checks[1]==z3.unsat:labels.append('False')
                elif all(x==z3.sat for x in checks):labels.append('Unknown')
                else:labels.append(None)
            resolves=sum(y in ('True','False') for y in labels)
            if all(feasible) and resolves:
                formula=name+'('+', '.join(args)+')' if arity else name
                candidates.append(dict(observation=formula,predicate=name,arguments=list(args),
                    if_observed_true=labels[0],if_observed_false=labels[1],resolving_outcomes=resolves,
                    scope='Conditional first-order checks under inferred formulas; no probability, cost, observability or optimal-query claim'))
        if examined>=limit:break
    candidates.sort(key=lambda r:(-r['resolving_outcomes'],r['observation']))
    return dict(candidates=candidates,examined=examined,limit=limit,
        scope='Finite pool of named-entity atoms with arity<=2. Absence of a candidate does not establish absence of useful observations. Query itself excluded.')
