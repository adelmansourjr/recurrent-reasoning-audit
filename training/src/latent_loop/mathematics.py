"""Small exact proof kernel for rational arithmetic and one-variable linear algebra.

Generated expressions are parsed as data. No eval, Python calls, attributes or
code execution. A checked calculation does not verify its language grounding.
"""
from __future__ import annotations
import ast
from fractions import Fraction
import json


def rational(text):
    token=str(text).strip()
    if token.count('/')==1:
        numerator,denominator=token.split('/')
        try:value=Fraction(numerator.strip())/Fraction(denominator.strip())
        except ZeroDivisionError as exc:raise ValueError('Division by zero') from exc
    else:value=Fraction(token)
    if max(abs(value.numerator).bit_length(), value.denominator.bit_length()) > 4096:
        raise ValueError('Number exceeds proof-kernel resource limit')
    return value


def display(value):
    return str(value.numerator) if value.denominator == 1 else str(value)


def expression(source, references=None, variable=False):
    """Return exact affine coefficients a,b for a*x+b and a reduction trace."""
    if not isinstance(source, str) or not source.strip() or len(source) > 1000:
        raise ValueError('Expected a bounded expression string')
    try:
        tree = ast.parse(source.strip(), mode='eval')
    except (SyntaxError, RecursionError) as exc:
        raise ValueError('Invalid arithmetic syntax') from exc
    if sum(1 for _ in ast.walk(tree)) > 160:
        raise ValueError('Expression too complex')
    refs = references or {}; trace = []
    def checked(a, b, rule, operands):
        a, b = rational(a), rational(b)
        trace.append(dict(rule=rule, operands=operands, result=dict(x=display(a), constant=display(b))))
        return a, b
    def visit(node, depth=0):
        if depth > 35: raise ValueError('Expression nesting too deep')
        if isinstance(node, ast.Constant) and type(node.value) in (int, float):
            token = ast.get_source_segment(source.strip(), node)
            return Fraction(0), rational(token)
        if isinstance(node, ast.Name):
            if node.id == 'x' and variable: return Fraction(1), Fraction(0)
            if node.id not in refs: raise ValueError(f'Unknown or forward reference: {node.id}')
            return Fraction(0), rational(refs[node.id])
        if isinstance(node, ast.UnaryOp) and isinstance(node.op, (ast.UAdd, ast.USub)):
            a,b=visit(node.operand,depth+1); sign=-1 if isinstance(node.op,ast.USub) else 1
            return checked(sign*a,sign*b,'sign',[display(a),display(b),sign])
        if not isinstance(node, ast.BinOp): raise ValueError('Unsupported arithmetic construct')
        a,b=visit(node.left,depth+1); c,d=visit(node.right,depth+1)
        operands=[dict(x=display(a),constant=display(b)),dict(x=display(c),constant=display(d))]
        if isinstance(node.op, ast.Add): return checked(a+c,b+d,'add',operands)
        if isinstance(node.op, ast.Sub): return checked(a-c,b-d,'subtract',operands)
        if isinstance(node.op, ast.Mult):
            if a and c: raise ValueError('Nonlinear multiplication is outside this kernel')
            return checked(a*d+b*c,b*d,'distribute_multiply',operands)
        if isinstance(node.op, ast.Div):
            if c: raise ValueError('Variable denominator needs domain analysis; unsupported')
            if not d: raise ValueError('Division by zero')
            return checked(a/d,b/d,'divide_nonzero_constant',operands)
        if isinstance(node.op, ast.Pow):
            if a or c or d.denominator != 1 or abs(d)>12:
                raise ValueError('Only bounded integer powers of rational constants are supported')
            if not b and d<0: raise ValueError('Division by zero')
            return checked(0,b**int(d),'integer_power',operands)
        raise ValueError('Unsupported arithmetic operator')
    a,b=visit(tree.body)
    return a,b,trace


def execute(program):
    """Check every operation, compute the answer, and return a replayable trace."""
    if not isinstance(program,list) or not 1<=len(program)<=32:
        raise ValueError('Program must contain 1 to 32 typed steps')
    refs={};proof=[];edges=[];concepts=set(['exact_rational_number'])
    for i,step in enumerate(program):
        if not isinstance(step,dict): raise ValueError('Each step must be an object')
        op=step.get('op'); result=dict(index=i,operation=op)
        if op=='calculate':
            if set(step)!={'op','expression'}: raise ValueError('Unexpected calculation fields')
            a,b,trace=expression(step['expression'],refs)
            if a: raise ValueError('Unresolved variable')
            value=b;result.update(expression=step['expression'],reductions=trace)
            concepts.update(t['rule'] for t in trace)
        elif op=='solve_linear':
            if set(step)!={'op','left','right'}: raise ValueError('Unexpected equation fields')
            a,b,lt=expression(step['left'],refs,True);c,d,rt=expression(step['right'],refs,True)
            slope=a-c;offset=d-b
            if not slope:
                raise ValueError('Equation has no unique solution: '+('identity' if not offset else 'inconsistent'))
            value=offset/slope
            if a*value+b != c*value+d: raise ValueError('Substitution check failed')
            result.update(left=step['left'],right=step['right'],left_reductions=lt,right_reductions=rt,
                normalized_equation=f'{display(slope)} * x = {display(offset)}',
                rule='subtract_equal_terms_then_divide_by_nonzero_coefficient',
                substitution=dict(left=display(a*value+b),right=display(c*value+d),equal=True))
            concepts.update(['equality','affine_expression','inverse_operation','equivalence_preserving_transformation','substitution'])
        else: raise ValueError('Unsupported operation; request a supported derivation')
        value=rational(value);refs[f's{i}']=value
        result['result']=display(value);proof.append(result)
        for field in ('expression','left','right'):
            if field not in step:continue
            tree=ast.parse(step[field].strip(),mode='eval')
            for name in sorted({n.id for n in ast.walk(tree) if isinstance(n,ast.Name) and n.id in refs}):
                if name!=f's{i}':edges.append(dict(source=name,target=f's{i}',role='checked_dependency'))
    return dict(answer=display(value),steps=proof,derivation_verified=True,grounding_verified=False,
        scope='Exact rational arithmetic and unique one-variable affine equations; conditional on the supplied program',
        proof_kernel='fraction-affine-v1',references={k:display(v) for k,v in refs.items()},
        mathematical_concepts=sorted(concepts),derivation_graph=dict(
            nodes=[dict(id=f's{s["index"]}',operation=s['operation'],value=s['result']) for s in proof],edges=edges),
        concept_origin='Deterministic proof-kernel annotations; not evidence of learned concept discovery')


def explain(result):
    lines=[]
    for step in result['steps']:
        if step['operation']=='calculate': lines.append(f"{step['expression']} = {step['result']}.")
        else: lines.append(f"{step['left']} = {step['right']}; reduce to {step['normalized_equation']}, so x = {step['result']}. Substitution agrees on both sides.")
    return ' '.join(lines)


if __name__=='__main__':
    import argparse
    from pathlib import Path
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('program');a=p.parse_args()
    result=execute(json.loads(Path(a.program).read_text()));result['text']=explain(result)
    print(json.dumps(result,indent=2))
