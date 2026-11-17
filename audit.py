"""Exact routing ceilings for four archived Qwen mathematics policies."""
from argparse import ArgumentParser
from fractions import Fraction
from hashlib import sha256
from itertools import combinations
import json
from math import comb
from pathlib import Path

ARCHIVES = {
    'one_pass': ('v8/math/evaluation.json', '5218e47b46d32da43be2b4f2738d4773afb4aae02a87b98480c352c3903cc7a9'),
    'three_refinements': ('v9/loop3/evaluation.json', '522f42f2aab611cff5a7692f6e5f0f65d9fc8ba84af089c6006b11ea44729e55'),
    'extra_training': ('v9/one-pass-compute/evaluation.json', '7b394fa628eb9f5f6420ee678a79c55f2d2d395c845a2c019f9edc578c3e0224'),
    'stock_qwen': ('v8/math/native-rescored.json', '0949e40936d56e2c385b0b2d8e19cb6569afacab9984365288eeb7eb5504c960'),
}
DOMAINS = {'GSM8K-test': 64, 'generated-linear': 24}


def correct(record):
    if record['answer'] is None:
        return False
    try:
        return Fraction(record['answer']) == Fraction(record['gold_answer'])
    except (ValueError, ZeroDivisionError, TypeError):
        return False


def archive_table(root):
    models = {}
    for label, (filename, digest) in ARCHIVES.items():
        raw = (root / filename).read_bytes()
        assert sha256(raw).hexdigest() == digest, filename
        data = json.loads(raw)
        assert data['complete'], filename
        rows = data['predictions']
        by_id = {r['id']: r for r in rows}
        assert len(rows) == len(by_id) == sum(DOMAINS.values())
        assert all(correct(row) == row['correct'] for row in rows), filename
        models[label] = by_id
    matching_ids = models['one_pass'].keys()
    assert all(case_map.keys() == matching_ids for case_map in models.values())
    output = {source: [] for source in DOMAINS}
    for case in sorted(matching_ids):
        records = {name: model[case] for name, model in models.items()}
        reference = records['one_pass']
        assert all(row['source'] == reference['source'] and
                   Fraction(row['gold_answer']) == Fraction(reference['gold_answer'])
                   for row in records.values()), case
        compile_row = {name: correct(row) for name, row in records.items()}
        compile_row['three_refinement_proof'] = bool(records['three_refinements'].get('derivation_verified', False))
        output[reference['source']].append(compile_row)
    return output


def envelope(rows, candidates):
    base = sum(r['one_pass'] for r in rows)
    fixes = sum(not r['one_pass'] and any(r[name] for name in candidates) for r in rows)
    return [base + min(quota, fixes) for quota in range(len(rows) + 1)]


def enumeration_check():
    cases = [
        {'one_pass': True, 'second': False, 'third': True},
        {'one_pass': False, 'second': True, 'third': False},
        {'one_pass': False, 'second': False, 'third': True},
        {'one_pass': False, 'second': False, 'third': False},
    ]
    optional = ('second', 'third')
    actual = envelope(cases, optional)
    for quota, expected in enumerate(actual):
        audited = max(
            sum(row['one_pass'] or (i in subset and any(row[x] for x in optional))
                for i, row in enumerate(cases))
            for subset in combinations(range(len(cases)), quota)
        )
        assert audited == expected


def report(panel):
    enumeration_check()
    outputs = {}
    for name, rows in panel.items():
        assert len(rows) == DOMAINS[name]
        assert all(set(x) == set(ARCHIVES) | {'three_refinement_proof'} for x in rows)
        assert all(type(v) is bool for x in rows for v in x.values())
        counts = {model: sum(x[model] for x in rows) for model in ARCHIVES}
        ceilings = {
            'refinement': envelope(rows, ('three_refinements',)),
            'training_control': envelope(rows, ('extra_training',)),
            'any_trained_adapter': envelope(rows, ('three_refinements', 'extra_training')),
            'stock_fallback': envelope(rows, ('stock_qwen',)),
        }
        fixes = sum(not x['one_pass'] and x['three_refinements'] for x in rows)
        breaks = sum(x['one_pass'] and not x['three_refinements'] for x in rows)
        dissents = fixes + breaks
        p = min(Fraction(1), Fraction(2 * sum(comb(dissents, i) for i in range(min(fixes, breaks) + 1)), 2**dissents))
        proof = [x for x in rows if x['three_refinement_proof']]
        outputs[name] = {
            'cases': len(rows), 'correct': counts, 'oracle_ceiling_by_quota': ceilings,
            'trained_oracle_minus_stock': ceilings['any_trained_adapter'][-1] - counts['stock_qwen'],
            'three_refinement_transition': {'fixes': fixes, 'breaks': breaks, 'exact_mcnemar_p': str(p)},
            'three_refinement_proof': {'verified': len(proof), 'verified_but_wrong': sum(not x['three_refinements'] for x in proof)},
        }
    return outputs


def as_json(x):
    return json.dumps(x, indent=2, sort_keys=True) + '\n'


if __name__ == '__main__':
    cli = ArgumentParser(description=__doc__)
    cli.add_argument('--archives', type=Path)
    cli.add_argument('--table', type=Path, required=True)
    output = cli.add_mutually_exclusive_group(required=True)
    output.add_argument('--output', type=Path)
    output.add_argument('--verify', type=Path)
    args = cli.parse_args()
    if args.archives:
        rows = archive_table(args.archives)
        if args.table.exists():
            assert args.table.read_text() == as_json(rows), 'Frozen case table does not match archives'
        else:
            args.table.write_text(as_json(rows))
    else:
        rows = json.loads(args.table.read_text())
    results = report(rows)
    result = {'protocol_sha256': sha256(Path(__file__).with_name('PROTOCOL.md').read_bytes()).hexdigest(),
              'case_table_sha256': sha256(args.table.read_bytes()).hexdigest(), 'results': results}
    if args.output:
        args.output.write_text(as_json(result))
    else:
        assert args.verify.read_text() == as_json(result), 'Replay differs from archived results'
    for domain, data in results.items():
        print(domain, data['correct'], data['trained_oracle_minus_stock'], data['three_refinement_transition'],
              data['three_refinement_proof'])
