"""Exploratory, baseline-conditioned singleton-chain diagnostic.

Separate from the frozen selection confirmation. Run only sequentially after
other model processes end. No population hypothesis test is performed.
"""
import argparse
from collections import Counter
import json
from pathlib import Path

import study
import selection_followup as followup

ROOT = Path(__file__).resolve().parent
PROTOCOL = ROOT / 'WITNESS_PROTOCOL.md'
FREEZE = ROOT / 'witness_followup_freeze.json'
TRACE = ROOT / 'witness_followup_trace.json'
PROGRESS = ROOT / 'witness_followup_progress.json'
SELECTION = ROOT / 'witness_followup_selection.json'
RESULTS = ROOT / 'witness_followup_results.json'


def freeze():
    if FREEZE.exists() or any(p.exists() for p in [TRACE, PROGRESS, SELECTION, RESULTS]):
        raise FileExistsError('Refuse to overwrite a diagnostic freeze or artifacts')
    initial = study.check_freeze()
    successor = followup.check()
    pre_path = ROOT / 'followup_preselect_trace.json'
    pre = json.loads(pre_path.read_text())
    assert pre['freeze'] == successor and len(pre['rows']) == 240
    receipt = {
        'scope': 'Exploratory training-pool diagnostic; no confirmation efficacy claim',
        'source_sha256': study.digest(__file__),
        'protocol_sha256': study.digest(PROTOCOL),
        'initial_freeze_sha256': study.digest(ROOT / 'freeze.json'),
        'followup_freeze_sha256': study.digest(ROOT / 'followup_freeze.json'),
        'preselect_trace_sha256': study.digest(pre_path),
        'pool_manifest_sha256': study.digest(ROOT / 'manifest.json'),
        'initial_freeze': initial,
        'followup_freeze': successor,
        'before_zero_shot_or_witness_inference': True,
        'max_witnesses': 12,
        'zero_shot_rotations': [0, 1, 2, 3],
        'edge_followup_rotations': [1, 2],
    }
    study.write(FREEZE, receipt)
    print(study.canonical(receipt))


def check():
    receipt = json.loads(FREEZE.read_text())
    for key, path in [
        ('source_sha256', Path(__file__)),
        ('protocol_sha256', PROTOCOL),
        ('initial_freeze_sha256', ROOT / 'freeze.json'),
        ('followup_freeze_sha256', ROOT / 'followup_freeze.json'),
        ('preselect_trace_sha256', ROOT / 'followup_preselect_trace.json'),
        ('pool_manifest_sha256', ROOT / 'manifest.json'),
    ]:
        assert receipt[key] == study.digest(path), key
    assert receipt['initial_freeze'] == study.check_freeze()
    assert receipt['followup_freeze'] == followup.check()
    return receipt


def argmax(values):
    return max(range(4), key=lambda i: (values[i], -i))


def raw_triples(matrix):
    return [(a, b, c) for a in range(16) for b in range(16) for c in range(16)
            if len({a, b, c}) == 3
            and matrix[a][b] and matrix[b][c] and not matrix[a][c]]


def choose(matrix, zero_correct):
    raw = raw_triples(matrix)
    eligible = [t for t in raw if not zero_correct[t[1], 0]
                and not zero_correct[t[2], 0]]
    return raw, eligible, eligible[:12]


def run(checkpoint):
    receipt = check()
    if any(p.exists() for p in [TRACE, PROGRESS, SELECTION, RESULTS]):
        raise FileExistsError('Refuse to overwrite diagnostic inference or results')
    bank = json.loads((ROOT / 'manifest.json').read_text())['pool']
    pre = json.loads((ROOT / 'followup_preselect_trace.json').read_text())
    matrix = pre['matrix']
    assert len(pre['rows']) == 240 and pre['freeze'] == receipt['followup_freeze']
    scorer = followup.PrefixedScorer(checkpoint)
    zero, edges = [], []
    selection = None

    def save_progress():
        study.write(PROGRESS, {'freeze': receipt, 'zero_shot_rows': zero,
                              'edge_rows': edges, 'selection': selection,
                              'meta': scorer.meta()})

    # Exactly 64 zero-shot observations, independent of any selected chain.
    for target in range(16):
        for rotation in range(4):
            result = scorer.score(study.make_prompt(bank[target], bank, [], rotation))
            _, gold, _ = study.display(bank[target], rotation)
            zero.append({'target': target, 'target_id': bank[target]['id'],
                         'rotation': rotation, 'selected': [], 'gold_label': gold,
                         'correct': result['prediction'] == gold, **result})
            save_progress()
        print('Witness zero-shot baseline', target + 1, '/16', flush=True)
    zero_correct = {(r['target'], r['rotation']): r['correct'] for r in zero}
    raw, eligible, chosen = choose(matrix, zero_correct)
    queries = sorted({(i, j, rot) for a, b, c in chosen for rot in (1, 2)
                      for i, j in [(a, b), (b, c), (a, c)]})
    # Selection is fully disclosed and saved BEFORE any rotated edge inference.
    selection = {'freeze': receipt, 'raw_nontransitive_triples': len(raw),
                 'eligible_after_canonical_zero_shot_filter': len(eligible),
                 'all_eligible_lexicographic_triples': [list(t) for t in eligible],
                 'first_twelve_or_fewer': [list(t) for t in chosen],
                 'deduplicated_edge_queries': [list(q) for q in queries],
                 'zero_shot_trace_sha256': study.sha256(study.canonical(zero).encode()).hexdigest(),
                 'before_rotated_edge_inference': True}
    study.write(SELECTION, selection)
    save_progress()
    for index, (demo, target, rotation) in enumerate(queries):
        result = scorer.score(study.make_prompt(bank[target], bank, [demo], rotation))
        _, gold, _ = study.display(bank[target], rotation)
        edges.append({'demo': demo, 'target': target, 'demo_id': bank[demo]['id'],
                      'target_id': bank[target]['id'], 'rotation': rotation,
                      'selected': [demo], 'gold_label': gold,
                      'correct': result['prediction'] == gold, **result})
        save_progress()
        if (index + 1) % 12 == 0:
            print('Witness rotated edge queries', index + 1, '/', len(queries), flush=True)
    study.write(TRACE, {'freeze': receipt, 'selection_sha256': study.digest(SELECTION),
                       'zero_shot_rows': zero, 'edge_rows': edges,
                       'meta': scorer.meta(), 'selection': selection})
    print('Saved exploratory witness trace; run analyse for all outcomes.', flush=True)


def analyse():
    receipt = check()
    trace = json.loads(TRACE.read_text())
    selection = json.loads(SELECTION.read_text())
    assert trace['freeze'] == receipt and selection == trace['selection']
    assert trace['selection_sha256'] == study.digest(SELECTION)
    bank = json.loads((ROOT / 'manifest.json').read_text())['pool']
    pre = json.loads((ROOT / 'followup_preselect_trace.json').read_text())
    matrix = pre['matrix']
    zero, edges = trace['zero_shot_rows'], trace['edge_rows']
    assert len(zero) == 64
    for rows in [zero, edges]:
        for row in rows:
            assert len(row['label_logits']) == 4
            assert row['prediction'] == argmax(row['label_logits'])
            assert row['gold_label'] == study.display(bank[row['target']], row['rotation'])[1]
            assert row['correct'] == (row['prediction'] == row['gold_label'])
    z = {(r['target'], r['rotation']): r['correct'] for r in zero}
    assert len(z) == 64 and set(z) == {(t, rot) for t in range(16) for rot in range(4)}
    raw, eligible, chosen = choose(matrix, z)
    assert selection['raw_nontransitive_triples'] == len(raw)
    assert selection['all_eligible_lexicographic_triples'] == [list(t) for t in eligible]
    assert selection['first_twelve_or_fewer'] == [list(t) for t in chosen]
    assert selection['eligible_after_canonical_zero_shot_filter'] == len(eligible)
    assert selection['zero_shot_trace_sha256'] == study.sha256(study.canonical(zero).encode()).hexdigest()
    required = {(i, j, rot) for a, b, c in chosen for rot in (1, 2)
                for i, j in [(a, b), (b, c), (a, c)]}
    observed = {(r['demo'], r['target'], r['rotation']): r for r in edges}
    assert len(observed) == len(edges) and set(observed) == required
    assert selection['deduplicated_edge_queries'] == [list(q) for q in sorted(required)]
    trials = []
    for a, b, c in chosen:
        for rotation in (1, 2):
            ab = observed[a, b, rotation]['correct']
            bc = observed[b, c, rotation]['correct']
            ac = observed[a, c, rotation]['correct']
            pattern = ab and bc and not ac
            controls = not z[b, rotation] and not z[c, rotation]
            trials.append({'triple': [a, b, c], 'rotation': rotation,
                           'a_to_b_correct': ab, 'b_to_c_correct': bc,
                           'a_to_c_correct': ac,
                           'zero_shot_b_correct': z[b, rotation],
                           'zero_shot_c_correct': z[c, rotation],
                           'nontransitive_pattern_retained': pattern,
                           'both_zero_shot_controls_wrong': controls,
                           'strict_survival': pattern and controls})
    costs = {name: {'calls': len(rows), 'input_tokens': sum(r['input_tokens'] for r in rows),
                    'elapsed_ns': sum(r['elapsed_ns'] for r in rows)}
             for name, rows in [('zero_shot', zero), ('rotated_edges', edges)]}
    return {'freeze': receipt, 'scope': 'Exploratory selected training-pool witnesses, not a population test or refutation of FEEDER',
            'raw_nontransitive_triples': len(raw),
            'eligible_baseline_conditioned_triples': len(eligible),
            'selected_triples': len(chosen),
            'rotated_configurations': len(trials),
            'nontransitive_pattern_retained': sum(r['nontransitive_pattern_retained'] for r in trials),
            'strict_surviving_configurations': sum(r['strict_survival'] for r in trials),
            'triples_strictly_surviving_both_rotations': sum(all(r['strict_survival'] for r in trials if r['triple'] == list(t)) for t in chosen),
            'trials': trials,
            'canonical_zero_shot_correct': sum(z[t, 0] for t in range(16)),
            'zero_shot_correct_by_rotation': {str(rot): sum(z[t, rot] for t in range(16)) for rot in range(4)},
            'zero_shot_label_counts': dict(Counter(study.LETTERS[r['prediction']] for r in zero)),
            'rotated_edge_label_counts': dict(Counter(study.LETTERS[r['prediction']] for r in edges)),
            'costs': costs, 'meta': trace['meta'],
            'limitations': ['Witnesses selected on canonical pool outcomes; dependent triples and rotations.',
                            'Definition is exact singleton correctness under one checkpoint and prompt.',
                            'Option rotations assess robustness to two ordering changes only.',
                            'No confirmation outcomes or model-selection efficacy are inferred.']}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest='command', required=True)
    sub.add_parser('freeze')
    sub.add_parser('check')
    run_parser = sub.add_parser('run')
    run_parser.add_argument('checkpoint', type=Path)
    analysis_parser = sub.add_parser('analyse')
    analysis_parser.add_argument('--verify', type=Path)
    args = parser.parse_args()
    if args.command == 'freeze':
        freeze()
    elif args.command == 'check':
        print(study.canonical(check()))
    elif args.command == 'run':
        run(args.checkpoint)
    else:
        result = analyse()
        if args.verify:
            assert result == json.loads(args.verify.read_text())
            print('Verified witness selection, zero-shot controls, rotated edges and all outcomes.')
        else:
            if RESULTS.exists():
                raise FileExistsError('Refuse to overwrite diagnostic results; use --verify')
            study.write(RESULTS, result)
            print(json.dumps(result, indent=2))


if __name__ == '__main__':
    main()
