"""Difficulty ladders for fixed candidate-choice evaluation (0097).

Each scenario has staged probes, ordered by `stage`.  We report the FIRST
unsuccessful stage and the longest successful prefix, rather than assigning a
single all-or-nothing score to an advanced multistep question.  These probes
are independently prompted; passing them does not establish a chain-of-thought.
"""
from collections import defaultdict


def summarize_progression(rows, *, correct_key='correct'):
    groups = defaultdict(list)
    by_difficulty = defaultdict(list)
    by_category = defaultdict(list)
    for row in rows:
        difficulty = row.get('difficulty', 'unspecified')
        category = row.get('category', row.get('challenge_type', 'unspecified'))
        by_difficulty[difficulty].append(row)
        by_category[category].append(row)
        if row.get('scenario_id') is not None:
            groups[row['scenario_id']].append(row)

    def summarize(subset):
        return {'count': len(subset),
                'accuracy': sum(bool(r[correct_key]) for r in subset)/len(subset)}

    ladders = []
    for scenario, records in sorted(groups.items()):
        stages = [int(r['stage']) for r in records]
        if any(s < 1 for s in stages) or len(stages) != len(set(stages)):
            raise ValueError('Duplicate/invalid stage in scenario '+str(scenario))
        records = sorted(records, key=lambda r: int(r['stage']))
        first_failure = next((int(r['stage']) for r in records if not r[correct_key]), None)
        passed_prefix = 0
        for r in records:
            if not r[correct_key]:
                break
            passed_prefix += 1
        ladders.append({'scenario_id': scenario,
                        'stages': [{'stage': int(r['stage']), 'id': r['id'],
                                    'correct': bool(r[correct_key])} for r in records],
                        'first_failed_stage': first_failure,
                        'passed_prefix_length': passed_prefix,
                        'all_passed': first_failure is None,
                        'highest_stage_tested': max(stages)})
    return {
        'by_difficulty': {k: summarize(v) for k,v in sorted(by_difficulty.items())},
        'by_category': {k: summarize(v) for k,v in sorted(by_category.items())},
        'ladders': ladders,
        'scenario_count': len(ladders),
        'all_stages_passed_rate': (sum(s['all_passed'] for s in ladders)/len(ladders)
                                   if ladders else None),
        'mean_passed_prefix': (sum(s['passed_prefix_length'] for s in ladders)/len(ladders)
                               if ladders else None),
        'note': ('Stages are independent likelihood probes; the first failure is a '
                 'diagnostic threshold, not proof of an internal reasoning step.'),
    }
