"""0096: regression tests for high-ceiling narrative tasks and sandbox diagnosis."""
import contextlib
import json
from collections import Counter
from pathlib import Path

import pytest

from benchmarks.capability_suite import _load, run_narrative
from benchmarks.live_suite import LiveBenchmarkScheduler

ROOT = Path(__file__).resolve().parents[1]
TASKS = ROOT/'benchmarks/tasks'
CONFIGS = ROOT/'benchmarks/configs'


def test_new_context_cases_difficulty_diversity_and_answers():
    tasks, checksum = _load(TASKS/'narrative_challenge_0096.json', 'narrative')
    assert len(tasks) == 126 and len(checksum) == 64
    groups = Counter(t['challenge_type'] for t in tasks)
    assert len(groups) == 7
    assert set(groups.values()) == {18}
    assert len({t['id'] for t in tasks}) == len(tasks)
    correct_positions = Counter(t['answer'] for t in tasks)
    assert all(20 <= correct_positions[i] <= 45 for i in range(4))
    for t in tasks:
        assert len(t['choices']) == 4
        assert len(set(t['choices'])) == 4
        assert len(t['recall_prompt']) > 140
        assert t['choices'][t['answer']].startswith(' ')
        assert t['difficulty'] == 'hard'
        assert not t['recall_prompt'].endswith(t['choices'][t['answer']])


def test_prior_stable_narrative_suite_unchanged():
    old, _ = _load(TASKS/'narrative_basic_0095.json', 'narrative')
    assert len(old) == 20
    harder, _ = _load(TASKS/'narrative_challenge_0096.json', 'narrative')
    assert not set(t['id'] for t in old) & set(t['id'] for t in harder)


def test_harder_references_fixed_and_diverse():
    data=json.loads((TASKS/'reference_texts_harder_0096.json').read_text())
    assert len(data['references']) >= 12
    assert len({r['id'] for r in data['references']}) == len(data['references'])
    assert all(len(r['text']) > 450 for r in data['references'])


def test_scheduler_accepts_0096_full_config():
    suitepath = CONFIGS/'foundation_capabilities_0096_challenges.json'
    suite=json.loads(suitepath.read_text())
    LiveBenchmarkScheduler(suite, object(), 9_800_000_000, suite_path=suitepath)
    old={e['id'] for e in json.loads((CONFIGS/'foundation_capabilities_0095_extended.json').read_text())['benchmarks']}
    new={e['id'] for e in suite['benchmarks']}
    assert old <= new
    events={e['id']:e for e in suite['benchmarks']}
    assert events['narrative_challenge_scored_0096']['parameters']['max_new_tokens']==0
    assert events['narrative_challenge_generation_0096']['parameters']['max_new_tokens']>0


def test_narrative_scoring_only_no_generator_and_grouped_metrics(monkeypatch):
    from benchmarks import capability_suite as mod
    tasks = [
       {'id':'c1', 'prompt':'A', 'recall_prompt':'After A', 'choices':[' alpha',' beta'],
        'answer':0,'challenge_type':'temporal'},
       {'id':'c2', 'prompt':'B', 'recall_prompt':'After B', 'choices':[' alpha',' beta'],
        'answer':1,'challenge_type':'relational'},
    ]
    monkeypatch.setattr(mod, '_load', lambda path, family: (tasks,'f'*64))
    monkeypatch.setattr(mod, '_plain_4096_model', lambda m: contextlib.nullcontext())
    monkeypatch.setattr(mod, '_preserve_global_rng', lambda: contextlib.nullcontext())
    monkeypatch.setattr(mod, '_generator', lambda *a, **kw: pytest.fail('must not create generator'))
    def score(model, tokenizer, prefix, choice, max_context):
        strength = {'After A':{' alpha':-0.1,' beta':-0.6},
                    'After B':{' alpha':-0.2,' beta':-0.5}}[prefix][choice]
        return {'sum_logprob': strength, 'mean_logprob':strength,'target_tokens':1}
    monkeypatch.setattr(mod, '_token_logprobs', score)
    out=run_narrative(None,None,tasks='irrelevant',max_new_tokens=0,seeds=[])
    assert out['count']==2 and out['fact_recall_accuracy']==0.5
    assert out['by_challenge_type']['temporal']['accuracy']==1.0
    assert out['by_challenge_type']['relational']['accuracy']==0.0
    assert out['by_challenge_type']['temporal']['mean_correct_choice_score_margin'] > 0
    assert out['by_challenge_type']['relational']['mean_correct_choice_score_margin'] < 0
    assert all(not r['generations'] for r in out['cases'])
    assert not out['generation_enabled']


def test_diagnostic_is_safe_and_reports_namespace_error(monkeypatch):
    from benchmarks import sandbox_diagnostics as mod
    monkeypatch.setattr(mod.shutil, 'which', lambda x: '/usr/bin/'+x)
    monkeypatch.setattr(mod, '_sysctl', lambda s:'1')
    monkeypatch.setattr(mod, '_sandbox', lambda *a,**kw:{'status':'sandbox_error',
                    'detail':'bwrap: loopback: Failed RTM_NEWADDR: Operation not permitted'})
    d=mod.report()
    assert not d['sandbox_ready']
    assert 'AppArmor' in d['suggestion']
    assert '--share-net' in d['suggestion']


def test_per_challenge_csv_columns_and_values(tmp_path):
    from benchmarks.live_report import narrative_detail_csv, summary_metrics, rebuild_summary_csv
    data={'count':2,'fact_recall_accuracy':0.5,
          'mean_correct_rank':2.0,'mean_correct_choice_score_margin':-0.25,
          'mean_correct_choice_relative_probability':0.4,
          'by_challenge_type': {'temporal':{'count':2,'accuracy':0.5,'mean_correct_rank':2.0,
                                'mean_correct_choice_score_margin':-0.25,
                                'mean_correct_choice_relative_probability':0.4}},
          'cases':[{'id':'one','challenge_type':'temporal','difficulty':'hard',
                    'recall_correct':True,'correct_rank':1,'correct_choice_score_margin':0.5,
                    'correct_choice_relative_probability':0.8},
                   {'id':'two','challenge_type':'temporal','difficulty':'hard',
                    'recall_correct':False,'correct_rank':3,'correct_choice_score_margin':-1,
                    'correct_choice_relative_probability':0.1}]}
    narrative_detail_csv(data,tmp_path)
    assert (tmp_path/'narrative_by_type.csv').is_file()
    assert (tmp_path/'narrative_per_case.csv').is_file()
    assert (tmp_path/'narrative_by_type.csv').read_text().count('\n')==2
    summary=summary_metrics('narrative',data)
    assert summary['mean_correct_rank']==2
    results=tmp_path/'step_00000001'/'example'
    results.mkdir(parents=True)
    (results/'result.json').write_text(json.dumps({'status':'complete','suite_hash':'a',
                  'optimizer_step':1,'benchmark_id':'narrative_test','summary_metrics':summary}))
    rebuild_summary_csv(tmp_path,'a')
    header=(tmp_path/'metrics.csv').read_text().splitlines()[0]
    assert 'mean_correct_choice_score_margin' in header
