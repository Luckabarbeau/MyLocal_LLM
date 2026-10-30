"""0095: versioned local suites, reference metrics and zero-checkpoint fast path."""
import json
import re
from pathlib import Path

import numpy as np
import pytest

from benchmarks.capability_suite import _load, _select_tasks, _check_python_syntax
from benchmarks.live_suite import LiveBenchmarkScheduler, run_reference
from benchmarks.reference_diagnostics import token_kind, summarize_rows, summarize_cases

ROOT = Path(__file__).resolve().parents[1]
TASKS = ROOT/'benchmarks/tasks'
CONFIGS = ROOT/'benchmarks/configs'


def test_big_suites_are_versioned_and_have_many_distinct_cases():
    expected = {'python':30, 'cpp':15, 'math':67, 'logic':52,
                'terminal':35, 'narrative':20}
    for name, size in expected.items():
        stem = {'python':'python_basic','cpp':'cpp_basic','math':'math_basic',
                'logic':'logic_basic','terminal':'terminal_basic',
                'narrative':'narrative_basic'}[name]
        cases, hash_value = _load(TASKS/f'{stem}_0095.json',name)
        assert len(cases) == size and len(hash_value)==64
        assert len({c['id'] for c in cases}) == size
    ref=json.loads((TASKS/'reference_texts_v2.json').read_text())
    assert len(ref['references'])==53
    assert len({x['id'] for x in ref['references']})==53
    assert len({x['category'] for x in ref['references']})>=7
    assert all(len(x['text']) > 80 for x in ref['references'])


def test_choice_answers_nonambiguous_and_balanced():
    for family in ['math', 'logic', 'terminal', 'narrative']:
        prefix={'math':'math_basic','logic':'logic_basic','terminal':'terminal_basic','narrative':'narrative_basic'}[family]
        cases,_ = _load(TASKS/f'{prefix}_0095.json',family)
        answers=[]
        for t in cases:
            assert len(t['choices'])==len(set(t['choices']))
            assert 0<=t['answer']<len(t['choices'])
            answers.append(t['answer'])
        # Require a mix of answer positions so models cannot exploit a fixed answer index.
        assert len(set(answers)) >= 3


def test_terminal_patterns_fullmatch_correct_answer():
    cases,_ = _load(TASKS/'terminal_basic_0095.json','terminal')
    for t in cases:
        correct = t['choices'][t['answer']]
        assert any(re.fullmatch(p,correct) is not None for p in t['accepted_patterns']), (t['id'],correct)


def test_python_original_completion_recognizes_redefinition():
    prompt='def f(x):\n    """Docstring."""\n'
    status=_check_python_syntax(prompt + 'def f(x):\n    return x+1\n','f')
    assert status['syntax_valid'] and status['nontrivial_body']
    assert status['duplicate_function_definitions']
    assert not status['original_function_completed']
    proper=_check_python_syntax(prompt + '    return x+1\n', 'f')
    assert proper['original_function_completed']
    assert not proper['duplicate_function_definitions']
    assert not _check_python_syntax(prompt + '    pass\n','f')['original_function_completed']


def test_task_ids_strict_selection():
    cases,_ = _load(TASKS/'math_basic_0095.json','math')
    assert len(_select_tasks(cases,['add_7_5']))==1
    with pytest.raises(ValueError, match='Unknown'):
        _select_tasks(cases,['wrong'])


def test_token_categories_and_content_summary():
    assert token_kind(' the')=='stopword'
    assert token_kind('+',category='python')=='operator'
    assert token_kind(',')=='punctuation'
    assert token_kind(' 17')=='number'
    rows=[]
    for i,(label,rank,prob) in enumerate([(' the',1,.8),(' prime',2,.1),('.',1,.7),(' 7',4,.05)],1):
        rows.append({'position':i,'actual_text':label,'actual_rank':rank,
                     'actual_probability':prob,'nll':float(-np.log(prob)),
                     'entropy_nats':.8,'top':[{'text':'alternative'}]})
    result=summarize_rows(rows,category='mathematics')
    assert result['all_tokens']['count']==4
    assert result['content_like_tokens']['count']==2
    assert result['all_tokens']['top1_accuracy']==.5
    assert result['content_like_tokens']['top1_accuracy']==0.0
    assert len(result['position_quartiles'])==4
    combined=summarize_cases([{'category':'mathematics','token_predictions':rows}])
    assert combined['total_scored_tokens']==4
    assert combined['micro']['content_like_tokens']['count']==2


def test_configs_have_true_full_generation_and_separate_reference_frequency():
    full=json.loads((CONFIGS/'foundation_capabilities_0095_extended.json').read_text())
    entries={e['id']:e for e in full['benchmarks']}
    p=entries['generation_full']['parameters']
    assert p['seeds']==[42,123,879]
    assert p['strategies']==['greedy','top-k']
    assert p['max_new_tokens']==128 and p['max_context']==512
    assert p['require_native_cuda_graph_decode'] is True
    assert entries['reference_token_only_0095']['parameters']['generation_tokens']==0
    assert entries['reference_full_0095']['parameters']['generation_tokens']>0
    assert entries['reference_token_only_0095']['every_supervised_tokens']==250_000_000
    assert entries['reference_full_0095']['every_supervised_tokens']==1_000_000_000
    assert len(full['benchmarks'])==12
    # Test registered arguments match their declared parameters, no runtime build needed.
    for name in ['foundation_capabilities_0095_extended.json', 'foundation_capabilities_0095_smoke.json']:
        path=CONFIGS/name
        LiveBenchmarkScheduler(json.loads(path.read_text()),object(),0,suite_path=path)


def test_old_task_suites_unchanged():
    for stem in ['math_basic','logic_basic','terminal_basic','narrative_basic',
                 'python_basic','cpp_basic']:
        assert (TASKS/f'{stem}_0093.json').is_file()
    assert (TASKS/'reference_texts_v1.json').is_file()


def test_reference_filtered_token_only_with_fake_model(tmp_path, monkeypatch):
    # Lightweight no-GPU test; model's native forward is exercised by existing 0092 tests.
    from benchmarks import live_suite
    from benchmarks.reference_continuation import analyze_reference
    class Tok:
        def encode(self,text): return [ord(x) for x in text]
        def decode(self,ids):return ''.join(map(chr,ids))
    class Model:
        def __init__(self):
            from types import SimpleNamespace
            self.config=SimpleNamespace(context_length=4096,memory_context=SimpleNamespace(enabled=False))
            self.historical_memory=None
        @property
        def hierarchical_memory_enabled(self): return False
        def forward(self, x, return_cache=False):
            a=np.asarray(x);out=np.zeros((*a.shape,128))
            for j,v in enumerate(a[0]):out[0,j,(int(v)+1)%128]=3
            return out
    filename=tmp_path/'references.json'
    filename.write_text(json.dumps({'references':[{'id':'a','category':'general','text':'ABCDEF'},
                     {'id':'b','category':'story','text':'XYZXYZ'}]}))
    result=run_reference(Model(),Tok(),references=filename,reference_ids=['a'],
                         generation_tokens=0,cutoffs=[],seeds=[],top_k_predictions=3)
    assert len(result['cases'])==1
    assert result['cases'][0]['cutoff_generations']==[]
    assert result['diagnostics']['total_references']==1
    with pytest.raises(ValueError,match='Unknown reference'):
        run_reference(Model(),Tok(),references=filename,reference_ids=['missing'],
                      generation_tokens=0)


def test_reference_generation_enables_native_graph_scope(tmp_path, monkeypatch):
    import os
    from types import SimpleNamespace
    from benchmarks import live_suite

    class Tok:
        def encode(self,text): return [ord(x) for x in text]
        def decode(self,ids): return ''.join(map(chr,ids))

    class Model:
        config=SimpleNamespace(context_length=4096,
            memory_context=SimpleNamespace(enabled=False))
        historical_memory=None
        hierarchical_memory_enabled=False
        def forward(self,x,return_cache=False):
            inputs=np.asarray(x)
            output=np.zeros((*inputs.shape,128),dtype=np.float32)
            output[...,65]=3.0
            return output

    observed=[]
    class Generator:
        def generate(self,**kwargs):
            observed.append((os.environ.get('MINI_LLM_NATIVE_CUDA_GRAPH_DECODE'),
                             os.environ.get('MINI_LLM_NATIVE_CUDA_GRAPH_DECODE_STRICT')))
            return 'A',{'generated_token_ids':[65],'eos_token_id':2}

    monkeypatch.delenv('MINI_LLM_NATIVE_CUDA_GRAPH_DECODE',raising=False)
    monkeypatch.delenv('MINI_LLM_NATIVE_CUDA_GRAPH_DECODE_STRICT',raising=False)
    monkeypatch.setattr(live_suite,'_generator',lambda model,tokenizer,max_context=None:Generator())
    filename=tmp_path/'references.json'
    filename.write_text(json.dumps({'references':[{'id':'a','text':'ABCDEF'}]}))
    run_reference(Model(),Tok(),references=filename,cutoffs=[0.5],generation_tokens=1,
                  seeds=[42],prefer_native_cuda_graph_decode=True)
    assert observed==[('1','0')]
    assert 'MINI_LLM_NATIVE_CUDA_GRAPH_DECODE' not in os.environ
    assert 'MINI_LLM_NATIVE_CUDA_GRAPH_DECODE_STRICT' not in os.environ


def test_0095_reference_result_artifacts_and_longitudinal_metrics(tmp_path):
    from types import SimpleNamespace
    class Tok:
        def encode(self,text):return [ord(c) for c in text]
        def decode(self,ids):return ''.join(chr(c) for c in ids)
    class Model:
        def __init__(self):
            self.config=SimpleNamespace(context_length=128,
                    memory_context=SimpleNamespace(enabled=False))
            self.historical_memory=None
        @property
        def hierarchical_memory_enabled(self):return False
        def forward(self,x,return_cache=False):
            inputs=np.asarray(x)
            out=np.zeros((*inputs.shape,128),dtype=np.float32)
            for j,v in enumerate(inputs[0]):out[0,j,(int(v)+1)%128]=2.0
            return out
    refs=tmp_path/'tiny_references.json'
    refs.write_text(json.dumps({'references':[{'id':'one','category':'python','text':'def f():\n    return 2\n'},
                                                {'id':'two','category':'narrative','text':'Mira held a small map.'}]}))
    suitefile=tmp_path/'suite.json'
    suitefile.write_text(json.dumps({'version':1,'output_root':str(tmp_path/'results'),
                     'benchmarks':[{'id':'reference_quick','type':'reference_continuation',
                       'every_optimizer_steps':20,'parameters':{'references':str(refs),
                                                    'generation_tokens':0,'cutoffs':[],
                                                    'reference_ids':['one','two']}}]}))
    scheduler=LiveBenchmarkScheduler(json.loads(suitefile.read_text()),Tok(),0,suite_path=suitefile)
    events=scheduler.run_selected(Model(),step=10,supervised_tokens_total=300,
                                  provenance='measured')
    assert events[0]['status']=='complete'
    assert len(list((tmp_path/'results').rglob('reference_per_case.csv')))==1
    assert len(list((tmp_path/'results').rglob('report.html')))==1
    columns=(tmp_path/'results'/'metrics.csv').read_text().splitlines()[0]
    assert 'reference_micro_nll' in columns
    assert 'content_top5_accuracy' in columns
    assert events[0]['summary_metrics']['reference_count']==2
