"""0097 regression / oracle / fake-inference tests; no model weights required."""
from __future__ import annotations
import contextlib
import json
import re
from collections import Counter
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

from benchmarks.capability_suite import _load
from benchmarks.long_retrieval_0097 import _build_prefix, _one_token_choices, run_long_retrieval
from benchmarks.progression_0097 import summarize_progression
from benchmarks.live_suite import LiveBenchmarkScheduler

BASE=Path(__file__).resolve().parents[1]
TASK=BASE/'benchmarks/tasks'
CONFIG=BASE/'benchmarks/configs'


def test_banks_have_unique_ids_and_difficulty_ramps():
    files={'math_ladders_0097.json':('math',180),
           'logic_ladders_0097.json':('logic',150),
           'narrative_ladders_0097.json':('narrative',216),
           'terminal_advanced_0097.json':('terminal',30),
           'python_advanced_0097.json':('python',20),
           'cpp_advanced_0097.json':('cpp',12),
           'long_retrieval_0097.json':('long_retrieval',8)}
    for file,(family,expected) in files.items():
        if family == 'long_retrieval':
            raw=(TASK/file).read_bytes()
            rows=json.loads(raw)['tasks']; digest=__import__('hashlib').sha256(raw).hexdigest()
        else:
            rows,digest=_load(TASK/file,family)
        assert len(rows)==expected and len(digest)==64
        assert len({r['id'] for r in rows})==len(rows)
        if family in {'math','logic','narrative'}:
            groups={}
            for r in rows:
                assert 1<=int(r['stage'])<=6
                assert r['difficulty'] in {'basic','intermediate','hard','expert','extreme'}
                assert len(r['choices'])==4 and len(set(r['choices']))==4
                assert r['choices'][r['answer']].startswith(' ')
                groups.setdefault(r['scenario_id'],set()).add(r['stage'])
            assert all(group==set(range(1,len(group)+1)) for group in groups.values())
            assert sum(len(group) for group in groups.values())==expected
            # Choice positions not predictably fixed by stage.
            assert all(Counter(r['answer'] for r in rows)[i]>20 for i in range(4))
    refs=json.loads((TASK/'reference_texts_advanced_0097.json').read_text())['references']
    assert len(refs)==64 and len({r['id'] for r in refs})==64
    assert len({r['category'] for r in refs})==8


def test_simple_and_hard_suites_remain_unchanged():
    assert len(_load(TASK/'narrative_basic_0095.json','narrative')[0])==20
    assert len(_load(TASK/'narrative_challenge_0096.json','narrative')[0])==126
    assert len(_load(TASK/'math_basic_0095.json','math')[0])==67
    assert len(_load(TASK/'logic_basic_0095.json','logic')[0])==52


def test_progression_can_recover_after_a_failed_stage_without_hiding_first_failure():
    cases=[{'id':f'case{i}', 'scenario_id':'one','stage':i,'correct':result,
            'category':'logic','difficulty':level}
           for i,(result,level) in enumerate([(True,'basic'),(False,'intermediate'),
                                               (True,'hard'),(False,'expert')],1)]
    diag=summarize_progression(cases)
    ladder=diag['ladders'][0]
    assert ladder['first_failed_stage']==2 and ladder['passed_prefix_length']==1
    assert ladder['stages'][2]['correct'] is True
    assert diag['by_difficulty']['basic']['accuracy']==1
    assert diag['all_stages_passed_rate']==0
    with pytest.raises(ValueError):
        summarize_progression(cases+[dict(cases[0],id='other')])


def test_config_accepts_new_type_and_preserves_previous_tests():
    old=json.loads((CONFIG/'foundation_capabilities_0096_challenges.json').read_text())
    new_path=CONFIG/'foundation_capabilities_0097_progressive.json'
    new=json.loads(new_path.read_text())
    ids={r['id'] for r in new['benchmarks']}
    assert {r['id'] for r in old['benchmarks']}<=ids
    generation=next(r for r in new['benchmarks'] if r['id']=='generation_full')
    assert generation['parameters']['use_kv_cache'] is True
    assert generation['parameters']['require_native_cuda_graph_decode'] is True
    assert generation['parameters']['prefer_native_cuda_graph_decode'] is True
    LiveBenchmarkScheduler(new, object(), 7_000_000_000, suite_path=new_path)
    assert any(r['type']=='long_retrieval' for r in new['benchmarks'])
    assert any(not r['enabled'] and r['id']=='long_retrieval_counterfactual_0097'
               for r in new['benchmarks'])
    smoke_path=CONFIG/'foundation_capabilities_0097_smoke.json'
    smoke=json.loads(smoke_path.read_text())
    assert len(smoke['benchmarks'])>=7
    LiveBenchmarkScheduler(smoke, object(), 7_000_000_000, suite_path=smoke_path)


class FakeTokenizer:
    def __init__(self):
        self.tokens={};self.texts={}
    def encode(self,text):
        parts=re.findall(r' ?[A-Za-z0-9]+|[^A-Za-z0-9 ]| +',text)
        result=[]
        for part in parts:
            if part not in self.tokens:
                index=len(self.tokens)+1
                self.tokens[part]=index;self.texts[index]=part
            result.append(self.tokens[part])
        return result
    def decode(self,ids):
        return ''.join(self.texts[int(i)] for i in ids)


class FakeImpl:
    def __init__(self,tokenizer):
        self.tok=tokenizer
        self.prefills=[]
    def create_generation_state(self,batch_size,max_length):
        return SimpleNamespace(max_length=max_length,long_memory_enabled=max_length>4096)
    def prefill(self,ids,state):
        text=self.tok.decode(ids[0])
        assert len(ids[0]) <= state.max_length
        self.prefills.append((len(ids[0]),state.long_memory_enabled))
        actual=re.search(r'Archived entry: project .*? has verification word (\w+)',text)
        assert actual, text[:130]
        logits=np.full((1,len(self.tok.tokens)+32),-9.0)
        target=self.tok.tokens[' '+actual.group(1)]
        logits[0,target]=9.0
        return logits


class FakeGen:
    def __init__(self,tok):
        self._inference_model=FakeImpl(tok)
    def _ensure_inference_model(self,reset=False):
        pass


def test_token_distance_retrieval_and_counterfactual(monkeypatch):
    from benchmarks import long_retrieval_0097 as mod
    tok=FakeTokenizer()
    fake=FakeGen(tok)
    monkeypatch.setattr(mod,'_generator',lambda model,tk,max_context=None:fake)
    cfg=SimpleNamespace(enabled=True,integration_mode='historical_read',memory_length=65536)
    model=SimpleNamespace(config=SimpleNamespace(context_length=4096,memory_context=cfg))
    out=run_long_retrieval(model,tok,tasks=TASK/'long_retrieval_0097.json',
        task_ids=['archive_00'],distances=[1000,4000,16000],include_counterfactual=True)
    assert out['count']==3
    assert [r['measured_distance_tokens'] for r in out['cases']]==[1000,4000,16000]
    assert all(r['correct'] for r in out['cases'])
    assert all(r['counterfactual']['follows_replacement'] for r in out['cases'])
    assert all(r['counterfactual']['original_answer_logprob_change'] < -10
               for r in out['cases'])
    assert out['by_distance']['16000']['accuracy']==1
    assert max(length for length,_ in fake._inference_model.prefills)>16000


def test_cannot_silently_truncate_long_prompt():
    tok=FakeTokenizer()
    cfg=SimpleNamespace(enabled=False,integration_mode='plain',memory_length=65536)
    model=SimpleNamespace(config=SimpleNamespace(context_length=4096,memory_context=cfg))
    with pytest.raises(NotImplementedError,match='never truncate'):
        run_long_retrieval(model,tok,tasks=TASK/'long_retrieval_0097.json',
                           task_ids=['archive_00'],distances=[16000])


def test_progression_metadata_roundtrip_is_stable():
    from benchmarks.capability_suite import run_choices
    from benchmarks import capability_suite as mod
    monkeypatch=pytest.MonkeyPatch()
    try:
        monkeypatch.setattr(mod,'_plain_4096_model',lambda model:contextlib.nullcontext())
        monkeypatch.setattr(mod,'_token_logprobs',lambda model,tok,prefix,choice,max_context:
            {'mean_logprob': -0.1 if choice.endswith(' 42') else -1.0, 'sum_logprob':-0.1 if choice.endswith(' 42') else -1.0})
        cases=[{'id':'stage1','prompt':'x','choices':[' 42',' 43'],'answer':0,
                'scenario_id':'s','stage':1,'category':'arithmetic','difficulty':'basic'},
               {'id':'stage2','prompt':'y','choices':[' 43',' 42'],'answer':0,
                'scenario_id':'s','stage':2,'category':'algebra','difficulty':'hard'}]
        monkeypatch.setattr(mod,'_load',lambda p,f:(cases,'f'*64))
        result=run_choices(None,None,tasks='not_used',family='math')
        assert result['progression']['ladders'][0]['first_failed_stage']==2
        assert result['progression']['ladders'][0]['passed_prefix_length']==1
    finally:
        monkeypatch.undo()


def test_real_small_historical_decoder_prefill_0097():
    # Verifies that the new scorer reaches the actual InferenceModel historical
    # prefill path; it does not assert that an untrained random model knows facts.
    from mini_llm.config import ModelConfig, MemoryContextConfig
    from mini_llm.model.decoder_lm import DecoderLanguageModel
    cfg=ModelConfig(tokenizer_vocab_size=512,context_length=16,n_layers=16,
      d_model=128,n_q_heads=16,n_kv_heads=4,d_head=8,n_experts=1,top_k=1,
      d_ff=256,ffn_mode='dense',memory_context=MemoryContextConfig(
      enabled=True,memory_length=528,recent_length=16,target_length=16,
      block_size=32,top_k_blocks=1,router_query_length=16,router_dim=16,
      read_heads=2,read_kv_heads=1,integration_mode='historical_read'))
    model=DecoderLanguageModel(cfg,dtype='float32')
    tok=FakeTokenizer()
    outcome=run_long_retrieval(model,tok,tasks=TASK/'long_retrieval_0097.json',
             task_ids=['archive_00'],distances=[80,160])
    assert len(outcome['cases'])==2
    assert [x['measured_distance_tokens'] for x in outcome['cases']]==[80,160]
    assert all(x['fact_in_historical_store'] for x in outcome['cases'])
    assert all(x['history_path_active'] for x in outcome['cases'])


def test_progression_csv_exports(tmp_path):
    from benchmarks.live_report import progression_detail_csv
    sample={'progression':summarize_progression(
        [{'id':'s1','scenario_id':'story','stage':1,'correct':True,
          'category':'recall','difficulty':'basic'},
         {'id':'s2','scenario_id':'story','stage':2,'correct':False,
          'category':'two_hop','difficulty':'expert'}])}
    progression_detail_csv(sample,tmp_path)
    assert 'first_failed_stage' in (tmp_path/'progression_by_scenario.csv').read_text()
    assert 'story,2,1,False,2' in (tmp_path/'progression_by_scenario.csv').read_text()
    assert 'expert,1,0.0' in (tmp_path/'progression_by_difficulty.csv').read_text()

def test_generated_banks_are_reproducible_without_regenerating_old_banks():
    from benchmarks.generate_challenge_banks_0097 import main
    from hashlib import sha256
    files=list(TASK.glob('*0097.json'))
    old_files=list(TASK.glob('*0095.json'))+list(TASK.glob('*0096.json'))
    before={f.name:sha256(f.read_bytes()).hexdigest() for f in files+old_files}
    main()
    after={f.name:sha256(f.read_bytes()).hexdigest() for f in files+old_files}
    assert before==after
