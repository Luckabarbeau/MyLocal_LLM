"""0093 local-benchmark, checkpoint, and sandbox-isolation contracts."""
import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

from benchmarks import capability_suite as capability
from benchmarks import live_suite
from benchmarks import run_checkpoint_suite as checkpoint_runner


class ByteTokenizer:
    def encode(self, text):
        return list(text.encode('utf-8'))
    def decode(self, ids):
        return bytes(ids).decode('utf-8', errors='replace')


class ToyModel:
    def __init__(self):
        self.config = SimpleNamespace(memory_context=SimpleNamespace(enabled=False),
                                      context_length=1024)
        self.historical_memory = None
        self.hierarchical_memory_enabled = False
    def forward(self, tokens, return_cache=False):
        assert return_cache is False
        # Predict next character 2 with high probability, others uniformly.
        a = np.asarray(tokens)
        logits = np.zeros((*a.shape, 256), dtype=np.float32)
        logits[:, :, ord('2')] = 2.
        return logits


class FakeGenerator:
    def __init__(self, output='    return n == 2\n'):
        self.output = output
    def generate(self, **kwargs):
        return self.output, {'generated_token_ids': [1, 2], 'eos_token_id': 2}


def test_all_seven_categories_defined_and_scheduled(tmp_path):
    root = Path(__file__).resolve().parents[1]
    suite_path = root/'benchmarks/configs/foundation_capabilities_0093.json'
    config = json.loads(suite_path.read_text())
    ids = {i['id'] for i in config['benchmarks']}
    assert {'python_syntax', 'python_correctness', 'cpp_code', 'math_basic',
            'logic_basic', 'terminal_basic', 'narrative_basic'} <= ids
    for bench in config['benchmarks']:
        if 'tasks' in bench['parameters']:
            tasks = root/bench['parameters']['tasks']
            assert tasks.is_file()
            cases, _ = capability._load(tasks, json.loads(tasks.read_text())['family'])
            assert len(cases) >= 3
    live_suite.LiveBenchmarkScheduler(config, ByteTokenizer(), 0, suite_path=suite_path)


def test_choice_likelihood_math_logic_no_generator():
    root = Path(__file__).resolve().parents[1]
    for family in ('math', 'logic'):
        result = capability.run_choices(ToyModel(), ByteTokenizer(),
            tasks=str(root/f'benchmarks/tasks/{family}_basic_0093.json'), family=family)
        assert len(result['cases']) == result['count']
        assert 0 <= result['accuracy'] <= 1
        assert all(len(case['scores']) >= 2 for case in result['cases'])


def test_code_syntax_and_no_unsandboxed_execution(tmp_path, monkeypatch):
    mock = FakeGenerator()
    monkeypatch.setattr(capability, '_generator', lambda *args, **kwargs: mock)
    monkeypatch.setattr(capability, '_bubblewrap_available', lambda: False)
    tasks = tmp_path/'python.json'
    tasks.write_text(json.dumps({'version':1,'family':'python', 'tasks':[
        {'id':'prime','function':'is_prime','prompt':'def is_prime(n):\n',
         'tests':[{'args':[2], 'expected':True}]}]}))
    run = capability.run_code(ToyModel(), ByteTokenizer(), tasks=str(tasks),
                              language='python', run_tests=True, max_new_tokens=10)
    assert run['syntax_or_compile_rate'] == 1
    assert run['unit_test_pass_rate'] is None
    assert run['unit_tests_evaluated'] == 0
    assert run['cases'][0]['checks']['unit_tests']['status'] == 'unavailable_no_bwrap'
    assert run['cases'][0]['checks']['nontrivial_body'] is True
    # No sandbox: compiled code never runs or compiles in host process.
    tasks.write_text(json.dumps({'version':1,'family':'cpp', 'tasks':[
        {'id':'cpp','prompt':'int f() {\n','test_harness':'int main(){return f();}'}]}))
    run = capability.run_code(ToyModel(), ByteTokenizer(), tasks=str(tasks),
                              language='cpp', run_tests=True, max_new_tokens=10)
    assert run['syntax_or_compile_rate'] is None
    assert run['unit_tests_evaluated'] == 0
    assert run['cases'][0]['checks']['status'] == 'unavailable_no_bwrap'


def test_terminal_and_narrative_factual_probes(tmp_path, monkeypatch):
    monkeypatch.setattr(capability, '_generator', lambda *args, **kwargs: FakeGenerator('pwd\n'))
    terminal = tmp_path/'terminal.json'
    terminal.write_text(json.dumps({'version':1, 'family':'terminal','tasks':[
        {'id':'pwd','prompt':'$ ','choices':['pwd','rm -rf .'],
         'answer':0,'accepted_patterns':['pwd']}]}))
    result = capability.run_terminal(ToyModel(), ByteTokenizer(), tasks=str(terminal))
    assert result['generated_static_match_rate'] == 1
    story = tmp_path/'narrative.json'
    story.write_text(json.dumps({'version':1, 'family':'narrative','tasks':[
        {'id':'story','prompt':'Mira took a key.','recall_prompt':'Mira took a',
         'choices':[' key',' dog'], 'answer':0,'anchors':['Mira']}]}))
    result = capability.run_narrative(ToyModel(), ByteTokenizer(), tasks=str(story))
    assert 'fact_recall_accuracy' in result
    assert 'lexical proxy' in result['warning']


def test_offline_run_checkpoint_select_ids_and_preserve_results(tmp_path, monkeypatch):
    ckpt = tmp_path/'ckpt'
    (ckpt/'model_params').mkdir(parents=True)
    (ckpt/'config.json').write_text('{}')
    (ckpt/'training_state.json').write_text(json.dumps({'step':100,
         'supervised_tokens_total':123456,'supervised_tokens_provenance':'measured'}))
    np.save(ckpt/'model_params/weight.npy', np.ones(4))
    suite = tmp_path/'suite.json'
    suite.write_text(json.dumps({'version':1,'output_root':str(tmp_path/'results'),
        'benchmarks':[{'id':'math','type':'math', 'every_optimizer_steps':10,
                       'parameters':{'tasks':'unused'}}]}))
    tok = ByteTokenizer()
    fake = SimpleNamespace(model=ToyModel(),tokenizer=tok)
    from benchmarks import evaluate_generation
    monkeypatch.setattr(evaluate_generation, 'load_generator',
                        lambda *args, **kw: (fake, {'step':100}))
    monkeypatch.setattr(capability, 'run_choices', lambda *args, **kwargs: {
        'count': 1, 'accuracy': .75, 'cases': [], 'family':'math'})
    events = checkpoint_runner.run_checkpoints(checkpoints=[ckpt], suite_path=suite,
                                                selected=['math'])
    assert len(events)==1 and events[0]['status']=='complete'
    assert events[0]['supervised_tokens_total']==123456
    assert events[0]['checkpoint_id'] is not None
    path = next((tmp_path/'results').rglob('result.json'))
    assert path.is_file()
    assert 'checkpoint_' in str(path)
    assert 'checkpoint_id' in (tmp_path/'results'/'metrics.csv').read_text()
    # Same checkpoint/suite idempotently reuses result.
    checkpoint_runner.run_checkpoints(checkpoints=[ckpt], suite_path=suite,
                                       selected=['math'])
    assert len(list((tmp_path/'results').rglob('result.json'))) == 1
    # Changed weights must get a distinct checkpoint result directory.
    np.save(ckpt/'model_params/weight.npy', np.ones(4)*2)
    checkpoint_runner.run_checkpoints(checkpoints=[ckpt], suite_path=suite, selected=['math'])
    assert len(list((tmp_path/'results').rglob('result.json'))) == 2


def test_legacy_checkpoint_counter_unknown_not_invented(tmp_path):
    ck = tmp_path/'legacy';ck.mkdir()
    (ck/'config.json').write_text('{}')
    (ck/'training_state.json').write_text('{"step":1234}')
    total, provenance, _ = checkpoint_runner._token_total(ck)
    assert total == 0 and provenance == 'unknown_legacy_checkpoint'
    with pytest.raises(ValueError):
        checkpoint_runner.run_checkpoints(checkpoints=[ck,ck], suite_path='missing',
                                          supervised_tokens=200)


def test_token_logprobs_preserve_reference_position_and_probability():
    value = capability._token_logprobs(ToyModel(), ByteTokenizer(), 'abc','2',max_context=100)
    assert value['target_tokens'] == 1
    assert value['tokenization'] == 'joined'
    other = capability._token_logprobs(ToyModel(), ByteTokenizer(), 'abc','z',max_context=100)
    assert value['sum_logprob'] > other['sum_logprob']
