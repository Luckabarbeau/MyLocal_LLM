"""0094: generation must match the established fast CUDA graph benchmark."""
import json
from pathlib import Path
from types import SimpleNamespace

import pytest

from benchmarks import live_suite


ROOT = Path(__file__).resolve().parents[1]


def test_full_generation_matches_production_cli():
    suite = json.loads((ROOT / 'benchmarks/configs/foundation_capabilities_0094_fast_generation.json').read_text())
    entries = [b for b in suite['benchmarks'] if b['id'] == 'generation_full']
    assert len(entries) == 1
    p = entries[0]['parameters']
    assert p['prompt_suite'] == 'benchmarks/generation_prompts_v1.json'
    assert 'prompt_ids' not in p
    assert p['seeds'] == [42, 123, 879]
    assert p['strategies'] == ['greedy', 'top-k']
    assert p['max_new_tokens'] == 128
    assert p['max_context'] == 512
    assert p['top_k'] == 40
    assert p['temperature'] == 0.8
    assert p['use_kv_cache'] is True
    assert p['require_native_cuda_graph_decode'] is True
    assert p['print_generations'] is True
    prompts = json.loads((ROOT / p['prompt_suite']).read_text())['prompts']
    assert len(prompts) * len(p['seeds']) * len(p['strategies']) == 144
    live_suite.LiveBenchmarkScheduler(suite, object(), 0,
        suite_path=ROOT / 'benchmarks/configs/foundation_capabilities_0094_fast_generation.json')


def test_live_generation_passes_context_and_logging(monkeypatch, tmp_path):
    path = tmp_path / 'prompts.json'
    path.write_text(json.dumps({'prompts': [{'id':'a', 'prompt': 'Once upon a time', 'domain':'story', 'length':'short', 'topic':'fiction', 'format':'prose'}]}))
    fake = SimpleNamespace()
    fake.calls = []
    fake._state = SimpleNamespace(_native_cuda_graph=object())
    def generate(**kwargs):
        fake.calls.append(kwargs)
        return 'story', {'generated_token_ids':[5,6], 'eos_token_id':7,
                         'total_time_s':0.5,'tokens_per_sec':4.0}
    fake.generate = generate
    contexts = []
    def mock_generator(model, tokenizer, *, max_context=None):
        contexts.append(max_context)
        return fake
    monkeypatch.setattr(live_suite, '_generator', mock_generator)
    result = live_suite.run_generation(object(), object(), prompt_suite=path,
        seeds=[42,123,879], strategies=['greedy','top-k'], max_context=512)
    assert contexts == [512]
    assert len(result['records']) == 6
    assert [c['seed'] for c in fake.calls] == [42,42,123,123,879,879]
    assert all(c['use_kv_cache'] and c['log_speed'] for c in fake.calls)
    assert result['records'][0]['timing']['tokens_per_sec'] == 4.0


def test_graph_requirement_refuses_slow_cpu_fallback(tmp_path):
    path = tmp_path / 'prompts.json'
    path.write_text(json.dumps({'prompts': [{'id':'a', 'prompt': 'hello', 'domain':'story', 'length':'short', 'topic':'fiction', 'format':'prose'}]}))
    with pytest.raises(RuntimeError, match='MINI_LLM_BACKEND=cupy'):
        live_suite.run_generation(object(), object(), prompt_suite=path,
            require_native_cuda_graph_decode=True)


def test_scoped_graph_flags_restored(monkeypatch):
    import os
    monkeypatch.delenv('MINI_LLM_NATIVE_CUDA_GRAPH_DECODE', raising=False)
    monkeypatch.setenv('MINI_LLM_NATIVE_CUDA_GRAPH_DECODE_STRICT', '0')
    from benchmarks.live_suite import _strict_graph_decode_for_generation
    try:
        with _strict_graph_decode_for_generation(True):
            assert os.environ['MINI_LLM_NATIVE_CUDA_GRAPH_DECODE'] == '1'
            assert os.environ['MINI_LLM_NATIVE_CUDA_GRAPH_DECODE_STRICT'] == '1'
    except RuntimeError as exc:
        # CPU backend intentionally cannot satisfy the graph requirement.
        assert 'MINI_LLM_BACKEND=cupy' in str(exc)
    assert 'MINI_LLM_NATIVE_CUDA_GRAPH_DECODE' not in os.environ
    assert os.environ['MINI_LLM_NATIVE_CUDA_GRAPH_DECODE_STRICT'] == '0'
