"""0103G: standardized benchmark metadata, session reuse and failure rules."""
from __future__ import annotations

from dataclasses import dataclass
import json
from pathlib import Path

import pytest

from mini_llm.evaluation import EvaluationEngine
from mini_llm.model_sources import ModelSpec
from benchmarks.run_unified import (
    _run, checkpoint_identity, prompts_from_file, _summary,
)


class DummyLocal:
    context_length = 48
    stop_ids = {99}
    def encode(self, prompt):
        return [1, 2]
    def decode(self, ids):
        return ''.join(chr(96 + i) for i in ids)
    def create_state(self, total_tokens):
        return object()
    def prefill(self, ids, state):
        import numpy as np
        return np.array([[-100., 100., -100., -100.]], dtype='float32')
    def decode_one(self, token_id, state):
        import numpy as np
        return np.array([[-100., 100., -100., -100.]], dtype='float32')
    def execution_report(self, state):
        return None


class DummyRemote:
    is_remote = True
    def __init__(self, usage=True):
        self.requests = []
        self.usage = usage
    def complete(self, payload, *, chat, public_id):
        self.requests.append((payload, chat, public_id))
        return {'choices': [{'message': {'content': 'hello'}, 'text': 'hello', 'finish_reason': 'stop'}],
                'usage': {'prompt_tokens': 7, 'completion_tokens': 2} if self.usage else {}}


@pytest.fixture
def suite(tmp_path):
    path = tmp_path / 'prompts.json'
    path.write_text(json.dumps({'prompts': [
        {'id': 'one', 'domain': 'math', 'prompt': 'One'},
        {'id': 'two', 'domain': 'code', 'prompt': 'Two'},
    ]}))
    return path


def test_local_session_usage_and_no_unverified_fast_claim():
    sample = EvaluationEngine(DummyLocal(), model_id='mock').generate('hi', max_new_tokens=2, seed=42)
    assert sample.text == 'aa'
    assert sample.token_ids == (1, 1)
    assert sample.prompt_tokens == 2
    assert sample.completion_tokens == 2
    assert sample.usage_source == 'local_tokenizer'
    assert sample.prefill_s is not None and sample.decode_s is not None
    assert sample.execution is None
    with pytest.raises(RuntimeError, match='strict-fast'):
        EvaluationEngine(DummyLocal(), model_id='mock', strict_fast=True).generate('hi', max_new_tokens=1)


def test_remote_usage_not_fabricated_and_no_token_metrics():
    remote = DummyRemote(usage=False)
    sample = EvaluationEngine(remote, model_id='lfm').generate('hi', max_new_tokens=4, seed=3)
    assert sample.usage_source == 'unknown'
    assert sample.completion_tokens is None and sample.token_ids is None
    assert sample.prefill_s is None and sample.decode_s is None
    assert remote.requests[0][0]['seed'] == 3
    assert remote.requests[0][0]['prompt'] == 'hi'
    assert remote.requests[0][1] is False
    with pytest.raises(ValueError, match='top_k'):
        EvaluationEngine(remote, model_id='lfm').generate('hi', max_new_tokens=4, top_k=40)
    with pytest.raises(ValueError, match='remote'):
        EvaluationEngine(remote, model_id='lfm', strict_fast=True)


def test_provider_usage_is_explicitly_labeled():
    sample = EvaluationEngine(DummyRemote(), model_id='lfm').generate('hi', max_new_tokens=2)
    assert sample.usage_source == 'provider_reported'
    assert sample.prompt_tokens == 7
    assert sample.completion_tokens == 2


def test_prompt_validation_and_unknown_ids(suite):
    assert len(prompts_from_file(suite, {'one'})) == 1
    with pytest.raises(ValueError, match='Unknown'):
        prompts_from_file(suite, {'absent'})
    suite.write_text(json.dumps({'prompts': [{'id': 'dupe', 'prompt': 'a', 'domain': 'b'},
                                             {'id': 'dupe', 'prompt': 'c', 'domain': 'd'}]}))
    with pytest.raises(ValueError, match='Duplicate'):
        prompts_from_file(suite)


def test_incremental_output_and_resume(tmp_path, suite, monkeypatch):
    from mini_llm import serving_models
    model = DummyLocal()
    monkeypatch.setattr(serving_models, 'load_model', lambda spec: model)
    path = tmp_path / 'model.gguf'
    path.write_bytes(b'abc')
    spec = ModelSpec(id='local', path=path, format='gguf')
    out = tmp_path / 'results'
    args = dict(model_spec=spec, prompts_path=suite, output_dir=out,
                prompt_ids=None, seeds=(3, 5), tokens=2,
                temperature=0.0, top_k=None, top_p=1.0, dry_run=False)
    result = _run(**args, resume=False)
    assert result['complete'] and result['completed'] == 4
    rows = [json.loads(s) for s in (out / 'results.jsonl').read_text().splitlines()]
    assert len(rows) == 4 and rows[0]['quality']['token_repetition'] is not None
    assert (out / 'manifest.json').is_file()
    assert _run(**args, resume=True)['completed'] == 4
    # Partial resume can repair a completed result after interruption.
    (out / 'results.jsonl').write_text('\n'.join(json.dumps(x) for x in rows[:2]) + '\n')
    assert _run(**args, resume=True)['completed'] == 4
    # Detect changed model and changed prompt suite before any execution.
    path.write_bytes(b'abcd')
    with pytest.raises(ValueError, match='Resume settings'):
        _run(**args, resume=True)


def test_corrupt_and_duplicate_result_records_rejected(tmp_path, suite, monkeypatch):
    from mini_llm import serving_models
    monkeypatch.setattr(serving_models, 'load_model', lambda spec: DummyLocal())
    path = tmp_path / 'tiny.gguf'
    path.write_bytes(b'abc')
    out = tmp_path / 'out'
    args = dict(model_spec=ModelSpec(id='x', path=path, format='gguf'),
                prompts_path=suite, output_dir=out, prompt_ids={'one'},
                seeds=(0,), tokens=1, temperature=0, top_k=None, top_p=1.0, dry_run=False)
    _run(**args, resume=False)
    with (out / 'results.jsonl').open('a') as f:
        f.write((out / 'results.jsonl').read_text())
    with pytest.raises(ValueError, match='Duplicate'):
        _run(**args, resume=True)


def test_metadata_no_content_hash_claim(tmp_path):
    p = tmp_path / 'model.gguf'
    p.write_bytes(b'abc')
    meta = checkpoint_identity(ModelSpec(id='m', path=p, format='gguf'))
    assert meta['identity_strength'] == 'file_size_mtime_not_content_hash'
    remote = checkpoint_identity(ModelSpec(id='r', format='remote',
        api_url='http://127.0.0.1:18080/v1', remote_model='lfm'))
    assert 'upstream weights may change' in remote['identity_strength']


def test_no_results_are_a_pass():
    assert not _summary([], 3)['complete']


def test_real_native_checkpoint_through_unified_cli(tmp_path, suite):
    """Catch wiring regressions that dummy adapter tests cannot detect."""
    from tests.test_canonical_model_loading_0103c import _tiny_checkpoint
    import subprocess, sys, os
    checkpoint = tmp_path / 'checkpoint'
    _tiny_checkpoint(checkpoint)
    output = tmp_path / 'cli_results'
    cmd = [sys.executable, '-m', 'benchmarks.run_unified',
           '--checkpoint', str(checkpoint), '--prompts', str(suite),
           '--prompt-ids', 'one', '--seeds', '42', '--tokens', '1',
           '--output-dir', str(output), '--execution-policy', 'reference']
    env = dict(os.environ, MINI_LLM_BACKEND='numpy')
    result = subprocess.run(cmd, cwd=Path(__file__).resolve().parents[1],
                            env=env, capture_output=True, text=True, timeout=45)
    assert result.returncode == 0, result.stdout + '\n' + result.stderr
    row = json.loads((output / 'results.jsonl').read_text())
    assert row['sample']['usage_source'] == 'local_tokenizer'
    assert row['sample']['execution']['backend'] == 'numpy'
    assert row['sample']['completion_tokens'] == 1
    assert row['performance']['end_to_end_wall_s'] >= 0


def test_existing_live_suite_bridge_uses_original_scoring(monkeypatch, suite, tmp_path):
    from benchmarks.run_unified import run_legacy_suite
    from benchmarks import run_checkpoint_suite
    called = []
    def mock_run(**kwargs):
        called.append(kwargs)
        return [{'status': 'complete', 'type': 'wikitext'}]
    monkeypatch.setattr(run_checkpoint_suite, 'run_checkpoints', mock_run)
    checkpoint = tmp_path / 'checkpoint'
    checkpoint.mkdir()
    (checkpoint / 'config.json').write_text('{}')
    live = tmp_path / 'live.json'
    live.write_text(json.dumps({'benchmarks': [{'id': 'wikitext_1024'}]}))
    destination = tmp_path / 'out'
    obj = run_legacy_suite(spec=ModelSpec(id='native', path=checkpoint, format='native'),
                           suite=live, output_dir=destination,
                           benchmark_ids=('wikitext_1024',))
    assert obj['completed_events'] == 1 and obj['all_reported_events_complete']
    assert called[0]['output_root'] == destination
    assert (destination / 'unified_legacy_index.json').exists()
    with pytest.raises(ValueError, match='Unknown legacy'):
        run_legacy_suite(spec=ModelSpec(id='native', path=checkpoint, format='native'),
                         suite=live, output_dir=destination, benchmark_ids=('unknown',))
    with pytest.raises(ValueError, match='require a native'):
        run_legacy_suite(spec=ModelSpec(id='remote', format='remote',
                         api_url='http://localhost:9999/v1', remote_model='lfm'),
                         suite=live, output_dir=destination)


def test_remote_chat_mode_is_labeled_and_not_prompt_equivalent():
    engine = DummyRemote()
    EvaluationEngine(engine, model_id='lfm', remote_mode='chat').generate('prompt', max_new_tokens=1)
    assert engine.requests[0][1] is True
    assert engine.requests[0][0]['messages'] == [{'role': 'user', 'content': 'prompt'}]
    assert 'prompt' not in engine.requests[0][0]


def test_output_directory_lock_prevents_concurrent_writers(tmp_path):
    from benchmarks.run_unified import output_lock
    with output_lock(tmp_path / 'out'):
        with pytest.raises(RuntimeError, match='already in use'):
            with output_lock(tmp_path / 'out'):
                pass
    with output_lock(tmp_path / 'out'):
        pass
