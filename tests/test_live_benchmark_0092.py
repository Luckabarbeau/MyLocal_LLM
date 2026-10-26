"""0092 CPU-side contracts; GPU integration still needs RTX 5080 smoke test."""
import csv
import dataclasses
import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

from benchmarks import live_suite
from benchmarks.reference_continuation import analyze_reference
from mini_llm.train_extended import ExtendedTrainer


@dataclasses.dataclass
class Memory:
    enabled: bool = True
    integration_mode: str = 'historical_read'


@dataclasses.dataclass
class Config:
    memory_context: Memory = dataclasses.field(default_factory=Memory)
    context_length: int = 20


class FakeModel:
    def __init__(self):
        self.config = Config()
        self.historical_memory = object()
        self.historical = self.historical_memory
        self.calls = []

    @property
    def hierarchical_memory_enabled(self):
        return self.historical_memory is not None

    def forward(self, inputs, return_cache=True):
        assert not return_cache
        self.calls.append(bool(self.hierarchical_memory_enabled))
        inputs = np.asarray(inputs)
        logits = np.zeros((*inputs.shape, 128), dtype=np.float32)
        for i in range(inputs.shape[1]):
            logits[0, i, (int(inputs[0, i]) + 1) % 128] = 4.0
        return logits


class SimpleTokenizer:
    def encode(self, text):
        return [ord(c) for c in text]

    def decode(self, ids):
        return ''.join(map(chr, ids))


def test_plain_model_restoration_on_error():
    model = FakeModel()
    with pytest.raises(RuntimeError):
        with live_suite._plain_4096_model(model):
            assert not model.hierarchical_memory_enabled
            assert not model.config.memory_context.enabled
            raise RuntimeError('intentional')
    assert model.historical_memory is model.historical
    assert model.config.memory_context.enabled


def test_reference_ranks_and_cutoffs():
    model = FakeModel()
    tok = SimpleTokenizer()
    with live_suite._plain_4096_model(model):
        result = analyze_reference(model, tok, 'abcd', top_k=3, max_context=20,
                                   cutoffs=[0.5], generator=None)
    assert result['reference_tokens'] == 4
    assert result['top1_accuracy'] == 1.0
    assert [r['actual_rank'] for r in result['token_predictions']] == [1, 1, 1]
    assert model.calls == [False]


def test_supervised_logging_legacy_and_new(tmp_path):
    for legacy in (False, True):
        trainer = object.__new__(ExtendedTrainer)
        trainer.log_file = tmp_path / ('legacy.csv' if legacy else 'new.csv')
        if legacy:
            with trainer.log_file.open('w', newline='') as f:
                csv.writer(f).writerow(ExtendedTrainer._LOG_COLUMNS_LEGACY)
        trainer._setup_logging()
        trainer._log({'step': 4, 'tokens_processed': 3000, 'train_loss': 2.5,
                      'supervised_tokens_step': 900,
                      'supervised_tokens_total': 1900,
                      'supervised_tokens_billions': 0.0000019,
                      'supervised_tokens_provenance': 'measured'})
        with trainer.log_file.open(newline='') as f:
            rows = list(csv.DictReader(f))
        assert rows[0]['tokens_processed'] == '3000'
        if not legacy:
            assert rows[0]['supervised_tokens_total'] == '1900'
        else:
            assert 'supervised_tokens_total' not in rows[0]
        with trainer._supervised_metrics_log_path().open(newline='') as f:
            rows = list(csv.DictReader(f))
        assert rows[0]['supervised_tokens_total'] == '1900'


def test_scheduler_once_per_crossed_threshold(tmp_path, monkeypatch):
    config = tmp_path / 'suite.json'
    config.write_text(json.dumps({
        'version': 1, 'output_root': str(tmp_path/'results'),
        'benchmarks': [{
            'id': 'w1', 'type': 'wikitext', 'every_supervised_tokens': 100,
            'parameters': {'data': 'unused', 'context_length': 2},
        }],
    }))
    calls = []
    def evaluate(model, tokenizer, **kwargs):
        calls.append(kwargs['context_length'])
        return {'cross_entropy': 2.0, 'perplexity': 7.389, 'bits_per_byte': 1.1, 'scored_tokens': 10}
    monkeypatch.setattr(live_suite, 'run_wikitext', evaluate)
    trainer = SimpleNamespace(step=10, supervised_tokens_total=95.,
        supervised_tokens_provenance='measured', model=object())
    scheduler = live_suite.LiveBenchmarkScheduler.from_file(config, object(), trainer)
    scheduler.maybe_run(trainer)
    assert calls == []
    trainer.step = 11
    trainer.supervised_tokens_total = 105
    scheduler.maybe_run(trainer)
    scheduler.maybe_run(trainer)
    assert calls == [2]
    assert json.loads((tmp_path/'results'/'step_00000011'/'w1'/'result.json').read_text())['status'] == 'complete'
    trainer.step = 12
    trainer.supervised_tokens_total = 205
    scheduler.maybe_run(trainer)
    assert calls == [2, 2]


def test_generation_config_prompt_ids_exist():
    root = Path(__file__).resolve().parents[1]
    suite = json.loads((root/'benchmarks/configs/foundation_live_0092.json').read_text())
    prompts = json.loads((root/'benchmarks/generation_prompts_v1.json').read_text())
    valid = {p['id'] for p in prompts['prompts']}
    for bench in suite['benchmarks']:
        if bench['type'] == 'generation':
            assert set(bench['parameters']['prompt_ids']) <= valid


def test_completed_steps_persist_actual_supervised_total(tmp_path):
    from mini_llm.config import ModelConfig
    from mini_llm.model.decoder_lm import DecoderLanguageModel
    from mini_llm.checkpoint import load_checkpoint

    model = DecoderLanguageModel(ModelConfig.tiny_inspection(), dtype='float32')
    shard = tmp_path / 'tiny.bin'
    (np.arange(4096, dtype='uint16') % 200).tofile(shard)
    checkpoint = tmp_path / 'ckpt'
    trainer = ExtendedTrainer(
        model, [str(shard)], [str(shard)],
        batch_size=1, seq_length=16, grad_accum_steps=2,
        total_steps=2, warmup_steps=0, val_interval=100,
        checkpoint_dir=str(checkpoint), log_file=str(tmp_path / 'train.csv'),
    )
    trainer.train(num_steps=2, log_interval=1)
    assert trainer.step == 2
    assert trainer.supervised_tokens_total == 64
    trainer.save()
    _, _, state = load_checkpoint(checkpoint, skip_optimizer=True)
    assert state['supervised_tokens_total'] == 64
    assert state['supervised_tokens_provenance'] == 'measured'
    with (tmp_path / 'train.supervised.csv').open(newline='') as f:
        rows = list(csv.DictReader(f))
    assert [float(r['supervised_tokens_total']) for r in rows] == [32., 64.]


def test_step_based_scheduler(tmp_path, monkeypatch):
    config = tmp_path / 'suite.json'
    config.write_text(json.dumps({
        'version': 1, 'output_root': str(tmp_path/'results'),
        'benchmarks': [{'id': 'w1', 'type': 'wikitext', 'every_optimizer_steps': 2,
                        'parameters': {'data': 'x', 'context_length': 2}}]}))
    monkeypatch.setattr(live_suite, 'run_wikitext', lambda *a, **kw: {
        'cross_entropy': 1., 'perplexity': 2.71828, 'bits_per_byte': 1., 'scored_tokens': 10})
    trainer = SimpleNamespace(step=11, supervised_tokens_total=3000,
        supervised_tokens_provenance='measured', model=object())
    scheduler = live_suite.LiveBenchmarkScheduler.from_file(config, object(), trainer)
    assert scheduler.next_due['w1']['value'] == 12
    scheduler.maybe_run(trainer)
    assert not (tmp_path/'results'/'metrics.csv').exists()
    trainer.step = 12
    scheduler.maybe_run(trainer)
    assert (tmp_path/'results'/'metrics.csv').exists()


def test_real_historical_read_plain_evaluation_restores_model(tmp_path):
    from mini_llm.config import ModelConfig, MemoryContextConfig
    from mini_llm.model.decoder_lm import DecoderLanguageModel
    from benchmarks.live_suite import run_reference

    cfg = ModelConfig(
        tokenizer_vocab_size=128, context_length=16, n_layers=16,
        d_model=32, n_q_heads=16, n_kv_heads=4, d_head=2,
        n_experts=1, top_k=1, d_ff=64, ffn_mode='dense',
        memory_context=MemoryContextConfig(
            enabled=True, memory_length=64, recent_length=16,
            target_length=16, block_size=8, top_k_blocks=1,
            router_query_length=16, router_dim=8, read_heads=2,
            read_kv_heads=1, integration_mode='historical_read',
        ),
    )
    model = DecoderLanguageModel(cfg, dtype='float32')
    suite = tmp_path/'references.json'
    suite.write_text(json.dumps({'references': [
        {'id': 'short', 'text': 'Hello there is a thing'}]}))
    result = run_reference(model, SimpleTokenizer(), references=str(suite),
                           cutoffs=[0.5], generation_tokens=2, seeds=[42],
                           max_context=24, top_k_predictions=3)
    assert len(result['cases']) == 1
    assert len(result['cases'][0]['cutoff_generations']) == 1
    assert model.historical_memory is not None
    assert model.config is cfg


def test_rng_isolation_without_numpy_style_cupy_get_state(monkeypatch):
    class Generator:
        def __init__(self, seed):
            self.seed = seed
    class RandomInterface:
        RandomState = Generator
        def __init__(self):
            self.live = Generator(999)
        def get_random_state(self):
            return self.live
        def set_random_state(self, state):
            self.live = state
        def seed(self, seed):
            self.live.seed = seed
    random = RandomInterface()
    fake_xp = SimpleNamespace(random=random)
    monkeypatch.setattr(live_suite, 'xp', fake_xp)
    original_global = np.random.get_state()
    old = random.get_random_state()
    with live_suite._preserve_global_rng():
        random.seed(42)
        np.random.seed(42)
        assert random.get_random_state().seed == 42
        assert random.get_random_state() is not old
    assert random.get_random_state() is old
    assert old.seed == 999
    after_global = np.random.get_state()
    assert original_global[0] == after_global[0]
    np.testing.assert_array_equal(original_global[1], after_global[1])
    assert original_global[2:] == after_global[2:]
