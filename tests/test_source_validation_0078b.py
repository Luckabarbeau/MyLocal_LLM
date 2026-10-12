"""0078B: fixed per-source held-out CE; no changes to training or mixed validation."""
import copy
import csv
import math

import numpy as np
import pytest

from mini_llm.config import ModelConfig, MemoryContextConfig
from mini_llm.model.decoder_lm import DecoderLanguageModel
from mini_llm.train_extended import ExtendedTrainer


def _trainer(tmp_path, monkeypatch, *, chunk=0, with_memory=False, batches=3):
    monkeypatch.setenv('MINI_LLM_LM_HEAD_CHUNK_TOKENS', str(chunk))
    # Deliberately distinctive source data, two shards per source.
    paths = {}
    for name, value in [('github', 4), ('fineweb', 20)]:
        paths[name] = []
        for i in range(2):
            p = tmp_path / f'{name}_{i}.bin'
            tokens = np.tile(np.array([value + i, value + 1, value + 2, 2], dtype=np.uint16), 300)
            tokens.tofile(p)
            paths[name].append(str(p))
    if with_memory:
        # Tiny actual EOS-bounded document-aware training/validation, not a mock.
        eos = 2
        mc = MemoryContextConfig(enabled=True, memory_length=8, recent_length=4,
                                  target_length=4, block_size=2, top_k_blocks=1,
                                  router_query_length=4, router_dim=4, read_heads=2,
                                  integration_mode='routed_prefix', memory_training='joint',
                                  retrieval_batch_probability=0.0,
                                  read_kv_heads=1, read_query_chunk=2)
        config = ModelConfig(tokenizer_vocab_size=32, context_length=mc.active_length,
                             n_layers=1, d_model=8, n_q_heads=2, n_kv_heads=1,
                             d_head=4, n_experts=2, top_k=1, d_ff=16,
                             dtype='float32', memory_context=mc)
        length = mc.source_input_length
    else:
        eos = None
        config = ModelConfig.tiny_inspection()
        length = 8
    return ExtendedTrainer(
        model=DecoderLanguageModel(config, rng_seed=18),
        train_shard_paths=list(paths['github']), val_shard_paths=list(paths['github']),
        train_source_shards=paths, val_source_shards=paths,
        source_weights={'github': .15, 'fineweb': .85},
        batch_size=2, seq_length=length, grad_accum_steps=1,
        total_steps=4, warmup_steps=1, val_steps=2,
        val_interval=1, save_interval=100,
        val_per_source_steps=batches, val_source_seed=78078,
        eos_token_id=eos, log_file=str(tmp_path / 'train.csv'),
    )


@pytest.mark.parametrize('chunk', [0, 4])
def test_fixed_sources_reproducible_and_state_neutral(tmp_path, monkeypatch, chunk):
    trainer = _trainer(tmp_path, monkeypatch, chunk=chunk)
    initial_train = copy.deepcopy(trainer.train_rng.bit_generator.state)
    initial_val = copy.deepcopy(trainer.val_rng.bit_generator.state)
    params = [np.array(p.data, copy=True) for p in trainer.optimization_parameters]
    first = trainer.compute_val_loss_by_source()
    assert list(first) == ['fineweb', 'github']
    assert first['fineweb']['sample_sha256'] != first['github']['sample_sha256']
    assert all(math.isfinite(r['ce_nats_per_target']) for r in first.values())
    assert all(r['supervised_tokens'] == 2 * 8 * 3 for r in first.values())
    assert all(r['target_fill'] == 1 for r in first.values())
    assert all(r['perplexity'] == pytest.approx(math.exp(r['ce_nats_per_target'])) for r in first.values())
    assert trainer.train_rng.bit_generator.state == initial_train
    assert trainer.val_rng.bit_generator.state == initial_val
    assert trainer.current_val_source_shard_idx == {'github': 0, 'fineweb': 0}
    assert trainer.val_source_batch_counts == {'github': 0, 'fineweb': 0}
    trainer.get_val_batch()
    second = trainer.compute_val_loss_by_source()
    for name in first:
        assert first[name]['sample_sha256'] == second[name]['sample_sha256']
        assert first[name]['ce_nats_per_target'] == pytest.approx(second[name]['ce_nats_per_target'], abs=1e-6)
    for p, original in zip(trainer.optimization_parameters, params):
        np.testing.assert_array_equal(p.data, original)
    trainer._log_val_source_losses(first)
    with (tmp_path / 'train.val_sources.csv').open() as f:
        rows = list(csv.DictReader(f))
    assert len(rows) == 2
    assert {r['source'] for r in rows} == {'fineweb', 'github'}
    assert all(int(r['validation_batches']) == 3 for r in rows)


def test_per_source_validation_runs_at_validation_interval(tmp_path, monkeypatch, capsys):
    trainer = _trainer(tmp_path, monkeypatch, batches=1)
    trainer.train(num_steps=1, log_interval=1)
    out = capsys.readouterr().out
    assert 'Per-source held-out validation CE' in out
    with (tmp_path / 'train.val_sources.csv').open() as f:
        rows = list(csv.DictReader(f))
    assert len(rows) == 2
    assert {int(row['step']) for row in rows} == {1}
    assert trainer.source_metrics  # from 0078A


def test_memory_masks_and_token_weighted_ce(tmp_path, monkeypatch):
    trainer = _trainer(tmp_path, monkeypatch, chunk=4, with_memory=True, batches=3)
    first = trainer.compute_val_loss_by_source()
    for rec in first.values():
        assert 0 < rec['supervised_token_weight'] <= rec['target_capacity_tokens']
        assert rec['supervised_tokens'] <= rec['target_capacity_tokens']
        assert 0 < rec['target_fill'] <= 1
        assert rec['ce_nats_per_target'] == pytest.approx(rec['loss_nll_sum'] / rec['supervised_token_weight'])
    # Repeated evaluations consume identical masked examples.
    second = trainer.compute_val_loss_by_source()
    assert {s: r['sample_sha256'] for s, r in first.items()} == {
        s: r['sample_sha256'] for s, r in second.items()
    }


def test_disabled_by_default_and_rejects_invalid(tmp_path, monkeypatch):
    trainer = _trainer(tmp_path, monkeypatch)
    trainer.val_per_source_steps = 0
    assert trainer.compute_val_loss_by_source() == {}
    trainer.val_per_source_steps = 2
    trainer.val_source_shards = None
    with pytest.raises(ValueError, match='requires local'):
        trainer.compute_val_loss_by_source()


def test_loss_is_weighted_by_valid_targets_not_batch_count(tmp_path, monkeypatch):
    trainer = _trainer(tmp_path, monkeypatch, batches=2)
    ce_values = iter([1.0, 3.0, 1.0, 3.0])
    def fake_eval(inputs, targets, metadata):
        val = next(ce_values)
        mask = np.zeros_like(targets, dtype=np.float32)
        mask.reshape(-1)[:2 if val == 1.0 else 6] = 1
        return val, targets, mask
    trainer._validation_batch_loss = fake_eval
    results = trainer.compute_val_loss_by_source()
    for result in results.values():
        assert result['supervised_tokens'] == 8
        assert result['ce_nats_per_target'] == pytest.approx((1 * 2 + 3 * 6) / 8)
        assert result['loss_nll_sum'] == pytest.approx(20)


def test_legacy_validation_mean_is_unchanged(tmp_path, monkeypatch):
    a = _trainer(tmp_path, monkeypatch, batches=2)
    b = _trainer(tmp_path, monkeypatch, batches=2)
    expected = []
    for _ in range(b.val_steps):
        x, y, m = b.get_val_batch(return_metadata=True)
        loss, _, _ = b._validation_batch_loss(x, y, m)
        expected.append(loss)
    assert a.compute_val_loss() == pytest.approx(np.mean(expected), abs=1e-5)
    assert a.val_source_batch_counts == b.val_source_batch_counts
