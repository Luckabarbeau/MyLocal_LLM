"""0078A: read-only token/loss/source attribution tests."""
import csv
import json

import numpy as np
import pytest

from mini_llm.config import ModelConfig
from mini_llm.model.decoder_lm import DecoderLanguageModel
from mini_llm.train_extended import ExtendedTrainer
from mini_llm.diagnostics.source_metrics import batch_coverage, empty_record, merge_record, row


def test_coverage_is_valid_target_weight_not_capacity():
    m = {
        'history_valid_starts': np.array([2, 5]),
        'target_valid_lengths': np.array([4, 2]),
        'document_starts': np.array([10, 40]),
        'document_ends': np.array([18, 44]),
    }
    mask = np.array([[1, 1, 1, 0], [1, 0.5, 0, 0]], dtype=np.float32)
    rec = batch_coverage(batch_size=2, source_length=10, target_shape=(2, 4),
                         loss_mask=mask, memory_length=6, target_length=4, metadata=m)
    assert rec['source_capacity_tokens'] == 20
    assert rec['valid_history_tokens'] == 5
    assert rec['valid_active_tokens'] == 6
    assert rec['valid_source_tokens'] == 11
    assert rec['target_capacity_tokens'] == 8
    assert rec['supervised_tokens'] == 5
    assert rec['supervised_token_weight'] == pytest.approx(4.5)
    assert rec['sampled_document_spans'] == 2
    assert rec['sampled_document_span_tokens_sum'] == 12
    assert rec['sampled_document_span_tokens_min'] == 4
    assert rec['sampled_document_span_tokens_max'] == 8


def test_ce_is_supervised_token_weighted_and_merge_exact():
    a, b = empty_record(), empty_record()
    a.update(supervised_tokens=2, supervised_token_weight=2., target_capacity_tokens=8, loss_nll_sum=8.0)
    b.update(supervised_tokens=8, supervised_token_weight=8., target_capacity_tokens=8, loss_nll_sum=16.0)
    merge_record(a, b)
    r = row('github', a, 46901, 'interval', 46900)
    assert r['mean_loss_nats_per_target'] == pytest.approx(2.4)
    assert r['target_fill'] == pytest.approx(10 / 16)
    assert r['supervised_tokens'] == 10


@pytest.mark.parametrize('chunked', [0, 4])
def test_source_metrics_survive_training_and_checkpoint(tmp_path, monkeypatch, chunked):
    monkeypatch.setenv('MINI_LLM_LM_HEAD_CHUNK_TOKENS', str(chunked))
    sources = {}
    for name, token in [('github', 4), ('fineweb', 19)]:
        p = tmp_path / f'{name}.bin'
        np.full(2048, token, dtype=np.uint16).tofile(p)
        sources[name] = [str(p)]

    trainer = ExtendedTrainer(
        model=DecoderLanguageModel(ModelConfig.tiny_inspection(), rng_seed=7),
        train_shard_paths=sum(sources.values(), []), val_shard_paths=sum(sources.values(), []),
        train_source_shards=sources, val_source_shards=sources,
        source_weights={'github': 0.5, 'fineweb': 0.5},
        batch_size=1, seq_length=8, grad_accum_steps=2,
        gradient_accumulation_weighting='supervised_tokens',
        warmup_steps=1, total_steps=4, val_interval=10, save_interval=10,
        checkpoint_dir=str(tmp_path / 'checkpoint'), log_file=str(tmp_path / 'train.csv'),
    )
    choices = iter(['github', 'fineweb'])
    trainer._choose_train_source = lambda _: next(choices)
    loss, _ = trainer.train_step()
    assert trainer.step == 1
    assert set(trainer.source_metrics) == {'github', 'fineweb'}
    assert trainer.source_metrics_since_step == 0
    for source in sources:
        stats = trainer.source_metrics[source]
        assert stats['microbatches'] == 1
        assert stats['supervised_tokens'] == 8
        assert stats['sampled_document_spans'] == 0
    total_nll = sum(x['loss_nll_sum'] for x in trainer.source_metrics.values())
    total_weight = sum(x['supervised_token_weight'] for x in trainer.source_metrics.values())
    assert total_nll / total_weight == pytest.approx(loss, abs=2e-5)
    trainer.save()
    state = json.loads((tmp_path / 'checkpoint' / 'training_state.json').read_text())
    assert state['source_metrics']['since_step'] == 0
    assert state['source_metrics']['sources']['github']['supervised_tokens'] == 8

    new = ExtendedTrainer(
        model=DecoderLanguageModel(ModelConfig.tiny_inspection(), rng_seed=7),
        train_shard_paths=sum(sources.values(), []), val_shard_paths=sum(sources.values(), []),
        train_source_shards=sources, val_source_shards=sources,
        source_weights={'github': 0.5, 'fineweb': 0.5}, batch_size=1, seq_length=8,
    )
    new.restore_source_metrics(state['source_metrics'])
    assert new.source_metrics == trainer.source_metrics
    assert new._source_metrics_interval == {}
    assert new.source_metrics_since_step == 0

    rows = [row(name, stat, trainer.step, 'interval', trainer.source_metrics_since_step)
            for name, stat in trainer._source_metrics_interval.items()]
    from mini_llm.diagnostics.source_metrics import write_source_rows
    write_source_rows(tmp_path/'train.sources.csv', rows)
    with (tmp_path / 'train.sources.csv').open() as f:
        lines = list(csv.DictReader(f))
    assert len(lines) == 2
    assert {r['source'] for r in lines} == {'github', 'fineweb'}


def test_actual_document_aware_routed_training_accounts_short_fragments(tmp_path, monkeypatch):
    """Exercise real packed EOS sampler with source metrics in a memory model."""
    import dataclasses
    from mini_llm.config import MemoryContextConfig

    monkeypatch.setenv('MINI_LLM_LM_HEAD_CHUNK_TOKENS', '0')
    eos = 31
    tokens = np.array(
        [4, 5, 6, eos, 7, 8, 9, 10, 11, eos, 12, 13, eos], dtype=np.uint16
    )
    shard = tmp_path / 'packed.bin'
    tokens.tofile(shard)
    cfg = ModelConfig.tiny_inspection()
    mc = MemoryContextConfig(
        enabled=True, memory_length=8, recent_length=4, target_length=4,
        block_size=2, top_k_blocks=1, router_query_length=4, router_dim=4,
        query_pooling='mean', history_pooling='mean', integration_mode='routed_prefix',
        memory_training='joint', retrieval_batch_probability=0.0,
        read_heads=1, read_kv_heads=1,
    )
    cfg = dataclasses.replace(cfg, context_length=mc.active_length, memory_context=mc)
    model = DecoderLanguageModel(cfg, rng_seed=11)
    trainer = ExtendedTrainer(
        model=model, train_shard_paths=[str(shard)], val_shard_paths=[str(shard)],
        train_source_shards={'cosmopedia_v2': [str(shard)]},
        val_source_shards={'cosmopedia_v2': [str(shard)]},
        source_weights={'cosmopedia_v2': 1.0}, eos_token_id=eos,
        batch_size=2, seq_length=mc.memory_length, grad_accum_steps=1,
        total_steps=2, warmup_steps=1,
    )
    loss, _ = trainer.train_step()
    assert np.isfinite(loss)
    record = trainer.source_metrics['cosmopedia_v2']
    assert record['supervised_tokens'] <= 8
    assert record['supervised_tokens'] > 0
    assert record['valid_source_tokens'] <= record['source_capacity_tokens']
    assert record['sampled_document_spans'] == 2
    assert 3 <= record['sampled_document_span_tokens_min'] <= 6
    assert record['sampled_document_span_tokens_max'] <= 6
    assert record['loss_nll_sum'] / record['supervised_token_weight'] == pytest.approx(loss, abs=2e-5)


def test_source_metrics_csv_is_written_at_log_interval(tmp_path, capsys):
    p = tmp_path / 's.bin'
    np.full(500, 3, dtype=np.uint16).tofile(p)
    trainer = ExtendedTrainer(
        model=DecoderLanguageModel(ModelConfig.tiny_inspection(), rng_seed=7),
        train_shard_paths=[str(p)], val_shard_paths=[str(p)],
        train_source_shards={'github': [str(p)]},
        val_source_shards={'github': [str(p)]}, source_weights={'github': 1},
        batch_size=1, seq_length=8, grad_accum_steps=1, warmup_steps=1,
        total_steps=2, val_interval=10, save_interval=10,
        log_file=str(tmp_path / 'train.csv'),
    )
    trainer.train(num_steps=2, log_interval=1)
    path = tmp_path / 'train.sources.csv'
    with path.open() as handle:
        rows = list(csv.DictReader(handle))
    assert [(int(r['step']), r['scope']) for r in rows] == [
        (1, 'interval'), (1, 'cumulative_since_patch'),
        (2, 'interval'), (2, 'cumulative_since_patch'),
    ]
    assert int(rows[-1]['supervised_tokens']) == 16
    assert int(rows[-1]['microbatches']) == 2
    assert float(rows[-1]['target_fill']) == pytest.approx(1.0)
    assert 'Per-source train CE' in capsys.readouterr().out
