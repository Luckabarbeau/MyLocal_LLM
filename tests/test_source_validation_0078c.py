"""0078C: representative held-out per-source evaluation without history filtering."""
import copy
import csv

import numpy as np
import pytest

from mini_llm.config import MemoryContextConfig, ModelConfig
from mini_llm.model.decoder_lm import DecoderLanguageModel
from mini_llm.train_extended import ExtendedTrainer


def _make_trainer(tmp_path, monkeypatch, *, steps=4):
    monkeypatch.setenv("MINI_LLM_LM_HEAD_CHUNK_TOKENS", "4")
    mc = MemoryContextConfig(
        enabled=True, memory_length=8, recent_length=4,
        target_length=4, block_size=2, top_k_blocks=1,
        router_query_length=4, router_dim=4, read_heads=2,
        integration_mode="routed_prefix", memory_training="joint",
        retrieval_batch_probability=0., read_kv_heads=1, read_query_chunk=2,
    )
    cfg = ModelConfig(
        tokenizer_vocab_size=32, context_length=mc.active_length,
        n_layers=1, d_model=8, n_q_heads=2, n_kv_heads=1,
        d_head=4, n_experts=2, top_k=1, d_ff=16,
        dtype="float32", memory_context=mc,
    )
    # Many short docs with too few target tokens, plus a handful of long docs
    # which the legacy forced-min-history sampler preferentially selects.
    short = [4, 5, 2]  # 2 supervised pairs per short document
    long_doc = ([6, 7, 8, 9, 10, 11, 12, 13, 14, 15] * 3) + [2]
    data = np.asarray(short * 150 + long_doc * 8, dtype=np.uint16)
    path = tmp_path / 'cosmopedia_0.bin'
    data.tofile(path)
    src = {"cosmopedia_v2": [str(path)]}
    return ExtendedTrainer(
        model=DecoderLanguageModel(cfg, rng_seed=17),
        train_shard_paths=[str(path)], val_shard_paths=[str(path)],
        train_source_shards=src, val_source_shards=src,
        source_weights={"cosmopedia_v2": 1.0},
        batch_size=2, seq_length=mc.source_input_length,
        grad_accum_steps=1, total_steps=4, warmup_steps=1,
        val_steps=1, val_interval=1, save_interval=100,
        val_per_source_steps=steps, val_source_seed=78078,
        val_representative_steps=steps, val_representative_seed=78078,
        eos_token_id=2, log_file=str(tmp_path / 'train.csv'),
    )


def test_representative_samples_short_docs_and_preserves_legacy(tmp_path, monkeypatch):
    trainer = _make_trainer(tmp_path, monkeypatch, steps=8)
    original_train_rng = copy.deepcopy(trainer.train_rng.bit_generator.state)
    original_val_rng = copy.deepcopy(trainer.val_rng.bit_generator.state)
    old = trainer.compute_val_loss_by_source()
    new = trainer.compute_val_loss_by_source_representative()
    old2 = trainer.compute_val_loss_by_source()
    new2 = trainer.compute_val_loss_by_source_representative()
    name = 'cosmopedia_v2'
    assert old[name]['sample_sha256'] == old2[name]['sample_sha256']
    assert new[name]['sample_sha256'] == new2[name]['sample_sha256']
    assert old[name]['sample_sha256'] != new[name]['sample_sha256']
    assert new[name]['target_fill'] < old[name]['target_fill']
    assert new[name]['target_fill'] < 0.90
    assert old[name]['target_fill'] == 1.0
    assert new[name]['sampled_document_length_median'] <= 4
    assert new[name]['short_document_fraction'] > 0.5
    assert new[name]['sampled_document_count'] == 16
    assert new[name]['document_selection_fallback_count'] == 0
    assert 0 < new[name]['unique_sampled_documents'] <= 16
    assert trainer.train_rng.bit_generator.state == original_train_rng
    assert trainer.val_rng.bit_generator.state == original_val_rng
    assert trainer.current_val_source_shard_idx[name] == 0
    assert trainer.val_source_batch_counts[name] == 0
    trainer._log_representative_source_losses(new)
    with (tmp_path / 'train.val_sources_representative.csv').open() as fh:
        rows = list(csv.DictReader(fh))
    assert len(rows) == 1
    assert rows[0]['sample_sha256'] == new[name]['sample_sha256']
    assert rows[0]['sampling_mode'] == 'representative_token_sampled'


def test_representative_no_perturbation_of_training_and_ordinary_validation(tmp_path, monkeypatch):
    trainer = _make_trainer(tmp_path, monkeypatch, steps=2)
    before = trainer.compute_val_loss_by_source()['cosmopedia_v2']['sample_sha256']
    trainer.train(num_steps=1, log_interval=1)
    # Separate output from unchanged historical mixed and 0078B source metrics.
    with (tmp_path / 'train.val_sources.csv').open() as fh:
        old_rows = list(csv.DictReader(fh))
    with (tmp_path / 'train.val_sources_representative.csv').open() as fh:
        new_rows = list(csv.DictReader(fh))
    assert len(old_rows) == len(new_rows) == 1
    assert old_rows[0]['sample_sha256'] == before
    assert new_rows[0]['sample_sha256'] != before
    assert trainer.step == 1


def test_representative_disabled_and_override_guardrails(tmp_path, monkeypatch):
    trainer = _make_trainer(tmp_path, monkeypatch, steps=2)
    trainer.val_representative_steps = 0
    assert trainer.compute_val_loss_by_source_representative() == {}
    with pytest.raises(ValueError, match='explicit source'):
        trainer.get_val_batch(min_history_override=0)
    with pytest.raises(ValueError, match='nonnegative'):
        trainer.get_val_batch(source_override='cosmopedia_v2', min_history_override=-1)
