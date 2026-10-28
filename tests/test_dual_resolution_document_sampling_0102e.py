"""0102E: EOS-bounded document-first full-resolution training contract."""
import numpy as np
import pytest

from mini_llm.data.token_shards import (
    build_packed_document_index, create_dual_resolution_document_minibatch,
)
from mini_llm.ops.dual_resolution_attention import (
    build_dual_resolution_metadata, dual_resolution_loss_mask,
)

EOS = 99


def sample(docs, *, cap=16, batch=1, seed=1):
    raw = np.asarray([v for doc in docs for v in [*doc, EOS]], dtype=np.uint16)
    index = build_packed_document_index(raw, EOS)
    return create_dual_resolution_document_minibatch(
        raw, batch, cap, rng=np.random.default_rng(seed),
        eos_token_id=EOS, document_index=index, return_metadata=True,
    )


@pytest.mark.parametrize('pairs', [1, 2, 7, 15, 16, 17, 32])
def test_variable_length_full_supervision_and_exact_shift(pairs):
    x, y, md = sample([list(range(1, pairs + 1))], cap=16)
    expected = min(pairs, 16)
    assert x.shape == y.shape == (1, expected)
    assert md['target_loss_mask'].sum() == expected
    assert md['document_sampling_policy'] == 'document_first_full_sequence'
    if pairs <= 16:
        np.testing.assert_array_equal(x[0], list(range(1, pairs + 1)))
        np.testing.assert_array_equal(y[0], list(range(2, pairs + 1)) + [EOS])
    assert int(md['executed_sequence_length']) == expected
    assert int(md['context_capacity']) == 16


def test_no_long_document_length_filter_and_no_cross_document_targets():
    docs = [[11, 12], list(range(20, 60)), [65, 66, 67]]
    x, y, md = sample(docs, cap=16, batch=500, seed=15)
    seen = set(map(int, md['document_starts']))
    assert len(seen) == 3
    assert np.all(md['target_valid_lengths'] > 0)
    for row in range(len(x)):
        length = int(md['target_valid_lengths'][row])
        assert md['target_loss_mask'][row, :length].sum() == length
        assert md['target_loss_mask'][row, length:].sum() == 0
        start = int(md['target_source_starts'][row])
        doc_end = int(md['document_ends'][row])
        assert start + length + 1 <= doc_end
        np.testing.assert_array_equal(x[row, :length],
                                      np.asarray([v for d in docs for v in [*d, EOS]], dtype=np.uint16)[start:start+length])


def test_exact_long_window_and_completed_block_visibility():
    x, y, md = sample([list(range(1, 49))], cap=32)
    assert x.shape == (1, 32)
    assert md['target_loss_mask'].sum() == 32
    meta = build_dual_resolution_metadata(x, eos_token_id=EOS, block_size=8)
    assert int(meta['block_valid'].sum()) == 4
    assert np.all(meta['document_ids'] == 0)
    assert np.all(meta['position_ids'][0] == np.arange(32))


def test_padded_batch_does_not_score_padding():
    x, y, md = sample([[1, 2], [3, 4, 5, 6, 7, 8, 9]], cap=16, batch=20)
    mask = dual_resolution_loss_mask(x, EOS, md['target_loss_mask'])
    np.testing.assert_array_equal(mask, md['target_loss_mask'])
    assert x.shape[1] == int(md['target_valid_lengths'].max())
    for row, length in enumerate(md['target_valid_lengths']):
        assert mask[row, :length].sum() == length
        assert not np.any(mask[row, length:])


def test_invalid_packed_shape_is_rejected():
    with pytest.raises(ValueError, match='1D packed'):
        create_dual_resolution_document_minibatch(np.zeros((4, 8), dtype=np.uint16),
                                                 1, 8, eos_token_id=EOS)


def _trainer(tmp_path, *, documents, cap=24, batch=1):
    from mini_llm.config import ModelConfig, DualResolutionConfig
    from mini_llm.model.decoder_lm import DecoderLanguageModel
    from mini_llm.train_extended import ExtendedTrainer
    data = np.asarray([v for doc in documents for v in [*doc, EOS]], dtype=np.uint16)
    path = tmp_path / 'tokens.bin'
    data.tofile(path)
    cfg = ModelConfig(
        tokenizer_vocab_size=128, context_length=cap, n_layers=2,
        d_model=32, d_ff=64, n_q_heads=4, n_kv_heads=2,
        d_head=8, ffn_mode='dense', n_experts=1, top_k=1,
        dtype='float32', dual_resolution=DualResolutionConfig(
            enabled=True, block_size=8, local_window=16,
            compression_dim=16, compression_heads=2, max_blocks=8,
            global_query_chunk=16,
        ),
    )
    return ExtendedTrainer(
        model=DecoderLanguageModel(cfg, rng_seed=12),
        train_shard_paths=[str(path)], val_shard_paths=[str(path)],
        batch_size=batch, seq_length=cap, grad_accum_steps=1,
        total_steps=4, warmup_steps=1, val_steps=1,
        val_interval=100, save_interval=100, eos_token_id=EOS,
        checkpoint_dir=str(tmp_path / 'ckpt'),
    )


def test_real_trainer_adapts_execution_and_scores_all_pairs(tmp_path):
    trainer = _trainer(tmp_path, documents=[[1, 2, 3, 4, 5]], cap=24)
    x, y, md = trainer.get_train_batch(return_metadata=True)
    assert x.shape == y.shape == (1, 5)
    assert trainer.tokens_processed == 5
    assert int(md['target_valid_lengths'][0]) == 5
    np.testing.assert_array_equal(md['target_loss_mask'], np.ones((1, 5)))
    loss, grad_norm = trainer.train_step()
    assert np.isfinite(loss) and np.isfinite(grad_norm)
    assert trainer._last_step_supervised_tokens == 5
    assert trainer._last_step_executed_tokens == 5


def test_real_trainer_scores_all_positions_with_completed_summaries(tmp_path):
    trainer = _trainer(tmp_path, documents=[list(range(1, 33))], cap=24)
    x, y, md = trainer.get_train_batch(return_metadata=True)
    assert x.shape == y.shape == (1, 24)
    assert md['target_loss_mask'].sum() == 24
    assert int(md['dual_resolution_metadata']['block_valid'].sum()) == 3
    loss, grad_norm = trainer.train_step()
    assert np.isfinite(loss) and np.isfinite(grad_norm)
    assert trainer._last_step_supervised_tokens == 24
    assert trainer._last_step_executed_tokens == 24
