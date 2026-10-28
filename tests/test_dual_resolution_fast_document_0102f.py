"""0102F: short context bypass, reusable metadata, and optional shape bucketing."""

import numpy as np
import pytest

from mini_llm.backend import RandomStream
from mini_llm.config import DualResolutionConfig
from mini_llm.data.token_shards import (
    _dual_resolution_geometric_length, build_packed_document_index,
    create_dual_resolution_document_minibatch,
)
from mini_llm.ops.dual_resolution_attention import (
    DualResolutionAttention, build_dual_resolution_metadata,
)


def attention(seed=3):
    return DualResolutionAttention(
        d_model=8, n_q_heads=2, n_kv_heads=1, d_head=4,
        input_std=.08, output_std=.08, rng=RandomStream(seed),
        dtype="float64", config=DualResolutionConfig(
            enabled=True, block_size=4, local_window=8,
            compression_dim=8, compression_heads=2,
            compression_depth=1, max_blocks=8, global_query_chunk=4,
            recompute_attention_probs=True,
        ),
    )


@pytest.mark.parametrize('length', [1, 2, 3, 4])
def test_no_compressor_gate_or_global_is_executed_when_no_visible_summary(length):
    a = attention()
    x = np.random.default_rng(2).normal(size=(1, length, 8))
    def forbidden(*args, **kwargs):
        raise AssertionError('unused global branch was executed')
    a._compress_forward = forbidden
    out, cache = a.forward(x)
    assert cache['local_only']
    assert cache['summaries'].shape == (1, 0, 8)
    assert not np.any(cache['global_available'])
    np.testing.assert_allclose(
        out, np.asarray(cache['local_context']).reshape(1, length, 8) @ a.Wo.data,
        atol=1e-12,
    )
    dx = a.backward(np.ones_like(out), cache)
    assert dx.shape == x.shape and np.isfinite(dx).all()
    # A branch with no causal path must not receive any gradient.
    assert np.count_nonzero(a.W_gate.grad) == 0
    assert np.count_nonzero(a.W_compress_in.grad) == 0
    assert np.count_nonzero(a.W_global_k.grad) == 0
    # A single-token causal softmax has no query dependence.
    assert np.count_nonzero(a.Wq.grad) > 0 if length > 1 else not np.count_nonzero(a.Wq.grad)
    assert np.count_nonzero(a.Wo.grad) > 0


def test_short_path_backward_finite_difference():
    a = attention(5)
    x = np.random.default_rng(4).normal(size=(1, 3, 8))
    upstream = np.random.default_rng(6).normal(size=x.shape)
    y, cache = a.forward(x)
    dx = a.backward(upstream, cache)
    direction = np.random.default_rng(7).normal(size=x.shape)
    eps = 1e-6
    actual = float(np.sum(dx * direction))
    numerical = (float(np.sum(a.forward(x + eps * direction, return_cache=False) * upstream))
                 - float(np.sum(a.forward(x - eps * direction, return_cache=False) * upstream))) / (2 * eps)
    np.testing.assert_allclose(actual, numerical, atol=1e-7, rtol=1e-6)


def test_document_metadata_reused_between_layers_and_replays():
    a = attention()
    tokens = np.array([[1, 2, 3, 4, 5, 6, 7, 8]], dtype=np.int64)
    md = build_dual_resolution_metadata(tokens, eos_token_id=99, block_size=4)
    first = a._prepare_metadata(md, 1, 8, None)
    second = a._prepare_metadata(md, 1, 8, None)
    assert second is first
    override = a._prepare_metadata(md, 1, 8, np.arange(8)[None, ::-1])
    assert override is not first
    np.testing.assert_array_equal(override['position_ids'], np.arange(8)[None, ::-1])
    corrupted = build_dual_resolution_metadata(tokens, eos_token_id=99, block_size=4)
    corrupted['block_indices'][0, 0, 0] = 100
    with pytest.raises(IndexError, match='valid block_indices'):
        a._prepare_metadata(corrupted, 1, 8, None)


@pytest.mark.parametrize('length,cap', [
    (1, 65_536), (35, 65_536), (120, 65_536), (129, 65_536),
    (520, 65_536), (921, 65_536), (16378, 65_536), (65536, 65536),
    (31, 31), (149, 149),
])
def test_geometric_buckets_do_not_drop_tokens(length, cap):
    padded = _dual_resolution_geometric_length(length, cap)
    assert length <= padded <= cap
    if length >= 64 and padded < cap:
        assert padded <= length * 1.5 + 32


def test_bucketed_document_preserves_genuine_targets_and_masks_padding():
    tokens = np.array([*range(1, 13), 99], dtype=np.uint16)
    index = build_packed_document_index(tokens, 99)
    original = create_dual_resolution_document_minibatch(
        tokens, 1, 64, rng=np.random.default_rng(1), eos_token_id=99,
        document_index=index, return_metadata=True, shape_bucket='none',
    )
    bucketed = create_dual_resolution_document_minibatch(
        tokens, 1, 64, rng=np.random.default_rng(1), eos_token_id=99,
        document_index=index, return_metadata=True, shape_bucket='geometric',
    )
    (x, y, meta), (xb, yb, mb) = original, bucketed
    assert xb.shape[1] >= x.shape[1]
    np.testing.assert_array_equal(xb[:, :x.shape[1]], x)
    np.testing.assert_array_equal(yb[:, :y.shape[1]], y)
    assert mb['target_loss_mask'].sum() == meta['target_loss_mask'].sum()
    assert not np.any(mb['target_loss_mask'][:, x.shape[1]:])
    a = attention()
    # Explicit padding mask: all fake EOS blocks must be invalidated.
    md = build_dual_resolution_metadata(xb, eos_token_id=99, block_size=4)
    md['block_valid'] &= md['block_indices'][:, :, -1] < mb['target_valid_lengths'][:, None]
    ref = build_dual_resolution_metadata(x, eos_token_id=99, block_size=4)
    full_out = a.forward(xb.astype(float)[..., None] * np.ones((1, 1, 8)),
                         dual_resolution_metadata=md, return_cache=False)
    orig_out = a.forward(x.astype(float)[..., None] * np.ones((1, 1, 8)),
                         dual_resolution_metadata=ref, return_cache=False)
    np.testing.assert_allclose(full_out[:, :x.shape[1]], orig_out, atol=1e-11, rtol=1e-11)


def test_real_trainer_geometric_mode_keeps_actual_supervision(tmp_path, monkeypatch):
    from mini_llm.config import ModelConfig
    from mini_llm.model.decoder_lm import DecoderLanguageModel
    from mini_llm.train_extended import ExtendedTrainer

    monkeypatch.setenv('MINI_LLM_DUAL_RESOLUTION_SHAPE_BUCKET', 'geometric')
    data = np.asarray([*range(1, 14), 99], dtype=np.uint16)
    path = tmp_path / 'tokens.bin'
    data.tofile(path)
    cfg = ModelConfig(
        tokenizer_vocab_size=128, context_length=32, n_layers=2,
        d_model=16, d_ff=32, n_q_heads=4, n_kv_heads=2,
        d_head=4, ffn_mode='dense', n_experts=1, top_k=1,
        dtype='float64', dual_resolution=DualResolutionConfig(
            enabled=True, block_size=4, local_window=8,
            compression_dim=8, compression_heads=2, max_blocks=8,
            global_query_chunk=4,
        ),
    )
    trainer = ExtendedTrainer(
        model=DecoderLanguageModel(cfg, rng_seed=12),
        train_shard_paths=[str(path)], val_shard_paths=[str(path)],
        batch_size=1, seq_length=32, grad_accum_steps=1,
        total_steps=4, warmup_steps=1, val_steps=1,
        val_interval=100, save_interval=100, eos_token_id=99,
        checkpoint_dir=str(tmp_path / 'ckpt'),
    )
    x, y, md = trainer.get_train_batch(return_metadata=True)
    assert x.shape[1] == 16  # 13 genuine pairs rounded up to 16
    assert int(md['target_valid_lengths'][0]) == 13
    assert md['target_loss_mask'].sum() == 13
    assert int(md['dual_resolution_metadata']['block_valid'].sum()) == 3
    loss, grad_norm = trainer.train_step()
    assert np.isfinite(loss) and np.isfinite(grad_norm)
    assert trainer._last_step_supervised_tokens == 13
    assert trainer._last_step_executed_tokens == 16
