"""0103K: adaptive 120-shape document-first training buckets through 64K."""

import numpy as np
import pytest

from mini_llm.data.token_shards import (
    _dual_resolution_fine_length,
    _dual_resolution_geometric_length,
    build_packed_document_index,
    create_dual_resolution_document_minibatch,
)
from mini_llm.data.dual_resolution_length_buckets import (
    regroup_dual_resolution_document_batches,
)
from mini_llm.ops.dual_resolution_attention import (
    build_dual_resolution_metadata,
    dual_resolution_loss_mask,
)

EOS = 65_535


@pytest.mark.parametrize("length,expected", [
    (1, 1), (33, 48), (65, 96), (128, 128),
    (129, 192), (193, 256), (500, 512), (1999, 2048),
    (2048, 2048), (2049, 2176), (3000, 3072), (4096, 4096),
    (4097, 4352), (4500, 4608), (6145, 6400), (6200, 6400),
    (7000, 7168), (8191, 8192), (8192, 8192),
    (8193, 8704), (10000, 10240), (16384, 16384),
    (16385, 17408), (19000, 19456), (32768, 32768),
    (32769, 34816), (50000, 51200), (65535, 65536),
    (65536, 65536),
])
def test_fine_bucket_boundaries(length, expected):
    assert _dual_resolution_fine_length(length, 65_536) == expected


def test_complete_120_bucket_ladder_and_geometric_dominance():
    expected = [1, 2, 4, 8, 16, 32, 48, 64, 96, 128]
    expected += list(range(192, 2049, 64))
    expected += list(range(2176, 4097, 128))
    expected += list(range(4352, 8193, 256))
    expected += list(range(8704, 16385, 512))
    expected += list(range(17408, 32769, 1024))
    expected += list(range(34816, 65537, 2048))
    assert len(expected) == 120
    observed = np.fromiter(
        (_dual_resolution_fine_length(length, 65_536)
         for length in range(1, 65_537)), dtype=np.int32,
    )
    assert sorted(set(observed.tolist())) == expected
    np.testing.assert_array_less(np.arange(1, 65_537), observed + 1)
    assert np.all(np.diff(observed) >= 0)
    assert all(int(width) <= _dual_resolution_geometric_length(length, 65_536)
               for length, width in enumerate(observed, start=1))


@pytest.mark.parametrize("cap", [1, 31, 127, 129, 149, 191, 350, 1000,
                                2048, 3000, 4096, 8192, 16384, 32768, 65536])
def test_fine_bucket_respects_non_aligned_context_caps(cap):
    previous = 0
    for length in range(1, cap + 1):
        width = _dual_resolution_fine_length(length, cap)
        assert length <= width <= cap
        assert previous <= width
        assert width <= _dual_resolution_geometric_length(length, cap)
        previous = width


@pytest.mark.parametrize("length,cap", [(0, 8192), (-1, 8192), (1, 0),
                                        (129, 128), (65537, 65537)])
def test_fine_bucket_rejects_invalid_lengths(length, cap):
    with pytest.raises(ValueError):
        _dual_resolution_fine_length(length, cap)


def _sample_with_policy(raw, cap, batch_size, policy, seed=17):
    index = build_packed_document_index(raw, EOS)
    return create_dual_resolution_document_minibatch(
        raw, batch_size, cap, rng=np.random.default_rng(seed),
        eos_token_id=EOS, document_index=index, return_metadata=True,
        shape_bucket=policy,
    )


@pytest.mark.parametrize("cap,document_pairs", [
    (8192, 6200), (16384, 10000), (32768, 19000), (65536, 50000),
])
def test_sampler_preserves_exact_tokens_and_provenance(cap, document_pairs):
    # EOS is chosen outside all genuine token values, including long documents.
    raw = np.concatenate((np.full(document_pairs, 7, dtype=np.uint16),
                          np.array([EOS], dtype=np.uint16)))
    baseline = _sample_with_policy(raw, cap, 2, "none")
    geometric = _sample_with_policy(raw, cap, 2, "geometric")
    fine = _sample_with_policy(raw, cap, 2, "fine")
    original_x, original_y, original_md = baseline
    for x, y, md in (geometric, fine):
        np.testing.assert_array_equal(md["document_starts"], original_md["document_starts"])
        np.testing.assert_array_equal(md["document_ends"], original_md["document_ends"])
        np.testing.assert_array_equal(md["target_source_starts"], original_md["target_source_starts"])
        np.testing.assert_array_equal(md["target_valid_lengths"], original_md["target_valid_lengths"])
        length = original_x.shape[1]
        np.testing.assert_array_equal(x[:, :length], original_x)
        np.testing.assert_array_equal(y[:, :length], original_y)
        np.testing.assert_array_equal(md["target_loss_mask"][:, :length],
                                      original_md["target_loss_mask"])
        assert np.all(x[:, length:] == EOS)
        assert np.all(y[:, length:] == EOS)
        assert np.all(md["target_loss_mask"][:, length:] == 0)
        assert md["target_loss_mask"].sum() == original_md["target_loss_mask"].sum()
    assert original_x.shape[1] <= fine[0].shape[1] <= geometric[0].shape[1]
    assert np.all(dual_resolution_loss_mask(fine[0], EOS, fine[2]["target_loss_mask"])
                  == fine[2]["target_loss_mask"])


def test_fine_bucket_does_not_create_valid_padded_blocks():
    original_length = 260  # geometric width=384, fine width=320
    raw = np.concatenate((np.full(original_length, 11, dtype=np.uint16),
                          np.array([EOS], dtype=np.uint16)))
    original = _sample_with_policy(raw, 512, 1, "none")
    geometric = _sample_with_policy(raw, 512, 1, "geometric")
    fine = _sample_with_policy(raw, 512, 1, "fine")
    assert [case[0].shape[1] for case in (original, fine, geometric)] == [260, 320, 384]
    for x, _, md in (original, fine, geometric):
        dual = build_dual_resolution_metadata(x, EOS, block_size=128)
        dual["block_valid"] &= (dual["block_indices"][:, :, -1]
                                < md["target_valid_lengths"][:, None])
        assert int(dual["block_valid"].sum()) == 2
        assert np.all(md["target_loss_mask"][:, original_length:] == 0)


def test_fine_mode_preserves_accumulation_repacking():
    def make_batch(lengths):
        width = _dual_resolution_fine_length(max(lengths), 8192)
        x = np.full((2, width), EOS, dtype=np.uint16)
        y = x.copy()
        mask = np.zeros((2, width), dtype=np.float32)
        for i, length in enumerate(lengths):
            x[i, :length] = 10 + i
            y[i, :length] = 20 + i
            mask[i, :length] = 1
        return ("fineweb", x, y, {
            "document_sampling_policy": "document_first_full_sequence",
            "document_aware": True,
            "target_valid_lengths": np.array(lengths),
            "target_loss_mask": mask,
            "context_capacity": 8192,
            "shape_bucket": "fine",
            "executed_sequence_length": width,
            "document_starts": np.array([1, 2]),
            "document_ends": np.array([10000, 10000]),
            "target_source_starts": np.array([1, 2]),
        })
    batches = [make_batch([6200, 500]), make_batch([5800, 600])]
    repacked, stats = regroup_dual_resolution_document_batches(
        batches, batch_size=2, context_length=8192, eos_token_id=EOS)
    assert stats["padding_saved_rows"] > 0
    assert len(repacked) == len(batches)
    before = sorted((int(n), int(x[i, 0]), int(y[i, 0]))
                    for _, x, y, md in batches
                    for i, n in enumerate(md["target_valid_lengths"]))
    after = sorted((int(n), int(x[i, 0]), int(y[i, 0]))
                   for _, x, y, md in repacked
                   for i, n in enumerate(md["target_valid_lengths"]))
    assert before == after
    for _, x, y, md in repacked:
        assert x.shape == y.shape == md["target_loss_mask"].shape
        assert md["executed_sequence_length"] == _dual_resolution_fine_length(
            max(md["target_valid_lengths"]), 8192)
        for i, n in enumerate(md["target_valid_lengths"]):
            assert np.all(md["target_loss_mask"][i, n:] == 0)
            assert np.all(x[i, n:] == EOS)


@pytest.mark.skipif(__import__('mini_llm.backend', fromlist=['BACKEND_NAME']).BACKEND_NAME
                    != 'numpy', reason='NumPy matrix-pool padding/gradient parity')
def test_matrix_pool_sparse_attention_padding_and_gradient_parity(monkeypatch):
    """Compare two execution widths using the *same* 260 genuine positions.

    Exercises local heads, stride-16 history (phase 5), completed-block
    summaries, global attention, and shared parameter/backward gradients.
    """
    from dataclasses import replace

    from mini_llm.backend import RandomStream
    from mini_llm.config import ModelConfig
    from mini_llm.ops.matrix_pool_dual_resolution_attention import (
        MatrixPoolDualResolutionAttention,
    )

    monkeypatch.setenv('MINI_LLM_DUAL_RESOLUTION_LOCAL_KERNEL', 'cupy')
    monkeypatch.setenv('MINI_LLM_DUAL_RESOLUTION_GLOBAL_KERNEL', 'cupy')
    monkeypatch.setenv('MINI_LLM_DUAL_RESOLUTION_FAST_PROJECTIONS', '0')
    base = ModelConfig.dual_resolution_matrix_pool_sparse_1024_20l().dual_resolution
    config = replace(base, block_size=16, local_window=32,
                     long_local_window=64, max_blocks=24,
                     global_query_chunk=32, global_head_dim=64,
                     sparse_retrieval_phase=5)
    operator = MatrixPoolDualResolutionAttention(
        d_model=256, n_q_heads=4, n_kv_heads=2, d_head=64,
        input_std=.05, output_std=.05, rng=RandomStream(100),
        dtype='float64', config=config,
    )
    assert operator.sparse_enabled
    rng = np.random.default_rng(4)
    valid = 260
    x = rng.normal(scale=.01, size=(1, valid, 256))
    residual = rng.normal(scale=.01, size=x.shape)
    upstream = rng.normal(scale=.01, size=x.shape)

    outputs = []
    for width in (320, 384):  # fine vs original geometric
        tokens = np.concatenate((np.ones(valid, dtype=np.int32),
                                 np.full(width-valid, 99, dtype=np.int32)))[None, :]
        metadata = build_dual_resolution_metadata(tokens, eos_token_id=99,
                                                   block_size=16)
        metadata['block_valid'] &= (metadata['block_indices'][:, :, -1]
                                    < valid)
        padding = ((0, 0), (0, width-valid), (0, 0))
        operator.zero_grad()
        result, cache = operator.forward(
            np.pad(x, padding), residual_input=np.pad(residual, padding),
            dual_resolution_metadata=metadata,
        )
        dx, dresidual = operator.backward(np.pad(upstream, padding), cache)
        outputs.append((result[:, :valid].copy(), dx[:, :valid].copy(),
                        dresidual[:, :valid].copy(),
                        [parameter.grad.copy() for parameter in operator.parameters()]))

    for fine, geometric in zip(outputs[0][:3], outputs[1][:3]):
        np.testing.assert_allclose(fine, geometric, rtol=1e-12, atol=1e-12)
    for fine, geometric in zip(outputs[0][3], outputs[1][3]):
        np.testing.assert_allclose(fine, geometric, rtol=1e-12, atol=1e-12)


@pytest.mark.skipif(__import__('mini_llm.backend', fromlist=['BACKEND_NAME']).BACKEND_NAME
                    != 'numpy', reason='CPU trainer acceptance test')
def test_trainer_accepts_fine_shape_bucket_from_environment(monkeypatch, tmp_path):
    from mini_llm.config import ModelConfig, DualResolutionConfig
    from mini_llm.model.decoder_lm import DecoderLanguageModel
    from mini_llm.train_extended import ExtendedTrainer

    monkeypatch.setenv('MINI_LLM_DUAL_RESOLUTION_SHAPE_BUCKET', 'fine')
    data = np.concatenate((np.full(260, 7, dtype=np.uint16),
                           np.array([99], dtype=np.uint16)))
    path = tmp_path / 'train.bin'
    data.tofile(path)
    config = ModelConfig(
        tokenizer_vocab_size=128, context_length=384, n_layers=2,
        d_model=16, d_ff=32, n_q_heads=4, n_kv_heads=2, d_head=4,
        ffn_mode='dense', n_experts=1, top_k=1, dtype='float64',
        dual_resolution=DualResolutionConfig(
            enabled=True, block_size=16, local_window=32,
            compression_dim=8, compression_heads=2, max_blocks=24,
            global_query_chunk=16,
        ),
    )
    trainer = ExtendedTrainer(
        model=DecoderLanguageModel(config, rng_seed=12),
        train_shard_paths=[str(path)], val_shard_paths=[str(path)],
        batch_size=1, seq_length=384, grad_accum_steps=1,
        total_steps=2, warmup_steps=1, val_steps=1,
        val_interval=100, save_interval=100, eos_token_id=99,
        checkpoint_dir=str(tmp_path / 'ckpt'),
    )
    inputs, targets, metadata = trainer.get_train_batch(return_metadata=True)
    assert trainer.dual_shape_bucket == 'fine'
    assert inputs.shape == targets.shape == (1, 320)
    assert int(metadata['target_valid_lengths'][0]) == 260
    assert int(metadata['target_loss_mask'].sum()) == 260
    assert not np.any(metadata['target_loss_mask'][:, 260:])
    loss, grad_norm = trainer.train_step()
    assert np.isfinite(loss) and np.isfinite(grad_norm)
    assert trainer._last_step_supervised_tokens == 260
    assert trainer._last_step_executed_tokens == 320
