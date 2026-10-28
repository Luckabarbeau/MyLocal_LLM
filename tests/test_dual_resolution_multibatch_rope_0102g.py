"""0102G: unequal document lengths must not feed invalid summary slots into RoPE."""

import numpy as np
import pytest

from mini_llm.backend import BACKEND_NAME, RandomStream
from mini_llm.config import DualResolutionConfig
from mini_llm.ops.dual_resolution_attention import (
    DualResolutionAttention,
    build_dual_resolution_metadata,
)


def _operator(*, backend="cupy"):
    if BACKEND_NAME == "cupy":
        width, q_heads, kv_heads, head_dim = 1024, 16, 4, 64
        block_size, compression_dim, compression_heads = 128, 512, 8
        dtype = "bfloat16"
    else:
        width, q_heads, kv_heads, head_dim = 8, 2, 1, 4
        block_size, compression_dim, compression_heads = 4, 8, 2
        dtype = "float64"
    attention = DualResolutionAttention(
        d_model=width, n_q_heads=q_heads, n_kv_heads=kv_heads,
        d_head=head_dim, input_std=0.02, output_std=0.02,
        rng=RandomStream(7), dtype=dtype,
        config=DualResolutionConfig(
            enabled=True, block_size=block_size, local_window=block_size * 2,
            compression_dim=compression_dim, compression_heads=compression_heads,
            compression_depth=1, max_blocks=8, global_query_chunk=block_size,
            recompute_attention_probs=True,
        ),
    )
    return attention, block_size


def _metadata(batch, block_size, eos=99):
    md = build_dual_resolution_metadata(batch, eos, block_size=block_size)
    # A padded row cannot have completed blocks from the padding.
    assert md["block_valid"][0].sum() > md["block_valid"][1].sum()
    assert np.any(md["block_end_positions"][1] == -1)
    return md


def test_masked_summary_rope_positions_preserve_original_visibility():
    attention, block = _operator()
    long = np.arange(1, 3 * block + 2, dtype=np.int64)
    short = np.full(long.shape, 99, dtype=np.int64)
    short[:block + 2] = np.arange(1, block + 3)
    tokens = np.stack([long, short])
    md = _metadata(tokens, block)
    before = md["block_end_positions"].copy()
    prepared = attention._prepare_metadata(md, 2, tokens.shape[1], None)
    safe = np.asarray(prepared["block_rope_positions"])
    np.testing.assert_array_equal(safe[md["block_valid"]], before[md["block_valid"]])
    assert np.all(safe[~md["block_valid"]] == 0)
    np.testing.assert_array_equal(md["block_end_positions"], before)
    assert attention._prepare_metadata(md, 2, tokens.shape[1], None) is prepared


def test_unequal_batches_match_single_row_forward_and_backward():
    attention, block = _operator()
    length = 3 * block + 1
    rng = np.random.default_rng(16)
    x = rng.normal(size=(2, length, attention.d_model)) * 0.1
    tokens = np.full((2, length), 99, dtype=np.int64)
    tokens[0] = np.arange(1, length + 1)
    valid_short = block + 2
    tokens[1, :valid_short] = np.arange(1, valid_short + 1)
    batch_md = _metadata(tokens, block)
    upstream = np.zeros_like(x)
    upstream[1, :valid_short] = rng.normal(size=(valid_short, attention.d_model))
    attention.zero_grad()
    out, cache = attention.forward(x, dual_resolution_metadata=batch_md)
    dx = attention.backward(upstream, cache)
    grads_batch = {p.name: np.array(p.grad, copy=True) for p in attention.parameters()}

    # Compare the genuine short document against its own unpadded execution.
    short_x = x[1:2, :valid_short]
    short_tokens = tokens[1:2, :valid_short]
    short_md = build_dual_resolution_metadata(short_tokens, 99, block_size=block)
    attention.zero_grad()
    short_out, short_cache = attention.forward(short_x, dual_resolution_metadata=short_md)
    short_dx = attention.backward(upstream[1:2, :valid_short], short_cache)
    np.testing.assert_allclose(out[1, :valid_short], short_out[0], rtol=1e-9, atol=1e-9)
    np.testing.assert_allclose(dx[1, :valid_short], short_dx[0], rtol=1e-9, atol=1e-9)
    for parameter in attention.parameters():
        np.testing.assert_allclose(grads_batch[parameter.name], parameter.grad,
                                   atol=1e-9, rtol=1e-9, err_msg=parameter.name)


@pytest.mark.skipif(BACKEND_NAME != "cupy", reason="RTX GPU/Triton smoke test")
def test_batched_bf16_all_triton_forward_backward(monkeypatch):
    pytest.importorskip("triton")
    import cupy as cp
    monkeypatch.setenv("MINI_LLM_DUAL_RESOLUTION_LOCAL_KERNEL", "triton")
    monkeypatch.setenv("MINI_LLM_DUAL_RESOLUTION_GLOBAL_KERNEL", "triton")
    monkeypatch.setenv("MINI_LLM_DUAL_RESOLUTION_COMPRESSION_KERNEL", "triton")
    attention, block = _operator()
    assert (attention.local_kernel, attention.global_kernel,
            attention.compression_kernel) == ("triton", "triton", "triton")
    length = 2 * block + 1
    tokens = np.full((2, length), 99, dtype=np.int64)
    tokens[0] = np.arange(1, length + 1)
    tokens[1, :block + 2] = np.arange(1, block + 3)
    metadata = build_dual_resolution_metadata(tokens, 99, block_size=block)
    np.random.seed(9)
    x = cp.asarray(np.random.randn(2, length, attention.d_model).astype("float32") * .01,
                   dtype=cp.dtype("bfloat16"))
    out, cache = attention.forward(x, dual_resolution_metadata=metadata)
    gradient = cp.ones(out.shape, dtype=out.dtype)
    gradient[1, block + 2:] = 0
    dx = attention.backward(gradient, cache)
    cp.cuda.get_current_stream().synchronize()
    assert bool(cp.all(cp.isfinite(out)))
    assert bool(cp.all(cp.isfinite(dx)))
    assert bool(cp.all(cp.isfinite(attention.W_global_k.grad)))
    assert bool(cp.all(cp.isfinite(attention.W_compress_in.grad)))


def test_tiny_document_first_training_supports_batch_two(tmp_path, monkeypatch):
    """Exercise the actual decoder/trainer path, not just isolated attention."""
    if BACKEND_NAME != "numpy":
        pytest.skip("Tiny CPU reference regression")
    from mini_llm.config import ModelConfig
    from mini_llm.model.decoder_lm import DecoderLanguageModel
    from mini_llm.train_extended import ExtendedTrainer

    monkeypatch.setenv("MINI_LLM_DUAL_RESOLUTION_SHAPE_BUCKET", "geometric")
    documents = [[*range(1, 6), 99], [*range(10, 31), 99]]
    shard = np.array([v for doc in documents for v in doc], dtype=np.uint16)
    shard_path = tmp_path / "tokens.bin"
    shard.tofile(shard_path)
    config = ModelConfig(
        tokenizer_vocab_size=128, context_length=24, n_layers=2,
        d_model=16, d_ff=32, n_q_heads=4, n_kv_heads=2, d_head=4,
        ffn_mode="dense", n_experts=1, top_k=1, dtype="float64",
        dual_resolution=DualResolutionConfig(
            enabled=True, block_size=4, local_window=8,
            compression_dim=8, compression_heads=2, max_blocks=8,
            global_query_chunk=8,
        ),
    )
    trainer = ExtendedTrainer(
        model=DecoderLanguageModel(config, rng_seed=2),
        train_shard_paths=[str(shard_path)], val_shard_paths=[str(shard_path)],
        batch_size=2, seq_length=24, grad_accum_steps=1,
        total_steps=2, warmup_steps=1, val_interval=100, save_interval=100,
        eos_token_id=99, checkpoint_dir=str(tmp_path / "ckpt"),
    )
    # Document-first sampler must allow unequal row lengths and mask padding.
    saw_unequal = False
    for _ in range(12):
        x, y, md = trainer.get_train_batch(return_metadata=True)
        if len(set(map(int, md["target_valid_lengths"]))) > 1:
            saw_unequal = True
            assert np.any(md["dual_resolution_metadata"]["block_end_positions"] < 0)
            break
    assert saw_unequal
    loss, gradient_norm = trainer.train_step()
    assert np.isfinite(loss) and np.isfinite(gradient_norm)
