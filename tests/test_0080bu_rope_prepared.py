"""0080BU: share per-context RoPE factors across deep layers without altering positions."""

import numpy as np
import pytest

from mini_llm.backend import asnumpy, xp, BACKEND_NAME, resolve_dtype
from mini_llm.config import ModelConfig, MemoryContextConfig
from mini_llm.model.decoder_lm import DecoderLanguageModel
from mini_llm.ops.rope import (
    prepare_rope_positions, rope_forward, rope_backward,
)


@pytest.mark.parametrize("batched", [False, True])
def test_prepared_rope_exact_forward_backward_and_reuses_factors(batched):
    rng = np.random.default_rng(80)
    positions_np = np.asarray(
        [[3, 4096, 4097, 65535, 12], [9, 2048, 10, 65534, 33]],
        dtype=np.int64,
    )
    if not batched:
        positions_np = positions_np[0]
    positions = xp.asarray(positions_np)
    x = xp.asarray(rng.normal(size=(2, 5, 3, 8)), dtype="float32")
    dy = xp.asarray(rng.normal(size=x.shape), dtype="float32")

    reference, ref_cache = rope_forward(x, base=1000000.0, position_ids=positions)
    prepared = prepare_rope_positions(
        positions, d_head=8, base=1000000.0, dtype=x.dtype,
    )
    actual, cache = rope_forward(x, base=1000000.0, position_ids=prepared)
    second, second_cache = rope_forward(x, base=1000000.0, position_ids=prepared)

    np.testing.assert_array_equal(asnumpy(actual), asnumpy(reference))
    np.testing.assert_array_equal(asnumpy(second), asnumpy(reference))
    np.testing.assert_array_equal(asnumpy(rope_backward(dy, cache)), asnumpy(rope_backward(dy, ref_cache)))
    assert cache["cos"] is second_cache["cos"] is prepared.cos
    assert cache["sin"] is second_cache["sin"] is prepared.sin


def test_prepared_rope_rejects_bad_ids_or_mismatched_input():
    x = xp.ones((1, 2, 1, 8), dtype="float32")
    with pytest.raises(ValueError, match="non-negative"):
        prepare_rope_positions(xp.asarray([1, -1]), d_head=8)
    with pytest.raises(TypeError, match="integer"):
        prepare_rope_positions(xp.asarray([1., 2.]), d_head=8)
    prepared = prepare_rope_positions(xp.asarray([10, 11]), d_head=8, dtype="float32")
    with pytest.raises(ValueError, match="base/dtype"):
        rope_forward(x, base=1000000, position_ids=prepared)
    with pytest.raises(ValueError, match="shape"):
        rope_forward(xp.ones((1, 3, 1, 8)), position_ids=prepared)


def _tiny_model():
    memory = MemoryContextConfig(
        enabled=True, memory_length=8, recent_length=4, target_length=3,
        block_size=2, top_k_blocks=2, router_query_length=4,
        router_dim=4, query_pooling="mean", history_pooling="mean",
        read_heads=2, read_kv_heads=1, read_query_chunk=2,
    )
    config = ModelConfig(
        tokenizer_vocab_size=32, context_length=memory.active_length,
        n_layers=2, d_model=8, n_q_heads=2, n_kv_heads=1, d_head=4,
        n_experts=2, top_k=1, d_ff=16, dtype="float32",
        memory_context=memory,
    )
    return DecoderLanguageModel(config, rng_seed=101)


@pytest.mark.parametrize("checkpoint", [False, True])
@pytest.mark.parametrize("use_retrieval", [False, True])
def test_deep_memory_trunk_prepared_position_parity(monkeypatch, checkpoint, use_retrieval):
    # Backprop must also match when the position table survives checkpoint
    # replay, not just in the first forward of every layer.
    source = xp.asarray(
        [[1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11],
         [2, 4, 6, 8, 10, 12, 14, 16, 18, 20, 22]], dtype=xp.int64,
    )
    kwargs = dict(
        activation_checkpoint=checkpoint,
        memory_metadata={"use_retrieval": use_retrieval},
        return_cache=True,
    )
    model_old = _tiny_model()
    model_new = _tiny_model()
    monkeypatch.setenv("MINI_LLM_ROPE_PREPARED_POSITIONS", "0")
    hidden_old, cache_old = model_old.forward_body(source, **kwargs)
    monkeypatch.setenv("MINI_LLM_ROPE_PREPARED_POSITIONS", "1")
    hidden_new, cache_new = model_new.forward_body(source, **kwargs)
    np.testing.assert_allclose(asnumpy(hidden_new), asnumpy(hidden_old), rtol=1e-6, atol=1e-6)

    dy = xp.ones_like(hidden_old)
    model_old.backward_body(dy, cache_old)
    model_new.backward_body(dy, cache_new)
    for old, new in zip(model_old.parameters(), model_new.parameters()):
        np.testing.assert_allclose(asnumpy(old.grad), asnumpy(new.grad), rtol=1e-5, atol=1e-6)


@pytest.mark.skipif(BACKEND_NAME != "cupy", reason="BF16 CUDA numerical parity test")
def test_prepared_rope_cuda_bf16_parity():
    rng = np.random.default_rng(81)
    dtype = resolve_dtype("bfloat16")
    positions = xp.asarray(
        [[61440, 61442, 63002, 65535], [2, 512, 4096, 65534]],
        dtype=xp.int64,
    )
    x = xp.asarray(rng.normal(size=(2, 4, 8, 64)), dtype=dtype)
    dy = xp.asarray(rng.normal(size=x.shape), dtype=xp.float32)
    ref, ref_cache = rope_forward(x, 1000000.0, position_ids=positions)
    prepared = prepare_rope_positions(positions, d_head=64, base=1000000.0, dtype=dtype)
    actual, cache = rope_forward(x, 1000000.0, position_ids=prepared)
    np.testing.assert_allclose(
        asnumpy(actual).astype(np.float32), asnumpy(ref).astype(np.float32),
        rtol=0.02, atol=0.02,
    )
    np.testing.assert_allclose(
        asnumpy(rope_backward(dy, cache)), asnumpy(rope_backward(dy, ref_cache)),
        rtol=0.02, atol=0.02,
    )
