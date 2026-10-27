"""GPU oracle tests for the native 0100B rectangular Triton attention."""

import importlib.util
import math
from types import SimpleNamespace

import numpy as np
import pytest


pytestmark = [
    pytest.mark.gpu,
    pytest.mark.skipif(
        importlib.util.find_spec("triton") is None,
        reason="requires the triton-attention optional dependency",
    ),
]


def _reference(q, k, v, upstream):
    q_len, k_len = q.shape[0], k.shape[0]
    history = k_len - q_len
    allow = np.arange(k_len)[None, :] <= history + np.arange(q_len)[:, None]
    scale = 1.0 / math.sqrt(q.shape[1])
    scores = q @ k.T * scale
    scores = np.where(allow, scores, -np.inf)
    shifted = scores - np.max(scores, axis=-1, keepdims=True)
    probabilities = np.where(allow, np.exp(shifted), 0.0)
    probabilities /= probabilities.sum(axis=-1, keepdims=True)
    output = probabilities @ v
    dv = probabilities.T @ upstream
    dp = upstream @ v.T
    ds = probabilities * (dp - np.sum(dp * probabilities, axis=-1, keepdims=True))
    ds = np.where(allow, ds * scale, 0.0)
    return output, ds @ k, ds.T @ q, dv


@pytest.mark.parametrize("query_length,history_length", [(1, 0), (8, 0), (8, 5), (33, 17)])
def test_native_rectangular_forward_backward_matches_reference(query_length, history_length):
    import cupy as cp
    from mini_llm.ops.triton_full_history_attention import (
        full_history_attention_backward,
        full_history_attention_forward,
    )

    rng = np.random.default_rng(10020 + query_length + history_length)
    q = rng.normal(0, .2, (query_length, 64)).astype(np.float32)
    k = rng.normal(0, .2, (query_length + history_length, 64)).astype(np.float32)
    v = rng.normal(0, .2, k.shape).astype(np.float32)
    upstream = rng.normal(0, .2, q.shape).astype(np.float32)
    expected = _reference(q, k, v, upstream)
    output, cache = full_history_attention_forward(
        cp.asarray(q), cp.asarray(k), cp.asarray(v), return_cache=True,
    )
    gradients = full_history_attention_backward(cp.asarray(upstream), cache)
    cp.cuda.get_current_stream().synchronize()
    actual = (cp.asnumpy(output), *(cp.asnumpy(value) for value in gradients))
    for observed, wanted in zip(actual, expected):
        np.testing.assert_allclose(observed, wanted, rtol=3e-4, atol=3e-5)


def test_native_bf16_forward_backward_matches_rounded_reference():
    import cupy as cp
    from mini_llm.ops.triton_full_history_attention import (
        full_history_attention_backward,
        full_history_attention_forward,
    )

    rng = np.random.default_rng(10064)
    q_host = rng.normal(0, .2, (8, 64)).astype(np.float32)
    k_host = rng.normal(0, .2, (13, 64)).astype(np.float32)
    v_host = rng.normal(0, .2, (13, 64)).astype(np.float32)
    upstream = rng.normal(0, .2, q_host.shape).astype(np.float32)
    q = cp.asarray(q_host).astype(cp.dtype("bfloat16"))
    k = cp.asarray(k_host).astype(cp.dtype("bfloat16"))
    v = cp.asarray(v_host).astype(cp.dtype("bfloat16"))
    rounded = tuple(cp.asnumpy(value.astype(cp.float32)) for value in (q, k, v))
    expected = _reference(*rounded, upstream)
    output, cache = full_history_attention_forward(q, k, v, return_cache=True)
    gradients = full_history_attention_backward(cp.asarray(upstream), cache)
    cp.cuda.get_current_stream().synchronize()
    actual = (cp.asnumpy(output), *(cp.asnumpy(value) for value in gradients))
    for observed, wanted in zip(actual, expected):
        assert np.all(np.isfinite(observed))
        np.testing.assert_allclose(observed, wanted, rtol=1e-2, atol=2e-3)


def test_full_history_head_cupy_forward_backward_without_torch():
    import cupy as cp
    from mini_llm.backend import RandomStream
    from mini_llm.ops.full_history_head import FullHistoryHead
    from mini_llm.parameter import Parameter

    rng = RandomStream(10065)
    embedding = Parameter(
        rng.normal((64, 64), std=.02, dtype="bfloat16"), name="embedding.W"
    )
    head = FullHistoryHead(
        d_model=64, d_head=64, embedding=embedding, rng=rng,
        init_std=.02, rope_base=10000, name="gpu.full_history", dtype="bfloat16",
    )
    head.o.data[...] = rng.normal(head.o.data.shape, std=.01, dtype="bfloat16")
    current = rng.normal((1, 8, 64), std=.1, dtype="bfloat16")
    positions = cp.arange(8, dtype=cp.int64)[None]
    output, cache = head.forward(current, None, positions, return_cache=True)
    dx = head.backward(cp.ones_like(output), cache)
    cp.cuda.get_current_stream().synchronize()
    assert output.shape == current.shape
    assert dx.shape == current.shape
    assert bool(cp.all(cp.isfinite(output)))
    assert bool(cp.all(cp.isfinite(dx)))
    assert float(cp.linalg.norm(head.q.grad)) > 0


def test_full_history_head_cupy_mixed_history_and_current_lengths():
    import cupy as cp
    from mini_llm.backend import RandomStream
    from mini_llm.ops.full_history_head import FullHistoryHead
    from mini_llm.parameter import Parameter

    rng = RandomStream(10066)
    embedding = Parameter(
        rng.normal((128, 64), std=.02, dtype="bfloat16"), name="embedding.W"
    )
    head = FullHistoryHead(
        d_model=64, d_head=64, embedding=embedding, rng=rng,
        init_std=.02, rope_base=10000, name="ragged.full_history", dtype="bfloat16",
    )
    head.o.data[...] = rng.normal(head.o.data.shape, std=.01, dtype="bfloat16")
    history_ids = cp.arange(96, dtype=cp.int32).reshape(3, 32) % 128
    valid = cp.zeros((3, 32), dtype=cp.bool_)
    valid[1, -8:] = True
    valid[2, :] = True
    history = SimpleNamespace(
        ids=history_ids, valid=valid, positions=cp.arange(32, dtype=cp.int64),
    )
    current = rng.normal((3, 8, 64), std=.1, dtype="bfloat16")
    positions = cp.broadcast_to(cp.arange(32, 40, dtype=cp.int64)[None], (3, 8))
    current_valid = cp.asarray([
        [True] * 8,
        [True] * 5 + [False] * 3,
        [False] * 8,
    ])
    output, cache = head.forward(
        current, history, positions, current_valid_mask=current_valid,
        return_cache=True,
    )
    dx = head.backward(cp.ones_like(output), cache)
    cp.cuda.get_current_stream().synchronize()
    assert bool(cp.all(cp.isfinite(output)))
    assert bool(cp.all(cp.isfinite(dx)))
    assert bool(cp.all(output[1, 5:] == 0))
    assert bool(cp.all(output[2] == 0))
    assert bool(cp.all(dx[1, 5:] == 0))
    assert bool(cp.all(dx[2] == 0))


def test_native_future_current_key_is_invisible():
    import cupy as cp
    from mini_llm.ops.triton_full_history_attention import full_history_attention_forward

    q = cp.zeros((4, 64), dtype=cp.float32)
    k = cp.zeros((7, 64), dtype=cp.float32)  # three history + four current
    v = cp.zeros((7, 64), dtype=cp.float32)
    v[3, 0] = 1.0
    v[4:, 0] = 1000.0
    output = full_history_attention_forward(q, k, v)
    cp.cuda.get_current_stream().synchronize()
    np.testing.assert_allclose(cp.asnumpy(output[0, 0]), .25, rtol=1e-5, atol=1e-6)
