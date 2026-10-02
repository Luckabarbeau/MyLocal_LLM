import numpy as np

from mini_llm.backend import xp
from mini_llm.ops.attention_selection import build_local_causal_plan
from mini_llm.ops.indexed_attention import (
    indexed_attention_backward,
    indexed_attention_forward,
    local_window_attention_backward,
    local_window_attention_forward,
)


def _arrays(seed=1, batch=2, seq=9, hq=4, hkv=2, dh=3, dtype="float64"):
    rng = np.random.default_rng(seed)
    q = xp.asarray(rng.normal(size=(batch, seq, hq, dh)), dtype=dtype)
    k = xp.asarray(rng.normal(size=(batch, seq, hkv, dh)), dtype=dtype)
    v = xp.asarray(rng.normal(size=(batch, seq, hkv, dh)), dtype=dtype)
    return q, k, v


def test_specialized_local_matches_indexed_forward_and_backward():
    q, k, v = _arrays()
    window = 5
    kv_map = (0, 0, 1, 1)
    plan = build_local_causal_plan(q.shape[0], q.shape[1], window)

    expected, expected_cache = indexed_attention_forward(
        q, k, v, plan, kv_head_indices=kv_map, query_chunk_size=3
    )
    actual, actual_cache = local_window_attention_forward(
        q, k, v, window, kv_head_indices=kv_map, query_chunk_size=3
    )
    np.testing.assert_allclose(np.asarray(actual), np.asarray(expected), rtol=2e-12, atol=2e-12)

    dout = xp.asarray(np.random.default_rng(2).normal(size=q.shape), dtype="float64")
    edq, edk, edv, _ = indexed_attention_backward(dout, expected_cache)
    adq, adk, adv = local_window_attention_backward(dout, actual_cache)
    np.testing.assert_allclose(np.asarray(adq), np.asarray(edq), rtol=4e-12, atol=4e-12)
    np.testing.assert_allclose(np.asarray(adk), np.asarray(edk), rtol=4e-12, atol=4e-12)
    np.testing.assert_allclose(np.asarray(adv), np.asarray(edv), rtol=4e-12, atol=4e-12)


def test_specialized_local_full_window_matches_indexed_dense_local():
    q, k, v = _arrays(seed=3, batch=1, seq=7, hq=2, hkv=1, dh=4)
    window = 64
    plan = build_local_causal_plan(1, 7, window)
    expected, _ = indexed_attention_forward(q, k, v, plan)
    actual, _ = local_window_attention_forward(q, k, v, window, query_chunk_size=2)
    np.testing.assert_allclose(np.asarray(actual), np.asarray(expected), rtol=2e-12, atol=2e-12)


def test_specialized_local_directional_derivative():
    q, k, v = _arrays(seed=4, batch=1, seq=8, hq=2, hkv=1, dh=2)
    window = 4
    coeff = xp.asarray(np.random.default_rng(5).normal(size=q.shape), dtype="float64")
    dq_dir = xp.asarray(np.random.default_rng(6).normal(size=q.shape), dtype="float64")
    dk_dir = xp.asarray(np.random.default_rng(7).normal(size=k.shape), dtype="float64")
    dv_dir = xp.asarray(np.random.default_rng(8).normal(size=v.shape), dtype="float64")

    out, cache = local_window_attention_forward(q, k, v, window, query_chunk_size=3)
    dq, dk, dv = local_window_attention_backward(coeff, cache)
    analytical = float(xp.sum(dq*dq_dir) + xp.sum(dk*dk_dir) + xp.sum(dv*dv_dir))

    def objective(qq, kk, vv):
        yy = local_window_attention_forward(
            qq, kk, vv, window, query_chunk_size=3, return_cache=False
        )
        return float(xp.sum(yy * coeff))

    eps = 1e-6
    finite = (
        objective(q + eps*dq_dir, k + eps*dk_dir, v + eps*dv_dir)
        - objective(q - eps*dq_dir, k - eps*dk_dir, v - eps*dv_dir)
    ) / (2*eps)
    np.testing.assert_allclose(analytical, finite, rtol=5e-6, atol=5e-7)
