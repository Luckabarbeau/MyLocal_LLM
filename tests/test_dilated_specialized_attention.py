import numpy as np

from mini_llm.backend import xp
from mini_llm.ops.attention_selection import build_dilated_causal_plan
from mini_llm.ops.indexed_attention import (
    indexed_attention_backward,
    indexed_attention_forward,
    dilated_attention_backward,
    dilated_attention_forward,
)


def _arrays(seed=1, batch=2, seq=13, hq=4, hkv=2, dh=3, dtype="float64"):
    rng = np.random.default_rng(seed)
    q = xp.asarray(rng.normal(size=(batch, seq, hq, dh)), dtype=dtype)
    k = xp.asarray(rng.normal(size=(batch, seq, hkv, dh)), dtype=dtype)
    v = xp.asarray(rng.normal(size=(batch, seq, hkv, dh)), dtype=dtype)
    return q, k, v


def _compare(window, dilation, offset, chunk=4):
    q, k, v = _arrays(seed=10 + offset + dilation)
    kv_map = (0, 0, 1, 1)
    plan = build_dilated_causal_plan(
        q.shape[0], q.shape[1], window, dilation, offset
    )
    expected, expected_cache = indexed_attention_forward(
        q, k, v, plan, kv_head_indices=kv_map, query_chunk_size=3
    )
    actual, actual_cache = dilated_attention_forward(
        q,
        k,
        v,
        window,
        dilation,
        offset,
        kv_head_indices=kv_map,
        query_chunk_size=chunk,
    )
    np.testing.assert_allclose(
        np.asarray(actual), np.asarray(expected), rtol=3e-12, atol=3e-12
    )

    dout = xp.asarray(
        np.random.default_rng(30 + offset).normal(size=q.shape), dtype="float64"
    )
    edq, edk, edv, _ = indexed_attention_backward(dout, expected_cache)
    adq, adk, adv = dilated_attention_backward(dout, actual_cache)
    np.testing.assert_allclose(np.asarray(adq), np.asarray(edq), rtol=6e-12, atol=6e-12)
    np.testing.assert_allclose(np.asarray(adk), np.asarray(edk), rtol=6e-12, atol=6e-12)
    np.testing.assert_allclose(np.asarray(adv), np.asarray(edv), rtol=6e-12, atol=6e-12)


def test_specialized_dilated_matches_indexed_offset_zero():
    _compare(window=11, dilation=3, offset=0)


def test_specialized_dilated_matches_indexed_nonzero_offset():
    _compare(window=10, dilation=4, offset=2)


def test_specialized_dilated_handles_phase_with_no_initial_key():
    _compare(window=8, dilation=5, offset=4, chunk=2)


def test_specialized_dilated_directional_derivative():
    q, k, v = _arrays(seed=40, batch=1, seq=11, hq=2, hkv=1, dh=2)
    window, dilation, offset = 9, 3, 1
    coeff = xp.asarray(np.random.default_rng(41).normal(size=q.shape), dtype="float64")
    dq_dir = xp.asarray(np.random.default_rng(42).normal(size=q.shape), dtype="float64")
    dk_dir = xp.asarray(np.random.default_rng(43).normal(size=k.shape), dtype="float64")
    dv_dir = xp.asarray(np.random.default_rng(44).normal(size=v.shape), dtype="float64")

    out, cache = dilated_attention_forward(
        q, k, v, window, dilation, offset, query_chunk_size=3
    )
    dq, dk, dv = dilated_attention_backward(coeff, cache)
    analytical = float(
        xp.sum(dq * dq_dir) + xp.sum(dk * dk_dir) + xp.sum(dv * dv_dir)
    )

    def objective(qq, kk, vv):
        yy = dilated_attention_forward(
            qq,
            kk,
            vv,
            window,
            dilation,
            offset,
            query_chunk_size=3,
            return_cache=False,
        )
        return float(xp.sum(yy * coeff))

    eps = 1e-6
    finite = (
        objective(q + eps*dq_dir, k + eps*dk_dir, v + eps*dv_dir)
        - objective(q - eps*dq_dir, k - eps*dk_dir, v - eps*dv_dir)
    ) / (2 * eps)
    np.testing.assert_allclose(analytical, finite, rtol=6e-6, atol=7e-7)
