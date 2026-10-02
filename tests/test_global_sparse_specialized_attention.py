import numpy as np

from mini_llm.backend import xp
from mini_llm.ops.attention_selection import build_global_sparse_causal_plan
from mini_llm.ops.indexed_attention import (
    indexed_attention_forward,
    indexed_attention_backward,
    global_sparse_attention_forward,
    global_sparse_attention_backward,
)


def _case(seed=900, seq_len=13, n_q_heads=4, n_kv_heads=2, d_head=3):
    rng = np.random.default_rng(seed)
    q = xp.asarray(rng.normal(size=(1, seq_len, n_q_heads, d_head)), dtype="float64")
    k = xp.asarray(rng.normal(size=(1, seq_len, n_kv_heads, d_head)), dtype="float64")
    v = xp.asarray(rng.normal(size=(1, seq_len, n_kv_heads, d_head)), dtype="float64")
    coeff = xp.asarray(rng.normal(size=q.shape), dtype="float64")
    return rng, q, k, v, coeff


def _compare_to_indexed(stride, offset, include_current):
    _, q, k, v, coeff = _case(seed=901 + stride + offset)
    kv_map = (0, 0, 1, 1)
    plan = build_global_sparse_causal_plan(
        q.shape[0], q.shape[1], stride, offset, include_current
    )
    ref, ref_cache = indexed_attention_forward(
        q, k, v, plan, kv_head_indices=xp.asarray(kv_map, dtype=xp.int64)
    )
    got, got_cache = global_sparse_attention_forward(
        q,
        k,
        v,
        stride,
        offset,
        include_current,
        kv_head_indices=kv_map,
    )
    np.testing.assert_allclose(np.asarray(got), np.asarray(ref), rtol=2e-12, atol=2e-12)

    ref_grads = indexed_attention_backward(coeff, ref_cache)[:3]
    got_grads = global_sparse_attention_backward(coeff, got_cache)
    for got_grad, ref_grad in zip(got_grads, ref_grads):
        np.testing.assert_allclose(
            np.asarray(got_grad), np.asarray(ref_grad), rtol=3e-12, atol=3e-12
        )


def test_global_sparse_specialized_matches_indexed_with_current():
    _compare_to_indexed(stride=4, offset=1, include_current=True)


def test_global_sparse_specialized_matches_indexed_without_current():
    _compare_to_indexed(stride=5, offset=3, include_current=False)


def test_global_sparse_specialized_stride_one_matches_dense_indexed():
    _compare_to_indexed(stride=1, offset=0, include_current=True)


def test_global_sparse_specialized_handles_no_early_anchor_without_current():
    rng = np.random.default_rng(906)
    q = xp.asarray(rng.normal(size=(1, 3, 2, 2)), dtype="float64")
    k = xp.asarray(rng.normal(size=(1, 3, 1, 2)), dtype="float64")
    v = xp.asarray(rng.normal(size=(1, 3, 1, 2)), dtype="float64")
    out, cache = global_sparse_attention_forward(
        q, k, v, stride=8, offset=6, include_current=False
    )
    np.testing.assert_array_equal(np.asarray(out), np.zeros_like(np.asarray(out)))
    dq, dk, dv = global_sparse_attention_backward(xp.ones_like(out), cache)
    np.testing.assert_array_equal(np.asarray(dq), np.zeros_like(np.asarray(dq)))
    np.testing.assert_array_equal(np.asarray(dk), np.zeros_like(np.asarray(dk)))
    np.testing.assert_array_equal(np.asarray(dv), np.zeros_like(np.asarray(dv)))


def test_global_sparse_specialized_qkv_directional_derivative():
    rng, q, k, v, coeff = _case(seed=907, seq_len=11)
    out, cache = global_sparse_attention_forward(
        q,
        k,
        v,
        stride=3,
        offset=1,
        include_current=True,
        kv_head_indices=(0, 0, 1, 1),
    )
    dq, dk, dv = global_sparse_attention_backward(coeff, cache)

    dq_dir = xp.asarray(rng.normal(size=q.shape), dtype="float64")
    dk_dir = xp.asarray(rng.normal(size=k.shape), dtype="float64")
    dv_dir = xp.asarray(rng.normal(size=v.shape), dtype="float64")
    norm = xp.sqrt(
        xp.sum(dq_dir * dq_dir) + xp.sum(dk_dir * dk_dir) + xp.sum(dv_dir * dv_dir)
    )
    dq_dir /= norm
    dk_dir /= norm
    dv_dir /= norm
    analytical = float(
        xp.sum(dq * dq_dir) + xp.sum(dk * dk_dir) + xp.sum(dv * dv_dir)
    )

    def objective(qx, kx, vx):
        y = global_sparse_attention_forward(
            qx,
            kx,
            vx,
            stride=3,
            offset=1,
            include_current=True,
            kv_head_indices=(0, 0, 1, 1),
            return_cache=False,
        )
        return float(xp.sum(y * coeff))

    eps = 1e-6
    finite_difference = (
        objective(q + eps * dq_dir, k + eps * dk_dir, v + eps * dv_dir)
        - objective(q - eps * dq_dir, k - eps * dk_dir, v - eps * dv_dir)
    ) / (2.0 * eps)
    np.testing.assert_allclose(analytical, finite_difference, rtol=4e-6, atol=1e-8)
