import math
import numpy as np

from mini_llm.backend import xp
from mini_llm.ops.attention_selection import (
    KeySelectionPlan,
    build_block_retrieval_plan,
)
from mini_llm.ops.indexed_attention import (
    indexed_attention_forward,
    indexed_attention_backward,
)


def _causal_full_plan(batch, n_heads, seq_len):
    keys = xp.broadcast_to(
        xp.arange(seq_len, dtype=xp.int64)[None, None, None, :],
        (batch, n_heads, seq_len, seq_len),
    ).copy()
    qpos = xp.arange(seq_len, dtype=xp.int64)[None, None, :, None]
    kpos = xp.arange(seq_len, dtype=xp.int64)[None, None, None, :]
    valid = xp.broadcast_to(kpos <= qpos, keys.shape).copy()
    return KeySelectionPlan(keys, valid)


def _dense_reference(q, k, v, kv_head_indices):
    q_np = np.asarray(q)
    k_np = np.asarray(k)
    v_np = np.asarray(v)
    batch, seq_len, n_q_heads, d_head = q_np.shape
    out = np.zeros_like(q_np)
    scale = 1.0 / math.sqrt(d_head)
    for b in range(batch):
        for h in range(n_q_heads):
            kh = int(kv_head_indices[h])
            for t in range(seq_len):
                logits = np.array([
                    np.dot(q_np[b, t, h], k_np[b, s, kh]) * scale
                    for s in range(t + 1)
                ])
                logits -= logits.max()
                p = np.exp(logits)
                p /= p.sum()
                out[b, t, h] = p @ v_np[b, : t + 1, kh]
    return out


def test_indexed_attention_matches_dense_causal_reference_with_gqa():
    rng = np.random.default_rng(100)
    q = xp.asarray(rng.normal(size=(2, 5, 4, 3)), dtype="float64")
    k = xp.asarray(rng.normal(size=(2, 5, 2, 3)), dtype="float64")
    v = xp.asarray(rng.normal(size=(2, 5, 2, 3)), dtype="float64")
    plan = _causal_full_plan(batch=2, n_heads=4, seq_len=5)
    kv_map = xp.asarray([0, 0, 1, 1], dtype=xp.int64)

    out, _ = indexed_attention_forward(q, k, v, plan, kv_head_indices=kv_map)
    ref = _dense_reference(q, k, v, np.asarray(kv_map))
    np.testing.assert_allclose(np.asarray(out), ref, rtol=2e-12, atol=2e-12)


def test_indexed_attention_supports_noncontiguous_query_to_kv_mapping():
    rng = np.random.default_rng(101)
    q = xp.asarray(rng.normal(size=(1, 4, 2, 3)), dtype="float64")
    k = xp.asarray(rng.normal(size=(1, 4, 2, 3)), dtype="float64")
    v = xp.asarray(rng.normal(size=(1, 4, 2, 3)), dtype="float64")
    plan = _causal_full_plan(batch=1, n_heads=2, seq_len=4)
    kv_map = xp.asarray([1, 0], dtype=xp.int64)

    out, _ = indexed_attention_forward(q, k, v, plan, kv_head_indices=kv_map)
    ref = _dense_reference(q, k, v, np.asarray(kv_map))
    np.testing.assert_allclose(np.asarray(out), ref, rtol=2e-12, atol=2e-12)


def test_indexed_attention_all_invalid_rows_are_exact_zero_and_finite():
    rng = np.random.default_rng(102)
    q = xp.asarray(rng.normal(size=(1, 3, 2, 4)), dtype="float64")
    k = xp.asarray(rng.normal(size=(1, 5, 1, 4)), dtype="float64")
    v = xp.asarray(rng.normal(size=(1, 5, 1, 4)), dtype="float64")
    indices = xp.zeros((1, 1, 3, 2), dtype=xp.int64)
    valid = xp.zeros(indices.shape, dtype=bool)
    plan = KeySelectionPlan(indices, valid)

    out, cache = indexed_attention_forward(q, k, v, plan)
    assert bool(xp.all(xp.isfinite(out)))
    np.testing.assert_array_equal(np.asarray(out), np.zeros_like(np.asarray(out)))
    np.testing.assert_array_equal(
        np.asarray(cache["probs"]), np.zeros_like(np.asarray(cache["probs"]))
    )

    dq, dk, dv, dbias = indexed_attention_backward(xp.ones_like(out), cache)
    np.testing.assert_array_equal(np.asarray(dq), np.zeros_like(np.asarray(dq)))
    np.testing.assert_array_equal(np.asarray(dk), np.zeros_like(np.asarray(dk)))
    np.testing.assert_array_equal(np.asarray(dv), np.zeros_like(np.asarray(dv)))
    assert dbias is None


def test_indexed_attention_qkv_directional_derivative_with_repeated_keys():
    rng = np.random.default_rng(103)
    q = xp.asarray(rng.normal(size=(1, 3, 2, 3)), dtype="float64")
    k = xp.asarray(rng.normal(size=(1, 4, 1, 3)), dtype="float64")
    v = xp.asarray(rng.normal(size=(1, 4, 1, 3)), dtype="float64")
    # Repeated key 1 deliberately exercises scatter-add in dk/dv.
    indices = xp.asarray(
        [[[[0, 1, 1], [0, 1, 2], [1, 2, 3]],
          [[0, 1, 1], [0, 1, 2], [1, 2, 3]]]],
        dtype=xp.int64,
    )
    valid = xp.ones(indices.shape, dtype=bool)
    plan = KeySelectionPlan(indices, valid)
    coeff = xp.asarray(rng.normal(size=q.shape), dtype="float64")

    out, cache = indexed_attention_forward(q, k, v, plan)
    dq, dk, dv, _ = indexed_attention_backward(coeff, cache)

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
        y, _ = indexed_attention_forward(qx, kx, vx, plan)
        return float(xp.sum(y * coeff))

    eps = 1e-6
    plus = objective(q + eps * dq_dir, k + eps * dk_dir, v + eps * dv_dir)
    minus = objective(q - eps * dq_dir, k - eps * dk_dir, v - eps * dv_dir)
    finite_difference = (plus - minus) / (2.0 * eps)
    np.testing.assert_allclose(
        analytical, finite_difference, rtol=3e-6, atol=1e-8
    )


def test_indexed_attention_logit_bias_gradient():
    rng = np.random.default_rng(104)
    q = xp.asarray(rng.normal(size=(1, 3, 2, 2)), dtype="float64")
    k = xp.asarray(rng.normal(size=(1, 4, 1, 2)), dtype="float64")
    v = xp.asarray(rng.normal(size=(1, 4, 1, 2)), dtype="float64")
    indices = xp.asarray(
        [[[[0, 1], [0, 2], [1, 3]],
          [[0, 1], [0, 2], [1, 3]]]],
        dtype=xp.int64,
    )
    valid = xp.ones(indices.shape, dtype=bool)
    bias = xp.asarray(rng.normal(size=indices.shape), dtype="float64") * 0.1
    plan = KeySelectionPlan(indices, valid, logit_bias=bias)
    coeff = xp.asarray(rng.normal(size=q.shape), dtype="float64")

    out, cache = indexed_attention_forward(q, k, v, plan)
    _, _, _, dbias = indexed_attention_backward(coeff, cache)

    direction = xp.asarray(rng.normal(size=bias.shape), dtype="float64")
    direction /= xp.sqrt(xp.sum(direction * direction))
    analytical = float(xp.sum(dbias * direction))

    def objective(b):
        y, _ = indexed_attention_forward(
            q, k, v, KeySelectionPlan(indices, valid, logit_bias=b)
        )
        return float(xp.sum(y * coeff))

    eps = 1e-6
    finite_difference = (
        objective(bias + eps * direction) - objective(bias - eps * direction)
    ) / (2.0 * eps)
    np.testing.assert_allclose(
        analytical, finite_difference, rtol=3e-6, atol=1e-8
    )


def test_manual_block_retrieval_plan_drives_full_resolution_attention():
    rng = np.random.default_rng(105)
    q = xp.asarray(rng.normal(size=(1, 12, 1, 3)), dtype="float64")
    k = xp.asarray(rng.normal(size=(1, 12, 1, 3)), dtype="float64")
    v = xp.asarray(rng.normal(size=(1, 12, 1, 3)), dtype="float64")

    selected = xp.asarray([[[0], [2]]], dtype=xp.int64)
    starts = xp.asarray([4, 8], dtype=xp.int64)
    plan = build_block_retrieval_plan(
        selected,
        starts,
        seq_len=12,
        block_size=2,
        routing_stride=4,
    )
    out, cache = indexed_attention_forward(q, k, v, plan)

    # No retrieval route exists before token 4, hence the retrieval head is 0.
    np.testing.assert_array_equal(
        np.asarray(out[:, :4]), np.zeros_like(np.asarray(out[:, :4]))
    )

    # Tokens 4..7 can only see full-resolution tokens from block 0: [0,1].
    scale = 1.0 / math.sqrt(3)
    q4 = np.asarray(q[0, 4, 0])
    kk = np.asarray(k[0, 0:2, 0])
    vv = np.asarray(v[0, 0:2, 0])
    logits = (kk @ q4) * scale
    p = np.exp(logits - logits.max())
    p /= p.sum()
    np.testing.assert_allclose(np.asarray(out[0, 4, 0]), p @ vv, rtol=1e-12)

    # Backward should remain finite with the unrouted zero rows present.
    dq, dk, dv, _ = indexed_attention_backward(xp.ones_like(out), cache)
    assert bool(xp.all(xp.isfinite(dq)))
    assert bool(xp.all(xp.isfinite(dk)))
    assert bool(xp.all(xp.isfinite(dv)))
