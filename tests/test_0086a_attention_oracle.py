"""0086A mathematical oracle and existing local-attention baseline tests."""

import numpy as np
import pytest

from benchmarks.reference_fused_attention_oracle import (
    backward_reference,
    forward_reference,
)
from mini_llm.backend import BACKEND_NAME, xp
from mini_llm.ops.indexed_attention import (
    local_window_attention_backward,
    local_window_attention_forward,
)


def _inputs(seed=8600, batch=2, length=9, n_q_heads=8, n_kv_heads=2, d_head=8):
    rng = np.random.default_rng(seed)
    q = rng.normal(0.0, 0.2, (batch, length, n_q_heads, d_head))
    k = rng.normal(0.0, 0.2, (batch, length, n_kv_heads, d_head))
    v = rng.normal(0.0, 0.2, (batch, length, n_kv_heads, d_head))
    d_out = rng.normal(0.0, 0.2, q.shape)
    return q, k, v, d_out


def test_oracle_qkv_central_differences_and_shared_kv_accumulation():
    q, k, v, d_out = _inputs()
    mapping = tuple(h // 4 for h in range(q.shape[2]))
    windows = (q.shape[1],) + (3,) * (q.shape[2] - 1)
    gradients = backward_reference(q, k, v, d_out, mapping, windows)

    checks = (
        (q, gradients[0], (0, 6, 3, 2)),
        (k, gradients[1], (1, 4, 0, 1)),
        (v, gradients[2], (1, 4, 1, 3)),
    )
    for tensor, analytical, index in checks:
        estimates = []
        original = tensor[index]
        for epsilon in (1e-4, 1e-5, 1e-6):
            tensor[index] = original + epsilon
            plus = np.sum(forward_reference(q, k, v, mapping, windows)[0] * d_out)
            tensor[index] = original - epsilon
            minus = np.sum(forward_reference(q, k, v, mapping, windows)[0] * d_out)
            estimates.append((plus - minus) / (2.0 * epsilon))
        tensor[index] = original
        errors = np.abs(np.asarray(estimates) - analytical[index])
        assert float(np.min(errors)) < 1e-8
        np.testing.assert_allclose(estimates[1], analytical[index], rtol=1e-5, atol=1e-7)


def test_oracle_window_boundaries_future_mask_and_window_one_gradients():
    q, k, v, d_out = _inputs(seed=8610, batch=1, length=7, n_q_heads=3, n_kv_heads=2)
    mapping = (0, 0, 1)
    windows = (3, 3, 1)
    output, _ = forward_reference(q, k, v, mapping, windows)
    query = 4

    outside = v.copy()
    outside[0, 1, 0] += 50.0  # query 4 with W=3 begins at key 2
    outside_output, _ = forward_reference(q, k, outside, mapping, windows)
    np.testing.assert_array_equal(outside_output[0, query, :2], output[0, query, :2])

    future = k.copy()
    future[0, 5, 0] += 50.0
    future_output, _ = forward_reference(q, future, v, mapping, windows)
    np.testing.assert_array_equal(future_output[0, query, :2], output[0, query, :2])

    boundary = v.copy()
    boundary[0, 2, 0] += 50.0  # exact i-W+1 boundary is valid
    boundary_output, _ = forward_reference(q, k, boundary, mapping, windows)
    assert np.max(np.abs(boundary_output[0, query, :2] - output[0, query, :2])) > 0.0

    np.testing.assert_allclose(output[:, :, 2], v[:, :, 1], rtol=0.0, atol=0.0)
    d_q, d_k, _ = backward_reference(q, k, v, d_out, mapping, windows, active_heads=(2,))
    np.testing.assert_allclose(d_q[:, :, 2], 0.0, rtol=0.0, atol=1e-16)
    np.testing.assert_allclose(d_k, 0.0, rtol=0.0, atol=1e-16)


@pytest.mark.skipif(BACKEND_NAME != "numpy", reason="CPU public-path comparison")
@pytest.mark.parametrize(
    "dtype,rtol,atol",
    (("float64", 5e-12, 5e-12), ("float32", 2e-5, 2e-6)),
)
def test_existing_public_path_matches_oracle_for_original_head_subset(dtype, rtol, atol):
    q_all, k, v, d_out_all = _inputs(
        seed=8620, batch=2, length=17, n_q_heads=16, n_kv_heads=4, d_head=8
    )
    original_heads = (1, 2, 3, 8, 12, 15)
    q = q_all[:, :, original_heads]
    d_out = d_out_all[:, :, original_heads]
    mapping = tuple(original_head // 4 for original_head in original_heads)
    assert mapping == (0, 0, 0, 2, 3, 3)
    windows = (3,) * len(original_heads)

    expected_out, _ = forward_reference(q, k, v, mapping, windows)
    expected_grads = backward_reference(q, k, v, d_out, mapping, windows)
    q_xp, k_xp, v_xp, d_out_xp = (
        xp.asarray(array, dtype=dtype) for array in (q, k, v, d_out)
    )
    actual_out, cache = local_window_attention_forward(
        q_xp,
        k_xp,
        v_xp,
        3,
        kv_head_indices=mapping,
        query_chunk_size=5,
        return_cache=True,
    )
    actual_grads = local_window_attention_backward(d_out_xp, cache)

    np.testing.assert_allclose(np.asarray(actual_out), expected_out, rtol=rtol, atol=atol)
    for actual, expected in zip(actual_grads, expected_grads):
        np.testing.assert_allclose(np.asarray(actual), expected, rtol=rtol * 2, atol=atol * 2)


def test_oracle_inactive_head_is_zero_and_lse_is_negative_infinity():
    q, k, v, d_out = _inputs(seed=8630, batch=1, length=5, n_q_heads=4, n_kv_heads=2)
    mapping = (0, 0, 1, 1)
    windows = (5, 3, 3, 3)
    active = (0, 1, 3)
    output, lse = forward_reference(q, k, v, mapping, windows, active_heads=active)
    gradients = backward_reference(
        q, k, v, d_out, mapping, windows, active_heads=active
    )
    assert np.all(output[:, :, 2] == 0.0)
    assert np.all(np.isneginf(lse[:, :, 2]))
    assert np.all(gradients[0][:, :, 2] == 0.0)
