"""CPU correctness gates for PATCH 0086C planning and mask semantics."""

from __future__ import annotations

import numpy as np

from benchmarks.cudnn_sdpa_0086c import (
    kv_split_plan,
    probe_cudnn_environment,
    regular16_plan,
    resolve_active_heads,
    zero_mask_local_dout,
)
from benchmarks.reference_fused_attention_oracle import (
    backward_reference,
    forward_reference,
)


def _direct_composite(q, k, v, d_out, active_heads, window):
    output = np.zeros_like(q)
    d_q = np.zeros_like(q)
    full_output, _ = forward_reference(
        q[:, :, 0:1], k, v, (0,), (q.shape[1],)
    )
    full_gradients = backward_reference(
        q[:, :, 0:1], k, v, d_out[:, :, 0:1], (0,), (q.shape[1],)
    )
    output[:, :, 0:1] = full_output
    d_q[:, :, 0:1] = full_gradients[0]

    local_q = q[:, :, active_heads]
    local_d_out = d_out[:, :, active_heads]
    mapping = tuple(head // 4 for head in active_heads)
    local_output, _ = forward_reference(
        local_q, k, v, mapping, (window,) * len(active_heads)
    )
    local_gradients = backward_reference(
        local_q, k, v, local_d_out, mapping, (window,) * len(active_heads)
    )
    output[:, :, active_heads] = local_output
    d_q[:, :, active_heads] = local_gradients[0]
    return (
        output,
        (d_q, full_gradients[1] + local_gradients[1], full_gradients[2] + local_gradients[2]),
    )


def _regular16_composite(q, k, v, d_out, active_heads, window):
    output = np.zeros_like(q)
    d_q = np.zeros_like(q)
    full_output, _ = forward_reference(
        q[:, :, 0:1], k, v, (0,), (q.shape[1],)
    )
    full_gradients = backward_reference(
        q[:, :, 0:1], k, v, d_out[:, :, 0:1], (0,), (q.shape[1],)
    )
    masked_d_out = zero_mask_local_dout(d_out, active_heads, xp=np)
    local_output, _ = forward_reference(
        q, k, v, tuple(head // 4 for head in range(16)), (window,) * 16
    )
    local_gradients = backward_reference(
        q,
        k,
        v,
        masked_d_out,
        tuple(head // 4 for head in range(16)),
        (window,) * 16,
    )
    output[:, :, 0:1] = full_output
    output[:, :, active_heads] = local_output[:, :, active_heads]
    d_q[:, :, 0:1] = full_gradients[0]
    d_q[:, :, active_heads] = local_gradients[0][:, :, active_heads]
    return (
        output,
        (d_q, full_gradients[1] + local_gradients[1], full_gradients[2] + local_gradients[2]),
        local_gradients,
    )


def _split_composite(q, k, v, d_out, active_heads, window):
    plan = kv_split_plan(active_heads)
    output = np.zeros_like(q)
    d_q = np.zeros_like(q)
    d_k = np.zeros_like(k)
    d_v = np.zeros_like(v)
    full_output, _ = forward_reference(
        q[:, :, 0:1], k, v, (0,), (q.shape[1],)
    )
    full_gradients = backward_reference(
        q[:, :, 0:1], k, v, d_out[:, :, 0:1], (0,), (q.shape[1],)
    )
    output[:, :, 0:1] = full_output
    d_q[:, :, 0:1] = full_gradients[0]
    d_k += full_gradients[1]
    d_v += full_gradients[2]
    for call in plan["calls"]:
        kv_head = call["kv_head"]
        heads = tuple(call["original_q_heads"])
        group_output, _ = forward_reference(
            q[:, :, heads],
            k[:, :, kv_head : kv_head + 1],
            v[:, :, kv_head : kv_head + 1],
            (0,) * len(heads),
            (window,) * len(heads),
        )
        group_gradients = backward_reference(
            q[:, :, heads],
            k[:, :, kv_head : kv_head + 1],
            v[:, :, kv_head : kv_head + 1],
            d_out[:, :, heads],
            (0,) * len(heads),
            (window,) * len(heads),
        )
        output[:, :, heads] = group_output
        d_q[:, :, heads] = group_gradients[0]
        d_k[:, :, kv_head : kv_head + 1] += group_gradients[1]
        d_v[:, :, kv_head : kv_head + 1] += group_gradients[2]
    return output, (d_q, d_k, d_v)


def test_head_strategy_plans_preserve_original_gqa_mapping():
    active = resolve_active_heads("13")
    assert active == tuple(range(1, 12)) + (14, 15)
    regular = regular16_plan(active)
    assert regular["kv_map"] == [head // 4 for head in range(16)]
    assert regular["dummy_original_q_heads"] == [0, 12, 13]

    scattered = resolve_active_heads("scattered")
    split = kv_split_plan(scattered)
    assert [call["kv_head"] for call in split["calls"]] == [0, 1, 2, 3]
    assert [call["original_q_heads"] for call in split["calls"]] == [
        [1, 3], [4, 7], [8, 10], [12, 15]
    ]
    assert all(call["relative_kv_map"] == [0, 0] for call in split["calls"])


def test_regular16_dummy_zero_dout_has_zero_dummy_gradients_and_correct_shared_kv_sum():
    rng = np.random.default_rng(86301)
    q = rng.normal(0.0, 0.2, (1, 6, 16, 64))
    k = rng.normal(0.0, 0.2, (1, 6, 4, 64))
    v = rng.normal(0.0, 0.2, (1, 6, 4, 64))
    d_out = rng.normal(0.0, 0.2, q.shape)
    active = resolve_active_heads("13")

    expected = _direct_composite(q, k, v, d_out, active, 4)
    actual_output, actual_gradients, local_gradients = _regular16_composite(
        q, k, v, d_out, active, 4
    )
    inactive = tuple(head for head in range(16) if head not in active)
    np.testing.assert_array_equal(local_gradients[0][:, :, inactive], 0.0)
    np.testing.assert_allclose(actual_output, expected[0], rtol=0.0, atol=0.0)
    for actual, reference in zip(actual_gradients, expected[1]):
        np.testing.assert_allclose(actual, reference, rtol=2e-14, atol=2e-14)


def test_kv_split_scatter_and_full_head_gradient_accumulation_match_direct_oracle():
    rng = np.random.default_rng(86302)
    q = rng.normal(0.0, 0.2, (1, 7, 16, 64))
    k = rng.normal(0.0, 0.2, (1, 7, 4, 64))
    v = rng.normal(0.0, 0.2, (1, 7, 4, 64))
    d_out = rng.normal(0.0, 0.2, q.shape)
    active = resolve_active_heads("scattered")

    expected = _direct_composite(q, k, v, d_out, active, 3)
    actual = _split_composite(q, k, v, d_out, active, 3)
    np.testing.assert_allclose(actual[0], expected[0], rtol=0.0, atol=0.0)
    for actual_gradient, expected_gradient in zip(actual[1], expected[1]):
        np.testing.assert_allclose(
            actual_gradient, expected_gradient, rtol=2e-14, atol=2e-14
        )
    assert np.linalg.norm(actual[1][1][:, :, 0]) > 0.0
    assert np.linalg.norm(actual[1][2][:, :, 0]) > 0.0


def test_window_512_sentinel_excludes_j7_includes_j8_in_forward_and_backward():
    q = np.zeros((1, 520, 1, 64), dtype=np.float64)
    k = np.zeros_like(q)
    v = np.zeros_like(q)
    v[0, 7, 0, 0] = 100.0
    v[0, 8, 0, 1] = 1.0
    output, _ = forward_reference(q, k, v, (0,), (512,))
    assert output[0, 519, 0, 0] == 0.0
    np.testing.assert_allclose(output[0, 519, 0, 1], 1.0 / 512.0, rtol=0.0, atol=1e-15)

    d_out = np.zeros_like(q)
    d_out[0, 519, 0, 2] = 1.0
    gradients = backward_reference(q, k, v, d_out, (0,), (512,))
    assert gradients[2][0, 7, 0, 2] == 0.0
    np.testing.assert_allclose(
        gradients[2][0, 8, 0, 2], 1.0 / 512.0, rtol=0.0, atol=1e-15
    )


def test_cudnn_capability_probe_is_noninvasive_and_explains_unavailability():
    capability = probe_cudnn_environment()
    assert "frontend_importable" in capability
    assert "pygraph_available" in capability
    assert "sdpa_backward_available" in capability
    assert "distributions" in capability
    if not capability["available"]:
        assert capability["reason"]
