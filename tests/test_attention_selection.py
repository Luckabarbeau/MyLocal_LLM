import numpy as np
import pytest

from mini_llm.backend import xp
from mini_llm.ops.attention_selection import (
    KeySelectionPlan,
    build_block_retrieval_plan,
)


def test_manual_block_retrieval_plan_expands_full_resolution_tokens():
    selected = xp.asarray(
        [[
            [[0, 1], [1, 0]],
            [[1, 2], [2, 1]],
        ]],
        dtype=xp.int64,
    )  # [B=1,R=2,H=2,Kb=2]
    starts = xp.asarray([4, 8], dtype=xp.int64)

    plan = build_block_retrieval_plan(
        selected,
        starts,
        seq_len=12,
        block_size=2,
        routing_stride=4,
        exclude_recent_tokens=0,
    )

    assert plan.key_indices.shape == (1, 2, 12, 4)
    assert plan.valid_mask.shape == plan.key_indices.shape

    # Before the first route no retrieved keys are visible.
    assert not bool(xp.any(plan.valid_mask[:, :, :4, :]))

    # Route 0 controls tokens 4..7 and reopens blocks [0,1] for head 0.
    expected_h0_r0 = np.array([0, 1, 2, 3])
    for token in range(4, 8):
        np.testing.assert_array_equal(
            np.asarray(plan.key_indices[0, 0, token]), expected_h0_r0
        )
        assert bool(xp.all(plan.valid_mask[0, 0, token]))

    # Route 1 controls tokens 8..11 and head 1 uses blocks [2,1].
    expected_h1_r1 = np.array([4, 5, 2, 3])
    for token in range(8, 12):
        np.testing.assert_array_equal(
            np.asarray(plan.key_indices[0, 1, token]), expected_h1_r1
        )


def test_manual_plan_keeps_gaps_invalid_when_routes_are_not_contiguous():
    selected = xp.asarray([[[0], [1]]], dtype=xp.int64)
    starts = xp.asarray([4, 10], dtype=xp.int64)
    plan = build_block_retrieval_plan(
        selected,
        starts,
        seq_len=14,
        block_size=2,
        routing_stride=2,
    )

    # First route covers 4,5. Tokens 6..9 must not silently reuse it.
    assert bool(xp.all(plan.valid_mask[0, 0, 4:6]))
    assert not bool(xp.any(plan.valid_mask[0, 0, 6:10]))
    assert bool(xp.all(plan.valid_mask[0, 0, 10:12]))
    assert not bool(xp.any(plan.valid_mask[0, 0, 12:14]))


def test_manual_plan_rejects_noncausal_or_too_recent_blocks():
    selected = xp.asarray([[[1]]], dtype=xp.int64)  # block 1 ends at token 4
    starts = xp.asarray([5], dtype=xp.int64)

    # With a two-token exclusion gap the selected block would need to end <= 3.
    with pytest.raises(ValueError, match="causal/recent-history"):
        build_block_retrieval_plan(
            selected,
            starts,
            seq_len=8,
            block_size=2,
            routing_stride=2,
            exclude_recent_tokens=2,
        )


def test_key_selection_plan_allows_one_shared_plan_head():
    indices = xp.zeros((2, 1, 5, 3), dtype=xp.int64)
    valid = xp.ones(indices.shape, dtype=bool)
    plan = KeySelectionPlan(indices, valid)
    plan.validate(batch_size=2, n_q_heads=4, query_length=5, key_length=7)


def test_weighted_block_plan_repeats_log_probability_bias_per_exact_token():
    from mini_llm.ops.attention_selection import build_weighted_block_retrieval_plan

    selected = xp.asarray([[[[0, 1]], [[1, 2]]]], dtype=xp.int64)
    weights = xp.asarray([[[[0.25, 0.75]], [[0.6, 0.4]]]], dtype="float64")
    starts = xp.asarray([4, 8], dtype=xp.int64)

    plan, _ = build_weighted_block_retrieval_plan(
        selected,
        weights,
        starts,
        seq_len=12,
        block_size=2,
        routing_stride=4,
        weight_scale=2.0,
        weight_eps=1e-12,
    )

    expected = np.array([
        2.0 * np.log(0.25 + 1e-12),
        2.0 * np.log(0.25 + 1e-12),
        2.0 * np.log(0.75 + 1e-12),
        2.0 * np.log(0.75 + 1e-12),
    ])
    np.testing.assert_allclose(np.asarray(plan.logit_bias[0, 0, 4]), expected)
    np.testing.assert_allclose(np.asarray(plan.logit_bias[0, 0, 7]), expected)
    np.testing.assert_array_equal(
        np.asarray(plan.logit_bias[0, 0, :4]),
        np.zeros_like(np.asarray(plan.logit_bias[0, 0, :4])),
    )


def test_weighted_block_plan_bias_backward_matches_directional_derivative():
    from mini_llm.ops.attention_selection import (
        build_weighted_block_retrieval_plan,
        weighted_block_retrieval_bias_backward,
    )

    selected = xp.asarray([[[[0, 1]], [[1, 2]]]], dtype=xp.int64)
    weights = xp.asarray([[[[0.35, 0.65]], [[0.55, 0.45]]]], dtype="float64")
    starts = xp.asarray([4, 8], dtype=xp.int64)
    coeff = xp.asarray(
        np.random.default_rng(120).normal(size=(1, 2, 12, 4)), dtype="float64"
    )
    # The router has one shared query but indexed attention may broadcast the
    # resulting plan over several Q heads.  Exercise that reduction explicitly.

    plan, cache = build_weighted_block_retrieval_plan(
        selected,
        weights,
        starts,
        seq_len=12,
        block_size=2,
        routing_stride=4,
        weight_scale=1.3,
        weight_eps=1e-7,
    )
    analytical_full = weighted_block_retrieval_bias_backward(coeff, cache)

    direction = xp.asarray(
        np.random.default_rng(121).normal(size=weights.shape), dtype="float64"
    )
    direction /= xp.sqrt(xp.sum(direction * direction))
    analytical = float(xp.sum(analytical_full * direction))

    def objective(w):
        p, _ = build_weighted_block_retrieval_plan(
            selected,
            w,
            starts,
            seq_len=12,
            block_size=2,
            routing_stride=4,
            weight_scale=1.3,
            weight_eps=1e-7,
        )
        # Broadcast the one router head to two attention heads exactly as the
        # indexed kernel does before returning dlogit_bias.
        bias = xp.broadcast_to(p.logit_bias, coeff.shape)
        return float(xp.sum(bias * coeff))

    eps = 1e-6
    finite_difference = (
        objective(weights + eps * direction) - objective(weights - eps * direction)
    ) / (2.0 * eps)
    np.testing.assert_allclose(analytical, finite_difference, rtol=2e-6, atol=1e-8)


def test_local_causal_plan_exact_window_visibility():
    from mini_llm.ops.attention_selection import build_local_causal_plan

    plan = build_local_causal_plan(batch_size=1, seq_len=6, window=3)
    expected_indices = np.array(
        [
            [0, 0, 0],
            [0, 0, 1],
            [0, 1, 2],
            [1, 2, 3],
            [2, 3, 4],
            [3, 4, 5],
        ],
        dtype=np.int64,
    )
    expected_valid = np.array(
        [
            [False, False, True],
            [False, True, True],
            [True, True, True],
            [True, True, True],
            [True, True, True],
            [True, True, True],
        ]
    )
    np.testing.assert_array_equal(np.asarray(plan.key_indices[0, 0]), expected_indices)
    np.testing.assert_array_equal(np.asarray(plan.valid_mask[0, 0]), expected_valid)
