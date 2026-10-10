import numpy as np

from mini_llm.backend import RandomStream, xp
from mini_llm.config import ContextRouterConfig
from mini_llm.ops.context_blocks import (
    HistoryBlockPooler,
    block_token_indices,
    complete_block_count,
    gather_full_resolution_blocks,
    scatter_full_resolution_blocks,
)


def test_context_router_config_validation_and_independent_stride():
    cfg = ContextRouterConfig(history_block_size=256, routing_stride=64)
    assert cfg.history_block_size == 256
    assert cfg.routing_stride == 64


def test_block_indices_and_complete_count():
    assert complete_block_count(17, 4) == 4
    blocks = xp.asarray([1, 3], dtype=xp.int64)
    np.testing.assert_array_equal(
        np.asarray(block_token_indices(blocks, 4)),
        np.array([[4, 5, 6, 7], [12, 13, 14, 15]]),
    )


def test_mean_pooling_ignores_partial_tail_and_backward():
    x = xp.asarray(np.arange(1 * 10 * 2).reshape(1, 10, 2), dtype="float64")
    pooler = HistoryBlockPooler(2, 4, strategy="mean")
    pooled, cache = pooler.forward(x)
    ref = np.asarray(x)[:, :8, :].reshape(1, 2, 4, 2).mean(axis=2)
    np.testing.assert_allclose(np.asarray(pooled), ref)

    dx = pooler.backward(xp.ones_like(pooled), cache)
    np.testing.assert_allclose(np.asarray(dx[:, :8]), 0.25)
    np.testing.assert_allclose(np.asarray(dx[:, 8:]), 0.0)


def test_learned_history_pooling_directional_derivative():
    rng = RandomStream(10)
    pooler = HistoryBlockPooler(
        3, 4, strategy="learned", rng=rng, input_std=0.1, dtype="float64"
    )
    x = xp.asarray(np.random.default_rng(11).normal(size=(1, 8, 3)), dtype="float64")
    coeff = xp.asarray(np.random.default_rng(12).normal(size=(1, 2, 3)), dtype="float64")
    direction = xp.asarray(np.random.default_rng(13).normal(size=x.shape), dtype="float64")
    direction /= xp.sqrt(xp.sum(direction * direction))

    pooled, cache = pooler.forward(x)
    pooler.zero_grad()
    dx = pooler.backward(coeff, cache)
    analytical = float(xp.sum(dx * direction))

    def objective(z):
        y, _ = pooler.forward(z)
        return float(xp.sum(y * coeff))

    eps = 1e-6
    finite_difference = (
        objective(x + eps * direction) - objective(x - eps * direction)
    ) / (2.0 * eps)
    np.testing.assert_allclose(analytical, finite_difference, rtol=2e-6, atol=1e-8)


def test_gather_and_scatter_add_with_repeated_blocks():
    x = xp.asarray(np.arange(1 * 12 * 2).reshape(1, 12, 2), dtype="float64")
    selected = xp.asarray([[[0, 2], [2, 2]]], dtype=xp.int64)
    gathered, cache = gather_full_resolution_blocks(x, selected, block_size=3)
    assert gathered.shape == (1, 2, 6, 2)

    np.testing.assert_array_equal(np.asarray(gathered[0, 0, :3]), np.asarray(x[0, 0:3]))
    np.testing.assert_array_equal(np.asarray(gathered[0, 0, 3:]), np.asarray(x[0, 6:9]))

    dx = scatter_full_resolution_blocks(xp.ones_like(gathered), cache)
    expected = np.zeros((1, 12, 2))
    expected[:, 0:3, :] += 1
    # Block 2 appears once in the first route and twice in the second route.
    expected[:, 6:9, :] += 3
    np.testing.assert_array_equal(np.asarray(dx), expected)


def test_bfloat16_block_backward_and_scatter_accumulate_in_float32():
    try:
        import ml_dtypes
    except ImportError:
        return

    x = xp.asarray(
        np.random.default_rng(80).normal(size=(1, 8, 3)),
        dtype=ml_dtypes.bfloat16,
    )
    pooler = HistoryBlockPooler(3, 4, strategy="mean")
    pooled, pool_cache = pooler.forward(x)
    dx_pool = pooler.backward(xp.ones_like(pooled), pool_cache)
    assert str(np.dtype(dx_pool.dtype)).lower() == "float32"

    selected = xp.asarray([[[0, 1]]], dtype=xp.int64)
    gathered, gather_cache = gather_full_resolution_blocks(x, selected, block_size=4)
    dx_gather = scatter_full_resolution_blocks(xp.ones_like(gathered), gather_cache)
    assert str(np.dtype(dx_gather.dtype)).lower() == "float32"
    np.testing.assert_allclose(np.asarray(dx_gather), 1.0, rtol=0, atol=0)
