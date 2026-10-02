import numpy as np

from mini_llm.backend import RandomStream, xp
from mini_llm.config import ContextRouterConfig
from mini_llm.ops.context_router import (
    CausalQueryPooler,
    ContextRouter,
    eligible_route_starts,
)


def _small_config(**overrides):
    values = dict(
        history_block_size=4,
        routing_stride=4,
        query_window=4,
        router_dim=3,
        top_k_blocks=2,
        exclude_recent_tokens=4,
        query_pooling="mean",
        history_pooling="mean",
        num_queries=1,
    )
    values.update(overrides)
    return ContextRouterConfig(**values)


def test_eligible_route_starts_respect_exclusion_and_topk():
    starts = eligible_route_starts(24, _small_config())
    np.testing.assert_array_equal(np.asarray(starts), np.array([12, 16, 20]))


def test_query_pooling_is_strictly_causal():
    x = xp.asarray(np.random.default_rng(1).normal(size=(1, 12, 3)), dtype="float64")
    starts = xp.asarray([4, 8], dtype=xp.int64)
    pooler = CausalQueryPooler(3, 4, strategy="mean")
    pooled_a, _ = pooler.forward(x, starts)

    changed = x.copy()
    changed[:, 4:, :] += 1000.0
    pooled_b, _ = pooler.forward(changed, starts)

    # Route at s=4 cannot observe token 4 or anything after it.
    np.testing.assert_allclose(np.asarray(pooled_a[:, 0]), np.asarray(pooled_b[:, 0]))


def test_learned_query_pooler_directional_derivative():
    rng = RandomStream(2)
    pooler = CausalQueryPooler(
        3, 5, strategy="learned", rng=rng, input_std=0.1, dtype="float64"
    )
    x = xp.asarray(np.random.default_rng(3).normal(size=(1, 10, 3)), dtype="float64")
    starts = xp.asarray([3, 7, 10], dtype=xp.int64)
    coeff = xp.asarray(np.random.default_rng(4).normal(size=(1, 3, 3)), dtype="float64")
    direction = xp.asarray(np.random.default_rng(5).normal(size=x.shape), dtype="float64")
    direction /= xp.sqrt(xp.sum(direction * direction))

    _, cache = pooler.forward(x, starts)
    pooler.zero_grad()
    dx = pooler.backward(coeff, cache)
    analytical = float(xp.sum(dx * direction))

    def objective(z):
        y, _ = pooler.forward(z, starts)
        return float(xp.sum(y * coeff))

    eps = 1e-6
    finite_difference = (
        objective(x + eps * direction) - objective(x - eps * direction)
    ) / (2.0 * eps)
    np.testing.assert_allclose(analytical, finite_difference, rtol=3e-6, atol=1e-8)


def test_context_router_masks_recent_and_future_blocks():
    router = ContextRouter(
        3, _small_config(), RandomStream(10), input_std=0.1, dtype="float64"
    )
    x = xp.asarray(np.random.default_rng(11).normal(size=(1, 24, 3)), dtype="float64")
    _, selected, starts, cache = router.forward(x)

    for route_i, start in enumerate(np.asarray(starts)):
        max_block_end = start - router.config.exclude_recent_tokens
        chosen = np.asarray(selected[0, route_i, 0])
        chosen_ends = (chosen + 1) * router.config.history_block_size
        assert np.all(chosen_ends <= max_block_end)

    mask = np.asarray(cache["candidate_mask"])
    assert mask.shape == (3, 6)
    np.testing.assert_array_equal(mask[0], np.array([True, True, False, False, False, False]))


def test_context_router_first_route_is_unchanged_by_current_or_future_tokens():
    cfg = _small_config(query_pooling="learned", history_pooling="learned")
    router = ContextRouter(3, cfg, RandomStream(20), input_std=0.1, dtype="float64")
    x = xp.asarray(np.random.default_rng(21).normal(size=(1, 24, 3)), dtype="float64")
    weights_a, selected_a, starts, _ = router.forward(x)
    first_start = int(np.asarray(starts)[0])

    changed = x.copy()
    changed[:, first_start:, :] += 500.0
    weights_b, selected_b, _, _ = router.forward(changed)

    np.testing.assert_array_equal(
        np.asarray(selected_a[:, 0]), np.asarray(selected_b[:, 0])
    )
    np.testing.assert_allclose(
        np.asarray(weights_a[:, 0]), np.asarray(weights_b[:, 0]), rtol=1e-12, atol=1e-12
    )


def test_context_router_input_directional_derivative_with_stable_topk():
    cfg = _small_config(query_pooling="mean", history_pooling="mean")
    router = ContextRouter(3, cfg, RandomStream(30), input_std=0.15, dtype="float64")
    x = xp.asarray(np.random.default_rng(31).normal(size=(1, 24, 3)), dtype="float64")
    weights, _, _, cache = router.forward(x)
    coeff = xp.asarray(np.random.default_rng(32).normal(size=weights.shape), dtype="float64")
    direction = xp.asarray(np.random.default_rng(33).normal(size=x.shape), dtype="float64")
    direction /= xp.sqrt(xp.sum(direction * direction))

    router.zero_grad()
    dx = router.backward(coeff, cache)
    analytical = float(xp.sum(dx * direction))

    def objective(z):
        w, _, _, _ = router.forward(z)
        return float(xp.sum(w * coeff))

    eps = 1e-6
    finite_difference = (
        objective(x + eps * direction) - objective(x - eps * direction)
    ) / (2.0 * eps)
    np.testing.assert_allclose(analytical, finite_difference, rtol=2e-5, atol=1e-8)


def test_context_router_query_projection_gradient():
    cfg = _small_config()
    router = ContextRouter(3, cfg, RandomStream(40), input_std=0.15, dtype="float64")
    x = xp.asarray(np.random.default_rng(41).normal(size=(1, 24, 3)), dtype="float64")
    weights, _, _, cache = router.forward(x)
    coeff = xp.asarray(np.random.default_rng(42).normal(size=weights.shape), dtype="float64")

    router.zero_grad()
    router.backward(coeff, cache)
    direction = xp.asarray(
        np.random.default_rng(43).normal(size=router.W_query.data.shape), dtype="float64"
    )
    direction /= xp.sqrt(xp.sum(direction * direction))
    analytical = float(xp.sum(router.W_query.grad * direction))

    original = router.W_query.data.copy()

    def objective(delta):
        router.W_query.data[...] = original + delta * direction
        w, _, _, _ = router.forward(x)
        return float(xp.sum(w * coeff))

    eps = 1e-6
    finite_difference = (objective(eps) - objective(-eps)) / (2.0 * eps)
    router.W_query.data[...] = original
    np.testing.assert_allclose(analytical, finite_difference, rtol=2e-5, atol=1e-8)


def test_context_router_accepts_auxiliary_full_score_gradient():
    cfg = _small_config()
    router = ContextRouter(3, cfg, RandomStream(50), input_std=0.1, dtype="float64")
    x = xp.asarray(np.random.default_rng(51).normal(size=(1, 24, 3)), dtype="float64")
    weights, _, _, cache = router.forward(x)
    dweights = xp.zeros_like(weights)
    dscores = xp.ones_like(cache["scores"])

    router.zero_grad()
    dx = router.backward(dweights, cache, dscores_extra=dscores)
    assert xp.any(dx != 0)
    assert xp.any(router.W_query.grad != 0)
    assert xp.any(router.W_history.grad != 0)
