import numpy as np

from mini_llm.backend import RandomStream, xp
from mini_llm.config import ContextRouterConfig
from mini_llm.ops.retrieval_attention import ContextRetrievalAttention


def _config(**overrides):
    values = dict(
        history_block_size=2,
        routing_stride=2,
        query_window=2,
        router_dim=3,
        top_k_blocks=2,
        exclude_recent_tokens=2,
        query_pooling="mean",
        history_pooling="mean",
        num_queries=2,
        router_weight_mode="logit_bias",
        router_weight_scale=1.0,
        router_weight_eps=1e-8,
    )
    values.update(overrides)
    return ContextRouterConfig(**values)


def _inputs(seed=200):
    rng = np.random.default_rng(seed)
    router_input = xp.asarray(rng.normal(size=(1, 12, 4)), dtype="float64")
    q = xp.asarray(rng.normal(size=(1, 12, 2, 3)), dtype="float64")
    k = xp.asarray(rng.normal(size=(1, 12, 1, 3)), dtype="float64")
    v = xp.asarray(rng.normal(size=(1, 12, 1, 3)), dtype="float64")
    return router_input, q, k, v


def test_context_retrieval_attention_is_zero_before_first_eligible_route():
    module = ContextRetrievalAttention(
        4, _config(), RandomStream(201), input_std=0.12, dtype="float64"
    )
    router_input, q, k, v = _inputs(202)
    out, routing, _ = module.forward(router_input, q, k, v)
    first_start = int(np.asarray(routing["route_starts"])[0])
    np.testing.assert_array_equal(
        np.asarray(out[:, :first_start]),
        np.zeros_like(np.asarray(out[:, :first_start])),
    )


def test_context_retrieval_attention_router_input_directional_derivative():
    module = ContextRetrievalAttention(
        4, _config(), RandomStream(203), input_std=0.18, dtype="float64"
    )
    router_input, q, k, v = _inputs(204)
    coeff = xp.asarray(
        np.random.default_rng(205).normal(size=q.shape), dtype="float64"
    )
    direction = xp.asarray(
        np.random.default_rng(206).normal(size=router_input.shape), dtype="float64"
    )
    direction /= xp.sqrt(xp.sum(direction * direction))

    out, routing, cache = module.forward(router_input, q, k, v)
    selected_ref = np.asarray(routing["selected_blocks"]).copy()
    module.zero_grad()
    dx, _, _, _ = module.backward(coeff, cache)
    analytical = float(xp.sum(dx * direction))

    def objective(x):
        y, info, _ = module.forward(x, q, k, v)
        # Directional derivative is only valid while the hard Top-K identity is
        # fixed.  A tiny perturbation should preserve it for this deterministic
        # test fixture.
        np.testing.assert_array_equal(
            np.asarray(info["selected_blocks"]), selected_ref
        )
        return float(xp.sum(y * coeff))

    eps = 1e-6
    finite_difference = (
        objective(router_input + eps * direction)
        - objective(router_input - eps * direction)
    ) / (2.0 * eps)
    np.testing.assert_allclose(analytical, finite_difference, rtol=3e-5, atol=1e-8)


def test_context_retrieval_attention_qkv_directional_derivative():
    module = ContextRetrievalAttention(
        4, _config(), RandomStream(207), input_std=0.15, dtype="float64"
    )
    router_input, q, k, v = _inputs(208)
    coeff = xp.asarray(
        np.random.default_rng(209).normal(size=q.shape), dtype="float64"
    )

    out, _, cache = module.forward(router_input, q, k, v)
    module.zero_grad()
    _, dq, dk, dv = module.backward(coeff, cache)

    rng = np.random.default_rng(210)
    q_dir = xp.asarray(rng.normal(size=q.shape), dtype="float64")
    k_dir = xp.asarray(rng.normal(size=k.shape), dtype="float64")
    v_dir = xp.asarray(rng.normal(size=v.shape), dtype="float64")
    norm = xp.sqrt(
        xp.sum(q_dir * q_dir) + xp.sum(k_dir * k_dir) + xp.sum(v_dir * v_dir)
    )
    q_dir /= norm
    k_dir /= norm
    v_dir /= norm
    analytical = float(
        xp.sum(dq * q_dir) + xp.sum(dk * k_dir) + xp.sum(dv * v_dir)
    )

    def objective(qx, kx, vx):
        y, _, _ = module.forward(router_input, qx, kx, vx)
        return float(xp.sum(y * coeff))

    eps = 1e-6
    plus = objective(q + eps * q_dir, k + eps * k_dir, v + eps * v_dir)
    minus = objective(q - eps * q_dir, k - eps * k_dir, v - eps * v_dir)
    finite_difference = (plus - minus) / (2.0 * eps)
    np.testing.assert_allclose(analytical, finite_difference, rtol=3e-6, atol=1e-8)


def test_context_retrieval_attention_router_projection_gradient_from_lm_path():
    module = ContextRetrievalAttention(
        4, _config(), RandomStream(211), input_std=0.2, dtype="float64"
    )
    router_input, q, k, v = _inputs(212)
    coeff = xp.asarray(
        np.random.default_rng(213).normal(size=q.shape), dtype="float64"
    )

    out, routing, cache = module.forward(router_input, q, k, v)
    selected_ref = np.asarray(routing["selected_blocks"]).copy()
    module.zero_grad()
    module.backward(coeff, cache)

    direction = xp.asarray(
        np.random.default_rng(214).normal(size=module.router.W_query.data.shape),
        dtype="float64",
    )
    direction /= xp.sqrt(xp.sum(direction * direction))
    analytical = float(xp.sum(module.router.W_query.grad * direction))
    original = module.router.W_query.data.copy()

    def objective(delta):
        module.router.W_query.data[...] = original + delta * direction
        y, info, _ = module.forward(router_input, q, k, v)
        np.testing.assert_array_equal(
            np.asarray(info["selected_blocks"]), selected_ref
        )
        return float(xp.sum(y * coeff))

    eps = 1e-6
    finite_difference = (objective(eps) - objective(-eps)) / (2.0 * eps)
    module.router.W_query.data[...] = original
    np.testing.assert_allclose(analytical, finite_difference, rtol=4e-5, atol=1e-8)
    assert abs(analytical) > 1e-10


def test_router_weight_none_has_no_lm_gradient_but_accepts_auxiliary_scores():
    module = ContextRetrievalAttention(
        4,
        _config(router_weight_mode="none"),
        RandomStream(215),
        input_std=0.15,
        dtype="float64",
    )
    router_input, q, k, v = _inputs(216)
    coeff = xp.asarray(
        np.random.default_rng(217).normal(size=q.shape), dtype="float64"
    )

    _, _, cache = module.forward(router_input, q, k, v)
    module.zero_grad()
    dx, _, _, _ = module.backward(coeff, cache)
    np.testing.assert_array_equal(np.asarray(dx), np.zeros_like(np.asarray(dx)))
    np.testing.assert_array_equal(
        np.asarray(module.router.W_query.grad),
        np.zeros_like(np.asarray(module.router.W_query.grad)),
    )

    scores = cache["router_cache"]["scores"]
    dscores_extra = xp.ones_like(scores)
    module.zero_grad()
    dx_aux, _, _, _ = module.backward(coeff, cache, dscores_extra=dscores_extra)
    assert bool(xp.any(dx_aux != 0))
    assert bool(xp.any(module.router.W_query.grad != 0))


def test_context_retrieval_attention_handles_no_eligible_history():
    module = ContextRetrievalAttention(
        4,
        _config(exclude_recent_tokens=32),
        RandomStream(218),
        input_std=0.15,
        dtype="float64",
    )
    router_input, q, k, v = _inputs(219)
    out, routing, cache = module.forward(router_input, q, k, v)
    assert routing["route_starts"].size == 0
    np.testing.assert_array_equal(np.asarray(out), np.zeros_like(np.asarray(out)))

    module.zero_grad()
    dx, dq, dk, dv = module.backward(xp.ones_like(out), cache)
    np.testing.assert_array_equal(np.asarray(dx), np.zeros_like(np.asarray(dx)))
    np.testing.assert_array_equal(np.asarray(dq), np.zeros_like(np.asarray(dq)))
    np.testing.assert_array_equal(np.asarray(dk), np.zeros_like(np.asarray(dk)))
    np.testing.assert_array_equal(np.asarray(dv), np.zeros_like(np.asarray(dv)))


def test_context_retrieval_attention_shared_router_query_broadcasts_to_heads():
    module = ContextRetrievalAttention(
        4,
        _config(num_queries=1),
        RandomStream(220),
        input_std=0.15,
        dtype="float64",
    )
    router_input, q, k, v = _inputs(221)
    coeff = xp.asarray(
        np.random.default_rng(222).normal(size=q.shape), dtype="float64"
    )

    _, routing, cache = module.forward(router_input, q, k, v)
    assert routing["weights"].shape[2] == 1
    module.zero_grad()
    dx, dq, dk, dv = module.backward(coeff, cache)
    assert bool(xp.all(xp.isfinite(dx)))
    assert bool(xp.all(xp.isfinite(dq)))
    assert bool(xp.all(xp.isfinite(dk)))
    assert bool(xp.all(xp.isfinite(dv)))
    assert bool(xp.any(module.router.W_query.grad != 0))
