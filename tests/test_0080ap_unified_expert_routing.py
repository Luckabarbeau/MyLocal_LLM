"""0080AP: one MoE routing equation and its derivative for every top_k."""
import numpy as np
import pytest

from mini_llm.backend import BACKEND_NAME, RandomStream
from mini_llm.ops.router import Router, expert_topk_forward, expert_topk_backward
from mini_llm.ops.router_inference import RouterInference

pytestmark = pytest.mark.skipif(BACKEND_NAME != 'numpy', reason='CPU finite differences')


@pytest.mark.parametrize('n_experts,k', [(2,1),(6,1),(6,2),(3,3)])
def test_exact_full_distribution_topk_forward_backward(n_experts, k):
    rng = np.random.default_rng(43 + k)
    logits = rng.normal(size=(2, 4, n_experts))
    weights, indices, cache = expert_topk_forward(logits, k)
    probs = np.exp(logits - logits.max(axis=-1, keepdims=True))
    probs /= probs.sum(axis=-1, keepdims=True)
    np.testing.assert_allclose(weights, np.take_along_axis(probs, indices, axis=-1))
    if k < n_experts:
        assert np.all(weights.sum(axis=-1) < 1.)
    else:
        np.testing.assert_allclose(weights.sum(axis=-1), 1., atol=1e-14)

    upstream = rng.normal(size=weights.shape)
    analytical = expert_topk_backward(upstream, cache)
    # The sorted expert choices must not swap under tiny finite differences.
    eps = 1e-6
    for idx in [(0,0,0),(1,2,n_experts-1),(1,3,0)]:
        z_plus = logits.copy(); z_plus[idx] += eps
        z_minus = logits.copy(); z_minus[idx] -= eps
        scalar_plus = (expert_topk_forward(z_plus,k)[0] * upstream).sum()
        scalar_minus = (expert_topk_forward(z_minus,k)[0] * upstream).sum()
        np.testing.assert_allclose(analytical[idx], (scalar_plus-scalar_minus)/(2*eps), atol=1e-8,rtol=1e-5)
    if k < n_experts:
        unused = ~np.isin(np.arange(n_experts),indices[0,0])
        assert np.any(np.abs(analytical[0,0,unused]) > 1e-10)


@pytest.mark.parametrize('n_experts,k', [(2,1),(6,2),(4,4)])
def test_all_top_k_modes_training_inference_parity_and_router_gradients(n_experts,k):
    router = Router(5,n_experts,k,0.1,RandomStream(37),dtype='float64')
    infer = RouterInference(5,n_experts,k,dtype='float64'); infer.set_weights(router)
    x = np.random.default_rng(31).normal(size=(2,3,5))
    w, ids, cache = router.forward(x)
    wi, idi = infer.forward(x)
    np.testing.assert_array_equal(ids, idi)
    np.testing.assert_allclose(w, wi, atol=0, rtol=0)
    router.zero_grad()
    signal = np.random.default_rng(65).normal(size=w.shape)
    dx = router.backward(signal,cache)
    assert np.isfinite(dx).all()
    assert np.any(np.abs(router.W_router_param.grad) > 0)
    def objective(inp):
        return (router.forward(inp)[0]*signal).sum()
    eps = 1e-6
    loc = (0,1,2)
    plus=x.copy(); plus[loc]+=eps
    minus=x.copy(); minus[loc]-=eps
    np.testing.assert_allclose(dx[loc],(objective(plus)-objective(minus))/(2*eps),rtol=1e-5,atol=1e-8)


def test_single_expert_degenerate_configuration_is_consistent():
    p, ids, cache=expert_topk_forward(np.array([[[12.]]]),1)
    np.testing.assert_array_equal(p,np.array([[[1.]]]))
    np.testing.assert_array_equal(expert_topk_backward(np.array([[[2.]]]),cache),np.zeros((1,1,1)))


def test_no_extra_moe_router_behavior_flags():
    import inspect
    from mini_llm.config import ModelConfig
    from mini_llm.blocks.moe_block import MoE
    from mini_llm.blocks.transformer_block import TransformerBlock
    assert not hasattr(ModelConfig(), 'top1_router_full_softmax')
    for constructor in (Router, RouterInference, MoE, TransformerBlock):
        assert 'full_softmax_top1' not in inspect.signature(constructor).parameters
        assert 'top1_router_full_softmax' not in inspect.signature(constructor).parameters
