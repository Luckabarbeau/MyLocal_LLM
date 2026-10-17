"""0080AO: deep 500M topology and differentiable sparse top-1 expert router."""

import numpy as np
import pytest

from mini_llm.backend import BACKEND_NAME, RandomStream
from mini_llm.blocks.moe_block import MoE
from mini_llm.config import ModelConfig
from mini_llm.ops.router import Router

pytestmark = pytest.mark.skipif(BACKEND_NAME != "numpy", reason="NumPy reference")


def test_deep_top1_preset_preserves_64k_memory_and_attention():
    old = ModelConfig.wide_500m_memory_64k()
    new = ModelConfig.deep_500m_memory_64k()
    assert (new.n_layers, new.n_experts, new.top_k, new.d_ff) == (32, 2, 1, 2368)
    assert new.estimated_parameter_count() == 499_213_120
    assert old.estimated_parameter_count() == 501_237_576
    assert new.d_model == old.d_model == 768
    assert (new.n_q_heads, new.n_kv_heads, new.d_head) == (12, 3, 64)
    assert new.memory_context == old.memory_context
    assert new.context_length == old.context_length == 6144
    assert len(new.attention_layers) == 32
    assert all(layer == old.attention_layers[0] for layer in new.attention_layers)
    # Verifies CLI preset dispatch independently from ModelConfig.
    from train_model import model_config_from_name
    assert model_config_from_name("deep-500m-memory-64k") == new


def test_top1_router_weights_use_full_softmax_not_degenerate_one():
    router = Router(4, 2, 1, 0.1, RandomStream(21), dtype="float64")
    router.W_router_param.data[...] = np.array(
        [[0.3, -0.2], [-0.1, 0.15], [0.05, -0.1], [0.2, -0.05]],
        dtype=np.float64,
    )
    router.b_router_param.data[...] = np.array([0.03, -0.02])
    x = np.array([[[1.0, 0.5, -0.1, 0.7], [-0.9, 1.0, 0.8, -0.4]]])
    w, indices, cache = router.forward(x)
    logits = x @ router.W_router_param.data + router.b_router_param.data
    exps = np.exp(logits - np.max(logits, axis=-1, keepdims=True))
    p = exps / exps.sum(axis=-1, keepdims=True)
    np.testing.assert_array_equal(indices, np.argmax(logits, axis=-1)[..., None])
    np.testing.assert_allclose(w, np.take_along_axis(p, indices, axis=-1))
    assert np.all(w > 0.5) and np.all(w < 1.0)
    assert "topk_cache" in cache


def test_top1_router_gradients_match_finite_differences_for_both_experts():
    router = Router(4, 2, 1, 0.1, RandomStream(27), dtype="float64")
    router.W_router_param.data[...] = np.array(
        [[0.3, -0.2], [-0.1, 0.15], [0.05, -0.1], [0.2, -0.05]],
        dtype=np.float64,
    )
    router.b_router_param.data[...] = np.array([0.03, -0.02])
    x = np.array([[[1.0, 0.5, -0.1, 0.7], [-0.9, 1.0, 0.8, -0.4]]])
    signal = np.array([[[0.7], [-0.35]]])
    router.zero_grad()
    _, _, cache = router.forward(x)
    dx = router.backward(signal, cache)
    assert np.any(np.abs(router.W_router_param.grad[:, 0]) > 1e-9)
    assert np.any(np.abs(router.W_router_param.grad[:, 1]) > 1e-9)
    def objective():
        return float(np.sum(router.forward(x)[0] * signal))
    eps = 1e-6
    def finite_diff(array, idx):
        old = array[idx]
        try:
            array[idx] = old + eps
            plus = objective()
            array[idx] = old - eps
            minus = objective()
        finally:
            array[idx] = old
        return (plus - minus) / (2 * eps)
    for ind in [(0, 0), (1, 1), (3, 0), (3, 1)]:
        np.testing.assert_allclose(router.W_router_param.grad[ind],
                                   finite_diff(router.W_router_param.data, ind),
                                   rtol=1e-5, atol=1e-8)
    for ind in [(0,), (1,)]:
        np.testing.assert_allclose(router.b_router_param.grad[ind],
                                   finite_diff(router.b_router_param.data, ind),
                                   rtol=1e-5, atol=1e-8)
    # Input gradient includes router derivatives for both candidate logits.
    dx_index = (0, 0, 0)
    np.testing.assert_allclose(dx[dx_index], finite_diff(x, dx_index),
                               rtol=1e-5, atol=1e-8)


def test_top1_moe_finite_difference_input_gradient():
    moe = MoE(d_model=4, d_ff=8, n_experts=2, k=1,
              input_std=.1, output_std=.05, rng=RandomStream(123), dtype="float64")
    x = np.array([[[1., 0.5, -.1, .7], [-0.7, 0.6, .3, -0.6]]])
    y, tape = moe.forward(x)
    grad_output = np.array([[[.2, -.3, .4, .8], [.1, .2, -.2, .15]]])
    moe.zero_grad()
    dx = moe.backward(grad_output, tape)
    def objective():
        return float(np.sum(moe.forward(x)[0] * grad_output))
    for ind in [(0, 0, 0), (0, 1, 1)]:
        old = x[ind]; eps = 1e-5
        try:
            x[ind] = old + eps
            plus = objective()
            x[ind] = old - eps
            minus = objective()
        finally:
            x[ind] = old
        np.testing.assert_allclose(dx[ind], (plus - minus)/(2*eps),
                                   rtol=1e-3, atol=1e-8)
    assert any(np.any(np.abs(p.grad) > 0) for p in moe.router.parameters())


def test_top2_router_uses_unrenormalized_full_probabilities():
    router = Router(4, 3, 2, .1, RandomStream(45), dtype="float64")
    x = np.array([[[1.0, -.2, .3, -.4]]])
    w, indices, cache = router.forward(x)
    assert indices.shape == (1, 1, 2)
    logits = x @ router.W_router_param.data + router.b_router_param.data
    p = np.exp(logits - logits.max(axis=-1, keepdims=True))
    p /= p.sum(axis=-1, keepdims=True)
    np.testing.assert_allclose(w, np.take_along_axis(p, indices, axis=-1))
    assert np.all(w.sum(axis=-1) < 1)
    assert "topk_cache" in cache


@pytest.mark.parametrize("n_experts,k", [(2, 1), (6, 2), (3, 3)])
def test_inference_router_matches_training(n_experts, k):
    from mini_llm.ops.router_inference import RouterInference
    x = np.array([[[0.3, -1., .5, .1], [-.2, .7, .1, 1.3]]])
    train = Router(4, n_experts, k, .1, RandomStream(83), dtype="float64")
    inference = RouterInference(4, n_experts, k, dtype="float64")
    inference.set_weights(train)
    weights, inds, _ = train.forward(x)
    inference_weights, inference_inds = inference.forward(x)
    np.testing.assert_array_equal(inds, inference_inds)
    np.testing.assert_allclose(weights, inference_weights, rtol=1e-14, atol=1e-14)
    if k < n_experts:
        assert np.all(weights.sum(axis=-1) < 1.0)
    else:
        np.testing.assert_allclose(weights.sum(axis=-1), 1.0)


def test_all_32_layers_execute_and_backpropagate_in_tiny_config():
    from dataclasses import replace
    from mini_llm.config import MemoryContextConfig
    from mini_llm.model.decoder_lm import DecoderLanguageModel
    new = ModelConfig.deep_500m_memory_64k()
    tiny = replace(
        new, tokenizer_vocab_size=43, context_length=16,
        d_model=24, n_q_heads=12, n_kv_heads=3, d_head=2, d_ff=32,
        attention_layers=None, memory_context=MemoryContextConfig(enabled=False),
    )
    model = DecoderLanguageModel(tiny, rng_seed=19, dtype="float64")
    assert len(model.blocks) == 32
    assert all(b.moe.router.k == 1 for b in model.blocks)
    token_ids = np.array([[1, 2, 3, 4, 5, 6]], dtype=np.int64)
    model.zero_grad()
    output, cache = model.forward_body(token_ids)
    assert output.shape == (1, 6, 24)
    grad = np.random.default_rng(19).normal(scale=0.1, size=output.shape)
    model.backward_body(grad, cache)
    for block in (model.blocks[0], model.blocks[15], model.blocks[31]):
        assert any(np.any(p.grad != 0) for p in block.moe.router.parameters())
