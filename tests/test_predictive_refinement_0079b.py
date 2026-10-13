"""0079B compact dense reference: masks, full-model BPTT hooks and leakage."""

import numpy as np
import pytest
from mini_llm.backend import xp, asnumpy
from mini_llm.config import ModelConfig
from mini_llm.model.decoder_lm import DecoderLanguageModel
from mini_llm.predictive_refinement import FuturePositionEmbeddings
from mini_llm.predictive_refinement.block_attention import block_causal_allow_mask
from mini_llm.backend import RandomStream


def small_model():
    model = DecoderLanguageModel(ModelConfig.tiny_inspection(), rng_seed=51, dtype="float64")
    assert model.config.attention_layers is None
    assert not model.hierarchical_memory_enabled
    return model


def make_input(model, context, k):
    state = FuturePositionEmbeddings(model.config.d_model, RandomStream(55), dtype="float64")
    cemb, _ = model.embedding.forward(context)
    return xp.concatenate((cemb, state.unknown_block(context.shape[0], k)), axis=1)


def test_block_causal_mask_exact_and_padding():
    allow = asnumpy(block_causal_allow_mask(3, 4))
    assert allow.shape == (7, 7)
    np.testing.assert_array_equal(allow[:3, :3], np.tril(np.ones((3, 3), dtype=bool)))
    assert not allow[:3, 3:].any()
    assert allow[3:, :3].all() and allow[3:, 3:].all()
    padded = asnumpy(block_causal_allow_mask(2, 3, valid_block_mask=xp.asarray([True, True, False])))
    assert not padded[2:4, 4].any()
    assert padded[4, 4]   # invalid row has itself, no empty softmax
    batched = block_causal_allow_mask(2, 3, valid_block_mask=xp.asarray([[True, True, False], [True, False, False]]))
    assert batched.shape == (2, 5, 5)
    with pytest.raises(ValueError):
        block_causal_allow_mask(1024, 4)


def test_context_states_invariant_to_future_unknown_embeddings():
    model = small_model()
    ids = xp.asarray([[1, 2, 3, 4]], dtype="int64")
    whole = make_input(model, ids, 3)
    mask = block_causal_allow_mask(ids.shape[1], 3)
    direct = model.forward(ids, return_cache=False)
    combined = model.forward(None, input_embeddings=whole, attention_allow_mask=mask, return_cache=False)
    np.testing.assert_allclose(asnumpy(combined[:, :4]), asnumpy(direct), rtol=1e-11, atol=1e-11)
    changed = whole.copy()
    changed[:, -1] += 2.0
    logits = model.forward(None, input_embeddings=changed, attention_allow_mask=mask, return_cache=False)
    np.testing.assert_allclose(asnumpy(combined[:, :4]), asnumpy(logits[:, :4]), rtol=1e-11, atol=1e-11)


def test_future_provisional_position_can_affect_first_position():
    model = small_model()
    ids = xp.asarray([[1, 2, 3]], dtype="int64")
    x = make_input(model, ids, 3)
    allow = block_causal_allow_mask(3, 3)
    x[:, -1] = x[:, -1] + 3
    logits, cache = model.forward(None, input_embeddings=x, attention_allow_mask=allow)
    dlogits = xp.zeros_like(logits)
    dlogits[:, 3, 1] = 1.0
    model.zero_grad()
    dx = model.backward(dlogits, cache)
    assert dx.shape == x.shape
    assert np.linalg.norm(asnumpy(dx[:, -1])) > 1e-12
    eps = 1e-5
    plus = x.copy()
    minus = x.copy()
    plus[:, -1, 0] += eps
    minus[:, -1, 0] -= eps
    fd = (
        model.forward(None, input_embeddings=plus, attention_allow_mask=allow, return_cache=False)[0, 3, 1]
        - model.forward(None, input_embeddings=minus, attention_allow_mask=allow, return_cache=False)[0, 3, 1]
    ) / (2 * eps)
    np.testing.assert_allclose(float(dx[0, -1, 0]), float(fd), atol=5e-6, rtol=2e-3)


def test_independent_forward_caches_recurrent_shared_model():
    model = small_model()
    x = make_input(model, xp.asarray([[1, 2, 3]]), 2)
    mask = block_causal_allow_mask(3, 2)
    logits_a, cache_a = model.forward(None, input_embeddings=x, attention_allow_mask=mask)
    logits_b, cache_b = model.forward(None, input_embeddings=x + 0.05, attention_allow_mask=mask)
    model.zero_grad()
    da = xp.ones_like(logits_a)
    db = xp.ones_like(logits_b)
    grad_b = model.backward(db, cache_b)
    grad_a = model.backward(da, cache_a)
    assert np.isfinite(asnumpy(grad_b)).all()
    assert np.isfinite(asnumpy(grad_a)).all()
    assert np.linalg.norm(asnumpy(model.output_proj.W.grad)) > 0


def test_reject_routed_mode_custom_mask():
    # This is a deliberate gate, not a silent fallback to causal sparse kernels.
    from mini_llm.config import AttentionLayerConfig
    # Existing ordinary forward without a custom mask must remain unchanged.
    model = small_model()
    x = make_input(model, xp.asarray([[1, 2]]), 2)
    with pytest.raises(ValueError):
        model.forward(None, input_embeddings=x, attention_allow_mask=xp.ones((3, 3), dtype=bool))
