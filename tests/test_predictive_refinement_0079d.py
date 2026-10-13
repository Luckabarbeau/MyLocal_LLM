"""0079D cache correctness, progressive commitment and three-block generation."""

import numpy as np
import pytest

from mini_llm.backend import xp, asnumpy
from mini_llm.config import ModelConfig
from mini_llm.inference_model import InferenceModel
from mini_llm.model.decoder_lm import DecoderLanguageModel
from mini_llm.predictive_refinement.block_attention import block_causal_allow_mask
from mini_llm.predictive_refinement.training_reference import (
    DenseRefinementTrainingReference, RefinementReferenceConfig,
)
from mini_llm.predictive_refinement.inference import (
    CachedPredictiveRefinementGenerator, RefinementGenerationConfig,
)


def setup(k=3):
    model = DecoderLanguageModel(ModelConfig.tiny_inspection(), rng_seed=53, dtype="float64")
    trained = DenseRefinementTrainingReference(
        model, RefinementReferenceConfig(block_size=k), rng_seed=9
    )
    inf = InferenceModel(model.config, dtype="float64")
    inf.set_weights(model)
    state = inf.create_generation_state(batch_size=1, max_length=16)
    prompt = xp.asarray([[4, 7, 11]], dtype=xp.int32)
    inf.prefill(prompt, state)
    gen = CachedPredictiveRefinementGenerator(
        inf, trained.states, RefinementGenerationConfig(block_size=k)
    )
    return model, trained, inf, state, prompt, gen


def uncached(model, context, active):
    emb = model.embedding.W.data[context]
    x = xp.concatenate((emb, active), axis=1)
    mask = block_causal_allow_mask(context.shape[1], active.shape[1])
    return model.forward(
        None, input_embeddings=x, attention_allow_mask=mask, return_cache=False
    )[:, context.shape[1]:, :]


def test_cached_provisional_matches_uncached_reference_and_cache_is_readonly():
    model, trained, inf, state, context, gen = setup()
    active = trained.states.unknown_block(1, 3)
    k_before = state.k_cache.copy()
    v_before = state.v_cache.copy()
    cached = inf.forward_provisional_block(active, state)
    from_scratch = uncached(model, context, active)
    np.testing.assert_allclose(asnumpy(cached), asnumpy(from_scratch), atol=1e-8, rtol=1e-8)
    np.testing.assert_array_equal(asnumpy(state.k_cache[:, :, :, :state.length]), asnumpy(k_before[:, :, :, :state.length]))
    np.testing.assert_array_equal(asnumpy(state.v_cache[:, :, :, :state.length]), asnumpy(v_before[:, :, :, :state.length]))
    # Any change in future soft-position embedding can change first prediction
    # but must not touch committed context KV.
    changed = active.copy()
    changed[:, -1, :] += 4.0
    changed_logits = inf.forward_provisional_block(changed, state)
    assert not np.allclose(asnumpy(cached[:, 0]), asnumpy(changed_logits[:, 0]))
    np.testing.assert_array_equal(asnumpy(state.k_cache[:, :, :, :state.length]), asnumpy(k_before[:, :, :, :state.length]))


def test_append_cache_is_identical_to_fresh_causal_prefill():
    model, trained, inf, state, prompt, gen = setup(k=2)
    more = xp.asarray([[24, 2, 17]], dtype=xp.int32)
    inf.append_causal_block(more, state)
    assert state.length == 6
    state2 = inf.create_generation_state(1, 16)
    inf.prefill(xp.concatenate((prompt, more), axis=1), state2)
    np.testing.assert_allclose(asnumpy(state.k_cache[:, :, :, :6]), asnumpy(state2.k_cache[:, :, :, :6]), rtol=1e-12, atol=1e-12)
    np.testing.assert_allclose(asnumpy(state.v_cache[:, :, :, :6]), asnumpy(state2.v_cache[:, :, :, :6]), rtol=1e-12, atol=1e-12)


def test_progressive_generation_uses_correct_passes_and_updates_cache_only_after():
    model, trained, inf, state, prompt, gen = setup(k=3)
    result = gen.generate_block(state, return_trace=True)
    assert result["passes"] == 4
    assert len(result["tokens"]) == 3
    assert len(result["trace"]) == 4
    assert state.length == 6
    expected = xp.concatenate((prompt, xp.asarray([result["tokens"]], dtype=xp.int32)), axis=1)
    state2 = inf.create_generation_state(1, 16)
    inf.prefill(expected, state2)
    np.testing.assert_allclose(asnumpy(state.k_cache[:, :, :, :6]), asnumpy(state2.k_cache[:, :, :, :6]), atol=1e-12)
    np.testing.assert_allclose(asnumpy(state.v_cache[:, :, :, :6]), asnumpy(state2.v_cache[:, :, :, :6]), atol=1e-12)
    # The refined token is chosen from pass j+1, not pass zero.
    for j, token in enumerate(result["tokens"]):
        assert token == int(xp.argmax(result["trace"][j + 1][0, j]))


def test_three_consecutive_blocks_match_fresh_uncached_logits():
    model, trained, inf, state, prompt, gen = setup(k=2)
    context = prompt.copy()
    for _ in range(3):
        unknown = trained.states.unknown_block(1, 2)
        cached = inf.forward_provisional_block(unknown, state)
        brute = uncached(model, context, unknown)
        np.testing.assert_allclose(asnumpy(cached), asnumpy(brute), atol=1e-8, rtol=1e-8)
        result = gen.generate_block(state)
        context = xp.concatenate((context, xp.asarray([result["tokens"]], dtype=xp.int32)), axis=1)
        assert state.length == context.shape[1]
    assert context.shape[1] == 9


def test_eos_ends_block_before_k_and_cache_is_finalized():
    model, trained, inf, state, prompt, _ = setup(k=4)
    # Establish first greedy refined token without mutating state.
    init = inf.forward_provisional_block(trained.states.unknown_block(1, 4), state)
    exp, _ = gen_expectation(inf, init)
    first_logits = inf.forward_provisional_block(trained.states.soft_block(exp), state)
    eos = int(xp.argmax(first_logits[0, 0]))
    gen = CachedPredictiveRefinementGenerator(
        inf, trained.states, RefinementGenerationConfig(block_size=4, eos_token_id=eos)
    )
    result = gen.generate_block(state)
    assert result == {"tokens": [eos], "passes": 2, "eos": True}
    assert state.length == 4


def gen_expectation(inf, logits):
    from mini_llm.predictive_refinement.soft_embeddings import SoftTokenExpectation
    return SoftTokenExpectation(inf.embedding.W).forward(logits)


def test_generation_no_capacity_and_k1():
    model, trained, inf, state, prompt, gen = setup(k=1)
    result = gen.generate_block(state)
    assert result["passes"] == 2
    state.length = state.max_length
    assert gen.generate_block(state)["tokens"] == []
    with pytest.raises(ValueError):
        inf.forward_provisional_block(trained.states.unknown_block(1, 1), state)
