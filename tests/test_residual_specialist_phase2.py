"""Small deterministic CPU prototype tests, not a GPU training benchmark."""
from dataclasses import replace

import numpy as np
import pytest

from mini_llm.config import ModelConfig
from mini_llm.model.decoder_lm import DecoderLanguageModel
from mini_llm.optim.hybrid_muon import muon_eligible
from mini_llm.specialists import ResidualSpecialist, SpecialistConfig, FrozenFoundation


def setup():
    base_cfg = ModelConfig(
        tokenizer_vocab_size=32, context_length=8, n_layers=1,
        d_model=16, n_q_heads=2, n_kv_heads=1, d_head=8,
        n_experts=1, top_k=1, d_ff=32, ffn_mode="dense", dtype="float32",
    )
    model = DecoderLanguageModel(base_cfg, rng_seed=99)
    # Model weight buffers still exist here; the frozen loader is independently
    # tested to remove them completely.
    frozen = FrozenFoundation(model=model)
    spec = SpecialistConfig(
        specialist_id="python", n_layers=2, d_model=16, d_ff=32,
        n_q_heads=2, n_kv_heads=1, d_head=8, full_window=8,
        local_window=4, dtype="float32",
    )
    specialist = ResidualSpecialist(spec, rng_seed=13)
    return frozen, specialist


def test_zero_output_projection_and_gradient_flow():
    foundation, specialist = setup()
    source = np.arange(8, dtype=np.int64)[None, :]
    hidden, _ = foundation.hidden(source)
    delta, cache = specialist.forward_delta(hidden)
    assert np.array_equal(delta, np.zeros_like(delta))
    predicted = foundation.project(specialist.corrected_hidden(hidden, delta))
    np.testing.assert_array_equal(predicted, foundation.predict(source))
    targets = (source + 1) % 32
    loss, loss_cache = foundation.head_loss(
        specialist.corrected_hidden(hidden, delta), targets, chunk_tokens=4
    )
    dh = foundation.head_backward(loss_cache)
    specialist.backward_delta(dh, cache)
    assert np.isfinite(loss)
    assert np.any(specialist.correction_proj.W.grad != 0)
    assert all(np.all(p.grad == 0) for block in specialist.blocks for p in block.parameters())
    # Simulate first optimizer update. Next step must propagate into blocks.
    specialist.correction_proj.W.data[...] -= 1e-2 * specialist.correction_proj.W.grad
    specialist.zero_grad()
    delta, cache = specialist.forward_delta(hidden)
    loss, loss_cache = foundation.head_loss(
        specialist.corrected_hidden(hidden, delta), targets, chunk_tokens=4
    )
    dh = foundation.head_backward(loss_cache)
    specialist.backward_delta(dh, cache)
    assert any(np.any(p.grad) for block in specialist.blocks for p in block.parameters())


def test_recompute_matches_nonrecompute():
    foundation, spec = setup()
    source = np.arange(8, dtype=np.int64)[None, :]
    hidden, _ = foundation.hidden(source)
    # Unfreeze just the specialist's output projection for informative gradients.
    spec.correction_proj.W.data[...] = np.eye(16, dtype=np.float32) * 0.01
    spec.zero_grad()
    delta_a, cache_a = spec.forward_delta(hidden, activation_checkpoint=False)
    spec.backward_delta(np.ones_like(delta_a), cache_a)
    first = [p.grad.copy() for p in spec.parameters()]
    spec.zero_grad()
    delta_b, cache_b = spec.forward_delta(hidden, activation_checkpoint=True)
    spec.backward_delta(np.ones_like(delta_b), cache_b)
    np.testing.assert_allclose(delta_a, delta_b, rtol=1e-5, atol=1e-6)
    for old, p in zip(first, spec.parameters()):
        np.testing.assert_allclose(old, p.grad, rtol=1e-4, atol=1e-6)


def test_causal_prefix_invariance():
    foundation, spec = setup()
    x = np.random.RandomState(3).normal(size=(1, 8, 16)).astype("float32")
    spec.correction_proj.W.data[...] = np.eye(16, dtype=np.float32) * 0.01
    y = spec.forward_delta(x, return_cache=False)
    changed = x.copy()
    changed[:, 5:, :] += 3.0
    new = spec.forward_delta(changed, return_cache=False)
    np.testing.assert_allclose(y[:, :5], new[:, :5], atol=1e-5, rtol=1e-5)


def test_optimizer_name_scoping_and_memory_mode():
    _, spec = setup()
    eligible = [p.name for p in spec.parameters() if muon_eligible(p)]
    assert any(".attention." in p for p in eligible)
    assert any(".dense_ffn." in p for p in eligible)
    assert not any("correction_proj" in p for p in eligible)
    with pytest.raises(NotImplementedError, match="private"):
        ResidualSpecialist(replace(spec.config, memory_mode="private"))


def test_output_projection_gradient_finite_difference():
    foundation, specialist = setup()
    source = np.arange(8, dtype=np.int64)[None, :]
    target = (source + 1) % 32
    h, _ = foundation.hidden(source)
    delta, cache = specialist.forward_delta(h)
    loss, loss_cache = foundation.head_loss(
        specialist.corrected_hidden(h, delta), target, chunk_tokens=8
    )
    gradient = foundation.head_backward(loss_cache)
    specialist.backward_delta(gradient, cache)
    analytic = specialist.correction_proj.W.grad[0, 0].item()
    weight = specialist.correction_proj.W.data
    eps = 1e-3
    def score(value):
        weight[0, 0] = value
        d = specialist.forward_delta(h, return_cache=False)
        result, _ = foundation.head_loss(
            specialist.corrected_hidden(h, d), target, chunk_tokens=8
        )
        return result
    plus, minus = score(eps), score(-eps)
    weight[0, 0] = 0
    finite_difference = (plus - minus) / (2 * eps)
    np.testing.assert_allclose(analytic, finite_difference, atol=4e-4, rtol=2e-2)


def test_twelve_layer_specialist_smoke():
    _, small = setup()
    config = replace(small.config, n_layers=12)
    large = ResidualSpecialist(config, rng_seed=17)
    assert len(large.blocks) == 12
    x = np.random.RandomState(0).normal(size=(1, 6, 16)).astype("float32")
    delta = large.forward_delta(x, return_cache=False)
    np.testing.assert_array_equal(delta, np.zeros_like(x))
