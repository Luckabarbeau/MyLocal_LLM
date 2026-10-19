"""0082R: exact row-weighted gradient accumulation and checkpoint continuity."""

import numpy as np
import pytest

from mini_llm.collective import FusionConfig, ProbabilityResidualCombiner
from mini_llm.optim.adamw import AdamW
from examples import train_probability_fusion_streaming_0082f as streaming


@pytest.mark.parametrize('context_heads', [0, 2])
@pytest.mark.parametrize('chunks', [(7,), (3, 3, 1), (2, 2, 2, 1)])
def test_microbatch_gradients_equal_full_batch(context_heads, chunks):
    rng = np.random.default_rng(81)
    cfg = FusionConfig(vocab_size=17, max_models=4, embed_dim=6,
                       member_width=4, decoder_width=7, seed=88,
                       context_heads=context_heads, context_head_dim=4,
                       context_hidden=5)
    z = rng.normal(size=(7, 2, 17)).astype(np.float32)
    y = rng.integers(0, 17, size=7)
    model_ids = [3, 1]
    context = {}
    if context_heads:
        context = {'context_tokens': rng.integers(0, 17, size=(7, 6), dtype=np.int32),
                   'context_lengths': np.array([1, 2, 3, 4, 5, 6, 3])}
    full = ProbabilityResidualCombiner(cfg)
    full.output_weight.data[:] = np.linspace(-.06, .07, cfg.decoder_width)
    full.reference_logits.data[3] = 0.7
    if context_heads:
        full.ctx_score.data[:] = np.linspace(-.3, .5, cfg.context_hidden)
    micro = ProbabilityResidualCombiner(cfg)
    micro.load_state_dict(full.state_dict())
    full.zero_grad()
    expected = full.loss_and_backward(z, model_ids, y, return_weighted=True, **context)
    micro.zero_grad()
    running = np.zeros(3)
    start = 0
    for count in chunks:
        end = start + count
        ctx_slice = {k: v[start:end] for k, v in context.items()}
        observed = micro.loss_and_backward(z[start:end], model_ids, y[start:end],
                                           gradient_scale=count/7,
                                           return_weighted=True, **ctx_slice)
        running += np.asarray(observed) * (count/7)
        start = end
    assert start == 7
    np.testing.assert_allclose(running, expected, rtol=3e-6, atol=3e-6)
    for name, p in full.parameters_by_name.items():
        np.testing.assert_allclose(micro.parameters_by_name[name].grad, p.grad,
                                   rtol=2e-4, atol=3e-6, err_msg=name)
    opt_full = AdamW(full.parameters(), lr=2e-4, weight_decay=0.01)
    opt_micro = AdamW(micro.parameters(), lr=2e-4, weight_decay=0.01)
    streaming._clip_grad_norm_(full.parameters(), 0.1, device_only=True)
    streaming._clip_grad_norm_(micro.parameters(), 0.1, device_only=True)
    opt_full.step()
    opt_micro.step()
    for name, p in full.parameters_by_name.items():
        np.testing.assert_allclose(micro.parameters_by_name[name].data, p.data,
                                   rtol=2e-4, atol=3e-6, err_msg=name)


def test_gradient_scale_rejects_invalid():
    model = ProbabilityResidualCombiner(FusionConfig(vocab_size=7, max_models=2))
    z = np.zeros((2, 2, 7), dtype=np.float32)
    for scale in (0, -1, np.inf, np.nan):
        with pytest.raises(ValueError, match='gradient_scale'):
            model.loss_and_backward(z, [0, 1], [1, 2], gradient_scale=scale)
