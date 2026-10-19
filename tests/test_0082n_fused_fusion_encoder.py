"""0082N CUDA encoder/aggregator regression against unfused FP32 operations."""

import numpy as np
import pytest

from mini_llm.backend import BACKEND_NAME, xp
from mini_llm.collective import FusionConfig, ProbabilityResidualCombiner


def test_token_embedding_vocabulary_gemm_matches_einsum():
    rng = np.random.default_rng(82)
    # Check the algebra used by the two large backward vocabulary contractions.
    for shape in [(2, 2, 13, 4), (3, 3, 9, 7)]:
        rows, members, vocab, d = shape
        a = rng.normal(size=(rows, members, vocab)).astype(np.float32)
        b = rng.normal(size=(rows, members, d)).astype(np.float32)
        expected = np.einsum('rnv,rnd->vd', a, b)
        gemm = a.reshape(-1, vocab).T @ b.reshape(-1, d)
        np.testing.assert_allclose(gemm, expected, rtol=2e-6, atol=2e-6)


def test_default_numpy_api_still_handles_variable_members_and_decoder_depth():
    if BACKEND_NAME != 'numpy':
        pytest.skip('NumPy reference backend')
    rng = np.random.default_rng(93)
    for n_models in (1, 2, 3):
        for depth in (1, 3):
            model = ProbabilityResidualCombiner(FusionConfig(
                vocab_size=17, max_models=4, decoder_layers=depth))
            p, cache = model.forward(rng.normal(size=(3, n_models, 17)).astype('float32'),
                                     list(range(n_models)), return_cache=True)
            assert cache['member'].shape == (3, n_models, 17, model.config.member_width)
            assert cache['member_mean'].shape == (3, 17, model.config.member_width)
            assert np.allclose(np.sum(p, axis=-1), 1.0, atol=1e-6)


@pytest.mark.skipif(BACKEND_NAME != 'cupy', reason='GPU validation requires CuPy')
def test_fused_cuda_member_matches_unfused_cupy_reference():
    from mini_llm.collective.fusion_cuda_kernels import member_encode_aggregate, member_backward
    rng = np.random.default_rng(82)
    for rows, n_models, vocab, hidden in ((2, 1, 11, 4), (3, 2, 13, 8), (2, 3, 17, 16)):
        ratio = xp.asarray(rng.normal(size=(rows, n_models, vocab)).astype('float32'))
        model_token = xp.asarray(rng.normal(size=(n_models, vocab)).astype('float32'))
        token_context = xp.asarray(rng.normal(size=(rows, n_models, vocab)).astype('float32'))
        entropy = xp.asarray(rng.normal(size=(rows, n_models)).astype('float32'))
        weights = [xp.asarray(rng.normal(size=(hidden,)).astype('float32')) for _ in range(5)]
        a, b, c, d, bias = weights
        pre = (ratio[..., None] * a + model_token[None, ..., None] * b +
               token_context[..., None] * c + entropy[:, :, None, None] * d + bias)
        reference = xp.tanh(pre)
        mean_ref = xp.mean(reference, axis=1)
        var_ref = xp.mean((reference - mean_ref[:, None])**2, axis=1)
        member, mean, var = member_encode_aggregate(
            ratio, model_token, token_context, entropy, *weights)
        xp.testing.assert_allclose(member, reference, rtol=3e-6, atol=3e-6)
        xp.testing.assert_allclose(mean, mean_ref, rtol=3e-6, atol=3e-6)
        xp.testing.assert_allclose(var, var_ref, rtol=3e-6, atol=3e-6)
        feat = xp.asarray(rng.normal(size=(rows, vocab, 2*hidden+4)).astype('float32'))
        # Verify the backwards gather from noncontiguous mean/variance slices.
        dmean, dvar = feat[..., :hidden], feat[..., hidden:2*hidden]
        dmember = (dmean[:, None] + 2*dvar[:, None]*(member - mean[:, None]))/n_models
        dpre_ref = dmember * (1.0 - member**2)
        dpre = member_backward(member, mean, feat)
        xp.testing.assert_allclose(dpre, dpre_ref, rtol=3e-6, atol=3e-6)


@pytest.mark.skipif(BACKEND_NAME != 'cupy', reason='GPU validation requires CuPy')
def test_fused_cuda_full_forward_backward_with_context():
    rng = np.random.default_rng(90)
    model = ProbabilityResidualCombiner(FusionConfig(
        vocab_size=19, max_models=4, context_heads=2, decoder_layers=2))
    logits = xp.asarray(rng.normal(size=(3, 2, 19)).astype('float32'))
    tokens = rng.integers(0, 19, size=(3, 9), dtype=np.int64)
    lengths = np.array([9, 3, 7], dtype=np.int64)
    model.zero_grad()
    ce, arithmetic, weighted = model.loss_and_backward(
        logits, [0, 1], [2, 4, 6], context_tokens=tokens, context_lengths=lengths,
        return_weighted=True)
    assert np.isfinite([ce, arithmetic, weighted]).all()
    for p in model.parameters():
        assert bool(xp.all(xp.isfinite(p.grad)))

@pytest.mark.skipif(BACKEND_NAME != 'cupy', reason='GPU validation requires CuPy')
def test_fused_full_parameter_gradients_match_unfused_cupy(monkeypatch):
    """On a real GPU compare the entire fused training step to CuPy reference."""
    import mini_llm.collective.probability_fusion as fusion_mod

    rng = np.random.default_rng(75)
    for n_models, n_layers in ((2, 1), (3, 2)):
        cfg = FusionConfig(vocab_size=23, max_models=4, decoder_layers=n_layers)
        model = ProbabilityResidualCombiner(cfg)
        # Nonzero residual ensures gradients flow to the whole neural encoder.
        model.output_weight.data[...] = xp.asarray(
            rng.normal(0, 0.2, size=model.output_weight.data.shape).astype('float32'))
        reference = ProbabilityResidualCombiner(cfg)
        reference.load_state_dict(model.state_dict())
        logits = xp.asarray(rng.normal(size=(3, n_models, 23)).astype('float32'))
        targets = np.array([2, 4, 6], dtype=np.int64)
        ids = list(range(n_models))
        model.zero_grad()
        reference.zero_grad()
        results_gpu = model.loss_and_backward(logits, ids, targets, return_weighted=True)
        # Only redirect the fusion member encoder branch; xp remains CuPy,
        # allowing the full old and new GPU implementations to be compared.
        with monkeypatch.context() as m:
            m.setattr(fusion_mod, 'BACKEND_NAME', 'numpy')
            results_ref = reference.loss_and_backward(logits, ids, targets, return_weighted=True)
        np.testing.assert_allclose(results_gpu, results_ref, rtol=2e-5, atol=2e-5)
        for name in model.parameters_by_name:
            xp.testing.assert_allclose(
                model.parameters_by_name[name].grad,
                reference.parameters_by_name[name].grad,
                rtol=5e-4, atol=5e-5, err_msg=name)
