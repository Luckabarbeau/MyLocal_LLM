"""0082Q: decoder GEMV reductions and fused default decoder backward.

Run GPU parity before trusting performance numbers:
  MINI_LLM_BACKEND=cupy python -m pytest -q tests/test_0082q_fusion_decoder_backward.py
"""

import numpy as np
import pytest

from mini_llm.backend import BACKEND_NAME, xp, asnumpy
from mini_llm.collective import FusionConfig, ProbabilityResidualCombiner


def test_decoder_reductions_are_exact_matrix_vector_contractions():
    rng = np.random.default_rng(82)
    for rows, vocab, width in ((2, 19, 8), (3, 13, 24), (2, 7, 48)):
        activation = rng.normal(0, 0.3, size=(rows, vocab, width)).astype('float32')
        dz = rng.normal(0, 0.1, size=(rows, vocab)).astype('float32')
        dpre = rng.normal(0, 0.1, size=(rows, vocab, width)).astype('float32')
        expected_output = np.sum(activation * dz[..., None], axis=(0, 1))
        expected_bias = np.sum(dpre, axis=(0, 1))
        got_output = activation.reshape(-1, width).T @ dz.reshape(-1)
        got_bias = dpre.reshape(-1, width).T @ np.ones(rows * vocab, dtype=np.float32)
        np.testing.assert_allclose(got_output, expected_output, rtol=2e-5, atol=5e-6)
        np.testing.assert_allclose(got_bias, expected_bias, rtol=2e-5, atol=5e-6)


def test_decoder_bias_reduction_retains_numpy_reference():
    if BACKEND_NAME != 'numpy':
        pytest.skip('NumPy reference only')
    cfg = FusionConfig(vocab_size=19, decoder_layers=3)
    model = ProbabilityResidualCombiner(cfg)
    rng = np.random.default_rng(19)
    for rows in (2, 4):
        data = rng.normal(size=(rows, 19, cfg.decoder_width)).astype('float32')
        np.testing.assert_array_equal(model._decoder_bias_gradient(data),
                                      np.sum(data, axis=(0, 1)))


@pytest.mark.skipif(BACKEND_NAME != 'cupy', reason='requires CUDA and CuPy')
def test_fp32_cublas_accumulation_matches_cupy_matmul():
    from mini_llm.collective.fusion_cublas import fp32_matmul_into

    rng = np.random.default_rng(31)
    for rows, inner, width in ((7, 5, 3), (19, 8, 24), (11, 16, 7)):
        a = xp.asarray(rng.normal(size=(rows, inner)).astype('float32'))
        b = xp.asarray(rng.normal(size=(inner, width)).astype('float32'))
        c = xp.asarray(rng.normal(size=(rows, inner)).astype('float32'))
        d = xp.asarray(rng.normal(size=(inner, width)).astype('float32'))
        out = xp.empty((rows, width), dtype=xp.float32)
        fp32_matmul_into(a, b, out)
        xp.testing.assert_allclose(out, a @ b, rtol=3e-6, atol=3e-6)
        fp32_matmul_into(c, d, out, accumulate=True)
        xp.testing.assert_allclose(out, a @ b + c @ d, rtol=3e-6, atol=3e-6)


@pytest.mark.skipif(BACKEND_NAME != 'cupy', reason='requires CUDA and CuPy')
def test_gpu_fused_decoder_propagation_matches_unfused():
    from mini_llm.collective.fusion_cuda_kernels import (
        decoder_activation_backward, decoder_bias_tanh_inplace,
        decoder_first_layer_backward, decoder_scalar_features_tanh_inplace)

    rng = np.random.default_rng(49)
    for rows, vocab, width in ((1, 13, 4), (3, 67, 24), (2, 17, 48)):
        decoder = xp.asarray(np.tanh(rng.normal(size=(rows, vocab, width))).astype('float32'))
        dz = xp.asarray(rng.normal(0, .1, (rows, vocab)).astype('float32'))
        weight = xp.asarray(rng.normal(0, .1, (width,)).astype('float32'))
        expected = (dz[..., None] * weight) * (1.0 - decoder ** 2)
        actual = decoder_first_layer_backward(decoder, dz, weight)
        xp.testing.assert_allclose(actual, expected, rtol=3e-6, atol=3e-6)
        upstream = xp.asarray(rng.normal(0, .1, decoder.shape).astype('float32'))
        expected_activation = upstream * (1.0 - decoder ** 2)
        actual_activation = decoder_activation_backward(decoder, upstream)
        xp.testing.assert_allclose(actual_activation, expected_activation,
                                   rtol=3e-6, atol=3e-6)
        bias = xp.asarray(rng.normal(0, .1, (width,)).astype('float32'))
        linear = xp.asarray(rng.normal(0, .2, decoder.shape).astype('float32'))
        expected_forward = xp.tanh(linear + bias)
        actual_forward = decoder_bias_tanh_inplace(linear.copy(), bias)
        xp.testing.assert_allclose(actual_forward, expected_forward,
                                   rtol=3e-6, atol=3e-6)
        log_pa = xp.asarray(rng.normal(size=(rows, vocab)).astype('float32'))
        difference = xp.asarray(rng.normal(size=(rows, vocab)).astype('float32'))
        variance = xp.asarray(rng.normal(size=(rows, vocab)).astype('float32'))
        scalar_weight = xp.asarray(rng.normal(size=(4, width)).astype('float32'))
        count = np.float32(0.37)
        expected_scalar = xp.tanh(
            linear + bias + (log_pa / 10.0)[..., None] * scalar_weight[0]
            + (difference / 10.0)[..., None] * scalar_weight[1]
            + (variance / 100.0)[..., None] * scalar_weight[2]
            + count * scalar_weight[3])
        actual_scalar = decoder_scalar_features_tanh_inplace(
            linear.copy(), log_pa, difference, variance, scalar_weight, bias, count)
        xp.testing.assert_allclose(actual_scalar, expected_scalar,
                                   rtol=3e-6, atol=3e-6)


@pytest.mark.skipif(BACKEND_NAME != 'cupy', reason='requires CUDA and CuPy')
def test_full_cuda_fusion_gradients_match_original_decoder(monkeypatch):
    import mini_llm.collective.probability_fusion as fusion_mod

    rng = np.random.default_rng(107)
    # Both native shallow GPU fast path and generic deeper decoder path,
    # including contextual attention which reuses decoder gradients.
    for members, depth, context, member_width in (
        (1, 1, 0, 8), (2, 1, 0, 8), (3, 3, 0, 8),
        (2, 2, 2, 8), (2, 4, 0, 64),
    ):
        cfg = FusionConfig(vocab_size=47, max_models=4, member_width=member_width,
                           decoder_layers=depth, context_heads=context)
        fast = ProbabilityResidualCombiner(cfg)
        fast.output_weight.data[...] = xp.asarray(
            rng.normal(0, .12, fast.output_weight.data.shape).astype('float32'))
        ref = ProbabilityResidualCombiner(cfg)
        ref.load_state_dict(fast.state_dict())
        logits = xp.asarray(rng.normal(size=(5, members, 47)).astype('float32'))
        targets = rng.integers(0, 47, size=5, dtype=np.int64)
        ids = list(range(members))
        kw = {}
        if context:
            kw['context_tokens'] = rng.integers(0, 47, size=(5, 11), dtype=np.int64)
            kw['context_lengths'] = np.array([11, 7, 8, 3, 10], dtype=np.int64)
        fast.zero_grad()
        ref.zero_grad()
        probability, cache = fast.forward(logits, ids, return_cache=True, **kw)
        xp.testing.assert_allclose(xp.exp(cache['log_final']), probability,
                                   rtol=3e-6, atol=3e-6)
        actual = fast.loss_and_backward(logits, ids, targets, return_weighted=True, **kw)
        with monkeypatch.context() as mp:
            mp.setattr(fusion_mod, 'BACKEND_NAME', 'numpy')
            expected = ref.loss_and_backward(logits, ids, targets, return_weighted=True, **kw)
        np.testing.assert_allclose(actual, expected, rtol=2e-5, atol=2e-5)
        for name in fast.parameters_by_name:
            np.testing.assert_allclose(asnumpy(fast.parameters_by_name[name].grad),
                                       asnumpy(ref.parameters_by_name[name].grad),
                                       rtol=2e-3, atol=6e-5, err_msg=name)


@pytest.mark.skipif(BACKEND_NAME != 'cupy', reason='requires CUDA and CuPy')
def test_gpu_decoder_profiler_has_individual_backward_stages():
    from examples.benchmark_probability_fusion_batches_0082m import run_batch
    elapsed, rate, stages, _ = run_batch(3, members=2, vocab=37,
                                         warmup=1, repeats=2, clip=1.0,
                                         profile=True)
    names = {s['stage'] for s in stages}
    assert {'backward.output_head.weight_grad',
            'backward.output_head.bias_grad',
            'backward.output_head.propagate_and_decoder_activation',
            'backward.decoder.weight_grad',
            'backward.decoder.bias_grad',
            'backward.decoder.features_grad'} <= names
    assert elapsed > 0 and rate > 0
