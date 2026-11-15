"""LM-head weight-gradient staging: gate, kernel and accuracy contract.

``chunked_lm_head_backward`` stages each chunk of the vocabulary-wide weight
gradient in contiguous FP32 scratch (``projection_wgrad(..., out=scratch)``) and
combines it once with the registered parameter gradient, through
``accumulate_transposed`` when the head is tied (the gradient is physically
``[Vocab, D]`` while the fast GEMM orientation produces ``[D, Vocab]``).

Two things are pinned here: the ``MINI_LLM_HIP_FP32_PROJECTION_WGRAD`` gate
selecting the staged path, and the fact that the staged path is numerically
closer to the FP64 reference than the previous BF16-product-plus-strided-add
form. The NumPy/CUDA fallback expression is unchanged.
"""

import os
from unittest.mock import patch

import numpy as np
import pytest

from mini_llm.model import decoder_lm
from mini_llm.model.decoder_lm import _hip_lm_head_weight_gradient_enabled
from mini_llm.ops.hip_projection_wgrad import (
    accumulate_transposed,
    projection_wgrad,
)

WGRAD_FLAG = "MINI_LLM_HIP_FP32_PROJECTION_WGRAD"


@pytest.fixture(autouse=True)
def _isolated_environment():
    with patch.dict(os.environ):
        for name in tuple(os.environ):
            if name.startswith("MINI_LLM_"):
                del os.environ[name]
        yield


def test_weight_gradient_gate_off_for_numpy_backend():
    with patch.object(decoder_lm, "BACKEND_NAME", "numpy"), \
            patch.object(decoder_lm, "xp", np):
        assert _hip_lm_head_weight_gradient_enabled(
            np.zeros((4, 8), np.float32), np.zeros((8, 16), np.float32)) is False


@pytest.mark.gpu
def test_weight_gradient_gate_on_hip_bf16_follows_the_existing_switch():
    import cupy as cp

    if not cp.cuda.runtime.is_hip:
        pytest.skip("the staged LM-head weight gradient requires HIP")

    bf16 = cp.dtype("bfloat16")
    activation = cp.zeros((4, 8), dtype=bf16)
    weight = cp.zeros((8, 16), dtype=bf16)
    with patch.object(decoder_lm, "BACKEND_NAME", "cupy"), \
            patch.object(decoder_lm, "xp", cp):
        assert _hip_lm_head_weight_gradient_enabled(activation, weight) is True
        os.environ[WGRAD_FLAG] = "0"
        assert _hip_lm_head_weight_gradient_enabled(activation, weight) is False
        os.environ[WGRAD_FLAG] = "1"
        assert _hip_lm_head_weight_gradient_enabled(activation, weight) is True
        assert _hip_lm_head_weight_gradient_enabled(
            activation.astype(cp.float32), weight) is False
        assert _hip_lm_head_weight_gradient_enabled(
            activation, weight.astype(cp.float32)) is False


def _tiled_shapes():
    return [(16, 16), (17, 16), (16, 17), (33, 31), (1, 64), (64, 1), (96, 48)]


@pytest.mark.gpu
def test_accumulate_transposed_matches_the_strided_expression():
    import cupy as cp

    if not cp.cuda.runtime.is_hip:
        pytest.skip("accumulate_transposed requires HIP")

    rng = np.random.default_rng(8113)
    for rows, cols in _tiled_shapes():
        destination_np = rng.normal(size=(rows, cols)).astype(np.float32)
        source_np = rng.normal(size=(cols, rows)).astype(np.float32)

        eager = cp.asarray(destination_np.copy())
        eager += cp.asarray(source_np).T
        staged = cp.asarray(destination_np.copy())
        accumulate_transposed(staged, cp.asarray(source_np))
        cp.cuda.get_current_stream().synchronize()

        assert bool(cp.array_equal(eager, staged)), (rows, cols)


@pytest.mark.gpu
def test_accumulate_transposed_validation():
    import cupy as cp

    if not cp.cuda.runtime.is_hip:
        pytest.skip("accumulate_transposed requires HIP")

    good_source = cp.ones((6, 4), dtype=cp.float32)
    with pytest.raises(TypeError, match="requires CuPy arrays"):
        accumulate_transposed(np.ones((4, 6), np.float32), good_source)
    with pytest.raises(ValueError, match="rank-2"):
        accumulate_transposed(cp.ones((4, 6, 1), cp.float32),
                              cp.ones((6, 4, 1), cp.float32))
    with pytest.raises(ValueError, match="C-contiguous"):
        accumulate_transposed(cp.ones((6, 4), cp.float32),
                              cp.ones((4, 6), cp.float32).T)
    with pytest.raises(TypeError, match="requires float32"):
        accumulate_transposed(cp.ones((4, 6), cp.float32),
                              cp.ones((6, 4), cp.dtype("bfloat16")))
    with pytest.raises(ValueError, match="shapes must be"):
        accumulate_transposed(cp.ones((4, 6), cp.float32),
                              cp.ones((5, 4), cp.float32))
    assert accumulate_transposed(cp.zeros((4, 6), cp.float32),
                                cp.zeros((6, 4), cp.float32)) is None


@pytest.mark.gpu
def test_out_accumulates_or_overwrites_per_caller():
    import cupy as cp

    if not cp.cuda.runtime.is_hip:
        pytest.skip("the projection wgrad bridge requires HIP")

    t, din, dout = 512, 128, 256
    rng = np.random.default_rng(4102)
    activation = (cp.asarray(rng.normal(size=(t, din)).astype(np.float32))
                  .astype(cp.dtype("bfloat16")))
    gradient = (cp.asarray(rng.normal(size=(t, dout)).astype(np.float32))
                .astype(cp.dtype("bfloat16")))

    seed = cp.full((din, dout), 0.25, dtype=cp.float32)
    fresh = projection_wgrad(activation, gradient)
    overwritten = projection_wgrad(activation, gradient, out=seed.copy(),
                                  accumulate=False)
    staged = projection_wgrad(activation, gradient, out=seed.copy(),
                              accumulate=True)
    once_more = projection_wgrad(activation, gradient, out=staged,
                                accumulate=True)
    cp.cuda.get_current_stream().synchronize()

    assert overwritten is not fresh
    assert once_more is staged
    # accumulate=True uses beta=1 for every part, so the 0.25 seed planted in the
    # output buffer survives the two accumulations and must be subtracted here.
    assert float(cp.abs(once_more - 0.25 - fresh * 2).max()) < 2e-2
    assert float(cp.abs(overwritten - fresh).max()) < 1e-6


@pytest.mark.gpu
def test_out_validation_rejects_incompatible_buffers():
    import cupy as cp

    if not cp.cuda.runtime.is_hip:
        pytest.skip("the projection wgrad bridge requires HIP")

    activation = cp.ones((8, 4), dtype=cp.dtype("bfloat16"))
    gradient = cp.ones((8, 6), dtype=cp.dtype("bfloat16"))
    with pytest.raises(TypeError, match="out must be a CuPy array"):
        projection_wgrad(activation, gradient, out=np.zeros((4, 6), np.float32))
    with pytest.raises(TypeError, match="out must have dtype float32"):
        projection_wgrad(activation, gradient,
                         out=cp.zeros((4, 6), cp.dtype("bfloat16")))
    with pytest.raises(ValueError, match="out must match"):
        projection_wgrad(activation, gradient, out=cp.zeros((4, 5), cp.float32))
    with pytest.raises(ValueError, match="out must be rank-2"):
        projection_wgrad(activation, gradient, out=cp.zeros((4, 6, 1), cp.float32))
    with pytest.raises(ValueError, match="out must be C-contiguous"):
        projection_wgrad(activation, gradient,
                         out=cp.zeros((6, 4), cp.float32).T)


@pytest.mark.gpu
def test_staged_wgrad_is_closer_than_the_bf16_product_plus_strided_add():
    import cupy as cp

    if not cp.cuda.runtime.is_hip:
        pytest.skip("the projection wgrad bridge requires HIP")

    # Production tied-head layout: the FP32 gradient is row-major [Vocab, D],
    # so the staged path ends in accumulate_transposed and the previous form
    # accumulated through the transpose view of that buffer.
    vocab, din, t = 4096, 128, 2048
    rng = np.random.default_rng(20260113)
    activation = (cp.asarray(rng.normal(size=(t, din)).astype(np.float32))
                  .astype(cp.dtype("bfloat16")))
    gradient = (cp.asarray(rng.normal(size=(t, vocab)).astype(np.float32))
                .astype(cp.dtype("bfloat16")))

    activation64 = cp.asnumpy(activation).astype(np.float64)
    gradient64 = cp.asnumpy(gradient).astype(np.float64)
    reference64 = activation64.T @ gradient64
    reference_norm = float(np.sqrt((reference64 * reference64).sum()))

    staged_scratch = cp.zeros((din, vocab), dtype=cp.float32)
    projection_wgrad(activation, gradient, out=staged_scratch)
    staged_grad = cp.zeros((vocab, din), dtype=cp.float32)
    accumulate_transposed(staged_grad, staged_scratch)

    previous_grad = cp.zeros((vocab, din), dtype=cp.float32)
    # CuPy cannot write through a transposed view, so this is the algebraically
    # identical form of production's old `W_grad.T.__iadd__(x.T @ d_logits)`.
    previous_grad += (activation.T @ gradient).T
    cp.cuda.get_current_stream().synchronize()

    def describe(candidate):
        values = cp.asnumpy(candidate.T).astype(np.float64)
        difference = values - reference64
        return {
            "max_absolute": float(np.abs(difference).max()),
            "relative_l2": float(np.sqrt((difference * difference).sum())
                                 / reference_norm),
        }

    staged_error = describe(staged_grad)
    previous_error = describe(previous_grad)
    assert bool(cp.isfinite(staged_grad).all())
    # FP32 staging sees GEMM accumulation error only; the previous form rounds
    # the whole [Din, Vocab] product to BF16 before adding it.
    assert staged_error["relative_l2"] < 1e-4, (staged_error, previous_error)
    assert previous_error["relative_l2"] > staged_error["relative_l2"] * 10.0, (
        staged_error, previous_error)
    assert previous_error["relative_l2"] < 5e-2, (staged_error, previous_error)
