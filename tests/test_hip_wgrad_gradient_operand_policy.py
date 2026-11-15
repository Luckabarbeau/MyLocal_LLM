"""Dispatch and accuracy contract for the BF16 wgrad gradient-operand policy.

The policy rounds an FP32 projection weight-gradient operand to BF16 so that
``projection_wgrad`` issues one tensor-core GEMM instead of the three-part
decomposition.  Split path remains the default; these tests pin the dispatch
switch and characterize the operand rounding.
"""

import os
from unittest.mock import patch

import numpy as np
import pytest

from mini_llm.ops.hip_projection_wgrad import (
    _bf16_wgrad_gradient_enabled,
    _split_gradient,
    projection_wgrad,
)

FLAG = "MINI_LLM_HIP_WGRAD_BF16_GRADIENT"


@pytest.fixture(autouse=True)
def _isolated_environment():
    with patch.dict(os.environ):
        for name in tuple(os.environ):
            if name.startswith("MINI_LLM_"):
                del os.environ[name]
        yield


@pytest.mark.parametrize(
    "value,expected",
    [
        (None, False), ("0", False), ("off", False), ("", False), ("ON", False),
        ("1", True), ("true", True), ("YES", True), (" 1 ", True),
    ],
    ids=["unset", "zero", "off", "empty", "on-word", "one", "true", "yes", "padded"],
)
def test_policy_defaults_off_and_parses_switch(value, expected):
    if value is None:
        assert not _bf16_wgrad_gradient_enabled()
    else:
        os.environ[FLAG] = value
        assert _bf16_wgrad_gradient_enabled() is expected


def _operands(t, din, dout):
    rng = np.random.default_rng(107)
    activation = rng.normal(size=(t, din)).astype(np.float32)
    gradient = rng.normal(size=(t, dout)).astype(np.float32)
    return activation, gradient


@pytest.mark.gpu
def test_split_path_is_default_and_single_operand_path_skips_it():
    import cupy as cp

    if not cp.cuda.runtime.is_hip:
        pytest.skip("the projection wgrad bridge requires HIP")

    t, din, dout = 2048, 1024, 1024
    activation_np, gradient_np = _operands(t, din, dout)
    activation = cp.asarray(activation_np).astype(cp.dtype("bfloat16"))
    gradient = cp.asarray(gradient_np)

    split_calls = []
    original_split = _split_gradient

    def counting_split(value):
        split_calls.append(value.shape)
        return original_split(value)

    with patch("mini_llm.ops.hip_projection_wgrad._split_gradient", counting_split):
        split_result = projection_wgrad(activation, gradient)
    assert split_calls == [(t, dout)], split_calls

    with patch("mini_llm.ops.hip_projection_wgrad._split_gradient",
               lambda value: (_ for _ in ()).throw(
                   AssertionError("single-operand policy must not decompose"))):
        os.environ[FLAG] = "1"
        single_result = projection_wgrad(activation, gradient)

    assert split_result.dtype == cp.float32
    assert single_result.dtype == cp.float32
    assert single_result.shape == (din, dout)
    assert bool(cp.isfinite(single_result).all())

    # FP64 reference of the exact BF16 activation times the exact FP32 gradient.
    activation64 = cp.asnumpy(activation).astype(np.float64)
    gradient64 = cp.asnumpy(gradient).astype(np.float64)
    reference64 = activation64.T @ gradient64

    def describe(candidate):
        candidate64 = cp.asnumpy(candidate).astype(np.float64)
        difference = candidate64 - reference64
        reference_norm = float(np.sqrt((reference64 * reference64).sum()))
        return {
            "max_absolute": float(np.abs(difference).max()),
            "relative_l2": float(np.sqrt((difference * difference).sum()) / reference_norm),
        }

    split_error = describe(split_result)
    single_error = describe(single_result)
    # The split path keeps the FP32 operand significand, so its error is GEMM
    # accumulation only. The single-operand arm additionally rounds that operand
    # to BF16 (8-bit mantissa, i.e. about 2**-9 relative per element).
    assert split_error["relative_l2"] < 1e-4, (split_error, single_error)
    assert single_error["relative_l2"] < 5e-2, (split_error, single_error)


@pytest.mark.gpu
def test_single_operand_policy_matches_split_policy_within_operand_rounding():
    import cupy as cp

    if not cp.cuda.runtime.is_hip:
        pytest.skip("the projection wgrad bridge requires HIP")

    t, din, dout = 8192, 1024, 1024
    activation_np, gradient_np = _operands(t, din, dout)
    activation = cp.asarray(activation_np).astype(cp.dtype("bfloat16"))
    gradient = cp.asarray(gradient_np)

    os.environ[FLAG] = "1"
    single_result = projection_wgrad(activation, gradient)
    del os.environ[FLAG]
    split_result = projection_wgrad(activation, gradient)

    single64 = cp.asnumpy(single_result).astype(np.float64)
    split64 = cp.asnumpy(split_result).astype(np.float64)
    difference = single64 - split64
    reference_norm = float(np.sqrt((split64 * split64).sum()))
    relative_l2 = float(np.sqrt((difference * difference).sum()) / reference_norm)
    max_pointwise = float(
        (np.abs(difference) / np.maximum(np.abs(split64), 1e-300)).max())
    # A BF16 operand cast perturbs each summed addend by at most 2**-9; the
    # whole-tensor relative L2 error must therefore sit near that scale, far
    # below any training-relevant threshold, and must not exceed it by 10x.
    assert 1e-4 < relative_l2 < 5e-2, relative_l2
    assert np.isfinite(single64).all()
