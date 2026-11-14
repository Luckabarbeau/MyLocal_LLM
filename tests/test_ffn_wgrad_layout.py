"""CPU-only dispatch contracts for the optional HIP FFN down-wgrad layout."""

from types import SimpleNamespace

import ml_dtypes
import numpy as np
import pytest

from mini_llm.ops import experts


OPTION = "MINI_LLM_HIP_FFN_TRANSPOSE_WGRAD"
BF16 = np.dtype(ml_dtypes.bfloat16)


class Operand:
    """Record layout operations without allocating or executing a GEMM."""

    def __init__(self, name, shape, dtype, calls):
        self.name = name
        self.shape = shape
        self.ndim = len(shape)
        self.dtype = np.dtype(dtype)
        self.calls = calls

    @property
    def T(self):
        self.calls.append(("transpose", self.name))
        return Operand(self.name + ".T", self.shape[::-1], self.dtype, self.calls)

    def __matmul__(self, other):
        self.calls.append(("matmul", self.name, other.name))
        # Invalid input shapes deliberately reach this mock: the helper's
        # contract is dispatch, not validation of the fallback matmul.
        return Operand(
            "product", (self.shape[0], other.shape[-1]), self.dtype, self.calls
        )


@pytest.fixture
def hip_mock(monkeypatch):
    monkeypatch.setenv(OPTION, "1")
    monkeypatch.setattr(experts, "BACKEND_NAME", "cupy")
    monkeypatch.setattr(
        experts, "xp",
        SimpleNamespace(cuda=SimpleNamespace(runtime=SimpleNamespace(is_hip=True))),
    )


def assert_dispatch(h_shape=(8192, 4096), dy_shape=(8192, 1024),
                    h_dtype=BF16, dy_dtype=BF16, swapped=False):
    calls = []
    h = Operand("h", h_shape, h_dtype, calls)
    dy = Operand("dy", dy_shape, dy_dtype, calls)
    result = experts._down_projection_wgrad(h, dy)
    expected = (
        [("transpose", "dy"), ("matmul", "dy.T", "h"),
         ("transpose", "product")]
        if swapped else
        [("transpose", "h"), ("matmul", "h.T", "dy")]
    )
    assert calls == expected
    assert result.name == ("product.T" if swapped else "product")
    return result


@pytest.mark.parametrize("rows", [8192, 16384, 8193])
def test_hip_bf16_long_shapes(hip_mock, rows):
    result = assert_dispatch((rows, 4096), (rows, 1024), swapped=True)
    assert result.shape == (4096, 1024)
    assert result.dtype == BF16


@pytest.mark.parametrize("flag", [None, "0", "false", "off", "true", "2"])
def test_flag_must_be_explicitly_one(hip_mock, monkeypatch, flag):
    if flag is None:
        monkeypatch.delenv(OPTION, raising=False)
    else:
        monkeypatch.setenv(OPTION, flag)
    assert_dispatch()


@pytest.mark.parametrize("rows", [0, 1, 1024, 8191])
def test_short_rows_use_original_layout(hip_mock, rows):
    assert_dispatch((rows, 4096), (rows, 1024))


@pytest.mark.parametrize("h_shape,dy_shape", [
    ((8192, 4096), (8193, 1024)),
    ((8193, 4096), (8192, 1024)),
    ((1, 8192, 4096), (8192, 1024)),
    ((8192, 4096), (1, 8192, 1024)),
    ((8192,), (8192, 1024)),
    ((8192, 4096), (8192,)),
    ((8192, 2048), (8192, 1024)),
    ((8192, 4096), (8192, 512)),
    ((8192, 1024), (8192, 4096)),
])
def test_shape_guards_use_original_layout(hip_mock, h_shape, dy_shape):
    assert_dispatch(h_shape, dy_shape)


@pytest.mark.parametrize("h_dtype,dy_dtype", [
    (BF16, np.float32),
    (np.float32, BF16),
    (BF16, np.float16),
    (np.float16, BF16),
    (np.float32, np.float32),
    (np.float16, np.float16),
])
def test_dtype_guards_use_original_layout(hip_mock, h_dtype, dy_dtype):
    assert_dispatch(h_dtype=h_dtype, dy_dtype=dy_dtype)


def test_cuda_uses_original_layout(hip_mock, monkeypatch):
    monkeypatch.setattr(experts.xp.cuda.runtime, "is_hip", False)
    assert_dispatch()


def test_numpy_short_circuits_gpu_runtime_access(hip_mock, monkeypatch):
    monkeypatch.setattr(experts, "BACKEND_NAME", "numpy")
    monkeypatch.setattr(experts, "xp", np)
    assert_dispatch()


def test_numpy_fallback_numerical_result(monkeypatch):
    monkeypatch.setenv(OPTION, "1")
    monkeypatch.setattr(experts, "BACKEND_NAME", "numpy")
    monkeypatch.setattr(experts, "xp", np)
    h = np.arange(15, dtype=np.float32).reshape(5, 3) / 10
    dy = np.arange(10, dtype=np.float32).reshape(5, 2) / 10
    result = experts._down_projection_wgrad(h, dy)
    np.testing.assert_array_equal(result, h.T @ dy)
    assert result.shape == (3, 2)
    assert result.dtype == np.float32
