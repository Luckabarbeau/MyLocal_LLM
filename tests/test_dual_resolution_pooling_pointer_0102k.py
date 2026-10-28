"""0102K: CUDA pooling launch must pass typed device pointers to Triton.

This is a CPU-only protocol test: the real pooling kernels are compared with
CuPy numerically by the existing 0102J GPU parity tests.  A fake launcher
catches passing a raw CuPy-like ndarray, which Triton 3.8 cannot specialize.
"""
from types import SimpleNamespace

import numpy as np
import pytest

from mini_llm.ops import triton_dual_resolution_pooling as pooling


class _FakeDeviceArray:
    def __init__(self, shape, dtype=np.float32):
        backing = np.empty(shape, dtype=dtype)
        self.shape = backing.shape
        self.ndim = backing.ndim
        self.strides = backing.strides
        self.itemsize = backing.itemsize
        self.dtype = backing.dtype
        self.data = SimpleNamespace(ptr=id(backing))
        self._backing = backing


class _FakeLauncher:
    def __init__(self, pointer_count):
        self.pointer_count = pointer_count
        self.launches = []

    def __getitem__(self, grid):
        def launch(*args, **kwargs):
            assert len(grid) == 2
            for ptr in args[:self.pointer_count]:
                assert isinstance(ptr, pooling._Pointer)
                assert isinstance(ptr.data_ptr(), int)
                assert ptr.dtype == 'fp32'
            self.launches.append((grid, args, kwargs))
        return launch


def test_pooling_forward_backward_use_typed_cuda_pointers(monkeypatch):
    """Both launches must wrap *all* device arrays, not just Q/K/V."""
    fake_tl = SimpleNamespace(float32='fp32', float16='fp16', bfloat16='bf16')
    fake_triton = SimpleNamespace(next_power_of_2=lambda n: 1 << (n - 1).bit_length())
    fake_cp = SimpleNamespace(
        float32=np.float32,
        empty=lambda shape, dtype: _FakeDeviceArray(shape, dtype),
        empty_like=lambda other: _FakeDeviceArray(other.shape, other.dtype),
        sum=lambda arr, axis: np.zeros((arr.shape[1], arr.shape[2]), dtype=np.float32),
    )
    fwd_spy = _FakeLauncher(5)
    bwd_spy = _FakeLauncher(8)
    monkeypatch.setattr(pooling, 'tl', fake_tl)
    monkeypatch.setattr(pooling, '_require_gpu', lambda: (fake_cp, fake_triton, fake_tl))
    monkeypatch.setattr(pooling, '_pool_forward', fwd_spy, raising=False)
    monkeypatch.setattr(pooling, '_pool_backward', bwd_spy, raising=False)

    k = _FakeDeviceArray((2, 8, 128, 64))
    v = _FakeDeviceArray((2, 8, 128, 64))
    q = _FakeDeviceArray((8, 64))
    context, probs = pooling.forward(k, v, q, 0.125)
    grads = pooling.backward(context, k, v, q, probs, 0.125)
    assert fwd_spy.launches[0][0] == (2, 8)
    assert bwd_spy.launches[0][0] == (2, 8)
    assert context.shape == (2, 8, 64)
    assert probs.shape == (2, 8, 128)
    assert grads[0].shape == k.shape
    assert grads[1].shape == v.shape
    assert grads[2].shape == q.shape


def test_pooling_pointer_types_and_validation(monkeypatch):
    monkeypatch.setattr(pooling, 'tl', SimpleNamespace(
        float32='fp32', float16='fp16', bfloat16='bf16'))
    for dtype, triton_type in ((np.float32, 'fp32'), (np.float16, 'fp16')):
        arr = _FakeDeviceArray((2,), dtype=dtype)
        wrapped = pooling._ptr(arr)
        assert wrapped.array is arr
        assert wrapped.dtype == triton_type
        assert wrapped.data_ptr() == arr.data.ptr
    arr = _FakeDeviceArray((2,), dtype=np.int32)
    with pytest.raises(TypeError, match='unsupported Triton pooling dtype'):
        pooling._ptr(arr)
