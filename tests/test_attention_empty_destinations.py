"""Exercise the real Triton dispatch's zero-work branches without GPU launches."""

from types import SimpleNamespace
import sys

import numpy as np
import pytest


@pytest.fixture
def attention(monkeypatch):
    pytest.importorskip("triton")
    from mini_llm.ops import triton_dual_resolution_dense as module
    # The tested branches must return without launching any device kernels.
    fake = SimpleNamespace(ndarray=np.ndarray, asarray=np.asarray, zeros=np.zeros,
                           float32=np.float32, int32=np.int32, uint8=np.uint8, bool_=np.bool_)
    monkeypatch.setitem(sys.modules, "cupy", fake)
    return module


@pytest.mark.parametrize("length,phase", [(1, 1), (5, 8), (8, 8), (15, 15)])
def test_empty_sparse_forward_overwrites_caller_head_and_lse(attention, length, phase):
    q = np.zeros((1, length, 1, 64), dtype=np.float32)
    keys = np.zeros_like(q)[:, phase::16]
    # Match actual integration: a strided slice of the skipped local head.
    storage = np.full((1, length, 16, 64), 91, dtype=np.float32)
    lse_storage = np.full((1, length, 16), 92, dtype=np.float32)
    out = storage[:, :, 1:2]
    lse = lse_storage[:, :, 1:2]
    result, cache = attention.forward(
        q, keys, keys, scale=0.125, documents=np.zeros((1, length), dtype=np.int32),
        bounded_causal=True, sparse_stride=16, sparse_phase=phase, out=out, lse=lse,
    )
    assert result is out and cache.lse is lse
    np.testing.assert_array_equal(out, 0)
    np.testing.assert_array_equal(lse, 0)
    np.testing.assert_array_equal(storage[:, :, 0], 91)
    np.testing.assert_array_equal(storage[:, :, 2:], 91)
    np.testing.assert_array_equal(lse_storage[:, :, 0], 92)


@pytest.mark.parametrize("add", [False, True])
@pytest.mark.parametrize("empty_keys", [False, True])
def test_zero_work_backward_initializes_owned_destinations(attention, add, empty_keys):
    q = np.zeros((1, 5, 1, 64), dtype=np.float32)
    k = np.zeros((1, 0 if empty_keys else 1, 1, 64), dtype=np.float32)
    cache = SimpleNamespace(q=q, k=k, v=k.copy(), bounded=True,
                            sparse_stride=16 if empty_keys else 0, history_block_size=128)
    backing = np.full((1, 5, 16, 64), 37, dtype=np.float32)
    dq = backing[:, :, 1:2]
    dk = np.full_like(k, 41)
    dv = np.full_like(k, 43)
    result = attention.backward(np.zeros_like(q), cache, dq=dq, dk=dk, dv=dv, add_dkdv=add)
    assert all(got is want for got, want in zip(result, (dq, dk, dv)))
    np.testing.assert_array_equal(dq, 0)
    np.testing.assert_array_equal(dk, 41 if add else 0)
    np.testing.assert_array_equal(dv, 43 if add else 0)
    np.testing.assert_array_equal(backing[:, :, 0], 37)
    np.testing.assert_array_equal(backing[:, :, 2:], 37)
