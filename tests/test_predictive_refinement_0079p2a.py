"""0079P2a: prevent cached CUDA gather lookups returning (module, kernel)."""
from types import SimpleNamespace

import numpy as np
import pytest

from mini_llm.backend import BACKEND_NAME, xp
from mini_llm.ops import shared_indexed_kv as resident_kv


@pytest.mark.parametrize("itemsize", [2, 4, 8])
def test_cached_split_gather_returns_callable_on_every_lookup(monkeypatch, itemsize):
    """GPU-independent regression: fake RawModule to exercise cache hits."""
    created = []

    class FakeModule:
        def __init__(self, **kwargs):
            created.append(self)
            self.function = lambda *args, **kw: None

        def get_function(self, name):
            assert name == 'gather_split'
            return self.function

    monkeypatch.setattr(resident_kv, 'xp', SimpleNamespace(RawModule=FakeModule))
    monkeypatch.setattr(resident_kv, '_SPLIT_GATHER_KERNELS', {})

    first = resident_kv._get_split_gather_kernel(itemsize)
    second = resident_kv._get_split_gather_kernel(itemsize)
    third = resident_kv._get_split_gather_kernel(itemsize)
    assert callable(first)
    assert first is second is third
    assert len(created) == 1
    assert resident_kv._SPLIT_GATHER_KERNELS[itemsize] == (created[0], first)


@pytest.mark.skipif(BACKEND_NAME != 'cupy', reason='requires CUDA/CuPy')
def test_cuda_gather_reuses_kernel_for_keys_and_values():
    """A real K then V gather triggers the cached lookup on CUDA."""
    context_k = xp.asarray(np.arange(6).reshape(1, 3, 1, 2), dtype=xp.float32)
    context_v = context_k + xp.float32(100)
    active_k = xp.asarray(np.arange(4).reshape(1, 2, 1, 2) + 20, dtype=xp.float32)
    active_v = active_k + xp.float32(100)
    doc_rows = xp.asarray([0], dtype=xp.int64)
    history_index = xp.asarray([[0, 2]], dtype=xp.int64)
    indices = xp.asarray([[[[0, 2, 1, 3]]]], dtype=xp.int64)
    kv_heads = xp.asarray([[[[0]]]], dtype=xp.int64)
    row_ids = xp.asarray([[[[0]]]], dtype=xp.int64)

    keys = resident_kv.SharedIndexedKV(context_k, None, doc_rows, history_index).with_active(active_k)
    values = resident_kv.SharedIndexedKV(context_v, None, doc_rows, history_index).with_active(active_v)
    actual_k = keys.gather(row_ids, indices, kv_heads)
    actual_v = values.gather(row_ids, indices, kv_heads)
    actual_k_again = keys.gather(row_ids, indices, kv_heads)
    expected_k = xp.stack([context_k[0, 0, 0], active_k[0, 0, 0],
                           context_k[0, 2, 0], active_k[0, 1, 0]], axis=0)
    xp.testing.assert_array_equal(actual_k[0, 0, 0], expected_k)
    xp.testing.assert_array_equal(actual_v[0, 0, 0], expected_k + 100)
    xp.testing.assert_array_equal(actual_k_again, actual_k)
