"""0080BW: ensure shared global-sparse geometry preserves exact semantics."""
import numpy as np
import pytest

from mini_llm.backend import xp, BACKEND_NAME, asnumpy, resolve_dtype
from mini_llm.ops.indexed_attention import (
    _global_sparse_causal_geometry,
    _global_sparse_kv_mapping,
    clear_global_sparse_geometry_cache,
    global_sparse_geometry_cache_stats,
    global_sparse_attention_forward,
    global_sparse_attention_backward,
)


def _run_reference(monkeypatch, *, enabled, stride, offset, include_current):
    monkeypatch.setenv("MINI_LLM_GLOBAL_SPARSE_GEOMETRY_CACHE", str(int(enabled)))
    rng = np.random.default_rng(818 + stride + offset)
    q = xp.asarray(rng.normal(size=(2, 25, 4, 8)), dtype=xp.float64)
    k = xp.asarray(rng.normal(size=(2, 25, 2, 8)), dtype=xp.float64)
    v = xp.asarray(rng.normal(size=(2, 25, 2, 8)), dtype=xp.float64)
    dout = xp.asarray(rng.normal(size=q.shape), dtype=xp.float64)
    out, cache = global_sparse_attention_forward(
        q, k, v, stride=stride, offset=offset,
        include_current=include_current, kv_head_indices=(0, 0, 1, 1),
    )
    grads = global_sparse_attention_backward(dout, cache)
    return tuple(np.asarray(asnumpy(item)) for item in (out, *grads))


@pytest.mark.parametrize('stride,offset,include_current', [
    (4, 0, True), (5, 3, True), (6, 2, False), (32, 31, True),
    (32, 31, False), (1, 0, True),
])
def test_geometry_cache_exact_forward_backward(monkeypatch, stride, offset, include_current):
    clear_global_sparse_geometry_cache()
    off = _run_reference(monkeypatch, enabled=False, stride=stride,
                         offset=offset, include_current=include_current)
    on = _run_reference(monkeypatch, enabled=True, stride=stride,
                        offset=offset, include_current=include_current)
    for before, after in zip(off, on):
        np.testing.assert_array_equal(after, before)


def test_geometry_cache_reuses_arrays_across_layers_and_keys(monkeypatch):
    monkeypatch.setenv("MINI_LLM_GLOBAL_SPARSE_GEOMETRY_CACHE", "1")
    clear_global_sparse_geometry_cache()
    start = global_sparse_geometry_cache_stats()
    a0, v0 = _global_sparse_causal_geometry(128, 8, 0, True)
    a1, v1 = _global_sparse_causal_geometry(128, 8, 0, True)
    assert a0 is a1 and v0 is v1
    _, v2 = _global_sparse_causal_geometry(128, 8, 4, True)
    assert v2 is not v0
    _, v3 = _global_sparse_causal_geometry(128, 8, 0, False)
    assert v3 is not v0
    _, v4 = _global_sparse_causal_geometry(96, 8, 0, True)
    assert v4 is not v0
    stats = global_sparse_geometry_cache_stats()
    assert stats['hits'] - start['hits'] == 1
    assert stats['misses'] - start['misses'] == 4
    assert stats['entries'] == 4
    assert stats['bytes'] < 16 * 1024 * 1024


def test_global_sparse_kv_mapping_reused(monkeypatch):
    monkeypatch.setenv("MINI_LLM_GLOBAL_SPARSE_GEOMETRY_CACHE", "1")
    clear_global_sparse_geometry_cache()
    before = global_sparse_geometry_cache_stats()
    a, host0 = _global_sparse_kv_mapping((0, 0, 1, 1), 4, 2)
    b, host1 = _global_sparse_kv_mapping((0, 0, 1, 1), 4, 2)
    assert a is b and host0 == host1
    after = global_sparse_geometry_cache_stats()
    assert after['kv_map_hits'] - before['kv_map_hits'] == 1
    assert after['kv_map_misses'] - before['kv_map_misses'] == 1


def test_geometry_cache_bounded_and_disable(monkeypatch):
    monkeypatch.setenv("MINI_LLM_GLOBAL_SPARSE_GEOMETRY_CACHE", "1")
    clear_global_sparse_geometry_cache()
    for length in range(128, 128 + 50 * 128, 128):
        _global_sparse_causal_geometry(length, 128, 0, True)
    stats = global_sparse_geometry_cache_stats()
    assert stats['entries'] <= 32
    assert stats['bytes'] <= 16 * 1024 * 1024
    monkeypatch.setenv("MINI_LLM_GLOBAL_SPARSE_GEOMETRY_CACHE", "0")
    a0, v0 = _global_sparse_causal_geometry(500, 32, 0, True)
    a1, v1 = _global_sparse_causal_geometry(500, 32, 0, True)
    assert a0 is not a1 and v0 is not v1
    clear_global_sparse_geometry_cache()
    assert global_sparse_geometry_cache_stats()['bytes'] == 0


@pytest.mark.skipif(BACKEND_NAME != "cupy", reason="Requires CUDA CuPy BF16")
def test_gpu_bf16_cache_parity(monkeypatch):
    import cupy as cp
    rng = np.random.default_rng(128)
    q = cp.asarray(rng.standard_normal((2, 256, 4, 64)), dtype=cp.float32).astype(resolve_dtype('bfloat16'))
    k = cp.asarray(rng.standard_normal((2, 256, 2, 64)), dtype=cp.float32).astype(resolve_dtype('bfloat16'))
    v = cp.asarray(rng.standard_normal((2, 256, 2, 64)), dtype=cp.float32).astype(resolve_dtype('bfloat16'))
    dout = cp.asarray(rng.standard_normal((2, 256, 4, 64)), dtype=cp.float32)
    results = []
    for enabled in (False, True):
        clear_global_sparse_geometry_cache()
        monkeypatch.setenv('MINI_LLM_GLOBAL_SPARSE_GEOMETRY_CACHE', str(int(enabled)))
        out, cache = global_sparse_attention_forward(
            q, k, v, stride=16, offset=8, kv_head_indices=(0, 0, 1, 1),
        )
        grads = global_sparse_attention_backward(dout, cache)
        results.append([cp.asnumpy(z.astype(cp.float32)) for z in (out, *grads)])
    for old, new in zip(*results):
        np.testing.assert_array_equal(old, new)
