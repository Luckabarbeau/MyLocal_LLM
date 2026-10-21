"""0080BA: local attention backward reconstructs probability chunks on demand."""

import numpy as np
import pytest

from mini_llm.backend import BACKEND_NAME, xp
from mini_llm.ops.indexed_attention import (
    local_window_attention_forward,
    local_window_attention_backward,
)


def _inputs(seed=31, batch=2, seq=15, hq=4, hkv=2, dh=5, dtype='float64'):
    rng = np.random.default_rng(seed)
    a = (
        xp.asarray(rng.normal(size=shape), dtype=dtype)
        for shape in (
            (batch, seq, hq, dh),
            (batch, seq, hkv, dh),
            (batch, seq, hkv, dh),
            (batch, seq, hq, dh),
        )
    )
    return tuple(a)


@pytest.mark.parametrize('window,chunk_size', [(1, 3), (4, 5), (9, 4), (40, 7)])
def test_recompute_matches_original_forward_and_manual_backward(monkeypatch, window, chunk_size):
    q, k, v, dout = _inputs()
    monkeypatch.setenv('MINI_LLM_LOCAL_ATTN_RECOMPUTE_PROBS', '0')
    out_ref, ref = local_window_attention_forward(
        q, k, v, window, kv_head_indices=(0, 0, 1, 1), query_chunk_size=chunk_size
    )
    grads_ref = local_window_attention_backward(dout, ref)
    assert any(probs is not None for *_, groups in ref['chunks'] for _, _, probs in groups)

    monkeypatch.setenv('MINI_LLM_LOCAL_ATTN_RECOMPUTE_PROBS', '1')
    out, cache = local_window_attention_forward(
        q, k, v, window, kv_head_indices=(0, 0, 1, 1), query_chunk_size=chunk_size
    )
    assert cache['recompute_local_probs'] is True
    assert all(probs is None for *_, groups in cache['chunks'] for _, _, probs in groups)
    grads = local_window_attention_backward(dout, cache)

    np.testing.assert_allclose(np.asarray(out), np.asarray(out_ref), atol=1e-12, rtol=1e-12)
    for new, old in zip(grads, grads_ref):
        np.testing.assert_allclose(np.asarray(new), np.asarray(old), atol=3e-12, rtol=3e-12)


def test_recompute_matches_directional_finite_difference(monkeypatch):
    monkeypatch.setenv('MINI_LLM_LOCAL_ATTN_RECOMPUTE_PROBS', '1')
    q, k, v, dout = _inputs(seed=33, batch=1, seq=7, hq=2, hkv=1, dh=3)
    drng = np.random.default_rng(34)
    directions = [xp.asarray(drng.normal(size=x.shape), dtype=x.dtype) for x in (q, k, v)]
    out, cache = local_window_attention_forward(
        q, k, v, 4, query_chunk_size=3
    )
    grads = local_window_attention_backward(dout, cache)
    calculated = float(sum(xp.sum(g * d) for g, d in zip(grads, directions)))

    def objective(delta):
        qq, kk, vv = (x + delta*d for x, d in zip((q, k, v), directions))
        y = local_window_attention_forward(qq, kk, vv, 4, query_chunk_size=3, return_cache=False)
        return float(xp.sum(y * dout))

    eps = 1e-6
    numerical = (objective(eps) - objective(-eps)) / (2*eps)
    np.testing.assert_allclose(calculated, numerical, atol=2e-7, rtol=5e-6)


def test_empty_cache_has_no_probabilities_even_across_many_chunks(monkeypatch):
    monkeypatch.setenv('MINI_LLM_LOCAL_ATTN_RECOMPUTE_PROBS', '1')
    q, k, v, dout = _inputs(batch=2, seq=37)
    _, cache = local_window_attention_forward(q, k, v, 18, query_chunk_size=7)
    assert len(cache['chunks']) == 6
    assert all(p is None for *_, groups in cache['chunks'] for _, _, p in groups)
    grads = local_window_attention_backward(dout, cache)
    assert all(np.isfinite(np.asarray(g)).all() for g in grads)


def test_auto_local_chunk_uses_actual_attention_geometry(monkeypatch):
    from mini_llm.ops.indexed_attention import _resolve_local_query_chunk_size
    monkeypatch.setenv('MINI_LLM_LOCAL_ATTN_QUERY_CHUNK', 'auto')
    monkeypatch.delenv('MINI_LLM_LOCAL_ATTN_AUTO_WORK_BUDGET', raising=False)

    resolve = _resolve_local_query_chunk_size
    assert resolve(4096, window=4096, batch=3, n_q_heads=1, d_head=64) == 768
    assert resolve(4096, window=512, batch=3, n_q_heads=15, d_head=64) == 256
    assert resolve(4096, window=512, batch=3, n_q_heads=9, d_head=64) == 384

    monkeypatch.setenv('MINI_LLM_LOCAL_ATTN_QUERY_CHUNK', '1024')
    assert resolve(4096, window=512, batch=3, n_q_heads=15, d_head=64) == 1024


@pytest.mark.skipif(BACKEND_NAME != 'cupy', reason='requires CUDA/CuPy')
def test_grouped_bf16_cuda_forward_backward_and_cache(monkeypatch):
    from mini_llm.ops import indexed_attention as local
    from mini_llm.ops.cublas_grouped import available, is_enabled
    monkeypatch.setenv('MINI_LLM_CUBLAS_GROUPED_LOCAL_GEMM', '1')
    monkeypatch.setenv('MINI_LLM_LOCAL_CUBLAS_MODE', 'grouped')
    monkeypatch.setenv('MINI_LLM_BF16_LOCAL_GEMM', '1')
    monkeypatch.setenv('MINI_LLM_FUSED_BF16_LOCAL_PIPELINE', '1')
    if not available() or not is_enabled():
        pytest.skip('grouped BF16 cuBLAS unavailable on this GPU')
    q, k, v, dout = _inputs(batch=1, seq=24, hq=4, hkv=2, dh=64, dtype='bfloat16')
    mapping = (0, 0, 1, 1)
    if not local._local_grouped_fastpath_ready(q, k, v, True):
        pytest.skip('GPU layout does not support grouped fastpath')
    monkeypatch.setenv('MINI_LLM_LOCAL_ATTN_RECOMPUTE_PROBS', '0')
    monkeypatch.setenv('MINI_LLM_LOCAL_GROUPED_ROW_STORE', '0')
    out_ref, ref = local_window_attention_forward(q, k, v, 12, kv_head_indices=mapping, query_chunk_size=8)
    assert ref.get('grouped_cublas_local')
    grads_ref = local_window_attention_backward(dout, ref)
    monkeypatch.setenv('MINI_LLM_LOCAL_ATTN_RECOMPUTE_PROBS', '1')
    monkeypatch.setenv('MINI_LLM_LOCAL_GROUPED_ROW_STORE', '1')
    out, cache = local_window_attention_forward(q, k, v, 12, kv_head_indices=mapping, query_chunk_size=8)
    assert cache.get('grouped_cublas_local')
    assert cache['recompute_local_probs']
    assert all('probs' not in c for c in cache['chunks'])
    grads = local_window_attention_backward(dout, cache)
    np.testing.assert_allclose(xp.asnumpy(out).astype('float32'), xp.asnumpy(out_ref).astype('float32'), rtol=1e-3, atol=1e-3)
    for actual, expected in zip(grads, grads_ref):
        np.testing.assert_allclose(xp.asnumpy(actual), xp.asnumpy(expected), rtol=5e-3, atol=5e-3)

@pytest.mark.skipif(BACKEND_NAME != 'numpy', reason='NumPy BF16 reference')
def test_numpy_bf16_recompute_matches_cached_backward(monkeypatch):
    import ml_dtypes
    dtype = ml_dtypes.bfloat16
    q, k, v, _ = _inputs(seed=57, batch=1, seq=11, hq=4, hkv=2, dh=4)
    q, k, v = (x.astype(dtype) for x in (q, k, v))
    rng = np.random.default_rng(58)
    dout = xp.asarray(rng.normal(size=q.shape), dtype=xp.float32)
    kwargs = dict(window=5, kv_head_indices=(0,0,1,1), query_chunk_size=4)
    monkeypatch.setenv('MINI_LLM_LOCAL_ATTN_RECOMPUTE_PROBS','0')
    old_output, old_cache = local_window_attention_forward(q, k, v, **kwargs)
    old_grad = local_window_attention_backward(dout, old_cache)
    monkeypatch.setenv('MINI_LLM_LOCAL_ATTN_RECOMPUTE_PROBS','1')
    new_output, new_cache = local_window_attention_forward(q, k, v, **kwargs)
    new_grad = local_window_attention_backward(dout, new_cache)
    np.testing.assert_allclose(np.asarray(new_output, dtype='float32'), np.asarray(old_output, dtype='float32'), rtol=0, atol=0)
    for new, old in zip(new_grad, old_grad):
        np.testing.assert_allclose(np.asarray(new), np.asarray(old), rtol=1e-4, atol=1e-4)
