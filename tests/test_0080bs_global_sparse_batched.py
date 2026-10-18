"""Optional 0080BS global-sparse batched BF16 projection tests."""
import numpy as np
import pytest

from mini_llm.backend import BACKEND_NAME, xp
from mini_llm.ops.indexed_attention import (
    _global_sparse_batched_gemm,
    global_sparse_attention_forward,
    global_sparse_attention_backward,
)


def test_batched_matrix_product_matches_per_head_on_cpu():
    rng = np.random.default_rng(80)
    a = rng.standard_normal((3, 2, 31, 64)).astype('float32')
    b = rng.standard_normal((3, 2, 64, 13)).astype('float32')
    expected = np.empty((3, 2, 31, 13), dtype='float32')
    for i in range(3):
        for j in range(2):
            expected[i, j] = a[i, j] @ b[i, j]
    got = _global_sparse_batched_gemm(xp.asarray(a), xp.asarray(b))
    np.testing.assert_allclose(np.asarray(got), expected, atol=2e-5, rtol=2e-5)

    # The real source uses transposed noncontiguous matrices in backward.
    got_t = _global_sparse_batched_gemm(
        xp.asarray(a).swapaxes(-2, -1), xp.asarray(a)
    )
    expected_t = np.stack([
        np.stack([a[i, j].T @ a[i, j] for j in range(2)])
        for i in range(3)
    ])
    np.testing.assert_allclose(np.asarray(got_t), expected_t, atol=2e-4, rtol=2e-5)


@pytest.mark.skipif(BACKEND_NAME != 'cupy', reason='requires CuPy BF16 GPU')
def test_global_sparse_bf16_batched_matches_existing(monkeypatch):
    from mini_llm.ops import indexed_attention as idx
    rng = np.random.default_rng(33)
    shape = (2, 97, 2, 64)
    q = xp.asarray(rng.standard_normal(shape), dtype='bfloat16')
    k = xp.asarray(rng.standard_normal((2, 97, 1, 64)), dtype='bfloat16')
    v = xp.asarray(rng.standard_normal((2, 97, 1, 64)), dtype='bfloat16')
    dy = xp.asarray(rng.standard_normal(shape), dtype='bfloat16')
    monkeypatch.setenv('MINI_LLM_FUSED_BF16_INDEXED_PIPELINE', '1')
    monkeypatch.setenv('MINI_LLM_CUBLAS_GROUPED_LOCAL_GEMM', '1')
    monkeypatch.setenv('MINI_LLM_CUBLAS_GROUPED_LOCAL_GEMM_STRICT', '1')
    monkeypatch.setenv('MINI_LLM_GLOBAL_SPARSE_BATCHED_BF16', '0')
    y0, c0 = global_sparse_attention_forward(
        q, k, v, stride=8, offset=2, include_current=True,
        kv_head_indices=(0, 0), return_cache=True,
    )
    if not c0.get('bf16_pipeline', False):
        pytest.skip('BF16 indexed GPU pipeline not available')
    g0 = global_sparse_attention_backward(dy, c0)
    monkeypatch.setenv('MINI_LLM_GLOBAL_SPARSE_BATCHED_BF16', '1')
    y1, c1 = global_sparse_attention_forward(
        q, k, v, stride=8, offset=2, include_current=True,
        kv_head_indices=(0, 0), return_cache=True,
    )
    g1 = global_sparse_attention_backward(dy, c1)
    np.testing.assert_allclose(
        xp.asnumpy(y1), xp.asnumpy(y0), atol=2e-2, rtol=2e-2
    )
    for actual, expected in zip(g1, g0):
        np.testing.assert_allclose(
            xp.asnumpy(actual), xp.asnumpy(expected), atol=3e-2, rtol=3e-2
        )


def test_batched_matrix_product_supports_strided_output_on_cpu():
    rng = np.random.default_rng(81)
    a = rng.standard_normal((2, 3, 17, 8)).astype('float32')
    b = rng.standard_normal((2, 3, 8, 11)).astype('float32')
    scores = xp.full((2, 3, 17, 12), -99, dtype=xp.float32)
    output = scores[..., :11]
    got = _global_sparse_batched_gemm(xp.asarray(a), xp.asarray(b), out=output)
    np.testing.assert_allclose(np.asarray(got), a @ b, atol=2e-5, rtol=2e-5)
    np.testing.assert_array_equal(np.asarray(scores)[..., 11], -99)


@pytest.mark.skipif(BACKEND_NAME != 'cupy', reason='requires CuPy BF16 GPU')
def test_global_sparse_bf16_grouped_cublas_is_exercised(monkeypatch):
    from mini_llm.ops import indexed_attention as idx
    from mini_llm.ops import cublas_grouped as grouped
    monkeypatch.setenv('MINI_LLM_CUBLAS_GROUPED_LOCAL_GEMM', '1')
    monkeypatch.setenv('MINI_LLM_CUBLAS_GROUPED_LOCAL_GEMM_STRICT', '1')
    if not grouped.available():
        pytest.skip('CUDA grouped cuBLAS interface unavailable')
    calls = []
    original = idx._cublas_grouped_bf16_gemm

    def recording_grouped(groups):
        ok = original(groups)
        calls.append(ok)
        return ok

    monkeypatch.setattr(idx, '_cublas_grouped_bf16_gemm', recording_grouped)
    rng = np.random.default_rng(82)
    a = xp.asarray(rng.standard_normal((2, 2, 64, 32)), dtype='bfloat16')
    b = xp.asarray(rng.standard_normal((2, 2, 32, 17)), dtype='bfloat16')
    expected = xp.empty((2, 2, 64, 17), dtype=a.dtype)
    for i in range(2):
        for j in range(2):
            expected[i, j] = a[i, j] @ b[i, j]
    destination = xp.zeros((2, 2, 64, 18), dtype=a.dtype)
    result = idx._global_sparse_batched_gemm(a, b, out=destination[..., :17])
    np.testing.assert_allclose(
        xp.asnumpy(result.astype('float32')),
        xp.asnumpy(expected.astype('float32')),
        atol=0.04, rtol=0.03,
    )
    assert calls and all(calls), 'Grouped cuBLAS must execute, not silently fall back'
    np.testing.assert_array_equal(
        xp.asnumpy(destination[..., 17].astype('float32')),
        0,
    )
