"""0080CA: the cuBLAS device pointer tables must not change in packed mode."""
from __future__ import annotations

import ctypes
import os
from unittest.mock import patch

import numpy as np
import pytest

from benchmarks.benchmark_grouped_dispatch_host_0080ca import (
    _MockXp, mock_cuda_dispatch, sample_groups,
)
from mini_llm.ops import cublas_grouped as grouped


def _read_table(ptr, n):
    return tuple((ctypes.c_size_t * n).from_address(ptr))


@pytest.mark.parametrize('packed', (0, 1))
@pytest.mark.parametrize('count', (1, 7, 51))
def test_packed_pointer_layout_is_identical(packed, count, monkeypatch):
    xp = _MockXp()
    monkeypatch.setattr(grouped, 'xp', xp)
    monkeypatch.setenv('MINI_LLM_CUBLAS_PACKED_POINTER_UPLOAD', str(packed))
    a = [0x1000 + i * 128 for i in range(count)]
    b = [0x2000 + i * 512 for i in range(count)]
    c = [0x3000 + i * 1024 for i in range(count)]
    (ap, bp, cp), owners = grouped._device_pointer_tables(a, b, c)
    assert _read_table(ap, count) == tuple(a)
    assert _read_table(bp, count) == tuple(b)
    assert _read_table(cp, count) == tuple(c)
    assert len(owners) == (1 if packed else 3)
    assert xp.submissions == (1 if packed else 3)


def test_packed_pointer_layout_rejects_unequal_tables(monkeypatch):
    monkeypatch.setenv('MINI_LLM_CUBLAS_PACKED_POINTER_UPLOAD', '1')
    with pytest.raises(ValueError, match='equal length'):
        grouped._device_pointer_tables([1, 2], [3], [4, 5])


@pytest.mark.parametrize('packed', (0, 1))
def test_grouped_cublas_receives_exact_same_three_swapped_tables(packed, monkeypatch):
    monkeypatch.setenv('MINI_LLM_CUBLAS_PACKED_POINTER_UPLOAD', str(packed))
    groups = sample_groups(chunks=4, batch=3, heads=4)
    with mock_cuda_dispatch(capture=True) as (xp, calls):
        assert grouped.grouped_bf16_gemm(groups)
    assert len(calls) == 1
    a, b, c = calls[0]
    problems = [p for g in groups for p in g.problems]
    assert a == tuple(p.b_ptr for p in problems)
    assert b == tuple(p.a_ptr for p in problems)
    assert c == tuple(p.c_ptr for p in problems)
    assert xp.submissions == (1 if packed else 3)


@pytest.mark.parametrize('packed', (0, 1))
def test_batched_cublas_pointer_offsets_across_groups(packed, monkeypatch):
    monkeypatch.setenv('MINI_LLM_CUBLAS_PACKED_POINTER_UPLOAD', str(packed))
    groups = sample_groups(chunks=3, batch=2, heads=3)
    expected = [tuple(p for p in group.problems) for group in groups]
    with mock_cuda_dispatch(capture=False) as (xp, _):
        captured = []
        def fake_batched(*args):
            n = int(args[17].value)
            tables = tuple(_read_table(int(args[i].value), n) for i in (7, 10, 14))
            captured.append(tables)
            return 0
        with patch.object(grouped, '_load_batched_symbol', return_value=fake_batched), \
             patch.object(grouped, 'local_cublas_mode', return_value='batched'):
            assert grouped.grouped_bf16_gemm(groups)
        assert xp.submissions == (1 if packed else 3)
    assert len(captured) == len(groups)
    for tables, problems in zip(captured, expected):
        assert tables == (
            tuple(p.b_ptr for p in problems),
            tuple(p.a_ptr for p in problems),
            tuple(p.c_ptr for p in problems),
        )


def test_cuda_grouped_gemm_packed_vs_original(monkeypatch):
    """Run the real BF16 GEMM on a CUDA machine; skipped on CPU-only systems."""
    from mini_llm.backend import BACKEND_NAME
    if BACKEND_NAME != 'cupy':
        pytest.skip('Requires MINI_LLM_BACKEND=cupy on a CUDA host')
    import cupy as cp
    import ml_dtypes
    if cp.cuda.runtime.getDeviceCount() <= 0:
        pytest.skip('No CUDA device')

    monkeypatch.setenv('MINI_LLM_CUBLAS_GROUPED_LOCAL_GEMM', '1')
    monkeypatch.setenv('MINI_LLM_CUBLAS_GROUPED_LOCAL_GEMM_STRICT', '1')
    monkeypatch.setenv('MINI_LLM_LOCAL_CUBLAS_MODE', 'grouped')
    rng = np.random.default_rng(12)
    inputs = []
    for rows in (8, 13, 24):
        a = cp.asarray(rng.standard_normal((rows, 64)).astype(np.float32)).astype(ml_dtypes.bfloat16)
        b = cp.asarray(rng.standard_normal((64, 32)).astype(np.float32)).astype(ml_dtypes.bfloat16)
        inputs.append((a, b))

    results = []
    for packed in ('0', '1'):
        monkeypatch.setenv('MINI_LLM_CUBLAS_PACKED_POINTER_UPLOAD', packed)
        outputs = [cp.zeros((a.shape[0], b.shape[1]), dtype=ml_dtypes.bfloat16) for a, b in inputs]
        groups = []
        for (a, b), out in zip(inputs, outputs):
            groups.append(grouped.RowMajorGemmGroup((grouped.RowMajorGemmProblem(
                a_ptr=int(a.data.ptr), b_ptr=int(b.data.ptr), c_ptr=int(out.data.ptr),
                m=int(a.shape[0]), n=int(b.shape[1]), k=int(a.shape[1]),
                lda=int(a.shape[1]), ldb=int(b.shape[1]), ldc=int(out.shape[1]),
            ),)))
        assert grouped.grouped_bf16_gemm(groups)
        cp.cuda.get_current_stream().synchronize()
        results.append([cp.asnumpy(out.astype(cp.float32)) for out in outputs])
    for old, new, (a, b) in zip(*results, inputs):
        np.testing.assert_array_equal(new, old)
        expected = cp.asnumpy(a.astype(cp.float32) @ b.astype(cp.float32))
        np.testing.assert_allclose(new, expected, rtol=0.015, atol=0.04)
