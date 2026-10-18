"""0080CB: compact immutable descriptors have identical grouped-cuBLAS ABI."""
import os
import subprocess
import sys
from unittest.mock import patch

import pytest

from benchmarks.benchmark_grouped_dispatch_host_0080ca import mock_cuda_dispatch, sample_groups
from mini_llm.ops import cublas_grouped as bridge


@pytest.mark.parametrize('trans_a,trans_b', [(False, False), (True, False), (False, True), (True, True)])
def test_compact_matches_original_fields(trans_a, trans_b):
    values = (123, 456, 789, 8, 16, 32, 64, 96, 128, trans_a, trans_b)
    standard = bridge.RowMajorGemmProblem(*values)
    compact = bridge.CompactRowMajorGemmProblem(*values)
    assert tuple(getattr(standard, key) for key in compact._fields) == tuple(compact)
    assert compact == bridge.CompactRowMajorGemmProblem(*values)
    with pytest.raises(AttributeError):
        compact.a_ptr = 3


@pytest.mark.parametrize('packed', ('0','1'))
def test_mocked_grouped_cublas_submission_is_identical(monkeypatch, packed):
    monkeypatch.setenv('MINI_LLM_CUBLAS_PACKED_POINTER_UPLOAD', packed)
    observed = []
    with mock_cuda_dispatch(capture=True) as (_, calls):
        for klass in (bridge.RowMajorGemmProblem, bridge.CompactRowMajorGemmProblem):
            with patch.object(bridge, 'RowMajorGemmProblem', klass):
                groups = sample_groups(chunks=4, batch=3, heads=4)
                assert bridge.grouped_bf16_gemm(groups)
                observed.append(calls[-1])
    assert observed[0] == observed[1]


def test_import_time_feature_flag_applies_to_indexed_attention():
    code = 'from mini_llm.ops import indexed_attention as a, cublas_grouped as b; print(a.RowMajorGemmProblem is b.CompactRowMajorGemmProblem)'
    base_env = dict(os.environ)
    base_env['MINI_LLM_BACKEND'] = 'numpy'
    for flag in ('0', '1'):
        env = {**base_env, 'MINI_LLM_CUBLAS_COMPACT_DESCRIPTORS': flag}
        result = subprocess.run([sys.executable, '-c', code], env=env,
                                capture_output=True, text=True, check=True)
        assert result.stdout.strip() == ('True' if flag == '1' else 'False')


def test_cuda_bf16_descriptor_parity(monkeypatch):
    from mini_llm.backend import BACKEND_NAME
    if BACKEND_NAME != 'cupy':
        pytest.skip('CUDA BF16 test requires MINI_LLM_BACKEND=cupy')
    import cupy as cp
    if cp.cuda.runtime.getDeviceCount() <= 0:
        pytest.skip('No CUDA device')
    import ml_dtypes
    import numpy as np
    monkeypatch.setenv('MINI_LLM_CUBLAS_GROUPED_LOCAL_GEMM', '1')
    monkeypatch.setenv('MINI_LLM_CUBLAS_GROUPED_LOCAL_GEMM_STRICT', '1')
    monkeypatch.setenv('MINI_LLM_LOCAL_CUBLAS_MODE', 'grouped')
    rng = np.random.default_rng(80)
    a = cp.asarray(rng.normal(size=(15, 64)).astype('float32')).astype(ml_dtypes.bfloat16)
    b = cp.asarray(rng.normal(size=(64, 40)).astype('float32')).astype(ml_dtypes.bfloat16)
    outputs = []
    for klass in (bridge.RowMajorGemmProblem, bridge.CompactRowMajorGemmProblem):
        c = cp.empty((15, 40), dtype=ml_dtypes.bfloat16)
        problem = klass(a_ptr=int(a.data.ptr), b_ptr=int(b.data.ptr), c_ptr=int(c.data.ptr),
                        m=15, n=40, k=64, lda=64, ldb=40, ldc=40)
        assert bridge.grouped_bf16_gemm((bridge.RowMajorGemmGroup((problem,)),))
        cp.cuda.get_current_stream().synchronize()
        outputs.append(cp.asnumpy(c.astype(cp.float32)))
    np.testing.assert_array_equal(*outputs)
