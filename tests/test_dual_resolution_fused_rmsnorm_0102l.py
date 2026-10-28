"""0102L compressor-only RMSNorm fusion / cache-equivalence regression tests."""
import numpy as np
import pytest

from mini_llm.backend import BACKEND_NAME
from mini_llm.ops.dual_resolution_attention import (
    _fused_compressor_rmsnorm_enabled, _rmsnorm_forward, _rmsnorm_backward,
)


class _Gamma:
    def __init__(self, value):
        self.data = value
        self.grad = value.astype('float32') * 0


def test_cpu_fused_rmsnorm_switch_never_calls_gpu(monkeypatch):
    if BACKEND_NAME != 'numpy':
        pytest.skip('NumPy only')
    monkeypatch.setenv('MINI_LLM_DUAL_RESOLUTION_FUSED_RMSNORM', '1')
    assert not _fused_compressor_rmsnorm_enabled()
    rng = np.random.default_rng(9102)
    x = rng.normal(size=(2, 4, 16)).astype('float32')
    g = _Gamma(rng.normal(size=(16,)).astype('float32'))
    y, cache = _rmsnorm_forward(x, g)
    assert not cache.get('triton_compressor_norm', False)
    dx = _rmsnorm_backward(np.ones_like(y), cache, g, True)
    assert np.all(np.isfinite(y)) and np.all(np.isfinite(dx))
    assert np.all(np.isfinite(g.grad))



def test_triton_rmsnorm_pointer_protocol_without_gpu(monkeypatch):
    """Prevent recurrence of 0102J's raw-CuPy-array Triton launch issue."""
    from mini_llm.ops import triton_dual_resolution_rmsnorm as kernel
    class FakeTL:
        float32 = object()
        float16 = object()
        bfloat16 = object()
    class FakeArray:
        class Data:
            ptr = 123456
        data = Data()
        def __init__(self, dtype):
            self.dtype = dtype
    monkeypatch.setattr(kernel, 'tl', FakeTL)
    for dtype in ['float32', 'float16', 'bfloat16']:
        ptr = kernel._ptr(FakeArray(dtype))
        assert ptr.data_ptr() == 123456
        assert ptr.dtype is getattr(FakeTL, dtype)
    with pytest.raises(TypeError):
        kernel._ptr(FakeArray('int32'))


@pytest.mark.skipif(BACKEND_NAME != 'cupy', reason='requires CuPy/CUDA')
@pytest.mark.parametrize('dtype', ['float32', 'bfloat16'])
@pytest.mark.parametrize('blocks', [1, 7, 32])
def test_rmsnorm_forward_backward_parity(dtype, blocks, monkeypatch):
    import cupy as cp
    pytest.importorskip('triton')
    rng = np.random.default_rng(491 + blocks)
    x = cp.asarray(rng.normal(size=(blocks, 128, 512)).astype('float32')).astype(dtype)
    dy = cp.asarray(rng.normal(size=(blocks, 128, 512)).astype('float32'))
    gamma_value = cp.asarray(rng.uniform(0.8, 1.2, size=(512,)).astype('float32')).astype(dtype)

    def run(fused, accumulate=True):
        monkeypatch.setenv('MINI_LLM_DUAL_RESOLUTION_FUSED_RMSNORM', '1' if fused else '0')
        gamma = _Gamma(gamma_value.copy())
        out, cache = _rmsnorm_forward(x, gamma)
        dx = _rmsnorm_backward(dy, cache, gamma, accumulate)
        return out.astype('float32'), dx.astype('float32'), gamma.grad

    y_ref, dx_ref, dg_ref = run(False)
    y, dx, dg = run(True)
    atol = 2e-2 if dtype == 'bfloat16' else 2e-4
    cp.testing.assert_allclose(y, y_ref, rtol=atol, atol=atol)
    cp.testing.assert_allclose(dx, dx_ref, rtol=2e-4, atol=2e-4)
    cp.testing.assert_allclose(dg, dg_ref, rtol=3e-3, atol=3e-3)
    _, no_accum_dx, no_accum_dg = run(True, accumulate=False)
    cp.testing.assert_allclose(no_accum_dx, dx, rtol=1e-5, atol=1e-5)
    assert bool(cp.all(no_accum_dg == 0))


@pytest.mark.skipif(BACKEND_NAME != 'cupy', reason='requires CuPy/CUDA')
def test_rmsnorm_integration_with_compressor_and_unequal_documents(monkeypatch):
    import cupy as cp
    pytest.importorskip('triton')
    from mini_llm.backend import RandomStream
    from mini_llm.config import DualResolutionConfig
    from mini_llm.ops.dual_resolution_attention import (
        DualResolutionAttention, build_dual_resolution_metadata,
    )
    cfg = DualResolutionConfig(
        enabled=True, block_size=128, local_window=512,
        compression_dim=512, compression_heads=8, compression_depth=1,
        max_blocks=8, global_query_chunk=128,
    )
    monkeypatch.setenv('MINI_LLM_DUAL_RESOLUTION_LOCAL_KERNEL', 'triton')
    monkeypatch.setenv('MINI_LLM_DUAL_RESOLUTION_GLOBAL_KERNEL', 'triton')
    monkeypatch.setenv('MINI_LLM_DUAL_RESOLUTION_COMPRESSION_KERNEL', 'triton')
    monkeypatch.setenv('MINI_LLM_DUAL_RESOLUTION_FAST_PROJECTIONS', '1')
    monkeypatch.setenv('MINI_LLM_DUAL_RESOLUTION_FUSED_POOLING', '1')
    tokens = np.full((2, 261), 3, dtype=np.int64)
    tokens[0, 255] = 0
    tokens[1, 127] = 0
    meta = build_dual_resolution_metadata(tokens, eos_token_id=0, block_size=128)
    rng = np.random.default_rng(9155)
    x = cp.asarray(rng.normal(scale=0.04, size=(2, 261, 1024)).astype('float32')).astype('bfloat16')
    dy = cp.asarray(rng.normal(scale=0.04, size=(2, 261, 1024)).astype('float32'))

    def run(fused):
        monkeypatch.setenv('MINI_LLM_DUAL_RESOLUTION_FUSED_RMSNORM', str(int(fused)))
        model = DualResolutionAttention(
            d_model=1024, n_q_heads=16, n_kv_heads=4, d_head=64,
            input_std=0.01, output_std=0.01, rng=RandomStream(9156),
            dtype='bfloat16', config=cfg,
        )
        y, cache = model.forward(x, dual_resolution_metadata=meta)
        dx = model.backward(dy, cache)
        return y.astype('float32'), dx.astype('float32'), [p.grad.copy() for p in model.parameters()]

    ref = run(False)
    opt = run(True)
    cp.testing.assert_allclose(opt[0], ref[0], atol=0.06, rtol=0.06)
    cp.testing.assert_allclose(opt[1], ref[1], atol=0.08, rtol=0.10)
    for got, expected in zip(opt[2], ref[2]):
        assert bool(cp.all(cp.isfinite(got)))
        cp.testing.assert_allclose(got, expected, atol=0.14, rtol=0.14)
