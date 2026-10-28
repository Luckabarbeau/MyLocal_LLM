"""0102M fused local/global gate: pointer contract and GPU parity."""
import numpy as np
import pytest

from mini_llm.backend import BACKEND_NAME, RandomStream
from mini_llm.ops.dual_resolution_attention import (
    _fused_gating_enabled, DualResolutionAttention,
    build_dual_resolution_metadata,
)
from mini_llm.config import DualResolutionConfig


def test_reference_default_and_explicit_opt_in(monkeypatch):
    monkeypatch.delenv('MINI_LLM_DUAL_RESOLUTION_FUSED_GATE', raising=False)
    assert not _fused_gating_enabled()
    monkeypatch.setenv('MINI_LLM_DUAL_RESOLUTION_FUSED_GATE', '1')
    assert _fused_gating_enabled() == (BACKEND_NAME == 'cupy')
    monkeypatch.setenv('MINI_LLM_DUAL_RESOLUTION_FUSED_GATE', '0')
    assert not _fused_gating_enabled()


def test_typed_pointer_adapter_no_cuda(monkeypatch):
    from mini_llm.ops import triton_dual_resolution_gate as module
    class FakeTL:
        float32 = object()
        float16 = object()
        bfloat16 = object()
        int1 = object()
    class FakeArray:
        class Data:
            ptr = 12345
        data = Data()
        def __init__(self, dtype):
            self.dtype = dtype
    monkeypatch.setattr(module, 'tl', FakeTL)
    for dtype in ('float32', 'float16', 'bfloat16', 'bool'):
        item = module._ptr(FakeArray(dtype))
        assert item.dtype is getattr(FakeTL, 'int1' if dtype == 'bool' else dtype)
        assert item.data_ptr() == 12345
    with pytest.raises(TypeError):
        module._ptr(FakeArray('int64'))


@pytest.mark.skipif(BACKEND_NAME != 'cupy', reason='requires CuPy and Triton')
@pytest.mark.parametrize('token_count', [1, 7, 256, 4096])
@pytest.mark.parametrize('logits_dtype', ['float32', 'bfloat16'])
def test_gate_direct_forward_backward(token_count, logits_dtype):
    import cupy as cp
    pytest.importorskip('triton')
    from mini_llm.ops.triton_dual_resolution_gate import forward, backward
    rng = np.random.default_rng(2100 + token_count)
    b, h, d = 2, 16, 64
    local = cp.asarray(rng.normal(size=(b, token_count, h, d)).astype('float32'))
    glob = cp.asarray(rng.normal(size=(b, token_count, h, d)).astype('float32'))
    logits = cp.asarray(rng.uniform(-3, 3, size=(b, token_count, h)).astype('float32')).astype(logits_dtype)
    available = cp.asarray(rng.uniform(size=(b, token_count)) > 0.4)
    dc = cp.asarray(rng.normal(size=local.shape).astype('float32'))
    gate_ref = 1 / (1 + cp.exp(-logits.astype('float32')))
    eff_ref = gate_ref * available[..., None]
    context_ref = local + eff_ref[..., None] * (glob - local)
    dlocal_ref = dc * (1 - eff_ref[..., None])
    dglobal_ref = dc * eff_ref[..., None]
    dlogit_ref = cp.sum(dc * (glob - local), axis=-1) * available[..., None] * gate_ref * (1 - gate_ref)
    context, gate, eff = forward(local, glob, logits, available)
    cp.testing.assert_allclose(context, context_ref, rtol=2e-5, atol=2e-5)
    cp.testing.assert_allclose(gate, gate_ref, rtol=2e-5, atol=2e-5)
    cp.testing.assert_allclose(eff, eff_ref, rtol=2e-5, atol=2e-5)
    dl, dg, dlogit = backward(dc, local, glob, gate, eff, available)
    cp.testing.assert_allclose(dl, dlocal_ref, rtol=2e-5, atol=2e-5)
    cp.testing.assert_allclose(dg, dglobal_ref, rtol=2e-5, atol=2e-5)
    cp.testing.assert_allclose(dlogit, dlogit_ref, rtol=4e-5, atol=4e-5)


@pytest.mark.skipif(BACKEND_NAME != 'cupy', reason='requires CuPy and Triton')
def test_full_attention_gate_parity_unequal_documents(monkeypatch):
    import cupy as cp
    pytest.importorskip('triton')
    config = DualResolutionConfig(
        enabled=True, block_size=128, local_window=512,
        compression_dim=512, compression_heads=8, compression_depth=1,
        max_blocks=8, global_query_chunk=128,
    )
    tokens = np.full((2, 260), 3, dtype=np.int64)
    tokens[0, 255] = 0
    tokens[1, 127] = 0
    metadata = build_dual_resolution_metadata(tokens, eos_token_id=0, block_size=128)
    rng = np.random.default_rng(2200)
    x = cp.asarray(rng.normal(scale=0.04, size=(2, 260, 1024)).astype('float32')).astype('bfloat16')
    dy = cp.asarray(rng.normal(scale=0.04, size=x.shape).astype('float32'))
    monkeypatch.setenv('MINI_LLM_DUAL_RESOLUTION_LOCAL_KERNEL', 'triton')
    monkeypatch.setenv('MINI_LLM_DUAL_RESOLUTION_GLOBAL_KERNEL', 'triton')
    monkeypatch.setenv('MINI_LLM_DUAL_RESOLUTION_COMPRESSION_KERNEL', 'triton')
    monkeypatch.setenv('MINI_LLM_DUAL_RESOLUTION_FAST_PROJECTIONS', '1')
    monkeypatch.setenv('MINI_LLM_DUAL_RESOLUTION_FUSED_POOLING', '1')
    def run(fused):
        monkeypatch.setenv('MINI_LLM_DUAL_RESOLUTION_FUSED_GATE', str(int(fused)))
        model = DualResolutionAttention(
            d_model=1024, n_q_heads=16, n_kv_heads=4, d_head=64,
            input_std=0.01, output_std=0.01, rng=RandomStream(2201),
            dtype='bfloat16', config=config,
        )
        out, cache = model.forward(x, dual_resolution_metadata=metadata)
        dx = model.backward(dy, cache)
        return out.astype('float32'), dx.astype('float32'), [p.grad.copy() for p in model.parameters()]
    ref = run(False)
    fused = run(True)
    cp.testing.assert_allclose(fused[0], ref[0], rtol=0.06, atol=0.06)
    cp.testing.assert_allclose(fused[1], ref[1], rtol=0.10, atol=0.08)
    for a, b in zip(fused[2], ref[2]):
        assert bool(cp.all(cp.isfinite(a)))
        cp.testing.assert_allclose(a, b, rtol=0.14, atol=0.14)
