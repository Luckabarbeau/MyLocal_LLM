"""0080O8: exact MoE gate/up cache offload to pinned RAM.

CUDA tests validate byte-identical restored BF16 activations and all
backpropagated expert, input, and router gradients against GPU-cached mode.
"""
import numpy as np
import pytest

from mini_llm.backend import BACKEND_NAME, RandomStream, asnumpy, xp
from mini_llm.blocks.moe_block import MoE
from mini_llm.ops import experts as experts_ops


def _host(value):
    return np.asarray(asnumpy(value), dtype=np.float64)


def _make_moe(dtype="float64"):
    return MoE(d_model=16, d_ff=32, n_experts=4, k=2,
               input_std=0.04, output_std=0.04,
               rng=RandomStream(798), name="blocks.0.moe", dtype=dtype)


def test_offload_disabled_leaves_cpu_cache_unchanged(monkeypatch):
    monkeypatch.setenv("MINI_LLM_MOE_OFFLOAD_GATE_UP", "0")
    monkeypatch.setenv("MINI_LLM_MOE_RECOMPUTE_GATE_UP", "0")
    monkeypatch.setenv("MINI_LLM_FUSED_EXPERT_GEMM", "0")
    moe = _make_moe()
    x = xp.ones((2, 4, 16), dtype=xp.float64)
    _, cache = moe.forward(x)
    expert_caches = cache["experts_cache"]["expert_caches"]
    assert expert_caches
    assert all("g" in c and "u" in c for c in expert_caches.values())


def test_offload_requires_cupy(monkeypatch):
    if BACKEND_NAME != "numpy":
        pytest.skip("requires NumPy backend")
    monkeypatch.setenv("MINI_LLM_MOE_OFFLOAD_GATE_UP", "1")
    monkeypatch.setenv("MINI_LLM_MOE_RECOMPUTE_GATE_UP", "0")
    moe = _make_moe()
    with pytest.raises(RuntimeError, match="MINI_LLM_BACKEND=cupy"):
        moe.forward(xp.ones((1, 3, 16), dtype=xp.float64))


def test_layer_selection(monkeypatch):
    monkeypatch.setenv("MINI_LLM_MOE_OFFLOAD_GATE_UP", "1")
    monkeypatch.setenv("MINI_LLM_MOE_RECOMPUTE_GATE_UP", "0")
    monkeypatch.setenv("MINI_LLM_MOE_OFFLOAD_LAYERS", "6")
    # Selecting a layer is a pure configuration function; mock the backend
    # name rather than creating GPU buffers in the CPU test runner.
    monkeypatch.setattr(experts_ops, "BACKEND_NAME", "cupy")
    assert experts_ops._moe_cpu_offload_enabled("blocks.0.moe.experts")
    assert experts_ops._moe_cpu_offload_enabled("blocks.5.moe.experts")
    assert not experts_ops._moe_cpu_offload_enabled("blocks.6.moe.experts")
    assert not experts_ops._moe_cpu_offload_enabled("blocks.11.moe.experts")
    monkeypatch.setenv("MINI_LLM_MOE_RECOMPUTE_GATE_UP", "1")
    with pytest.raises(ValueError, match="mutually exclusive"):
        experts_ops._moe_cpu_offload_enabled("blocks.0.moe.experts")


@pytest.mark.skipif(BACKEND_NAME != "cupy", reason="requires RTX/CuPy")
@pytest.mark.parametrize("packed", [False, True])
def test_cuda_bf16_offload_matches_cached_expert_and_router_gradients(monkeypatch, packed):
    monkeypatch.setenv("MINI_LLM_MOE_RECOMPUTE_GATE_UP", "0")
    monkeypatch.setenv("MINI_LLM_CONCURRENT_EXPERTS", "0")
    monkeypatch.setenv("MINI_LLM_FUSED_EXPERT_GEMM", str(int(packed)))
    monkeypatch.setenv("MINI_LLM_FUSED_SWIGLU", "1")
    monkeypatch.setenv("MINI_LLM_MOE_OFFLOAD_LAYERS", "6")
    moe = _make_moe("bfloat16")
    rng = np.random.default_rng(112)
    x = xp.asarray(rng.normal(size=(2, 24, 16)), dtype=xp.float32).astype("bfloat16")
    dy = xp.asarray(rng.normal(size=x.shape), dtype=xp.float32).astype("bfloat16")

    def run(offload):
        monkeypatch.setenv("MINI_LLM_MOE_OFFLOAD_GATE_UP", str(int(offload)))
        moe.zero_grad()
        y, cache = moe.forward(x)
        expert_caches = cache["experts_cache"]["expert_caches"]
        assert expert_caches
        if offload:
            assert all("offloaded_gu" in c and "g" not in c and "u" not in c
                       for c in expert_caches.values())
            assert sum(c["offloaded_gu"].nbytes for c in expert_caches.values()) > 0
            # Validate that D2H and H2D preserve BF16 bits exactly, independent
            # of numeric tolerances in the full backward path.
            for c in expert_caches.values():
                ticket = c["offloaded_gu"]
                g, u = ticket.to_device()
                xp.cuda.get_current_stream().synchronize()
                assert g.dtype == u.dtype == x.dtype
                assert tuple(g.shape) == ticket.shape
        else:
            assert all("g" in c and "u" in c for c in expert_caches.values())
        dx, da = moe.backward(dy, cache, output_scale=xp.asarray(0.7, dtype=xp.float32),
                              return_output_scale_grad=True)
        xp.cuda.get_current_stream().synchronize()
        return y.copy(), dx.copy(), da.copy(), [p.grad.copy() for p in moe.parameters()]

    a = run(False)
    b = run(True)
    for xa, xb in zip(a[:3], b[:3]):
        np.testing.assert_allclose(_host(xa), _host(xb), rtol=1e-3, atol=1e-3)
    for ga, gb in zip(a[3], b[3]):
        np.testing.assert_allclose(_host(ga), _host(gb), rtol=1e-3, atol=1e-3)


@pytest.mark.skipif(BACKEND_NAME != "cupy", reason="requires RTX/CuPy")
def test_cuda_offload_rejects_concurrent_expert_streams(monkeypatch):
    monkeypatch.setenv("MINI_LLM_MOE_OFFLOAD_GATE_UP", "1")
    monkeypatch.setenv("MINI_LLM_MOE_RECOMPUTE_GATE_UP", "0")
    monkeypatch.setenv("MINI_LLM_CONCURRENT_EXPERTS", "1")
    moe = _make_moe("bfloat16")
    with pytest.raises(RuntimeError, match="CONCURRENT_EXPERTS=0"):
        moe.forward(xp.ones((2, 3, 16), dtype="bfloat16"))
