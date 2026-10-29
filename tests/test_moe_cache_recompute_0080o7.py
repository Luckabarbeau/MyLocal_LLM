"""0080O7: expert projection rematerialization must preserve the MoE function.

No context-length changes, token dropping, history masking, or stochastic replay:
only G and U are recomputed from the saved routing plan and MoE input.
"""
import numpy as np
import pytest

from mini_llm.backend import BACKEND_NAME, RandomStream, asnumpy, xp
from mini_llm.ops.experts import (
    ExpertFFN,
    Experts,
    _moe_recompute_gate_up_enabled,
)
from mini_llm.blocks.moe_block import MoE


def host(x):
    return np.asarray(asnumpy(x)).astype(np.float64)


def check_same(lhs, rhs, *, rtol=1e-6, atol=1e-7):
    np.testing.assert_allclose(host(lhs), host(rhs), rtol=rtol, atol=atol)


def make_expert(seed, dtype):
    return ExpertFFN(
        d_model=8, d_ff=13, input_std=0.07, output_std=0.05,
        rng=RandomStream(seed), name="o7.expert", dtype=dtype,
    )


def test_gate_up_recompute_layer_limit(monkeypatch):
    monkeypatch.setenv("MINI_LLM_MOE_RECOMPUTE_GATE_UP", "1")
    monkeypatch.setenv("MINI_LLM_MOE_RECOMPUTE_GATE_UP_LAYERS", "17")
    assert _moe_recompute_gate_up_enabled("blocks.0.dense_ffn.expert.W_gate")
    assert _moe_recompute_gate_up_enabled("blocks.16.dense_ffn.expert.W_gate")
    assert not _moe_recompute_gate_up_enabled("blocks.17.dense_ffn.expert.W_gate")
    assert not _moe_recompute_gate_up_enabled("blocks.19.dense_ffn.expert.W_gate")
    assert _moe_recompute_gate_up_enabled()

    monkeypatch.setenv("MINI_LLM_MOE_RECOMPUTE_GATE_UP_LAYERS", "invalid")
    with pytest.raises(ValueError, match="must be an integer"):
        _moe_recompute_gate_up_enabled("blocks.0.dense_ffn.expert.W_gate")


@pytest.mark.parametrize("packed", [False, True])
@pytest.mark.parametrize("dtype", ["float32", "float64"])
def test_expert_backward_and_cache_parity(monkeypatch, dtype, packed):
    monkeypatch.setenv("MINI_LLM_FUSED_EXPERT_GEMM", str(int(packed)))
    model = make_expert(77, dtype)
    gen = np.random.default_rng(88)
    x = xp.asarray(gen.normal(size=(6, 8)), dtype=dtype)
    dy = xp.asarray(gen.normal(size=(6, 8)), dtype=dtype)
    weights = xp.asarray(gen.uniform(.2, .8, size=(6,)), dtype=xp.float32)

    def run(recompute, *, cache_input):
        monkeypatch.setenv("MINI_LLM_MOE_RECOMPUTE_GATE_UP", str(int(recompute)))
        model.zero_grad()
        y, cache = model.forward(x, cache_input=cache_input)
        if recompute:
            assert "g" not in cache and "u" not in cache
            assert cache["recompute_gate_up"] is True
        else:
            assert "g" in cache and "u" in cache
        dx, dweight = model.backward(
            dy, cache, x=None if cache_input else x,
            route_weights=weights, return_route_dweights=True,
        )
        return y.copy(), dx.copy(), dweight.copy(), [p.grad.copy() for p in model.parameters()]

    base = run(False, cache_input=False)
    compact = run(True, cache_input=False)
    direct = run(True, cache_input=True)
    for got in (compact, direct):
        for a, b in zip(got[:3], base[:3]):
            check_same(a, b)
        for a, b in zip(got[3], base[3]):
            check_same(a, b)


@pytest.mark.parametrize("packed", [False, True])
def test_complete_moe_routing_and_gradients(monkeypatch, packed):
    monkeypatch.setenv("MINI_LLM_FUSED_EXPERT_GEMM", str(int(packed)))
    model = MoE(
        d_model=8, d_ff=13, n_experts=4, k=2,
        input_std=0.08, output_std=0.05, rng=RandomStream(991),
        name="o7.moe", dtype="float64",
    )
    gen = np.random.default_rng(91)
    x = xp.asarray(gen.normal(size=(2, 7, 8)), dtype="float64")
    dy = xp.asarray(gen.normal(size=(2, 7, 8)), dtype="float64")

    def run(recompute):
        monkeypatch.setenv("MINI_LLM_MOE_RECOMPUTE_GATE_UP", str(int(recompute)))
        model.zero_grad()
        y, cache = model.forward(x)
        expert_caches = cache["experts_cache"]["expert_caches"]
        assert expert_caches
        projection_bytes = sum(
            sum(t.nbytes for t in (c["g"], c["u"]))
            for c in expert_caches.values()
        ) if not recompute else 0
        if recompute:
            assert all("g" not in c and "u" not in c for c in expert_caches.values())
        dx = model.backward(dy, cache)
        return y.copy(), dx.copy(), [p.grad.copy() for p in model.parameters()], projection_bytes

    a = run(False)
    b = run(True)
    assert a[-1] > 0 and b[-1] == 0
    check_same(a[0], b[0])
    check_same(a[1], b[1])
    for pa, pb in zip(a[2], b[2]):
        check_same(pa, pb)


def test_moe_residual_gate_gradient_parity(monkeypatch):
    monkeypatch.setenv("MINI_LLM_FUSED_EXPERT_GEMM", "0")
    model = MoE(
        d_model=8, d_ff=13, n_experts=4, k=2,
        input_std=0.08, output_std=0.05, rng=RandomStream(996),
        name="o7.gated", dtype="float64",
    )
    gen = np.random.default_rng(97)
    x = xp.asarray(gen.normal(size=(2, 7, 8)), dtype="float64")
    dy = xp.asarray(gen.normal(size=(2, 7, 8)), dtype="float64")
    def run(recompute):
        monkeypatch.setenv("MINI_LLM_MOE_RECOMPUTE_GATE_UP", str(int(recompute)))
        model.zero_grad()
        y, cache = model.forward(x)
        dx, dalpha = model.backward(
            dy, cache, output_scale=xp.asarray(.7, dtype=xp.float64),
            return_output_scale_grad=True,
        )
        return y.copy(), dx.copy(), dalpha.copy(), [p.grad.copy() for p in model.parameters()]
    a, b = run(False), run(True)
    for p, q in zip(a[:3], b[:3]):
        check_same(p, q)
    for p, q in zip(a[3], b[3]):
        check_same(p, q)


@pytest.mark.skipif(BACKEND_NAME != "cupy", reason="requires RTX/CuPy")
@pytest.mark.parametrize("packed", [False, True])
def test_cuda_bf16_moe_recompute_parity(monkeypatch, packed):
    monkeypatch.setenv("MINI_LLM_FUSED_EXPERT_GEMM", str(int(packed)))
    monkeypatch.setenv("MINI_LLM_CONCURRENT_EXPERTS", "0")
    monkeypatch.setenv("MINI_LLM_FUSED_SWIGLU", "1")
    model = MoE(
        d_model=32, d_ff=64, n_experts=4, k=2,
        input_std=0.03, output_std=0.03, rng=RandomStream(993),
        name="o7.cuda", dtype="bfloat16",
    )
    gen = np.random.default_rng(92)
    x = xp.asarray(gen.normal(size=(2, 16, 32)), dtype=xp.float32).astype("bfloat16")
    dy = xp.asarray(gen.normal(size=(2, 16, 32)), dtype=xp.float32).astype("bfloat16")

    def run(recompute):
        monkeypatch.setenv("MINI_LLM_MOE_RECOMPUTE_GATE_UP", str(int(recompute)))
        model.zero_grad()
        y, cache = model.forward(x)
        assert all(("g" not in c and "u" not in c) == bool(recompute)
                   for c in cache["experts_cache"]["expert_caches"].values())
        dx = model.backward(dy, cache)
        xp.cuda.Stream.null.synchronize()
        return y.copy(), dx.copy(), [p.grad.copy() for p in model.parameters()]

    a = run(False)
    b = run(True)
    for qa, qb in zip((a[0], a[1]), (b[0], b[1])):
        check_same(qa, qb, rtol=1e-3, atol=1e-3)
    for qa, qb in zip(a[2], b[2]):
        check_same(qa, qb, rtol=1e-3, atol=1e-3)
