"""CuPy BF16 smoke for nondegenerate top-1 MoE router and pinned offload."""

import numpy as np
import pytest

from mini_llm.backend import BACKEND_NAME, RandomStream, asnumpy, xp
from mini_llm.blocks.moe_block import MoE

pytestmark = pytest.mark.skipif(BACKEND_NAME != "cupy", reason="requires CuPy GPU")


@pytest.mark.parametrize("offload", [0, 1])
def test_top1_bf16_router_and_moe_backward_with_optional_offload(monkeypatch, offload):
    monkeypatch.setenv("MINI_LLM_MOE_OFFLOAD_GATE_UP", str(offload))
    monkeypatch.setenv("MINI_LLM_MOE_OFFLOAD_LAYERS", "32")
    monkeypatch.setenv("MINI_LLM_MOE_RECOMPUTE_GATE_UP", "0")
    monkeypatch.setenv("MINI_LLM_CONCURRENT_EXPERTS", "0")
    moe = MoE(d_model=16, d_ff=32, n_experts=2, k=1,
              input_std=.05, output_std=.05,
              rng=RandomStream(125), name="blocks.0.moe", dtype="bfloat16",
              )
    r = np.random.default_rng(125)
    x = xp.asarray(r.normal(size=(2, 18, 16)), dtype=xp.float32).astype("bfloat16")
    dy = xp.asarray(r.normal(size=x.shape), dtype=xp.float32).astype("bfloat16")
    moe.zero_grad()
    weights, indices, _ = moe.router.forward(x)
    assert tuple(weights.shape) == (2, 18, 1)
    w = asnumpy(weights).astype("float32")
    assert np.all(w < 1.) and np.all(w >= .5)
    y, cache = moe.forward(x)
    dx = moe.backward(dy, cache)
    for a in (y, dx, *[p.grad for p in moe.parameters()]):
        assert np.isfinite(asnumpy(a).astype("float32")).all()
    assert any(np.any(asnumpy(p.grad) != 0) for p in moe.router.parameters())
