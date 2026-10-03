import os

import numpy as np

from mini_llm.backend import RandomStream, xp
from mini_llm.ops.experts import ExpertFFN


def _run(expert, x, dy, enabled):
    old = os.environ.get("MINI_LLM_FUSED_EXPERT_GEMM")
    os.environ["MINI_LLM_FUSED_EXPERT_GEMM"] = "1" if enabled else "0"
    try:
        expert.zero_grad()
        y, cache = expert.forward(x)
        dx = expert.backward(dy, cache)
        grads = [p.grad.copy() for p in expert.parameters()]
        return y.copy(), dx.copy(), grads
    finally:
        if old is None:
            os.environ.pop("MINI_LLM_FUSED_EXPERT_GEMM", None)
        else:
            os.environ["MINI_LLM_FUSED_EXPERT_GEMM"] = old


def test_fused_expert_gemm_matches_legacy_path():
    rng = RandomStream(123)
    expert = ExpertFFN(
        d_model=8,
        d_ff=12,
        input_std=0.1,
        output_std=0.1,
        rng=rng,
        dtype="float32",
    )
    x = xp.asarray(np.random.default_rng(1).normal(size=(7, 8)).astype(np.float32))
    dy = xp.asarray(np.random.default_rng(2).normal(size=(7, 8)).astype(np.float32))

    y_old, dx_old, grads_old = _run(expert, x, dy, False)
    y_new, dx_new, grads_new = _run(expert, x, dy, True)

    np.testing.assert_allclose(np.asarray(y_new), np.asarray(y_old), rtol=1e-6, atol=1e-6)
    np.testing.assert_allclose(np.asarray(dx_new), np.asarray(dx_old), rtol=1e-6, atol=1e-6)
    for new, old in zip(grads_new, grads_old):
        np.testing.assert_allclose(np.asarray(new), np.asarray(old), rtol=1e-6, atol=1e-6)
