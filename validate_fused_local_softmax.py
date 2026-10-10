"""GPU differential check for the fused local-attention softmax kernels.

Run from the repository root with:
  MINI_LLM_BACKEND=cupy MINI_LLM_FUSED_LOCAL_SOFTMAX=1 \
      python validate_fused_local_softmax.py
"""
import math
import os

os.environ.setdefault("MINI_LLM_FUSED_LOCAL_SOFTMAX", "1")

from mini_llm.backend import xp, BACKEND_NAME
from mini_llm.ops.indexed_attention import (
    _local_causal_softmax_forward_cuda,
    _local_causal_softmax_backward_cuda,
    _masked_softmax_forward,
    _softmax_backward,
)


def _max_abs(a, b):
    return float(xp.max(xp.abs(a - b)).item())


def run_case(batch, heads, q_start, q_len, window):
    key_start = max(0, q_start - window + 1)
    key_end = q_start + q_len
    k_len = key_end - key_start

    # Deterministic, nontrivial FP32 scores without depending on a backend RNG API.
    n = batch * heads * q_len * k_len
    base = xp.arange(n, dtype=xp.float32).reshape(batch, heads, q_len, k_len)
    scores = xp.sin(base * xp.float32(0.013)) * xp.float32(2.0)
    scores = xp.ascontiguousarray(scores)

    q_positions = xp.arange(q_start, q_start + q_len, dtype=xp.int64)[:, None]
    k_positions = xp.arange(key_start, key_end, dtype=xp.int64)[None, :]
    valid = (
        (k_positions <= q_positions)
        & (k_positions >= (q_positions - window + 1))
    )[None, None, :, :]

    fused = _local_causal_softmax_forward_cuda(
        scores, q_start, key_start, window, 1.0
    )
    if fused is None:
        raise RuntimeError("fused forward returned None; CUDA kernel is disabled or unavailable")
    reference = _masked_softmax_forward(scores, valid, logit_multiplier=1.0)
    f_err = _max_abs(fused, reference)

    dprobs = xp.cos(base * xp.float32(0.017)).astype(xp.float32)
    dprobs = xp.ascontiguousarray(dprobs)
    probs = xp.ascontiguousarray(fused)
    fused_bwd = _local_causal_softmax_backward_cuda(
        dprobs, probs, q_start, key_start, window
    )
    if fused_bwd is None:
        raise RuntimeError("fused backward returned None; CUDA kernel declined the test layout")
    ref_bwd = _softmax_backward(dprobs, probs)
    ref_bwd = xp.where(valid, ref_bwd, xp.float32(0.0))
    b_err = _max_abs(fused_bwd, ref_bwd)

    print(
        f"B={batch} H={heads} q_start={q_start:4d} Q={q_len:4d} "
        f"K={k_len:4d} W={window:4d} | "
        f"forward max_abs={f_err:.3e} backward max_abs={b_err:.3e}"
    )
    return f_err, b_err


def main():
    if BACKEND_NAME != "cupy":
        raise RuntimeError(f"this check requires MINI_LLM_BACKEND=cupy, got {BACKEND_NAME!r}")

    cases = [
        (1, 1, 0, 17, 8),
        (2, 4, 0, 128, 128),
        (2, 4, 1024, 128, 1024),
        (2, 4, 2048, 128, 1024),
        (2, 4, 3072, 128, 1024),
        (2, 4, 1024, 512, 1024),
    ]
    worst_f = 0.0
    worst_b = 0.0
    for case in cases:
        f_err, b_err = run_case(*case)
        worst_f = max(worst_f, f_err)
        worst_b = max(worst_b, b_err)

    print(f"worst forward max_abs={worst_f:.3e}")
    print(f"worst backward max_abs={worst_b:.3e}")
    if not math.isfinite(worst_f) or not math.isfinite(worst_b):
        raise RuntimeError("non-finite fused/reference difference")
    if worst_f > 1e-5 or worst_b > 2e-5:
        raise RuntimeError(
            "fused local softmax does not match the reference closely enough; "
            "do not use it for training yet"
        )
    print("PASS: fused local softmax matches the reference on all tested CUDA cases")


if __name__ == "__main__":
    main()
