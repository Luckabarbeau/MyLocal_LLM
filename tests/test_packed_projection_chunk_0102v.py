"""0102V packed projection backward chunking: exact-shape reference checks.

The GPU BF16 path requires the user's RTX 5080 for authoritative parity and
performance measurements.  These CPU tests verify layouts, summation and
parameter-gradient semantics without assuming that hardware is available.
"""

import numpy as np
import pytest

from mini_llm.ops.dual_resolution_attention import _PackedProjection


class _Param:
    def __init__(self, data):
        self.data = data
        self.grad = np.zeros(data.shape, dtype=np.float32)


def _run(value, gradients, weights, chunk, monkeypatch, accumulate):
    monkeypatch.setenv("MINI_LLM_DUAL_RESOLUTION_PACKED_BACKWARD_CHUNK_TOKENS", str(chunk))
    params = [_Param(w.copy()) for w in weights]
    op = _PackedProjection(params)
    op.refresh()
    dx = op.backward(gradients, value, accumulate)
    return dx.copy(), [p.grad.copy() for p in params]


@pytest.mark.parametrize("rows,chunk", [(3, 2), (9, 4), (32, 7), (33, 64)])
@pytest.mark.parametrize("accumulate", [False, True])
def test_chunked_packed_backward_matches_reference(monkeypatch, rows, chunk, accumulate):
    rng = np.random.default_rng(10221 + rows)
    value = rng.normal(size=(1, rows, 12)).astype(np.float32)
    widths = (7, 3, 3)
    weights = [rng.normal(size=(12, width)).astype(np.float32) for width in widths]
    grads = [rng.normal(size=(1, rows, width)).astype(np.float32) for width in widths]
    expected = _run(value, grads, weights, 0, monkeypatch, accumulate)
    actual = _run(value, grads, weights, chunk, monkeypatch, accumulate)
    np.testing.assert_allclose(actual[0], expected[0], atol=3e-6, rtol=3e-6)
    for a, b in zip(actual[1], expected[1]):
        np.testing.assert_allclose(a, b, atol=3e-5, rtol=3e-5)


def test_packed_backward_rejects_negative_chunks(monkeypatch):
    a = np.ones((2, 3), dtype=np.float32)
    op = _PackedProjection([_Param(np.ones((3, 2), dtype=np.float32))])
    monkeypatch.setenv("MINI_LLM_DUAL_RESOLUTION_PACKED_BACKWARD_CHUNK_TOKENS", "-1")
    with pytest.raises(ValueError, match="nonnegative"):
        op.backward((a,), np.ones((2, 3), dtype=np.float32), False)


@pytest.mark.skipif(
    __import__("mini_llm.backend", fromlist=["BACKEND_NAME"]).BACKEND_NAME != "cupy",
    reason="requires CuPy/CUDA BF16",
)
def test_chunked_bf16_gpu_packed_projection_parity(monkeypatch):
    """Check BF16 tensor-core chunked matmuls against the packed full GEMM."""
    import cupy as cp

    class _GpuParam:
        def __init__(self, data):
            self.data = data
            self.grad = cp.zeros(data.shape, dtype=cp.float32)

    rng = np.random.default_rng(10222)
    rows, width = 1025, 64
    weights = [
        cp.asarray(rng.normal(scale=.02, size=(width, n)).astype("float32")).astype("bfloat16")
        for n in (64, 16, 16)
    ]
    value = cp.asarray(rng.normal(scale=.02, size=(1, rows, width)).astype("float32")).astype("bfloat16")
    grads = [
        cp.asarray(rng.normal(scale=.02, size=(1, rows, n)).astype("float32"))
        for n in (64, 16, 16)
    ]

    def run(chunk):
        monkeypatch.setenv("MINI_LLM_DUAL_RESOLUTION_BF16_PROJECTION_GRADS", "1")
        monkeypatch.setenv("MINI_LLM_DUAL_RESOLUTION_PACKED_BACKWARD_CHUNK_TOKENS", str(chunk))
        op = _PackedProjection([_GpuParam(w.copy()) for w in weights])
        op.refresh()
        dx = op.backward(grads, value, True)
        return dx, [p.grad.copy() for p in op.parameters]

    expected = run(0)
    actual = run(128)
    cp.testing.assert_allclose(actual[0].astype(cp.float32), expected[0].astype(cp.float32), rtol=.08, atol=.005)
    for a, b in zip(actual[1], expected[1]):
        cp.testing.assert_allclose(a, b, rtol=.12, atol=.015)
