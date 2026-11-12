"""GPU-only 0102B local-attention parity: CuPy baseline vs Triton BF16.

Run with MINI_LLM_BACKEND=cupy python -m pytest -q
    tests/test_triton_dual_resolution_local_0102b.py
"""
import numpy as np
import pytest

from mini_llm.backend import BACKEND_NAME

pytestmark = pytest.mark.skipif(
    BACKEND_NAME != "cupy", reason="Triton GPU parity requires CuPy backend"
)


def _available():
    if BACKEND_NAME != "cupy":
        pytest.skip("CuPy backend required")
    pytest.importorskip("triton")
    import cupy as cp
    if not cp.cuda.is_available():
        pytest.skip("CUDA device required")
    return cp


def _run(length, *, split_docs=False):
    cp = _available()
    from mini_llm.backend import RandomStream
    from mini_llm.config import DualResolutionConfig
    from mini_llm.ops.dual_resolution_attention import DualResolutionAttention
    rng = cp.random.RandomState(1234 + length)
    q = (rng.standard_normal((1, length, 16, 64), dtype=cp.float32) * 0.35).astype(cp.dtype('bfloat16'))
    k = (rng.standard_normal((1, length, 4, 64), dtype=cp.float32) * 0.35).astype(cp.dtype('bfloat16'))
    v = (rng.standard_normal((1, length, 4, 64), dtype=cp.float32) * 0.35).astype(cp.dtype('bfloat16'))
    d_out = rng.standard_normal((1, length, 16, 64), dtype=cp.float32) * 0.1
    docs = cp.zeros((1, length), dtype=cp.int64)
    if split_docs:
        docs[0, 111:] = 1
        docs[0, 299:] = 2
        docs[0, 449:] = 3

    config = DualResolutionConfig(
        enabled=True, block_size=128, local_window=512,
        compression_dim=64, compression_heads=1,
        compression_depth=1, max_blocks=8, global_query_chunk=256,
    )
    # Test direct local kernels without allocating a complete 20-layer model.
    operator = DualResolutionAttention(
        d_model=1024, n_q_heads=16, n_kv_heads=4, d_head=64,
        input_std=0.01, output_std=0.01,
        rng=RandomStream(121), dtype="float32", config=config,
    )
    operator.local_kernel = "cupy"
    expected, reference_cache = operator._local_forward(q, k, v, docs)
    edq, edk, edv = operator._local_backward(
        d_out, q, k, v, reference_cache, docs,
    )

    operator.local_kernel = "triton"
    actual, triton_cache = operator._local_forward(q, k, v, docs)
    adq, adk, adv = operator._local_backward(
        d_out, q, k, v, triton_cache, docs,
    )
    cp.cuda.get_current_stream().synchronize()

    for name, got, want, absolute, relative in (
        ("output", actual, expected, 0.035, 0.035),
        ("dQ", adq, edq, 0.065, 0.10),
        ("dK", adk, edk, 0.065, 0.10),
        ("dV", adv, edv, 0.065, 0.10),
    ):
        got_cpu, want_cpu = cp.asnumpy(got), cp.asnumpy(want)
        np.testing.assert_allclose(
            got_cpu, want_cpu, atol=absolute, rtol=relative,
            err_msg=f"{name} CuPy/Triton parity at length={length} split={split_docs}",
        )


def test_all_16_heads_and_shared_kv_gradients():
    _run(129)


def test_512_window_and_multiple_document_boundaries():
    _run(640, split_docs=True)


def test_longer_sliding_window():
    _run(1024)



def test_integrated_dual_resolution_forward_backward():
    """Exercise projections, local/global fusion, compressor, and input gradients."""
    cp = _available()
    from mini_llm.backend import RandomStream
    from mini_llm.config import DualResolutionConfig
    from mini_llm.ops.dual_resolution_attention import DualResolutionAttention

    config = DualResolutionConfig(
        enabled=True, block_size=128, local_window=512,
        compression_dim=64, compression_heads=1,
        compression_depth=1, max_blocks=8, global_query_chunk=128,
        recompute_attention_probs=True,
    )
    attention = DualResolutionAttention(
        d_model=1024, n_q_heads=16, n_kv_heads=4, d_head=64,
        input_std=0.01, output_std=0.01, rng=RandomStream(42),
        dtype='bfloat16', config=config,
    )
    random = cp.random.RandomState(100)
    x = (random.standard_normal((1, 257, 1024), dtype=cp.float32)
         * 0.04).astype(cp.dtype('bfloat16'))
    dy = (random.standard_normal((1, 257, 1024), dtype=cp.float32)
          * 0.02).astype(cp.dtype('bfloat16'))
    # A completed 128-token block must contribute through the global path.
    def run(kernel):
        attention.local_kernel = kernel
        attention.zero_grad()
        output, cache = attention.forward(x, return_cache=True)
        dx = attention.backward(dy, cache)
        return (cp.asnumpy(output.astype(cp.float32)),
                cp.asnumpy(dx.astype(cp.float32)),
                {p.name: cp.asnumpy(p.grad.astype(cp.float32)).copy()
                 for p in attention.parameters()})

    baseline_y, baseline_dx, baseline_grads = run('cupy')
    triton_y, triton_dx, triton_grads = run('triton')
    np.testing.assert_allclose(triton_y, baseline_y, atol=0.07, rtol=0.12)
    np.testing.assert_allclose(triton_dx, baseline_dx, atol=0.07, rtol=0.12)
    for name in baseline_grads:
        np.testing.assert_allclose(
            triton_grads[name], baseline_grads[name],
            atol=0.1, rtol=0.15, err_msg=f"gradient mismatch: {name}",
        )


@pytest.mark.parametrize('batch,length', [(2, 65), (1, 641)])
@pytest.mark.parametrize('skip_head', [-1, 0])
def test_mixed_window_packed_backward_against_fp64(batch, length, skip_head):
    cp = _available()
    from mini_llm.ops.triton_dual_resolution_local import (
        local_attention_forward, local_attention_backward,
    )
    rng = np.random.default_rng(107 + length)
    # Match packed Q/K/V row strides, with non-tile-aligned EOS boundaries.
    packed = cp.asarray(rng.normal(0, 0.2, (batch, length, 24, 64)),
                        dtype=cp.float32).astype('bfloat16')
    q, k, v = packed[:, :, :16], packed[:, :, 16:20], packed[:, :, 20:24]
    dy_np = rng.normal(0, 0.1, q.shape).astype(np.float32)
    dy = cp.asarray(dy_np)
    docs = np.zeros((batch, length), dtype=np.int32)
    docs[:, 97 if length > 97 else 17:] = 1
    output, cache = local_attention_forward(
        q, k, v, cp.asarray(docs), long_window=4096, long_heads=1,
        skip_head=skip_head,
    )
    actual = local_attention_backward(dy, cache)
    qf, kf, vf = [cp.asnumpy(x).astype(np.float64).transpose(0, 2, 1, 3)
                  for x in (q, k, v)]
    kf, vf = np.repeat(kf, 4, axis=1), np.repeat(vf, 4, axis=1)
    upstream = dy_np.astype(np.float64).transpose(0, 2, 1, 3)
    positions = np.arange(length)
    windows = np.full(16, 512)
    windows[0] = 4096
    allowed = (
        (positions[None, :] <= positions[:, None])[None, None]
        & (positions[None, None, None, :] >=
           positions[None, None, :, None] - windows[None, :, None, None] + 1)
        & (docs[:, None, :, None] == docs[:, None, None, :])
    )
    scores = np.where(allowed, (qf @ kf.swapaxes(-2, -1)) * 0.125, -np.inf)
    probabilities = np.exp(scores - scores.max(axis=-1, keepdims=True))
    probabilities /= probabilities.sum(axis=-1, keepdims=True)
    if skip_head >= 0:
        probabilities[:, skip_head] = 0
    expected_output = probabilities @ vf
    # Reverse O=P V, P=softmax(S), S=Q K^T / sqrt(64), then sum GQA groups.
    dp = upstream @ vf.swapaxes(-2, -1)
    ds = probabilities * (dp - (dp * probabilities).sum(axis=-1, keepdims=True))
    dq = (ds @ kf) * 0.125
    dk = ((ds.swapaxes(-2, -1) @ qf) * 0.125).reshape(batch, 4, 4, length, 64).sum(axis=2)
    dv = (probabilities.swapaxes(-2, -1) @ upstream).reshape(batch, 4, 4, length, 64).sum(axis=2)
    # The caller owns/replaces skipped output and dQ heads; their storage is
    # intentionally unwritten here. KV gradients must still exclude that head.
    owned = np.arange(16) != skip_head
    np.testing.assert_allclose(cp.asnumpy(output)[:, :, owned],
                               expected_output.transpose(0, 2, 1, 3)[:, :, owned],
                               atol=0.003, rtol=0.06)
    for index, (got, expected) in enumerate(zip(actual, (dq, dk, dv))):
        assert str(got.dtype) == 'float32'
        got_np, expected_np = cp.asnumpy(got), expected.transpose(0, 2, 1, 3)
        if index == 0:
            got_np, expected_np = got_np[:, :, owned], expected_np[:, :, owned]
        np.testing.assert_allclose(got_np, expected_np, atol=0.003, rtol=0.06)
