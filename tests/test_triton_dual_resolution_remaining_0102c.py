"""0102C CUDA parity tests for global summary and bidirectional block attention.

Run with MINI_LLM_BACKEND=cupy python -m pytest -v
    tests/test_triton_dual_resolution_remaining_0102c.py
"""
import numpy as np
import pytest

from mini_llm.backend import BACKEND_NAME

pytestmark = pytest.mark.skipif(BACKEND_NAME != 'cupy', reason='CuPy/CUDA required')


def _gpu():
    pytest.importorskip('triton')
    import cupy as cp
    if not cp.cuda.is_available():
        pytest.skip('CUDA device unavailable')
    return cp


def _operator(*, compression_dim=512, compression_heads=8, block_size=128):
    from mini_llm.backend import RandomStream
    from mini_llm.config import DualResolutionConfig
    from mini_llm.ops.dual_resolution_attention import DualResolutionAttention
    config = DualResolutionConfig(
        enabled=True, block_size=block_size, local_window=512,
        compression_dim=compression_dim, compression_heads=compression_heads,
        compression_depth=1, max_blocks=8, global_query_chunk=128,
        recompute_attention_probs=True,
    )
    return DualResolutionAttention(
        d_model=1024, n_q_heads=16, n_kv_heads=4, d_head=64,
        input_std=0.01, output_std=0.01,
        rng=RandomStream(101), dtype='bfloat16', config=config,
    )


def _compare(name, got, want, *, atol=0.06, rtol=0.08):
    cp = _gpu()
    got, want = cp.asnumpy(got.astype(cp.float32)), cp.asnumpy(want.astype(cp.float32))
    np.testing.assert_allclose(got, want, atol=atol, rtol=rtol, err_msg=name)


@pytest.mark.parametrize('packed', [False, True])
def test_global_summary_masked_forward_and_backward(packed):
    """Guard the causal summary boundary and all four shared query heads."""
    cp = _gpu()
    rng = cp.random.RandomState(701 + int(packed))
    model = _operator()
    length = 385
    summary_count = 3
    q = (rng.standard_normal((1, length, 16, 64), dtype=cp.float32)*0.17).astype(cp.dtype('bfloat16'))
    k = (rng.standard_normal((1, summary_count, 4, 64), dtype=cp.float32)*0.17).astype(cp.dtype('bfloat16'))
    v = (rng.standard_normal((1, summary_count, 4, 64), dtype=cp.float32)*0.17).astype(cp.dtype('bfloat16'))
    upstream = rng.standard_normal((1, length, 16, 64), dtype=cp.float32)*0.2
    documents = cp.zeros((1, length), dtype=cp.int64)
    positions = cp.broadcast_to(cp.arange(length, dtype=cp.int64)[None], (1, length)).copy()
    bdocs = cp.zeros((1, 3), dtype=cp.int64)
    ends = cp.asarray([[127,255,383]], dtype=cp.int64)
    valid = cp.asarray([[True,True,True]])
    if packed:
        # Multiple EOS-delimited docs, with different block end positions.
        documents[0, 257:] = 1
        positions[0, 257:] -= 257
        bdocs[0, 2] = 1
        ends[0, 2] = 127
        valid[0, 1] = False  # exercise an explicitly absent summary

    def run(backend):
        model.global_kernel = backend
        output, available, cache = model._global_forward(
            q, k, v, documents, positions, valid, bdocs, ends,
        )
        dq, dk, dv = model._global_backward(
            upstream, q, k, v, cache, {
                'document_ids': documents, 'position_ids': positions,
                'block_valid': valid, 'block_document_ids': bdocs,
                'block_end_positions': ends,
            },
        )
        return output, available, dq, dk, dv
    reference = run('cupy')
    candidate = run('triton')
    cp.cuda.get_current_stream().synchronize()
    for title, actual, expected in zip(
        ('global output', 'availability', 'global dq', 'global dk', 'global dv'),
        candidate, reference,
    ):
        if title == 'availability':
            np.testing.assert_array_equal(cp.asnumpy(actual), cp.asnumpy(expected))
        else:
            _compare(title, actual, expected, atol=0.07, rtol=0.10)
    np.testing.assert_array_equal(cp.asnumpy(candidate[0][0, :128]), 0.0)


def test_full_block_attention_forward_backward():
    """Batched independent bidirectional 128-token blocks, no causal masking."""
    cp = _gpu()
    rng = cp.random.RandomState(904)
    model = _operator()
    blocks, heads, length, head_dim = 3, 8, 128, 64
    q = rng.standard_normal((blocks, heads, length, head_dim), dtype=cp.float32)*0.13
    k = rng.standard_normal((blocks, heads, length, head_dim), dtype=cp.float32)*0.13
    v = rng.standard_normal((blocks, heads, length, head_dim), dtype=cp.float32)*0.13
    dout = rng.standard_normal((blocks, heads, length, head_dim), dtype=cp.float32)*0.09
    def run(backend):
        model.compression_kernel = backend
        context, cache = model._compress_attention_forward(q,k,v,retain_probabilities=False)
        return context, *model._compress_attention_backward(dout,q,k,v,cache)
    reference = run('cupy')
    candidate = run('triton')
    for title, got, want in zip(('context','dQ','dK','dV'),candidate,reference):
        _compare('compression '+title,got,want,atol=0.035,rtol=0.06)


def test_complete_attention_integration_and_parameter_gradients():
    """Global+compressor Triton together, local CuPy to isolate new paths."""
    cp = _gpu()
    model = _operator()
    rng = cp.random.RandomState(530)
    x = (rng.standard_normal((1, 257, 1024),dtype=cp.float32)*0.04).astype(cp.dtype('bfloat16'))
    dy = (rng.standard_normal((1, 257, 1024),dtype=cp.float32)*0.02).astype(cp.dtype('bfloat16'))
    def run(kernel):
        model.local_kernel = 'cupy'
        model.global_kernel = kernel
        model.compression_kernel = kernel
        model.zero_grad()
        out, cache = model.forward(x, return_cache=True)
        dx = model.backward(dy,cache)
        return out,dx,{p.name:p.grad.copy() for p in model.parameters()}
    baseline = run('cupy')
    experiment = run('triton')
    _compare('integrated output',experiment[0],baseline[0],atol=0.08,rtol=0.1)
    _compare('integrated dx',experiment[1],baseline[1],atol=0.10,rtol=0.15)
    for name, original in baseline[2].items():
        _compare('integrated gradient '+name,experiment[2][name],original,atol=0.15,rtol=0.2)
