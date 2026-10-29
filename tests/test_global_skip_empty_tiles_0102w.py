"""0102W: exact global-attention tile skipping, including empty history and packed docs.

Run on RTX 5080:
  MINI_LLM_BACKEND=cupy python -m pytest -v tests/test_global_skip_empty_tiles_0102w.py
"""

import numpy as np
import pytest
from mini_llm.backend import BACKEND_NAME

pytestmark = pytest.mark.skipif(BACKEND_NAME != "cupy", reason="CuPy/CUDA required")


def _gpu():
    pytest.importorskip("triton")
    import cupy as cp
    if not cp.cuda.is_available():
        pytest.skip("CUDA device unavailable")
    return cp


@pytest.mark.parametrize("packed", [False, True])
@pytest.mark.parametrize("nq,nk", [(96, 8), (385, 16), (1024, 32)])
def test_skip_empty_summary_tiles_matches_unoptimized(monkeypatch, packed, nq, nk):
    """Masked tiles must be skipped in forward, dQ, and dK/dV without changing gradients."""
    cp = _gpu()
    from mini_llm.ops.triton_dual_resolution_dense import forward, backward

    seed = 4211 + int(packed) * 7 + nq
    rng = cp.random.RandomState(seed)
    q = (rng.standard_normal((1, nq, 2, 64), dtype=cp.float32) * 0.1).astype(cp.dtype("bfloat16"))
    k = (rng.standard_normal((1, nk, 2, 64), dtype=cp.float32) * 0.1).astype(cp.dtype("bfloat16"))
    v = (rng.standard_normal((1, nk, 2, 64), dtype=cp.float32) * 0.1).astype(cp.dtype("bfloat16"))
    dout = rng.standard_normal(q.shape, dtype=cp.float32) * 0.1

    document_ids = cp.zeros((1, nq), dtype=cp.int32)
    positions = cp.arange(nq, dtype=cp.int32).reshape(1, -1)
    block_document_ids = cp.zeros((1, nk), dtype=cp.int32)
    # Include summary keys from future blocks and invalid summaries; most early queries
    # have no completed block, even when nk > 0.
    ends = cp.asarray(np.arange(nk, dtype=np.int32)[None] * 128 + 127)
    valid = cp.ones((1, nk), dtype=cp.int32)
    valid[0, ::9] = 0
    if packed and nq >= 385:
        split = 257
        document_ids[0, split:] = 1
        positions[0, split:] -= split
        block_document_ids[0, nk // 2:] = 1
        ends[0, nk // 2:] = (cp.arange(nk - nk // 2, dtype=cp.int32) * 128 + 127)

    def execute(enabled):
        monkeypatch.setenv("MINI_LLM_DUAL_RESOLUTION_GLOBAL_SKIP_EMPTY_TILES", str(int(enabled)))
        output, cache = forward(
            q, k, v, scale=0.125,
            documents=document_ids, positions=positions,
            block_documents=block_document_ids,
            block_ends=ends, block_valid=valid,
        )
        dq, dk, dv = backward(dout, cache)
        cp.cuda.get_current_stream().synchronize()
        return tuple(cp.asnumpy(t) for t in (output, cache.available, dq, dk, dv))

    before = execute(False)
    after = execute(True)
    for name, reference, optimized in zip(
        ("forward", "availability", "dQ", "dK", "dV"), before, after
    ):
        if name == "availability":
            np.testing.assert_array_equal(optimized, reference)
        else:
            np.testing.assert_allclose(optimized, reference, rtol=3e-3, atol=3e-3, err_msg=name)
    # First document's initial block cannot see any completed block.
    np.testing.assert_array_equal(after[0][0, :min(nq, 128)], 0.0)


@pytest.mark.parametrize("nq", [129, 385, 1025, 4096])
def test_bounded_contiguous_global_attention_matches_full_loops(monkeypatch, nq):
    cp = _gpu()
    from mini_llm.ops.triton_dual_resolution_dense import forward, backward

    blocks = nq // 128
    nk = blocks * 4
    rng = cp.random.RandomState(5100 + nq)
    q = (rng.standard_normal((1, nq, 2, 64), dtype=cp.float32) * 0.1).astype(cp.dtype("bfloat16"))
    k = (rng.standard_normal((1, nk, 2, 64), dtype=cp.float32) * 0.1).astype(cp.dtype("bfloat16"))
    v = (rng.standard_normal((1, nk, 2, 64), dtype=cp.float32) * 0.1).astype(cp.dtype("bfloat16"))
    dout = rng.standard_normal(q.shape, dtype=cp.float32) * 0.1
    documents = cp.zeros((1, nq), dtype=cp.int32)
    positions = cp.arange(nq, dtype=cp.int32)[None]
    block_documents = cp.zeros((1, nk), dtype=cp.int32)
    ends = cp.repeat(
        cp.arange(blocks, dtype=cp.int32) * 128 + 127, 4
    )[None]
    valid = cp.ones((1, nk), dtype=cp.int32)

    def execute(enabled):
        monkeypatch.setenv(
            "MINI_LLM_DUAL_RESOLUTION_GLOBAL_BOUNDED", str(int(enabled))
        )
        output, cache = forward(
            q, k, v, scale=0.125,
            documents=documents, positions=positions,
            block_documents=block_documents, block_ends=ends, block_valid=valid,
            bounded_causal=True, history_block_size=128, summaries_per_block=4,
        )
        gradients = backward(dout, cache)
        cp.cuda.get_current_stream().synchronize()
        return output, cache.available, *gradients

    reference = execute(False)
    bounded = execute(True)
    for name, got, expected in zip(
        ("forward", "availability", "dQ", "dK", "dV"), bounded, reference
    ):
        if name == "availability":
            np.testing.assert_array_equal(cp.asnumpy(got), cp.asnumpy(expected))
        else:
            np.testing.assert_allclose(
                cp.asnumpy(got), cp.asnumpy(expected),
                rtol=3e-3, atol=3e-3, err_msg=name,
            )


def test_no_valid_history_has_zero_global_gradients(monkeypatch):
    cp = _gpu()
    from mini_llm.ops.triton_dual_resolution_dense import forward, backward
    monkeypatch.setenv("MINI_LLM_DUAL_RESOLUTION_GLOBAL_SKIP_EMPTY_TILES", "1")
    q = cp.ones((1, 64, 2, 64), dtype=cp.dtype("bfloat16"))
    k = cp.ones((1, 4, 2, 64), dtype=cp.dtype("bfloat16"))
    v = cp.ones_like(k)
    output, cache = forward(
        q, k, v, scale=0.125,
        documents=cp.zeros((1, 64), cp.int32),
        positions=cp.arange(64, dtype=cp.int32)[None],
        block_documents=cp.zeros((1, 4), cp.int32),
        block_ends=cp.full((1, 4), 127, cp.int32),
        block_valid=cp.ones((1, 4), cp.int32),
    )
    dq, dk, dv = backward(cp.ones(output.shape, cp.float32), cache)
    for arr in (output, dq, dk, dv):
        np.testing.assert_array_equal(cp.asnumpy(arr), 0.0)
    np.testing.assert_array_equal(cp.asnumpy(cache.available), False)
