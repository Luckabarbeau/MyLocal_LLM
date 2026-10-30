"""0102Z: causal deterministic strided original-token retrieval tests."""
from dataclasses import replace
import numpy as np
import pytest

from mini_llm.config import ModelConfig, DualResolutionConfig
from mini_llm.ops.sparse_stride_history_0102z import sparse_forward, sparse_backward


def _sample():
    rng = np.random.default_rng(1602)
    q = rng.normal(size=(2, 43, 1, 64)).astype(np.float32) * .15
    k = rng.normal(size=(2, 43, 1, 64)).astype(np.float32) * .15
    v = rng.normal(size=(2, 43, 1, 64)).astype(np.float32) * .15
    docs = np.zeros((2, 43), dtype=np.int32)
    docs[:, 11:] = 1
    docs[:, 26:] = 2
    return q, k, v, docs


def test_architecture_separate_no_new_parameters_and_all_phases():
    base = ModelConfig.dual_resolution_matrix_pool_1024_20l()
    sparse = ModelConfig.dual_resolution_matrix_pool_sparse_1024_20l()
    assert base.dual_resolution.sparse_retrieval_stride == 0
    assert sparse.dual_resolution.sparse_retrieval_stride == 16
    assert sparse.dual_resolution.sparse_retrieval_head == 1
    assert len(sparse.dual_resolution.sparse_retrieval_layers) == 16
    phases = (0, 8, 4, 12, 2, 10, 6, 14, 1, 9, 5, 13, 3, 11, 7, 15)
    assert sorted(phases) == list(range(16))
    assert sparse.d_model == base.d_model == 1024
    assert sparse.n_layers == base.n_layers == 20
    assert sparse.dual_resolution.long_local_heads == 1
    assert sparse.dual_resolution.long_local_window == 4096
    assert sparse.dual_resolution.summaries_per_block == 4
    with pytest.raises(ValueError, match="cannot replace"):
        replace(sparse.dual_resolution, sparse_retrieval_head=0)


def test_reference_exact_causal_document_mask_and_distant_gradient():
    q, k, v, docs = _sample()
    out, cache = sparse_forward(q, k, v, docs, stride=16, phase=5,
                                scale=.125, kernel='cupy')
    assert out.shape == q.shape
    # All positions before first selected key in a document have no sparse
    # history, and the head must produce zero rather than NaN.
    np.testing.assert_array_equal(out[:, :5], 0)
    assert np.isfinite(out).all()
    # Phase 5 key in first document cannot leak into the next document.
    np.testing.assert_array_equal(out[:, 11:21], 0)
    changed = v.copy()
    changed[:, 37:, :, :] += 10.
    out_changed, _ = sparse_forward(q, k, changed, docs, stride=16, phase=5,
                                    scale=.125, kernel='cupy')
    np.testing.assert_allclose(out_changed[:, :37], out[:, :37], atol=1e-7)
    grad = np.zeros_like(out)
    grad[:, -1] = 1
    dq, dk, dv = sparse_backward(grad, cache, kernel='cupy')
    assert dq.shape == q.shape
    assert dk.shape == dv.shape == (2, 3, 1, 64)
    # Last token's document sees only the phase-5 key at index 37; the
    # earlier selected indices 5 and 21 must receive no gradient.
    np.testing.assert_array_equal(dk[:, :2], 0)
    np.testing.assert_array_equal(dv[:, :2], 0)
    assert np.max(np.abs(dv[:, 2])) > 0


def test_reference_finite_differences_qkv():
    q, k, v, docs = _sample()
    grad = np.random.default_rng(42).normal(size=q.shape).astype(np.float32)
    out, cache = sparse_forward(q, k, v, docs, stride=16, phase=5,
                                scale=.125, kernel='cupy')
    dq, dk, dv = sparse_backward(grad, cache, kernel='cupy')
    def objective():
        y, _ = sparse_forward(q, k, v, docs, stride=16, phase=5,
                              scale=.125, kernel='cupy')
        return np.sum(y * grad)
    eps = 2e-3
    for arr, index, expected in (
        (q, (0, 25, 0, 12), dq[0, 25, 0, 12]),
        (k, (0, 21, 0, 8), dk[0, 1, 0, 8]),
        (v, (0, 37, 0, 14), dv[0, 2, 0, 14]),
    ):
        original = arr[index].item()
        arr[index] = original + eps
        plus = objective()
        arr[index] = original - eps
        minus = objective()
        arr[index] = original
        np.testing.assert_allclose((plus-minus)/(2*eps), expected, rtol=.035, atol=.002)


def test_inference_sparse_head_respects_zero_history_and_selected_kv():
    from mini_llm.ops.dual_resolution_attention_inference import DualResolutionAttentionInference
    config = DualResolutionConfig(
        enabled=True, compression_mode='matrix_pool', long_local_heads=1,
        long_local_window=4096, global_heads=2, summaries_per_block=4,
        sparse_retrieval_stride=16, sparse_retrieval_head=1,
        sparse_retrieval_phase=5,
    )
    inference = DualResolutionAttentionInference(1024, 16, 4, 64,
                                                   dual_config=config)
    q = np.ones((1, 1, 16, 64), dtype=np.float32)
    k = np.ones((4, 1, 64), dtype=np.float32)
    v = np.ones((4, 1, 64), dtype=np.float32)
    empty = inference._matrix_local_attention(q, k, v,
                                               sparse_k=np.zeros((1,0,1,64), dtype=np.float32),
                                               sparse_v=np.zeros((1,0,1,64), dtype=np.float32))
    assert np.all(empty[..., 1, :] == 0.)
    selected_k = np.ones((1,2,1,64), dtype=np.float32)
    selected_v = np.stack((np.ones((64,),dtype=np.float32),
                           np.ones((64,),dtype=np.float32)*3), axis=0).reshape(1,2,1,64)
    populated = inference._matrix_local_attention(q, k, v,
                           sparse_k=selected_k, sparse_v=selected_v)
    np.testing.assert_allclose(populated[..., 1, :], 2., rtol=1e-5)


def test_sparse_backward_reuses_buffers_and_accumulates_kv():
    """Catch the reference bug that dropped sparse gradients in Model B."""
    q, k, v, docs = _sample()
    upstream = np.random.default_rng(92).normal(size=q.shape).astype(np.float32)
    _, cache = sparse_forward(q, k, v, docs, stride=16, phase=5,
                              scale=.125, kernel='cupy')
    expected_q, expected_k, expected_v = sparse_backward(
        upstream, cache, kernel='cupy')

    dq = np.full(expected_q.shape, 19., dtype=np.float32)
    dk = np.full(expected_k.shape, 3., dtype=np.float32)
    dv = np.full(expected_v.shape, -7., dtype=np.float32)
    result = sparse_backward(upstream, cache, kernel='cupy',
                             dq=dq, dk=dk, dv=dv, add_dkdv=True)
    assert all(actual is supplied for actual, supplied in zip(result, (dq, dk, dv)))
    np.testing.assert_allclose(dq, expected_q, rtol=1e-6, atol=1e-6)
    np.testing.assert_allclose(dk, expected_k + 3., rtol=1e-6, atol=1e-6)
    np.testing.assert_allclose(dv, expected_v - 7., rtol=1e-6, atol=1e-6)

    # When add_dkdv=False existing values must be overwritten, as in Triton.
    sparse_backward(upstream, cache, kernel='cupy', dq=dq, dk=dk, dv=dv)
    np.testing.assert_allclose(dk, expected_k, rtol=1e-6, atol=1e-6)
    np.testing.assert_allclose(dv, expected_v, rtol=1e-6, atol=1e-6)


def test_model_b_reference_sparse_gradients_reach_gqa_buffers():
    """Integration regression: local attention must retain sparse K/V grads."""
    from mini_llm.backend import RandomStream
    from mini_llm.ops.matrix_pool_dual_resolution_attention import MatrixPoolDualResolutionAttention
    config = replace(ModelConfig.dual_resolution_matrix_pool_sparse_1024_20l().dual_resolution,
                     sparse_retrieval_phase=8)
    operator = MatrixPoolDualResolutionAttention(
        d_model=1024, n_q_heads=16, n_kv_heads=4, d_head=64,
        input_std=.02, output_std=.02, rng=RandomStream(1901),
        dtype='float32', config=config,
    )
    rng = np.random.default_rng(1903)
    q = rng.normal(0, .1, size=(1, 32, 16, 64)).astype(np.float32)
    k = rng.normal(0, .1, size=(1, 32, 4, 64)).astype(np.float32)
    v = rng.normal(0, .1, size=(1, 32, 4, 64)).astype(np.float32)
    docs = np.zeros((1, 32), dtype=np.int32)
    upstream = np.zeros_like(q)
    upstream[0, 31, 1] = rng.normal(size=(64,)).astype(np.float32)
    operator.local_kernel = 'cupy'
    _, cache = operator._local_forward(q, k, v, docs, True)
    dq, dk, dv = operator._local_backward(upstream, q, k, v, cache, docs)
    _, sparse_cache = operator._sparse_forward(q, k, v, docs)
    ref_dq, ref_dk, ref_dv = sparse_backward(
        upstream[:, :, 1:2, :], sparse_cache, kernel='cupy')
    np.testing.assert_allclose(dq[:, :, 1:2, :], ref_dq, atol=1e-6)
    np.testing.assert_allclose(dk[:, 8::16, 0:1, :], ref_dk, atol=1e-6)
    np.testing.assert_allclose(dv[:, 8::16, 0:1, :], ref_dv, atol=1e-6)
    assert np.max(np.abs(dv[:, 8, 0, :])) > 0.
    assert np.max(np.abs(dv[:, 24, 0, :])) > 0.
    np.testing.assert_array_equal(dv[:, 9, 0, :], 0.)


@pytest.mark.skipif(__import__('mini_llm.backend',fromlist=['BACKEND_NAME']).BACKEND_NAME != 'cupy',
                    reason='GPU-specific Triton parity')
def test_triton_parity_and_strided_kv_gradient():
    cp = pytest.importorskip('cupy')
    pytest.importorskip('triton')
    q, k, v, docs = _sample()
    # BF16 execution uses approximate tensor-core reductions.
    qg, kg, vg = [cp.asarray(x).astype(cp.dtype('bfloat16')) for x in (q, k, v)]
    dg = cp.asarray(docs, dtype=cp.int32)
    grad = cp.asarray(np.random.default_rng(89).normal(size=q.shape), dtype=cp.float32)
    out, cache = sparse_forward(qg, kg, vg, dg, stride=16, phase=5, scale=.125,
                                kernel='triton')
    dq, dk, dv = sparse_backward(grad, cache, kernel='triton')
    # With MINI_LLM_BACKEND=cupy the reference also uses CuPy softmax;
    # passing NumPy arrays here is invalid and never exercised the kernel.
    ref_q, ref_k, ref_v = [x.astype(cp.float32) for x in (qg, kg, vg)]
    ref_o, ref_cache = sparse_forward(ref_q, ref_k, ref_v, dg,
                                      stride=16, phase=5, scale=.125, kernel='cupy')
    ref_dq, ref_dk, ref_dv = sparse_backward(grad, ref_cache, kernel='cupy')
    for actual, expected in ((out, ref_o), (dq,ref_dq), (dk,ref_dk), (dv,ref_dv)):
        np.testing.assert_allclose(cp.asnumpy(actual), cp.asnumpy(expected),
                                   rtol=.12, atol=.10)

@pytest.mark.skipif(__import__('mini_llm.backend',fromlist=['BACKEND_NAME']).BACKEND_NAME != 'numpy',
                    reason='NumPy incremental reference')
def test_two_layer_training_vs_incremental_inference(monkeypatch):
    from mini_llm.model.decoder_lm import DecoderLanguageModel
    from mini_llm.inference_model import InferenceModel
    monkeypatch.setenv('MINI_LLM_DUAL_RESOLUTION_FAST_PROJECTIONS','0')
    base = ModelConfig.dual_resolution_matrix_pool_sparse_1024_20l()
    dual = replace(base.dual_resolution, block_size=4, local_window=16,
                   long_local_window=32, max_blocks=20, global_query_chunk=16,
                   sparse_retrieval_layers=(0,1))
    config = replace(base, context_length=48, n_layers=2, d_ff=32,
                     tokenizer_vocab_size=31, dual_resolution=dual,
                     dtype='float64')
    training = DecoderLanguageModel(config, rng_seed=1012, dtype='float64')
    assert training.blocks[0].attention.sparse_phase == 0
    assert training.blocks[1].attention.sparse_phase == 8
    rng = np.random.default_rng(100)
    tokens = rng.integers(0,config.vocab_size,size=(1,config.context_length),dtype=np.int32)
    reference,_ = training.forward(tokens)
    inference = InferenceModel(config, dtype='float64')
    inference.set_weights(training)
    state = inference.create_generation_state(1, config.context_length)
    legacy_state = inference.create_generation_state(1, config.context_length)
    legacy_state.dual_host_counters = None
    for i in range(config.context_length):
        logits=inference.decode_one(tokens[:,i],state)
        legacy_logits=inference.decode_one(tokens[:,i],legacy_state)
        np.testing.assert_array_equal(logits,legacy_logits)
        np.testing.assert_allclose(logits,reference[:,i],rtol=5e-6,atol=5e-7,
                                   err_msg=f'sparse inference mismatch at token {i}')
    for name in ('dual_local_start', 'dual_local_count', 'dual_sparse_count',
                 'dual_summary_start', 'dual_summary_count',
                 'dual_incomplete_count', 'dual_document_positions'):
        np.testing.assert_array_equal(getattr(state,name),getattr(legacy_state,name))
    assert int(state.dual_sparse_count[0,0])==3
    assert int(state.dual_sparse_count[1,0])==3

@pytest.mark.skipif(__import__('mini_llm.backend',fromlist=['BACKEND_NAME']).BACKEND_NAME != 'cupy',
                    reason='GPU-only integrated attention parity')
def test_model_b_local_sparse_triton_matches_reference():
    cp = pytest.importorskip('cupy')
    pytest.importorskip('triton')
    from mini_llm.backend import RandomStream
    from mini_llm.ops.matrix_pool_dual_resolution_attention import MatrixPoolDualResolutionAttention
    config = replace(ModelConfig.dual_resolution_matrix_pool_sparse_1024_20l().dual_resolution,
                     sparse_retrieval_phase=8)
    operator = MatrixPoolDualResolutionAttention(
        d_model=1024, n_q_heads=16, n_kv_heads=4, d_head=64,
        input_std=.02, output_std=.02, rng=RandomStream(1901),
        dtype='bfloat16', config=config,
    )
    rng = np.random.default_rng(1902)
    def bf16(shape):
        return cp.asarray(rng.normal(size=shape).astype(np.float32)*.13).astype(cp.dtype('bfloat16'))
    q, k, v = bf16((1,280,16,64)),bf16((1,280,4,64)),bf16((1,280,4,64))
    docs = cp.zeros((1,280),dtype=cp.int32)
    docs[:,131:] = 1
    dout=cp.asarray(rng.normal(size=(1,280,16,64)).astype(np.float32))*.025
    operator.local_kernel='cupy'
    ref, cache_ref = operator._local_forward(q,k,v,docs,True)
    dr = operator._local_backward(dout,q,k,v,cache_ref,docs)
    operator.local_kernel='triton'
    got, cache_gpu = operator._local_forward(q,k,v,docs,True)
    dg = operator._local_backward(dout,q,k,v,cache_gpu,docs)
    # Name the failing tensor so a numerical kernel discrepancy is not
    # confused with missing GQA/sparse-gradient accumulation in the oracle.
    for name, actual, expected in (
        ('forward', got, ref),
        ('dQ', dg[0], dr[0]),
        ('dK', dg[1], dr[1]),
        ('dV', dg[2], dr[2]),
    ):
        np.testing.assert_allclose(cp.asnumpy(actual), cp.asnumpy(expected),
                                   rtol=.15, atol=.12, err_msg=name)
