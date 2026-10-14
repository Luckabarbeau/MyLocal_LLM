"""0080O5: memory-limited softmax proposal feedback, no reduced vocabulary."""
import os
import numpy as np
import pytest

from mini_llm.backend import BACKEND_NAME, xp, asnumpy
from mini_llm.parameter import Parameter
from mini_llm.predictive_refinement.soft_embeddings import stable_softmax, SoftTokenExpectation


@pytest.mark.parametrize('temperature', [0.65, 1., 1.7])
@pytest.mark.parametrize('shape', [(3, 4, 31), (7, 65280)])
def test_default_softmax_preserves_reference_and_normalization(monkeypatch, temperature, shape):
    monkeypatch.delenv('MINI_LLM_FUSED_SOFTMAX_FORWARD', raising=False)
    rng = np.random.default_rng(8105)
    logits = xp.asarray(rng.normal(0, 2, size=shape).astype(np.float32))
    result = asnumpy(stable_softmax(logits, temperature))
    z = np.asarray(asnumpy(logits), dtype=np.float32) / temperature
    z -= np.max(z, axis=-1, keepdims=True)
    p = np.exp(z)
    p /= np.sum(p, axis=-1, keepdims=True)
    np.testing.assert_allclose(result, p, rtol=1e-6, atol=2e-7)
    np.testing.assert_allclose(result.sum(axis=-1), 1., atol=2e-6)


@pytest.mark.skipif(BACKEND_NAME != 'cupy', reason='requires RTX/CuPy')
@pytest.mark.parametrize('dtype', ['float32', 'bfloat16'])
@pytest.mark.parametrize('temperature', [0.63, 1., 1.8])
def test_fused_forward_cuda_exact_vocab_matches_reference(monkeypatch, dtype, temperature):
    import ml_dtypes
    rng = np.random.default_rng(8106)
    logits = xp.asarray(rng.normal(0, 3.5, size=(3, 4, 65280)),
                        dtype=ml_dtypes.bfloat16 if dtype=='bfloat16' else xp.float32)
    monkeypatch.delenv('MINI_LLM_FUSED_SOFTMAX_FORWARD', raising=False)
    reference = asnumpy(stable_softmax(logits, temperature))
    monkeypatch.setenv('MINI_LLM_FUSED_SOFTMAX_FORWARD', '1')
    actual = asnumpy(stable_softmax(logits, temperature))
    np.testing.assert_allclose(actual, reference, rtol=4e-5, atol=1e-7)
    np.testing.assert_allclose(actual.sum(axis=-1), 1., atol=3e-6)
    assert np.isfinite(actual).all()
    # Noncontiguous inputs fall back to the original implementation.
    noncontig = logits[:, ::2, ::3]
    monkeypatch.delenv('MINI_LLM_FUSED_SOFTMAX_FORWARD', raising=False)
    a = asnumpy(stable_softmax(noncontig, temperature))
    monkeypatch.setenv('MINI_LLM_FUSED_SOFTMAX_FORWARD', '1')
    b = asnumpy(stable_softmax(noncontig, temperature))
    np.testing.assert_array_equal(a, b)


@pytest.mark.skipif(BACKEND_NAME != 'cupy', reason='requires RTX/CuPy')
@pytest.mark.parametrize('dtype', ['float32', 'bfloat16'])
def test_consume_probability_cache_inplace_after_embedding_gradient(monkeypatch, dtype):
    import ml_dtypes
    rng = np.random.default_rng(8107)
    vocab = 65280
    d = 48
    e_data = rng.normal(0, .02, size=(vocab, d)).astype(np.float32)
    logits_np = rng.normal(0, 1, size=(2, 4, vocab)).astype(np.float32)
    dz_np = rng.normal(0, 1, size=(2, 4, d)).astype(np.float32)
    tensor_dtype = ml_dtypes.bfloat16 if dtype=='bfloat16' else xp.float32
    x = xp.asarray(logits_np, dtype=tensor_dtype)
    grad = xp.asarray(dz_np, dtype=tensor_dtype)
    monkeypatch.setenv('MINI_LLM_FUSED_SOFTMAX_VJP', '1')
    monkeypatch.delenv('MINI_LLM_FUSED_SOFTMAX_FORWARD', raising=False)
    monkeypatch.delenv('MINI_LLM_INPLACE_EXPECTATION_VJP', raising=False)
    ref_w = Parameter(xp.asarray(e_data, dtype=tensor_dtype))
    ref_op = SoftTokenExpectation(ref_w)
    ref_plan, ref_cache = ref_op.forward(x, temperature=0.77)
    ref_grad = ref_op.backward(grad, ref_cache)
    monkeypatch.setenv('MINI_LLM_FUSED_SOFTMAX_FORWARD', '1')
    monkeypatch.setenv('MINI_LLM_INPLACE_EXPECTATION_VJP', '1')
    got_w = Parameter(xp.asarray(e_data, dtype=tensor_dtype))
    got_op = SoftTokenExpectation(got_w)
    got_plan, got_cache = got_op.forward(x, temperature=0.77)
    got_grad = got_op.backward(grad, got_cache)
    assert got_grad is got_cache.probs  # no new full-vocab FP32 VJP output
    np.testing.assert_allclose(asnumpy(got_plan), asnumpy(ref_plan), rtol=4e-3, atol=3e-4)
    np.testing.assert_allclose(asnumpy(got_grad), asnumpy(ref_grad), rtol=4e-3, atol=1e-6)
    np.testing.assert_allclose(asnumpy(got_w.grad), asnumpy(ref_w.grad), rtol=4e-3, atol=2e-5)
    assert np.isfinite(asnumpy(got_grad)).all()

@pytest.mark.skipif(BACKEND_NAME != 'cupy', reason='requires RTX/CuPy')
def test_joint_decoder_gpu_parity_fused_forward_and_consumed_vjp(monkeypatch):
    """Ensure residual-prior, E, horizon and decoder gradients all survive."""
    import ml_dtypes
    from mini_llm.config import ModelConfig
    from mini_llm.model.decoder_lm import DecoderLanguageModel
    from mini_llm.block_proposal.proposals import HorizonProposalHead
    from mini_llm.block_proposal.decoder import BlockPrefixDecoder, TwoStageBlockModel

    def make():
        cfg = ModelConfig(tokenizer_vocab_size=64, context_length=16,
            n_layers=1, d_model=32, n_q_heads=4, n_kv_heads=2,
            d_head=8, d_ff=64, n_experts=2, top_k=1, dtype='bfloat16')
        main = DecoderLanguageModel(cfg, rng_seed=11, dtype='bfloat16')
        head = HorizonProposalHead(main, 4, hidden_width=32, rng_seed=80)
        dec = BlockPrefixDecoder(main, 4, width=32, n_heads=4, n_layers=2,
                                 rng_seed=81)
        return TwoStageBlockModel(head, dec, lambda_proposal=.5)

    hidden = xp.asarray(np.random.default_rng(81).normal(size=(3, 32)),
                        dtype=ml_dtypes.bfloat16)
    labels = xp.asarray([[1, 2, 3, 4], [3, 5, 7, 9], [4, 5, 0, 0]])
    mask = xp.asarray([[1]*4, [1]*4, [1, 1, 0, 0]], dtype=xp.bool_)
    outputs = []
    for optimized in (False, True):
        monkeypatch.setenv('MINI_LLM_FUSED_SOFTMAX_VJP', '1')
        monkeypatch.setenv('MINI_LLM_FUSED_SOFTMAX_FORWARD', str(int(optimized)))
        monkeypatch.setenv('MINI_LLM_INPLACE_EXPECTATION_VJP', str(int(optimized)))
        model = make()
        model.zero_grad()
        metrics, tape = model.forward(hidden, labels, mask)
        dx = model.backward(tape)
        names = ('embedding.W', 'output_proj.W', 'block_decoder.plan.W',
                 'block_decoder.correction.W', 'block_proposal.horizon_back')
        grads = {p.name: asnumpy(p.grad).copy() for p in model.parameters()
                 if p.name in names}
        outputs.append((metrics['total_loss'], asnumpy(dx).copy(), grads))
    loss_0, dx_0, grads_0 = outputs[0]
    loss_1, dx_1, grads_1 = outputs[1]
    np.testing.assert_allclose(loss_1, loss_0, rtol=1e-3, atol=2e-3)
    np.testing.assert_allclose(dx_1, dx_0, rtol=3e-2, atol=1e-4)
    assert grads_0.keys() == grads_1.keys()
    assert grads_0
    for name in grads_0:
        np.testing.assert_allclose(grads_1[name], grads_0[name],
                                   rtol=3e-2, atol=1e-4, err_msg=name)
