"""0080O6: owned final FP32 logits may be consumed by CE, not proposals."""
import numpy as np
import pytest

from mini_llm.backend import BACKEND_NAME, asnumpy, xp
from mini_llm.ops.loss import cross_entropy_forward, cross_entropy_backward


@pytest.mark.parametrize('shape', [(2, 4, 17), (3, 4, 65280)])
@pytest.mark.parametrize('use_mask', [False, True])
def test_fp32_ce_consume_matches_reference_loss_rows_and_grad(shape, use_mask):
    gen = np.random.default_rng(80806)
    logits_np = gen.normal(loc=-.1, scale=1.8, size=shape).astype('float32')
    targets_np = gen.integers(0, shape[-1], size=shape[:-1], dtype=np.int32)
    mask_np = gen.random(shape[:-1]) > .3 if use_mask else None
    if mask_np is not None:
        mask_np.reshape(-1)[0] = True
    mask = xp.asarray(mask_np) if mask_np is not None else None
    x1 = xp.asarray(logits_np.copy())
    x2 = xp.asarray(logits_np.copy())
    y = xp.asarray(targets_np)

    normalizer = float(mask_np.sum()) if mask_np is not None else float(y.size)
    loss1, c1 = cross_entropy_forward(x1, y, loss_mask=mask,
        return_row_losses=True, normalizer_count=normalizer)
    loss2, c2 = cross_entropy_forward(x2, y, loss_mask=mask,
        return_row_losses=True, normalizer_count=normalizer,
        reuse_fp32_logits=True)
    assert bool(c2['consume_fp32_probs'])
    assert c2['probs'] is x2.reshape(-1, shape[-1]) or xp.shares_memory(c2['probs'], x2)
    np.testing.assert_allclose(loss2, loss1, rtol=8e-6, atol=3e-6)
    # Recovered stable per-token NLL must agree with the reference CE.
    reference_p = asnumpy(c1['probs']).reshape(-1, shape[-1])
    np_target = targets_np.reshape(-1)
    nll = -np.log(reference_p[np.arange(len(np_target)), np_target] + 1e-30)
    if mask_np is not None:
        nll = nll * mask_np.reshape(-1)
    np.testing.assert_allclose(asnumpy(c2['row_losses']), nll, atol=2e-5, rtol=1e-5)
    np.testing.assert_array_equal(asnumpy(x1), logits_np)  # default not mutated
    assert not np.array_equal(asnumpy(x2), logits_np)
    g1 = asnumpy(cross_entropy_backward(c1))
    g2 = asnumpy(cross_entropy_backward(c2))
    np.testing.assert_allclose(g2, g1, atol=2e-8, rtol=5e-5)
    assert 'probs' not in c2  # ownership consumed


def test_non_owned_fp32_view_does_not_mutate():
    rng = np.random.default_rng(86)
    arr = xp.asarray(rng.normal(size=(4, 1, 11)).astype('float32'))
    view = arr[::2]
    before = asnumpy(arr).copy()
    loss, cache = cross_entropy_forward(view, xp.zeros((2, 1), dtype='int32'),
                                        reuse_fp32_logits=True)
    assert 'consume_fp32_probs' not in cache
    np.testing.assert_array_equal(asnumpy(arr), before)
    assert np.isfinite(loss)


def test_float64_unchanged_if_reuse_requested():
    a = xp.zeros((2, 4, 7), dtype='float64')
    loss, c = cross_entropy_forward(a, xp.zeros((2, 4), dtype='int32'),
                                    reuse_fp32_logits=True)
    assert 'consume_fp32_probs' not in c
    assert np.isfinite(loss)


@pytest.mark.parametrize('consume', [False, True])
def test_joint_full_decoder_final_ce_no_proposal_corruption(monkeypatch, consume):
    """Check the actual two-stage main->proposal and decoder gradient path."""
    from mini_llm.config import ModelConfig
    from mini_llm.model.decoder_lm import DecoderLanguageModel
    from mini_llm.block_proposal.proposals import HorizonProposalHead
    from mini_llm.block_proposal.decoder import BlockPrefixDecoder, TwoStageBlockModel

    def make():
        cfg = ModelConfig(tokenizer_vocab_size=31, context_length=16,
            n_layers=1, d_model=32, n_q_heads=4, n_kv_heads=2,
            d_head=8, d_ff=64, n_experts=2, top_k=1, dtype='float32')
        main = DecoderLanguageModel(cfg, rng_seed=11, dtype='float32')
        head = HorizonProposalHead(main, 4, hidden_width=32, rng_seed=80)
        dec = BlockPrefixDecoder(main, 4, width=32, n_heads=4,
                                 n_layers=2, rng_seed=81)
        return TwoStageBlockModel(head, dec, lambda_proposal=.5)
    hidden = xp.asarray(np.random.default_rng(81).normal(size=(3, 32)).astype('float32'))
    labels = xp.asarray([[1, 2, 3, 4], [3, 5, 7, 9], [4, 5, 0, 0]])
    mask = xp.asarray([[1]*4, [1]*4, [1, 1, 0, 0]], dtype=xp.bool_)

    results = []
    for flag in ['0', '1']:
        monkeypatch.setenv('MINI_LLM_INPLACE_FP32_FINAL_CE', flag)
        model = make()
        model.zero_grad()
        metric, tape = model.forward(hidden, labels, mask)
        assert bool(tape.final_ce_cache.get('consume_fp32_probs')) is (flag == '1')
        assert 'consume_fp32_probs' not in tape.proposal_ce_cache
        dx = model.backward(tape)
        keys = ('embedding.W', 'output_proj.W', 'block_decoder.plan.W',
                'block_decoder.correction.W', 'block_proposal.horizon_back')
        params = {p.name: asnumpy(p.grad).copy() for p in model.parameters()
                  if p.name in keys}
        results.append((metric['total_loss'], asnumpy(dx), params))
    (l0,d0,p0), (l1,d1,p1) = results
    np.testing.assert_allclose(l1, l0, atol=5e-6, rtol=1e-6)
    np.testing.assert_allclose(d1, d0, atol=5e-5, rtol=1e-5)
    assert p0.keys() == p1.keys() and p0
    for name in p0:
        np.testing.assert_allclose(p1[name], p0[name], atol=5e-5, rtol=1e-5, err_msg=name)
    assert np.isfinite(l1)
