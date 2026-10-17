"""0080AG: isolated teacher-forced full-soft read; no CUDA/v2/generation claim."""

import numpy as np
import pytest

from mini_llm.backend import BACKEND_NAME
from mini_llm.block_proposal.checkpoint import architecture_metadata
from mini_llm.block_proposal.decoder import BlockPrefixDecoder, TwoStageBlockModel
from mini_llm.block_proposal.proposals import HorizonProposalHead
from mini_llm.block_proposal.soft_interface_config import SoftInterfaceConfig
from mini_llm.block_proposal.training import SinglePassBlockTrainer
from mini_llm.model.decoder_lm import DecoderLanguageModel
from mini_llm.optim.adamw import AdamW
from test_block_proposal_0080l_baselines import _configuration, _window

pytestmark = pytest.mark.skipif(BACKEND_NAME != 'numpy', reason='CPU mathematical integration')


def _build(k=4, layers=2, read_layer=0, read_scale=1.0):
    main = DecoderLanguageModel(_configuration(), rng_seed=8080, dtype='float64')
    baseline = BlockPrefixDecoder(main, k, width=12, n_layers=layers, n_heads=3,
                                  rng_seed=8082)
    enhanced = BlockPrefixDecoder(
        main, k, width=12, n_layers=layers, n_heads=3, rng_seed=8082,
        soft_interface_config=SoftInterfaceConfig(
            mode='full_vocab', read_heads=4, read_layer=read_layer,
            read_scale=read_scale), experimental_cpu_full_vocab=True)
    return main, baseline, enhanced


def _inputs(k=4):
    gen = np.random.default_rng(801)
    logits = gen.normal(size=(2, k, 101)).astype(np.float64)
    targets = gen.integers(0, 101, size=(2, k), dtype=np.int64)
    valid = np.ones((2, k), dtype=bool)
    return logits, targets, valid


def _activate(decoder):
    gen = np.random.default_rng(8091)
    w = decoder.vocab_read.output_projection.W
    w.data[...] = gen.normal(0.0, .035, size=w.shape)


def _finite_difference(parameter, index, f, step=1e-5):
    initial = float(parameter[index])
    try:
        parameter[index] = initial + step
        plus = float(f())
        parameter[index] = initial - step
        minus = float(f())
    finally:
        parameter[index] = initial
    return (plus - minus) / (2 * step)


def test_zero_initialization_parity_original_parameter_order_and_only_output_grad():
    main, baseline, dec = _build()
    original = baseline.parameters()
    upgraded = dec.parameters()
    assert [p.name for p in upgraded[:len(original)]] == [p.name for p in original]
    for a, b in zip(original, upgraded):
        np.testing.assert_array_equal(a.data, b.data)
    assert len({id(p) for p in upgraded}) == len(upgraded)
    assert len({p.name for p in upgraded}) == len(upgraded)
    assert all(np.count_nonzero(p.data) == 0 for p in dec.vocab_read.output_projection.parameters())
    logits, targets, valid = _inputs()
    ref, _ = baseline.forward(logits, targets, valid)
    got, tape = dec.forward(logits, targets, valid)
    np.testing.assert_array_equal(ref, got)
    dec.zero_grad()
    main.embedding.W.zero_grad()
    g = np.random.default_rng(21).normal(size=got.shape)
    dec.backward(g, tape)
    assert np.any(dec.vocab_read.output_projection.W.grad != 0)
    np.testing.assert_array_equal(dec.vocab_read.query_projection.W.grad, 0)


def test_integrated_backward_matches_finite_differences_for_all_paths():
    main, _, dec = _build()
    _activate(dec)
    logits, targets, valid = _inputs()
    upstream = np.random.default_rng(918).normal(0.0, 0.1, size=logits.shape)
    dec.zero_grad()
    main.embedding.W.zero_grad()
    main.output_proj.W.zero_grad()
    fwd, tape = dec.forward(logits, targets, valid)
    analytic_logits = dec.backward(upstream, tape)

    def objective():
        f, _ = dec.forward(logits, targets, valid)
        return np.sum(f * upstream)

    # Includes direct prior, legacy P@E, and dynamic read contribution.
    for index in ((0, 0, 3), (1, 2, 41), (1, 3, 5)):
        actual = _finite_difference(logits, index, objective)
        np.testing.assert_allclose(analytic_logits[index], actual, atol=2e-8, rtol=2e-5)
    for par, inds in (
        (dec.vocab_read.query_projection.W, ((0, 0), (9, 12))),
        (dec.vocab_read.output_projection.W, ((0, 0), (14, 7))),
        (dec.layers[0].norm1.gamma, ((0,),)),
        (main.embedding.W, ((3, 0), (29, 11))),
    ):
        for index in inds:
            numeric = _finite_difference(par.data, index, objective)
            np.testing.assert_allclose(par.grad[index], numeric, atol=5e-8, rtol=5e-4,
                                       err_msg=f'{par.name} {index}')
    assert np.max(np.abs(dec.vocab_read.query_projection.W.grad)) > 0
    assert np.max(np.abs(main.embedding.W.grad)) > 0


@pytest.mark.parametrize('read_layer', [0, 1])
def test_future_true_labels_do_not_change_current_or_past_predictions(read_layer):
    _, _, dec = _build(read_layer=read_layer)
    _activate(dec)
    logits, targets, valid = _inputs()
    reference, _ = dec.forward(logits, targets, valid)
    for changed_at in range(4):
        altered = targets.copy()
        altered[:, changed_at] = (altered[:, changed_at] + 17) % 101
        changed, _ = dec.forward(logits, altered, valid)
        # y_j becomes available only at DEC_(j+1), never at DEC_<=j.
        np.testing.assert_allclose(reference[:, :changed_at+1],
                                   changed[:, :changed_at+1], atol=0, rtol=0)


def test_prefix_masking_and_read_scale_zero_well_defined():
    main, _, dec = _build(read_scale=0.0)
    _activate(dec)
    logits, targets, valid = _inputs()
    out_normal, tape = dec.forward(logits, targets, valid)
    out_masked, _ = dec.forward(logits, targets, valid, prefix_condition='masked')
    assert out_masked.shape == out_normal.shape
    dec.zero_grad()
    main.embedding.W.zero_grad()
    dec.backward(np.ones_like(out_normal), tape)
    np.testing.assert_array_equal(dec.vocab_read.query_projection.W.grad, 0)
    assert np.any(dec.vocab_read.output_projection.W.grad)


def test_one_shared_body_backward_with_full_vocabulary_read():
    main = DecoderLanguageModel(_configuration(), rng_seed=8080, dtype='float64')
    head = HorizonProposalHead(main, 8, hidden_width=12, rng_seed=8081)
    dec = BlockPrefixDecoder(main, 8, width=12, n_layers=2, n_heads=3, rng_seed=8082,
                             soft_interface_config=SoftInterfaceConfig(mode='full_vocab'),
                             experimental_cpu_full_vocab=True)
    _activate(dec)
    net = TwoStageBlockModel(head, dec, lambda_proposal=.5)
    trainer = SinglePassBlockTrainer(net)
    source, targets, valid, metadata = _window()
    metrics = trainer.train_window(source, targets, valid_target_mask=valid,
                                   memory_metadata=metadata, block_group_size=2)
    assert metrics['main_forward_passes'] == 1
    assert metrics['main_backward_passes'] == 1
    assert metrics['unique_supervised_tokens'] == 13
    assert int(np.sum(metrics['position_counts'])) == 13
    assert np.all(np.isfinite(metrics['final_position_ce_sums']))
    assert np.max(np.abs(dec.vocab_read.output_projection.W.grad)) > 0
    assert np.max(np.abs(dec.vocab_read.query_projection.W.grad)) > 0
    assert len({id(p) for p in net.parameters()}) == len(net.parameters())
    assert len({p.name for p in net.parameters()}) == len(net.parameters())
    optimizer = AdamW(net.parameters(), lr=1e-3, weight_decay=.1)
    previous = {p.name: p.data.copy() for p in dec.vocab_read.parameters()}
    optimizer.step()
    assert optimizer.step_index == 1
    for p in dec.vocab_read.parameters():
        assert np.max(np.abs(previous[p.name] - p.data)) > 0, p.name


def test_non_v1_generation_and_v2_checkpoint_metadata():
    main, _, dec = _build()
    logits, _, _ = _inputs()
    plan=dec.prepare_plan_cache(logits)
    assert dec.cached_step(plan,None).shape==(logits.shape[0],logits.shape[-1])
    net = TwoStageBlockModel(HorizonProposalHead(main, 4, hidden_width=12), dec)
    from mini_llm.block_proposal.checkpoint import ARCHITECTURE_V2
    assert architecture_metadata(net)['type']==ARCHITECTURE_V2


def test_boundary_condition_is_independently_available():
    main = DecoderLanguageModel(_configuration(), rng_seed=8080, dtype='float64')
    decoder = BlockPrefixDecoder(main, 4, width=12, n_layers=2, n_heads=3,
        soft_interface_config=SoftInterfaceConfig(mode='expectation',boundary_condition='dec_residual'),
        experimental_cpu_full_vocab=True)
    assert decoder.boundary_projection is not None
    assert decoder.vocab_read is None
