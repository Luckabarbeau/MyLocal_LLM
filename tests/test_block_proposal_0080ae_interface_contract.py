"""0080AE: no-op configuration and hard gates for unimplemented interfaces."""

from dataclasses import replace
from types import SimpleNamespace

import numpy as np
import pytest

from mini_llm.backend import BACKEND_NAME, asnumpy
from mini_llm.block_proposal.checkpoint import ARCHITECTURE, architecture_metadata
from mini_llm.block_proposal.decoder import BlockPrefixDecoder
from mini_llm.block_proposal.soft_interface_config import SoftInterfaceConfig
from test_block_proposal_0080l_baselines import _configuration, _make, _window
from mini_llm.model.decoder_lm import DecoderLanguageModel
from train_model import build_training_objective

pytestmark = pytest.mark.skipif(BACKEND_NAME != 'numpy', reason='fixed NumPy reference')


def _explicit_baseline():
    main = DecoderLanguageModel(_configuration(), rng_seed=8080, dtype='float64')
    from mini_llm.block_proposal.proposals import HorizonProposalHead
    from mini_llm.block_proposal.decoder import TwoStageBlockModel
    from mini_llm.block_proposal.training import SinglePassBlockTrainer
    net = TwoStageBlockModel(
        HorizonProposalHead(main, 8, hidden_width=12, rng_seed=8081),
        BlockPrefixDecoder(main, 8, width=12, n_layers=1, n_heads=3,
                           rng_seed=8082, soft_interface_config=SoftInterfaceConfig()),
        lambda_proposal=0.5,
    )
    return SinglePassBlockTrainer(net)


def test_explicit_default_is_bitwise_identical_after_forward_backward():
    implicit = _make()
    explicit = _explicit_baseline()
    a = implicit.net.parameters()
    b = explicit.net.parameters()
    assert [p.name for p in a] == [p.name for p in b]
    assert len(a) == len({id(p) for p in a}) == len({p.name for p in a})
    for p, q in zip(a, b):
        np.testing.assert_array_equal(asnumpy(p.data), asnumpy(q.data))
    # Pin exact logits (not just the same CE), including the independent
    # autoregressive cached-generation implementation and its PLAN prefill.
    boundary = np.linspace(-0.2, 0.3, 32, dtype=np.float64).reshape(2, 16)
    proposal_a = implicit.net.proposal_head.forward(boundary, return_cache=False)
    proposal_b = explicit.net.proposal_head.forward(boundary, return_cache=False)
    np.testing.assert_array_equal(asnumpy(proposal_a), asnumpy(proposal_b))
    forced = np.mod(np.arange(16).reshape(2, 8) * 7 + 3, 101)
    fully_valid = np.ones((2, 8), dtype=bool)
    final_a, _ = implicit.net.decoder.forward(proposal_a, forced, fully_valid)
    final_b, _ = explicit.net.decoder.forward(proposal_b, forced, fully_valid)
    np.testing.assert_array_equal(asnumpy(final_a), asnumpy(final_b))
    cache_a = implicit.net.decoder.prepare_plan_cache(proposal_a)
    cache_b = explicit.net.decoder.prepare_plan_cache(proposal_b)
    for j in range(8):
        previous = None if j == 0 else forced[:, j-1]
        np.testing.assert_array_equal(
            asnumpy(implicit.net.decoder.cached_step(cache_a, previous)),
            asnumpy(explicit.net.decoder.cached_step(cache_b, previous)),
        )

    source, labels, valid, metadata = _window()
    x = implicit.train_window(source, labels, valid_target_mask=valid,
                              memory_metadata=metadata, block_group_size=2)
    y = explicit.train_window(source, labels, valid_target_mask=valid,
                              memory_metadata=metadata, block_group_size=2)
    for name in ('final_ce', 'proposal_ce', 'total_loss'):
        assert x[name] == y[name], name
    for name in ('final_position_ce_sums', 'proposal_position_ce_sums',
                 'position_counts', 'decoder_position_gain'):
        np.testing.assert_array_equal(asnumpy(x[name]), asnumpy(y[name]))
    for p, q in zip(a, b):
        np.testing.assert_array_equal(asnumpy(p.grad), asnumpy(q.grad), err_msg=p.name)
    assert architecture_metadata(implicit.net) == architecture_metadata(explicit.net)
    assert architecture_metadata(explicit.net)['type'] == ARCHITECTURE
    assert 'soft_interface' not in architecture_metadata(explicit.net)


@pytest.mark.parametrize('overrides, error, message', [
    ({'mode': 'full_vocab'}, NotImplementedError, 'scaffolding only'),
    ({'boundary_condition': 'dec_residual'}, NotImplementedError, 'scaffolding only'),
    ({'mode': 'full_vocab', 'boundary_condition': 'dec_residual'}, NotImplementedError, 'scaffolding only'),
    ({'mode': 'invalid'}, ValueError, 'soft interface'),
    ({'source_scope': 'all'}, ValueError, 'require full_vocab'),
    ({'read_heads': 0}, ValueError, 'positive integer'),
    ({'read_layer': 99}, ValueError, 'existing decoder layer'),
    ({'temperature': 0}, ValueError, 'temperature'),
    ({'temperature': float('nan')}, ValueError, 'temperature'),
    ({'read_scale': -0.1}, ValueError, 'query scale'),
    ({'read_scale': float('inf')}, ValueError, 'query scale'),
    ({'mode': 'full_vocab', 'read_heads': 5}, ValueError, 'divisible'),
])
def test_invalid_and_future_interfaces_fail_before_allocation(overrides, error, message):
    interface = replace(SoftInterfaceConfig(), **overrides)
    with pytest.raises(error, match=message):
        interface.validate(model_width=16, decoder_layers=1, block_size=4).require_implemented()


def test_direct_decoder_constructor_refuses_future_modes():
    main = DecoderLanguageModel(_configuration(), rng_seed=8080, dtype='float64')
    for mode, condition in [('full_vocab', 'off'), ('expectation', 'dec_residual')]:
        with pytest.raises(NotImplementedError, match='0080AE scaffolding'):
            BlockPrefixDecoder(main, 8, width=12, n_layers=1, n_heads=3,
                               soft_interface_config=SoftInterfaceConfig(
                                   mode=mode, boundary_condition=condition))


def test_cli_factory_refuses_future_modes_and_v1_mismatched_resume():
    main = DecoderLanguageModel(_configuration(), rng_seed=8080, dtype='float64')
    args = SimpleNamespace(objective='block-proposal', block_proposal_group_size=2,
                           block_proposal_k=8, block_proposal_decoder_width=12,
                           block_proposal_decoder_layers=1, block_proposal_decoder_heads=3)
    baseline = build_training_objective(main, args)
    assert baseline.net.decoder.soft_interface_config == SoftInterfaceConfig()
    assert baseline.checkpoint_metadata()['type'] == ARCHITECTURE
    args.block_proposal_soft_interface = 'full_vocab'
    from mini_llm.block_proposal.checkpoint import ARCHITECTURE_V2
    assert build_training_objective(main,args).checkpoint_metadata()['type']==ARCHITECTURE_V2
    args.block_proposal_soft_interface = 'expectation'
    args.block_proposal_boundary_condition = 'dec_residual'
    from mini_llm.block_proposal.checkpoint import ARCHITECTURE_V2
    assert build_training_objective(main,args).checkpoint_metadata()['type']==ARCHITECTURE_V2
    args.block_proposal_boundary_condition = 'off'
    resume = {'architecture': baseline.checkpoint_metadata(),
              'prediction_objective': 'block-proposal'}
    restored = build_training_objective(main, args, checkpoint_state=resume)
    assert restored.checkpoint_metadata() == baseline.checkpoint_metadata()
    args.block_proposal_soft_interface = 'full_vocab'
    with pytest.raises(ValueError, match='resume mismatch'):
        build_training_objective(main, args, checkpoint_state=resume)


def test_ar_rejects_experimental_flags_not_just_old_flags():
    main = DecoderLanguageModel(_configuration(), rng_seed=8080, dtype='float64')
    args = SimpleNamespace(objective='ar', block_proposal_soft_interface='expectation')
    with pytest.raises(ValueError, match='require --objective block-proposal'):
        build_training_objective(main, args)
