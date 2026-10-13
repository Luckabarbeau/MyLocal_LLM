"""Q3: learned per-layer retrieval into the ONE shared routed causal context."""
import numpy as np
import pytest

from mini_llm.backend import xp, asnumpy
from mini_llm.predictive_refinement.routed_retrieval import (
    build_group_retrieval_tasks, accumulate_retrieval_bias_gradient,
)
from tests.test_predictive_refinement_0079q2 import make_trainer, _source


def evaluate(trainer, group_size, *, batch=1):
    trainer.zero_grad()
    m = trainer.train_routed_window(
        _source(batch), group_size=group_size, compact_history=True,
        compact_active_blocks=True, resident_history=True,
        max_history_keys=32, max_window_tokens=16, max_active_queries=32,
        memory_metadata={'use_retrieval': True, 'router_stochastic': False,
                         'history_valid_starts': xp.zeros((batch,), dtype=xp.int64)},
    )
    grads = [(p.name, asnumpy(p.grad).copy()) for p in trainer.parameters()]
    return m, grads


@pytest.mark.parametrize('k', [4, 8])
@pytest.mark.parametrize('batch', [1, 2])
def test_learned_heads_k8_and_gradients_independent_of_grouping(k, batch):
    trainer = make_trainer(k=k, learned=True)
    a, ga = evaluate(trainer, 1, batch=batch)
    b, gb = evaluate(trainer, 2, batch=batch)
    for key in ('total_loss', 'commitment_ce', 'initial_ce'):
        np.testing.assert_allclose(float(a[key]), float(b[key]), rtol=2e-9, atol=2e-9)
    assert a['refinement_passes'] == (k + 1) * (16//k)
    assert b['refinement_passes'] == (k + 1) * ((16//k + 1)//2)
    for (name_a, grad_a), (name_b, grad_b) in zip(ga, gb):
        assert name_a == name_b
        np.testing.assert_allclose(grad_a, grad_b, rtol=2e-7, atol=2e-8, err_msg=name_a)
    assert any('retrieval' in n and np.linalg.norm(g) > 1e-9 for n, g in gb)


def test_learned_route_requires_resident_history_and_pre_window_metadata():
    trainer = make_trainer(k=8, learned=True)
    opts = dict(group_size=1, compact_history=True,
                compact_active_blocks=True, max_history_keys=32,
                max_window_tokens=16)
    with pytest.raises(ValueError, match='resident_history'):
        trainer.train_routed_window(_source(1), resident_history=False, **opts)
    with pytest.raises(ValueError, match='history_valid_starts'):
        trainer.train_routed_window(_source(1), resident_history=True, **opts)


def test_early_retrieval_route_is_unaffected_by_future_target_tokens():
    trainer = make_trainer(k=8, learned=True)
    source = _source(1)
    later = source.copy()
    later[:, -6:] = (later[:, -6:] + 11) % 99
    meta = dict(use_retrieval=True, router_stochastic=False,
                causal_prefix_query=True, history_valid_starts=xp.zeros((1,),dtype=xp.int64))
    inputs = []
    for tokens in (source, later):
        _, c = trainer.model.forward_body(tokens, memory_metadata=meta, return_cache=True)
        full = tuple((a['attn_cache']['k'],a['attn_cache']['v']) for a in c['block_caches'])
        route = build_group_retrieval_tasks(
            trainer.model,c,full,None,xp.asarray([0]),
            xp.asarray([int(trainer.model.config.memory_context.distant_memory_length)]),
            8,xp.ones((1,8),dtype=xp.bool_), compute_gradients=False)
        inputs.append(route)
    for a,b in zip(inputs[0],inputs[1]):
        for first, second in zip(a,b):
            np.testing.assert_array_equal(asnumpy(first['plan'].key_indices),
                                          asnumpy(second['plan'].key_indices))
            np.testing.assert_allclose(asnumpy(first['route_weights']),
                                       asnumpy(second['route_weights']),rtol=0,atol=0)
            assert np.all(asnumpy(first['plan'].key_indices[first['plan'].valid_mask]) <
                          8)  # routed-prefix slots in this small test


def test_routing_prior_gradient_matches_scalar_finite_difference():
    """Checks log(w) prior derivative and router head sharing/scatter."""
    from mini_llm.ops.attention_selection import KeySelectionPlan
    weights=np.array([[[[0.4,0.6]]]],dtype='float64')
    other=np.array([[[[0.7,0.3]]]],dtype='float64')
    # two active heads share one router weight vector
    dbias=xp.asarray(np.array([[[[1.,2.,3.,4.]],[[5.,6.,7.,8.]]]],dtype='float64'))
    extra=xp.zeros((1,2,1,2),dtype=xp.float64)
    task=dict(extra_dweights=extra,router_heads=1,block_size=2,
              route_weights=xp.asarray(weights[0]),route_valid=xp.asarray([True]),
              weight_scale=1.2,weight_eps=1e-8,
              doc_rows=xp.asarray([0]),route_slots=xp.asarray([1]))
    accumulate_retrieval_bias_gradient(task,dbias)
    expected=np.array([1.2*(1+2+5+6)/(0.4+1e-8),
                       1.2*(3+4+7+8)/(0.6+1e-8)])
    np.testing.assert_allclose(asnumpy(extra[0,1,0]),expected,rtol=1e-12)


def test_learned_router_weight_gradient_matches_finite_difference():
    """A full nine-pass K=8 run must differentiate through reused route weights."""
    trainer = make_trainer(k=8, learned=True)
    ids = _source(1)
    opts = dict(group_size=2, compact_history=True,
                compact_active_blocks=True, resident_history=True,
                max_history_keys=32, max_window_tokens=16,
                max_active_queries=32,
                memory_metadata=dict(use_retrieval=True, router_stochastic=False,
                    history_valid_starts=xp.zeros((1,), dtype=xp.int64)))
    trainer.zero_grad()
    trainer.train_routed_window(ids, **opts)
    parameter = trainer.model.blocks[0].attention._retrieval_groups[0]['module'].router.W_query
    index = (0, 0)
    analytic = float(parameter.grad[index])
    original = float(parameter.data[index])
    epsilon = 1e-2  # CE implementation includes finite-precision projections
    parameter.data[index] = original + epsilon
    plus = float(trainer.train_routed_window(ids, compute_gradients=False, **opts)['total_loss'])
    parameter.data[index] = original - epsilon
    minus = float(trainer.train_routed_window(ids, compute_gradients=False, **opts)['total_loss'])
    parameter.data[index] = original
    numerical = (plus - minus) / (2 * epsilon)
    assert abs(analytic) > 1e-7
    np.testing.assert_allclose(analytic, numerical, rtol=0.03, atol=1e-8)


def test_bf16_cuda_learned_retrieval_group_parity():
    from mini_llm.backend import BACKEND_NAME
    from dataclasses import replace
    from mini_llm.model.decoder_lm import DecoderLanguageModel
    from mini_llm.predictive_refinement.training_reference import RefinementReferenceConfig
    from mini_llm.predictive_refinement.window_training import WindowRefinementTrainer
    if BACKEND_NAME != 'cupy':
        pytest.skip('requires RTX/CuPy to execute Q3 BF16 shared scatter')
    baseline = make_trainer(k=8, learned=True)
    cfg = replace(baseline.model.config, dtype='bfloat16')
    model = DecoderLanguageModel(cfg, rng_seed=98, dtype='bfloat16')
    trainer = WindowRefinementTrainer(model, RefinementReferenceConfig(
        block_size=8, enable_shared_context_training=True,
        compact_recurrent_tape=True), rng_seed=95)
    a, ga = evaluate(trainer, 1)
    b, gb = evaluate(trainer, 2)
    np.testing.assert_allclose(float(a['total_loss']),float(b['total_loss']),rtol=0.015)
    for (n1, g1), (n2, g2) in zip(ga,gb):
        assert n1 == n2
        np.testing.assert_allclose(g1,g2,atol=0.035,rtol=0.09,err_msg=n1)


def test_smoke_supervision_limit_does_not_shorten_causal_encoder():
    trainer = make_trainer(k=8, learned=True)
    source = _source(1)
    trainer.zero_grad()
    out = trainer.train_routed_window(
        source, group_size=1, compact_history=True, compact_active_blocks=True,
        resident_history=True, max_history_keys=32, max_window_tokens=16,
        max_active_queries=8, max_supervised_blocks=1,
        memory_metadata=dict(use_retrieval=True,router_stochastic=False,
            history_valid_starts=xp.zeros((1,),dtype=xp.int64)),
    )
    assert out['supervised_tokens'] == 8
    assert out['refinement_batches'] == 1
    assert out['refinement_passes'] == 9
    assert out['routed_prefix_slots'] == 8
