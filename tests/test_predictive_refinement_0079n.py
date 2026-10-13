"""0079N: one full causal encoding, compact per-boundary K/V, exact BPTT."""
from dataclasses import replace

import numpy as np
import pytest

from mini_llm.backend import asnumpy, xp
from mini_llm.config import (ModelConfig, AttentionLayerConfig,
                             LocalAttentionConfig, DilatedAttentionConfig,
                             GlobalSparseAttentionConfig)
from mini_llm.model.decoder_lm import DecoderLanguageModel
from mini_llm.predictive_refinement.compact_history import pack_group_history
from mini_llm.predictive_refinement.compact_sparse import grouped_provisional_plan
from mini_llm.predictive_refinement.training_reference import RefinementReferenceConfig
from mini_llm.predictive_refinement.window_training import WindowRefinementTrainer


def trainer(sparse, context_length, layers=1):
    cfg = ModelConfig(tokenizer_vocab_size=32, context_length=context_length,
                      n_layers=layers, d_model=16, n_q_heads=2, n_kv_heads=1,
                      d_head=8, n_experts=2, top_k=1, d_ff=32, dtype='float64')
    if sparse:
        heads = (LocalAttentionConfig(window=8),
                 DilatedAttentionConfig(window=80, dilation=4))
        attention = AttentionLayerConfig(heads=heads)
        cfg = replace(cfg, attention_layers=tuple(attention for _ in range(layers)))
    return WindowRefinementTrainer(
        DecoderLanguageModel(cfg, rng_seed=793, dtype='float64'),
        RefinementReferenceConfig(block_size=2, enable_shared_context_training=True),
        rng_seed=794)


def metrics_and_grads(t, ids, valid, *, compact_history):
    t.zero_grad()
    metrics = t.train_window(ids, prefix_length=260, valid_token_mask=valid,
                             group_size=3, max_window_tokens=320,
                             compact_active_blocks=True,
                             compact_history=compact_history,
                             record_row_ce=True, compute_gradients=True)
    grads = [asnumpy(p.grad).copy() for p in t.parameters()]
    return metrics, grads


@pytest.mark.parametrize('sparse', [False, True])
def test_extended_context_exact_train_gradients(sparse):
    t = trainer(sparse, 320)
    tokens = (np.arange(2 * 268).reshape(2, 268) % 29 + 2).astype(np.int64)
    valid = np.arange(268)[None, :] < np.asarray([268, 265])[:, None]
    tokens[~valid] = -100
    ids, v = xp.asarray(tokens), xp.asarray(valid)
    a, ga = metrics_and_grads(t, ids, v, compact_history=False)
    b, gb = metrics_and_grads(t, ids, v, compact_history=True)
    for name in ('total_loss', 'initial_ce', 'commitment_ce'):
        np.testing.assert_allclose(float(a[name]), float(b[name]), atol=2e-10)
    for name in ('position_counts', 'initial_position_ce_sums',
                 'commitment_position_ce_sums', 'row_commitment_ce_sums'):
        np.testing.assert_allclose(asnumpy(a[name]), asnumpy(b[name]), atol=2e-10)
    for p, g0, g1 in zip(t.parameters(), ga, gb):
        np.testing.assert_allclose(g0, g1, atol=3e-9, rtol=1e-9, err_msg=p.name)
    assert b['causal_context_encodes'] == 1
    assert b['refinement_passes'] == 6
    assert b['context_keys_per_group_max'] <= a['context_keys_per_group_max']
    if sparse:
        assert b['context_keys_per_group_max'] < 100


def test_plan_indices_exact_and_future_inaccessible():
    starts = (5, 12, 20)
    b, k = 2, 3
    mask = np.ones((b, len(starts), k), dtype=bool)
    mask[1, 2, 1:] = False
    layouts = (({'kind': 'local', 'window': 6, 'dilation': None,
                 'offset': 0, 'stride': None},
                {'kind': 'global_sparse', 'window': None, 'dilation': None,
                 'offset': 1, 'stride': 4}),)
    history, hv, layers = pack_group_history(starts, b, k, xp.asarray(mask), layouts)
    hist, hmask = asnumpy(history), asnumpy(hv)
    width = hist.shape[1]
    for group_num, spec in enumerate(layouts[0]):
        old = grouped_provisional_plan(20, starts, b, k, valid_block_mask=xp.asarray(mask),
                                       **spec)
        newer = layers[0][group_num]
        before = asnumpy(old.key_indices)
        old_valid = asnumpy(old.valid_mask)
        after = asnumpy(newer.key_indices)
        new_valid = asnumpy(newer.valid_mask)
        for row in range(b * len(starts)):
            start = starts[row % len(starts)]
            for j in range(k):
                before_keys = before[row, 0, j][old_valid[row, 0, j]]
                mapped_keys = after[row, 0, j][new_valid[row, 0, j]]
                recovered = [hist[row, p] if p < width else 20 + p - width
                             for p in mapped_keys]
                assert list(before_keys) == recovered
                assert all(p < start or p >= 20 for p in recovered)
                assert all(p < start for p in hist[row, hmask[row]])


def test_empty_history_and_read_only_mask():
    valid = xp.ones((2, 1, 2), dtype=xp.bool_)
    layouts = (({'kind': 'local', 'window': 4, 'dilation': None,
                 'offset': 0, 'stride': None},),)
    h, hv, plans = pack_group_history((0,), 2, 2, valid, layouts)
    assert h.shape == (2, 0)
    assert hv.shape == (2, 0)
    assert np.all(asnumpy(plans[0][0].valid_mask))
    with pytest.raises(ValueError, match='max_history_keys'):
        pack_group_history((20,), 1, 2, xp.ones((1, 1, 2), dtype=xp.bool_),
                           layouts, max_history_keys=1)


def test_4096_sparse_history_plan_stays_bounded():
    # Planner test only; do not allocate a 500M-parameter model on the CPU.
    from mini_llm.predictive_refinement.scale_config import wide_static_trunk_config
    cfg = wide_static_trunk_config(context_length=4096)
    groups = []
    used = set()
    for head in cfg.attention_layers[0].heads:
        signature = (head.kind, getattr(head, 'window', None),
                     getattr(head, 'dilation', None),
                     getattr(head, 'offset', 0), getattr(head, 'stride', None))
        if signature in used:
            continue
        used.add(signature)
        groups.append(dict(zip(('kind', 'window', 'dilation', 'offset', 'stride'), signature)))
    starts = (3780, 3880, 3980, 4080)
    mask = xp.ones((2, 4, 4), dtype=xp.bool_)
    history, valid, plans = pack_group_history(
        starts, 2, 4, mask, (tuple(groups),), max_history_keys=4096)
    h = asnumpy(history)
    v = asnumpy(valid)
    assert h.shape[1] < 4096
    assert h.shape[0] == 8
    assert len(plans[0]) == len(groups)
    for row in range(8):
        assert np.all(h[row][v[row]] < starts[row % 4])
        assert np.all(h[row][v[row]][1:] > h[row][v[row]][:-1])


def test_training_cli_requires_explicit_long_window_and_budget():
    from examples.train_predictive_refinement_text import parse_args
    cli = ['--shard-dir', '/tmp/nothing', '--prefix-length', '260',
           '--block-size', '2', '--window-blocks', '4']
    with pytest.raises(SystemExit):
        parse_args(cli)
    with pytest.raises(SystemExit):
        parse_args(cli + ['--max-window-tokens', '512'])
    args = parse_args(cli + ['--max-window-tokens', '512', '--compact-history',
                             '--compact-active-blocks', '--max-active-queries', '64'])
    assert args.compact_history and args.max_window_tokens == 512
