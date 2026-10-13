"""0079M: compact nonempty document/block rows without changing refinement BPTT."""
from dataclasses import replace

import numpy as np
import pytest

from mini_llm.backend import xp, asnumpy
from mini_llm.config import (
    ModelConfig, AttentionLayerConfig, LocalAttentionConfig,
    DilatedAttentionConfig,
)
from mini_llm.model.decoder_lm import DecoderLanguageModel
from mini_llm.predictive_refinement.training_reference import RefinementReferenceConfig
from mini_llm.predictive_refinement.window_training import WindowRefinementTrainer
from mini_llm.predictive_refinement.telemetry import observe_microbatch, rates


def make_trainer(sparse):
    cfg = ModelConfig(tokenizer_vocab_size=64, context_length=32, n_layers=2,
                      d_model=16, n_q_heads=2, n_kv_heads=1, d_head=8,
                      n_experts=2, top_k=1, d_ff=32, dtype='float64')
    if sparse:
        layer = AttentionLayerConfig(heads=(LocalAttentionConfig(window=6),
                                           DilatedAttentionConfig(window=12, dilation=2)))
        cfg = replace(cfg, attention_layers=(layer, layer))
    return WindowRefinementTrainer(
        DecoderLanguageModel(cfg, rng_seed=79, dtype='float64'),
        RefinementReferenceConfig(block_size=2, enable_shared_context_training=True),
        rng_seed=80)


def ragged():
    # Three separate documents with 10, 4 and 3 legitimate tokens.
    tokens = np.arange(30, dtype=np.int64).reshape(3, 10) % 50 + 4
    valid = np.arange(10)[None, :] < np.array([10, 4, 3])[:, None]
    tokens[~valid] = -100
    return xp.asarray(tokens), xp.asarray(valid, dtype=xp.bool_)


@pytest.mark.parametrize('sparse', [False, True])
@pytest.mark.parametrize('group', [1, 3, 5])
@pytest.mark.parametrize('backward', [False, True])
def test_ragged_parallel_compaction_matches_padded_reference(sparse, group, backward):
    ref = make_trainer(sparse)
    tokens, valid = ragged()
    ref.zero_grad()
    baseline = ref.train_window(tokens, prefix_length=2, valid_token_mask=valid,
                                group_size=group, compute_gradients=backward,
                                record_row_ce=True)
    base_grad = [asnumpy(p.grad).copy() for p in ref.parameters()]
    ref.zero_grad()
    compact = ref.train_window(tokens, prefix_length=2, valid_token_mask=valid,
                               group_size=group, compute_gradients=backward,
                               record_row_ce=True, compact_active_blocks=True)
    for key in ('total_loss', 'commitment_ce', 'initial_ce'):
        np.testing.assert_allclose(float(baseline[key]), float(compact[key]), atol=1e-11, rtol=1e-11)
    for key in ('initial_position_ce_sums','commitment_position_ce_sums',
                'position_counts','row_commitment_ce_sums'):
        np.testing.assert_allclose(asnumpy(baseline[key]), asnumpy(compact[key]), atol=1e-10)
    for expected, parameter in zip(base_grad, ref.parameters()):
        np.testing.assert_allclose(expected, asnumpy(parameter.grad), rtol=1e-9,
                                   atol=1e-10, err_msg=parameter.name)
    assert compact['supervised_tokens'] == 11
    assert compact['active_block_queries'] < baseline['active_block_queries']
    assert compact['skipped_empty_block_rows'] > 0
    assert sum(compact['executed_active_rows_per_pass']) * 3 == compact['active_block_queries']
    telemetry = observe_microbatch(asnumpy(tokens), asnumpy(valid), 2, compact,
                                   configured_prefix=2, window_blocks=4, block_size=2,
                                   collect_sources=False)
    r = rates(telemetry, 1.)
    assert r['active_block_row_fill'] < 1
    assert r['active_queries_per_pass_max'] <= group * 3 * 2


def test_large_explicit_group_and_query_guard():
    ref = make_trainer(False)
    token = xp.asarray(np.arange(2, 24, dtype=np.int64)[None, :])
    small = ref.train_window(token, prefix_length=2, group_size=1,
                             compute_gradients=False)
    big = ref.train_window(token, prefix_length=2, group_size=20,
                           compact_active_blocks=True, max_active_queries=40,
                           compute_gradients=False)
    assert float(small['commitment_ce']) == pytest.approx(float(big['commitment_ce']), abs=1e-10)
    assert big['refinement_batches'] == 1
    assert big['executed_active_rows_per_pass'] == (20,)
    with pytest.raises(ValueError, match='active query guard'):
        ref.train_window(token, prefix_length=2, group_size=20,
                         compact_active_blocks=True, max_active_queries=10,
                         compute_gradients=False)
    with pytest.raises(ValueError, match='group_size'):
        ref.train_window(token, prefix_length=2, group_size=65, compute_gradients=False)
