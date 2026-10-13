"""0079H: grouped target-block training with one shared causal tape."""
from dataclasses import replace

import numpy as np
import pytest

from mini_llm.backend import asnumpy, xp
from mini_llm.config import (
    ModelConfig, AttentionLayerConfig, LocalAttentionConfig,
    DilatedAttentionConfig, GlobalSparseAttentionConfig,
)
from mini_llm.model.decoder_lm import DecoderLanguageModel
from mini_llm.predictive_refinement.compact_sparse import grouped_provisional_plan
from mini_llm.predictive_refinement.training_reference import (
    DenseRefinementTrainingReference, RefinementReferenceConfig,
)
from mini_llm.predictive_refinement.window_training import WindowRefinementTrainer


def pair(*, sparse=False, k=3, aux=0.13, dtype="float64"):
    cfg = ModelConfig.tiny_inspection()
    if sparse:
        cfg = replace(cfg, attention_layers=(AttentionLayerConfig(heads=(
            LocalAttentionConfig(window=5),
            DilatedAttentionConfig(window=12, dilation=3),
        )),))
    def make(cls):
        return cls(
            DecoderLanguageModel(cfg, rng_seed=319, dtype=dtype),
            RefinementReferenceConfig(
                block_size=k, initial_loss_weight=aux,
                enable_shared_context_training=True,
            ), rng_seed=320,
        )
    return make(DenseRefinementTrainingReference), make(WindowRefinementTrainer)


def sequences(*, length=11, prefix=2, k=3):
    seq = np.arange(2 * length).reshape(2, length).astype(np.int64) % 40 + 3
    valid = np.ones_like(seq, dtype=bool)
    valid[1, length - 2:] = False
    seq[1, length - 2:] = -100
    return xp.asarray(seq), xp.asarray(valid), prefix


def independent_blocks(ref, seq, valid, *, prefix, k):
    nvalid = int(xp.sum(valid[:, prefix:]))
    loss = {"total_loss": 0.0, "initial_ce": 0.0, "commitment_ce": 0.0}
    ref.zero_grad()
    b, length = seq.shape
    for s in range(prefix, length, k):
        c = xp.where(valid[:, :s], seq[:, :s], 0)
        targets = xp.zeros((b, k), dtype=xp.int64)
        mask = xp.zeros((b, k), dtype=xp.bool_)
        actual = min(k, length - s)
        targets[:, :actual] = xp.where(valid[:, s:s+actual], seq[:, s:s+actual], 0)
        mask[:, :actual] = valid[:, s:s+actual]
        if not bool(xp.any(mask)):
            continue
        metrics, tape = ref.forward(c, targets, valid_target_mask=mask,
                                    loss_normalization_tokens=nvalid)
        ref.backward(tape)
        for key in loss:
            loss[key] += float(metrics[key])
    return loss


@pytest.mark.parametrize("sparse", [False, True])
@pytest.mark.parametrize("group_size", [1, 2, 3])
@pytest.mark.parametrize("k,prefix", [(3, 2), (1, 0), (4, 3)])
def test_bounded_groups_match_independent_blocks(sparse, group_size, k, prefix):
    ref, grouped = pair(sparse=sparse, k=k)
    ids, valid, _ = sequences(prefix=prefix)
    baseline = independent_blocks(ref, ids, valid, prefix=prefix, k=k)
    grouped.zero_grad()
    actual = grouped.train_window(ids, prefix_length=prefix,
                                  valid_token_mask=valid, group_size=group_size)
    for key in baseline:
        np.testing.assert_allclose(float(actual[key]), baseline[key], atol=1e-11, rtol=1e-11)
    assert actual["supervised_tokens"] == int(asnumpy(valid[:, prefix:]).sum())
    assert actual["causal_context_encodes"] == 1
    for p0, p1 in zip(ref.parameters(), grouped.parameters()):
        np.testing.assert_allclose(asnumpy(p0.grad), asnumpy(p1.grad), atol=1e-9,
                                   rtol=1e-8, err_msg=p0.name)


def test_grouped_selection_contains_only_own_causal_prefix_and_own_active_slots():
    b, k, c = 2, 4, 9
    starts = (2, 5, 9)
    valid = xp.asarray(np.ones((b, len(starts), k), dtype=bool))
    valid[1, 1, 2:] = False
    for kind, kwargs in (("dense", {}), ("local", {"window": 5}),
                         ("dilated", {"window": 12, "dilation": 2}),
                         ("global_sparse", {"stride": 3})):
        plan = grouped_provisional_plan(c, starts, b, k, kind=kind,
                                        valid_block_mask=valid, **kwargs)
        keys = asnumpy(plan.key_indices).reshape(b, len(starts), 1, k, -1)
        mask = asnumpy(plan.valid_mask).reshape(b, len(starts), 1, k, -1)
        for row in range(b):
            for block, start in enumerate(starts):
                for j in range(k):
                    selected = keys[row, block, 0, j][mask[row, block, 0, j]]
                    assert np.all((selected < start) | ((selected >= c) & (selected < c+k)))
                    assert not np.any((selected >= start) & (selected < c))
                    if valid[row, block, j]:
                        # Bidirectional provisional interactions include later soft positions.
                        expected = np.flatnonzero(asnumpy(valid[row, block])) + c
                        assert set(expected).issubset(set(selected))


def test_full_future_targets_never_leak_into_earlier_group_predictions():
    _, grouped = pair(k=3, sparse=True)
    ids, valid, prefix = sequences()
    model = grouped.model
    observed = []
    fn = model.forward_shared_context_block
    def capture(*args, **kwargs):
        result = fn(*args, **kwargs)
        observed.append(asnumpy(result[0]).copy())
        return result
    model.forward_shared_context_block = capture
    try:
        grouped.zero_grad()
        grouped.train_window(ids, prefix_length=prefix, valid_token_mask=valid, group_size=3)
        first = [a.copy() for a in observed]
        observed.clear()
        changed = ids.copy()
        # Change only targets in the FINAL target block. Its own initial
        # predictions and earlier blocks must be identical before commitment.
        changed[0, -2:] = xp.asarray([77, 78])
        grouped.zero_grad()
        grouped.train_window(changed, prefix_length=prefix, valid_token_mask=valid, group_size=3)
        assert len(observed) == len(first)
        for earlier, later in zip(first[:2], observed[:2]):
            np.testing.assert_array_equal(earlier, later)
        # In subsequent passes, for first two blocks no target changes can
        # influence their logits (each block's contextual prefix is fixed).
        for earlier, later in zip(first, observed):
            np.testing.assert_array_equal(earlier[:2], later[:2])
    finally:
        model.forward_shared_context_block = fn


def test_grouped_finite_difference_through_prior_target_context():
    _, trainer = pair(k=2, sparse=True, aux=0)
    seq = xp.asarray([[3, 7, 11, 15, 19, 23]], dtype=xp.int64)
    param = trainer.model.embedding.W
    trainer.zero_grad()
    trainer.train_window(seq, prefix_length=2, group_size=2)
    analytical = float(param.grad[11, 2])
    eps = 1e-5
    param.data[11, 2] += eps
    plus = float(trainer.train_window(seq, prefix_length=2, group_size=2)["total_loss"])
    param.data[11, 2] -= 2 * eps
    minus = float(trainer.train_window(seq, prefix_length=2, group_size=2)["total_loss"])
    param.data[11, 2] += eps
    np.testing.assert_allclose(analytical, (plus-minus)/(2*eps), rtol=0.01, atol=2e-6)


def test_exact_eos_short_document_and_invalid_tail_invariance():
    _, trainer = pair(k=4, sparse=True)
    seq = xp.asarray([[3, 4, 5, 6, 7, 8, 9, 10, 11],
                      [4, 5, 6, 1, -100, -100, -100, -100, -100]], dtype=xp.int64)
    valid = xp.asarray([[True]*9, [True]*4 + [False]*5])
    trainer.zero_grad()
    baseline = trainer.train_window(seq, prefix_length=2, valid_token_mask=valid, group_size=2)
    grads = [asnumpy(p.grad).copy() for p in trainer.parameters()]
    changed = seq.copy()
    changed[1, 4:] = 52
    trainer.zero_grad()
    actual = trainer.train_window(changed, prefix_length=2, valid_token_mask=valid, group_size=2)
    assert actual["supervised_tokens"] == 9
    np.testing.assert_allclose(float(actual["total_loss"]), float(baseline["total_loss"]))
    for arr, param in zip(grads, trainer.parameters()):
        np.testing.assert_allclose(arr, asnumpy(param.grad), atol=1e-11, rtol=1e-11)


def test_oversized_reference_invalid_input_and_empty_supervision_are_rejected():
    _, trainer = pair(k=4)
    ids = xp.ones((1, 12), dtype=xp.int64)
    with pytest.raises(ValueError, match="bounded"):
        trainer.train_window(ids, max_window_tokens=8)
    with pytest.raises(ValueError, match="group_size"):
        trainer.train_window(ids, group_size=65)
    with pytest.raises(ValueError, match="supervised"):
        trainer.train_window(ids, prefix_length=2,
                             valid_token_mask=xp.asarray([[True,True]+[False]*10]))
    with pytest.raises(ValueError, match="prefix-contiguous"):
        trainer.train_window(ids, valid_token_mask=xp.asarray([[True,False]+[True]*10]))


def test_two_layer_window_training_matches_independent_full_backward():
    cfg = ModelConfig(
        tokenizer_vocab_size=64, context_length=32,
        n_layers=2, d_model=32, n_q_heads=4, n_kv_heads=2,
        d_head=8, n_experts=2, top_k=1, d_ff=64, dtype="float32",
    )
    cfg = replace(cfg, attention_layers=tuple(AttentionLayerConfig(heads=(
        LocalAttentionConfig(window=8), LocalAttentionConfig(window=8),
        DilatedAttentionConfig(window=16, dilation=2),
        DilatedAttentionConfig(window=16, dilation=2),
    )) for _ in range(cfg.n_layers)))
    def make(cls):
        return cls(
            DecoderLanguageModel(cfg, rng_seed=281, dtype="float32"),
            RefinementReferenceConfig(block_size=4, enable_shared_context_training=True),
            rng_seed=282,
        )
    seq = xp.asarray(np.arange(24).reshape(2, 12) + 3, dtype=xp.int64)
    baseline, grouped = make(DenseRefinementTrainingReference), make(WindowRefinementTrainer)
    values = independent_blocks(baseline, seq, xp.ones(seq.shape, dtype=xp.bool_), prefix=4, k=4)
    grouped.zero_grad()
    actual = grouped.train_window(seq, prefix_length=4, group_size=2)
    np.testing.assert_allclose(float(actual["total_loss"]), values["total_loss"], atol=2e-6)
    for p0, p1 in zip(baseline.parameters(), grouped.parameters()):
        np.testing.assert_allclose(asnumpy(p0.grad), asnumpy(p1.grad), atol=4e-5,
                                   rtol=3e-4, err_msg=p0.name)


def test_grouped_checkpoint_optimizer_resume_matches_uninterrupted(tmp_path):
    from mini_llm.optim.adamw import AdamW
    from mini_llm.predictive_refinement.checkpoint import (
        save_refinement_checkpoint, load_refinement_checkpoint,
    )
    _, left = pair(k=3, dtype="float32")
    _, right = pair(k=3, dtype="float32")
    left_optim = AdamW(left.parameters(), lr=0.002, weight_decay=0.0)
    right_optim = AdamW(right.parameters(), lr=0.002, weight_decay=0.0)
    seq = xp.asarray([[3, 5, 7, 9, 11, 13, 15, 17],
                      [4, 6, 8, 10, 12, 14, 16, 18]], dtype=xp.int64)
    for trainer, optim in ((left, left_optim),):
        trainer.zero_grad()
        result = trainer.train_window(seq, prefix_length=2, group_size=2)
        optim.step()
    save_refinement_checkpoint(
        tmp_path / "grouped_checkpoint", left, optimizer=left_optim,
        training_step=1, supervised_tokens=result["supervised_tokens"],
        data_provenance={"synthetic": True, "window_blocks": 2},
    )
    load_refinement_checkpoint(tmp_path / "grouped_checkpoint", right, optimizer=right_optim)
    for trainer, optim in ((left, left_optim), (right, right_optim)):
        trainer.zero_grad()
        trainer.train_window(seq, prefix_length=2, group_size=1)
        optim.step()
    for l, r in zip(left.parameters(), right.parameters()):
        np.testing.assert_array_equal(asnumpy(l.data), asnumpy(r.data))
