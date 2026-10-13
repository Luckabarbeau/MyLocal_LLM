"""0079G: exact shared-context training, including gradients into context KV."""

from dataclasses import replace
import numpy as np
import pytest

from mini_llm.backend import xp, asnumpy
from mini_llm.config import (
    ModelConfig, AttentionLayerConfig, DenseAttentionConfig, LocalAttentionConfig,
    DilatedAttentionConfig, GlobalSparseAttentionConfig,
)
from mini_llm.model.decoder_lm import DecoderLanguageModel
from mini_llm.predictive_refinement.training_reference import (
    DenseRefinementTrainingReference, RefinementReferenceConfig,
)


def pair(*, k=3, sparse=False, c=5, aux=0.15):
    cfg = ModelConfig.tiny_inspection()
    if sparse:
        cfg = replace(cfg, attention_layers=(AttentionLayerConfig(heads=(
            LocalAttentionConfig(window=4), DilatedAttentionConfig(window=10, dilation=2),
        )),))
    unshared = DenseRefinementTrainingReference(
        DecoderLanguageModel(cfg, rng_seed=79, dtype="float64"),
        RefinementReferenceConfig(
            block_size=k, initial_loss_weight=aux,
            enable_compact_static_attention=sparse,
        ), rng_seed=83,
    )
    shared = DenseRefinementTrainingReference(
        DecoderLanguageModel(cfg, rng_seed=79, dtype="float64"),
        RefinementReferenceConfig(
            block_size=k, initial_loss_weight=aux,
            enable_shared_context_training=True,
        ), rng_seed=83,
    )
    context = xp.asarray(np.arange(2 * c).reshape(2, c) % 60 + 1, dtype=xp.int32)
    targets = xp.asarray(np.arange(2 * k).reshape(2, k) % 60 + 61, dtype=xp.int32)
    return unshared, shared, context, targets


@pytest.mark.parametrize("sparse", [False, True])
@pytest.mark.parametrize("k,c", [(1, 4), (3, 5), (4, 1), (2, 0)])
def test_shared_matches_all_logits_losses_and_parameter_gradients(sparse, k, c):
    unshared, shared, context, target = pair(sparse=sparse, k=k, c=c)
    valid = xp.ones((2, k), dtype=xp.bool_)
    if k > 1:
        valid[1, -1] = False
        target[1, -1] = -100
    values0, tape0 = unshared.forward(context, target, valid_target_mask=valid)
    values1, tape1 = shared.forward(context, target, valid_target_mask=valid)
    for key in ("initial_ce", "commitment_ce", "total_loss"):
        np.testing.assert_allclose(asnumpy(values0[key]), asnumpy(values1[key]), rtol=1e-12, atol=1e-12)
    assert values0["supervised_tokens"] == values1["supervised_tokens"]
    assert values1["full_model_passes"] == k + 1
    assert values1["causal_context_encodes"] == (1 if c else 0)
    for a, b in zip(tape0["passes"], tape1["passes"]):
        np.testing.assert_allclose(asnumpy(a.logits[:, c:]), asnumpy(b.logits), atol=1e-11, rtol=1e-11)
    unshared.zero_grad()
    unshared.backward(tape0)
    shared.zero_grad()
    shared.backward(tape1)
    for p0, p1 in zip(unshared.parameters(), shared.parameters()):
        np.testing.assert_allclose(asnumpy(p0.grad), asnumpy(p1.grad), atol=5e-10, rtol=5e-9,
                                   err_msg=f"shared context gradient mismatch: {p0.name}")


def test_kv_cache_is_read_only_during_refinement():
    _, shared, ctx, tgt = pair(k=4, sparse=True, c=6)
    # Instrument only the causal context branch, then run all active passes.
    model = shared.model
    head, context_cache = model.forward_body(ctx, return_cache=True)
    kv = [(cache["attn_cache"]["k"], cache["attn_cache"]["v"])
          for cache in context_cache["block_caches"]]
    original = [(asnumpy(k).copy(), asnumpy(v).copy()) for k, v in kv]
    for j in range(5):
        embeddings = shared.states.unknown_block(2, 4)
        logits, forward_cache = model.forward_shared_context_block(
            embeddings, kv, context_length=6)
        model.backward_shared_context_block(xp.ones_like(logits), forward_cache)
        for (k, v), (ref_k, ref_v) in zip(kv, original):
            np.testing.assert_array_equal(asnumpy(k), ref_k)
            np.testing.assert_array_equal(asnumpy(v), ref_v)


def test_actual_context_encoding_only_once_and_no_context_logits():
    _, shared, ctx, tgt = pair(k=4, sparse=True, c=7)
    attention = shared.model.blocks[0].attention
    causal_calls, block_calls = [], []
    ordinary = attention.forward
    active = attention.forward_shared_context_block

    def instrument_causal(x, *args, **kwargs):
        causal_calls.append(x.shape[1])
        return ordinary(x, *args, **kwargs)

    def instrument_active(x, *args, **kwargs):
        block_calls.append(x.shape[1])
        return active(x, *args, **kwargs)

    attention.forward = instrument_causal
    attention.forward_shared_context_block = instrument_active
    try:
        metrics, tape = shared.forward(ctx, tgt)
        shared.backward(tape)
    finally:
        attention.forward = ordinary
        attention.forward_shared_context_block = active
    assert causal_calls == [7]
    assert block_calls == [4] * 5
    assert all(record.logits.shape[1] == 4 for record in tape["passes"])
    assert metrics["causal_context_encodes"] == 1


@pytest.mark.parametrize("sparse", [False, True])
def test_context_weight_gradient_from_refinement_matches_finite_differences(sparse):
    _, shared, ctx, tgt = pair(k=2, sparse=sparse, c=4, aux=0)
    values, tape = shared.forward(ctx, tgt)
    shared.zero_grad()
    shared.backward(tape)
    # Check the context embedding and K projection. Disabling cross-stream
    # backward would fail both tests despite correct forward logits.
    selected = [
        (shared.model.embedding.W, (int(ctx[0, 1]), 2)),
        (shared.model.blocks[0].attention.Wk, (1, 0)),
        (shared.model.blocks[0].attention.Wv, (1, 0)),
        (shared.states.unknown, (2,)),
    ]
    eps = 2e-5
    for p, idx in selected:
        analytical = float(p.grad[idx])
        p.data[idx] += eps
        plus = float(shared.forward(ctx, tgt)[0]["total_loss"])
        p.data[idx] -= 2 * eps
        minus = float(shared.forward(ctx, tgt)[0]["total_loss"])
        p.data[idx] += eps
        np.testing.assert_allclose(analytical, (plus - minus) / (2 * eps), atol=8e-6, rtol=6e-3,
                                   err_msg=p.name)


def test_future_target_values_cannot_affect_pass_0_or_pass_1():
    _, shared, ctx, tgt = pair(k=4, c=5)
    alt = tgt.copy()
    alt[:, 2] = 107
    _, tape_a = shared.forward(ctx, tgt)
    _, tape_b = shared.forward(ctx, alt)
    for i in (0, 1, 2):  # Third pass has only y0 committed.
        np.testing.assert_array_equal(asnumpy(tape_a["passes"][i].logits),
                                      asnumpy(tape_b["passes"][i].logits))


def test_invalid_targets_produce_identical_gradients():
    _, shared, ctx, tgt = pair(k=4, sparse=True, c=5)
    valid = xp.asarray([[True, True, True, True], [True, False, False, False]])
    alt = tgt.copy()
    alt[1, 1:] = -100
    v0, tape0 = shared.forward(ctx, tgt, valid_target_mask=valid)
    shared.zero_grad()
    shared.backward(tape0)
    grads = [asnumpy(p.grad).copy() for p in shared.parameters()]
    v1, tape1 = shared.forward(ctx, alt, valid_target_mask=valid)
    shared.zero_grad()
    shared.backward(tape1)
    np.testing.assert_allclose(float(v0["total_loss"]), float(v1["total_loss"]))
    for a, p in zip(grads, shared.parameters()):
        np.testing.assert_allclose(a, asnumpy(p.grad), atol=1e-11, rtol=1e-11)


def test_parameter_default_does_not_enable_shared_context():
    ref = RefinementReferenceConfig()
    assert not ref.enable_shared_context_training


def test_two_layer_float32_and_checkpoint_resume(tmp_path):
    from mini_llm.optim.adamw import AdamW
    from mini_llm.predictive_refinement.checkpoint import (
        save_refinement_checkpoint, load_refinement_checkpoint,
    )

    cfg = ModelConfig(
        tokenizer_vocab_size=64, context_length=64,
        n_layers=2, d_model=32, n_q_heads=4, n_kv_heads=2,
        d_head=8, n_experts=2, top_k=1, d_ff=64, dtype="float32",
    )

    def create(shared):
        ref = DenseRefinementTrainingReference(
            DecoderLanguageModel(cfg, rng_seed=794, dtype="float32"),
            RefinementReferenceConfig(block_size=3, enable_shared_context_training=shared),
            rng_seed=795,
        )
        opt = AdamW(ref.parameters(), lr=0.003, weight_decay=0.0)
        return ref, opt

    ctx = xp.asarray([[3, 5, 7], [4, 6, 8]], dtype=xp.int32)
    tgt = xp.asarray([[11, 13, 15], [12, 14, 16]], dtype=xp.int32)
    baseline, _ = create(False)
    cached, opt = create(True)
    orig, _ = baseline.forward(ctx, tgt)
    new, tape = cached.forward(ctx, tgt)
    np.testing.assert_allclose(float(orig["total_loss"]), float(new["total_loss"]), atol=1e-6)
    cached.zero_grad()
    cached.backward(tape)
    opt.step()
    path = tmp_path / "0079g"
    save_refinement_checkpoint(path, cached, optimizer=opt, training_step=1, supervised_tokens=6)
    resumed, resumed_opt = create(True)
    load_refinement_checkpoint(path, resumed, optimizer=resumed_opt)
    for p, q in zip(cached.parameters(), resumed.parameters()):
        np.testing.assert_array_equal(asnumpy(p.data), asnumpy(q.data))
    for ref, optimizer in ((cached, opt), (resumed, resumed_opt)):
        ref.zero_grad()
        _, tape = ref.forward(ctx, tgt)
        ref.backward(tape)
        optimizer.step()
    for p, q in zip(cached.parameters(), resumed.parameters()):
        np.testing.assert_array_equal(asnumpy(p.data), asnumpy(q.data))
