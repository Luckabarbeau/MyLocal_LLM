"""0079F static sparse block-causal plans and exact recurrent backward."""

from dataclasses import replace
import numpy as np
import pytest

from mini_llm.backend import RandomStream, asnumpy, xp
from mini_llm.config import (
    AttentionLayerConfig, DenseAttentionConfig, DilatedAttentionConfig,
    GlobalSparseAttentionConfig, LocalAttentionConfig, ModelConfig,
    RetrievalAttentionConfig, ContextRouterConfig,
)
from mini_llm.ops.attention import GQAAttention
from mini_llm.model.decoder_lm import DecoderLanguageModel
from mini_llm.predictive_refinement.compact_sparse import compact_provisional_plan
from mini_llm.predictive_refinement.training_reference import (
    DenseRefinementTrainingReference, RefinementReferenceConfig,
)


def numpy_plan(c, k, kind, **kwargs):
    return compact_provisional_plan(c, k, 1, kind=kind, **kwargs)


def keys(plan, j, b=0):
    indices = asnumpy(plan.key_indices)[b, 0, j]
    valid = asnumpy(plan.valid_mask)[b, 0, j]
    return indices[valid].tolist()


def test_compact_static_topology_exact_positions():
    assert keys(numpy_plan(10, 3, 'dense'), 0) == list(range(13))
    assert keys(numpy_plan(10, 3, 'local', window=4), 0) == [7, 8, 9, 10, 11, 12]
    assert keys(numpy_plan(10, 3, 'local', window=4), 2) == [9, 10, 11, 12]
    assert keys(numpy_plan(10, 3, 'dilated', window=9, dilation=3, offset=1), 0) == [3, 6, 9, 10, 11, 12]
    assert keys(numpy_plan(10, 3, 'global_sparse', stride=4, offset=1), 1) == [1, 5, 9, 10, 11, 12]


def test_compact_masks_and_empty_row_guard():
    plan = compact_provisional_plan(2, 4, 2, kind='local', window=2,
        valid_block_mask=xp.asarray([[1, 1, 0, 0], [1, 0, 0, 0]], dtype=bool))
    assert keys(plan, 0, 0) == [1, 2, 3]
    assert keys(plan, 0, 1) == [1, 2]
    # Padded positions cannot influence any valid provisional row.
    assert 4 not in keys(plan, 0, 0) and 5 not in keys(plan, 0, 0)
    empty = compact_provisional_plan(0, 2, 1, kind='global_sparse', stride=2, offset=1,
        valid_block_mask=xp.asarray([[True, False]]))
    assert keys(empty, 0) == [0]
    assert keys(empty, 1) == [0, 1]  # invalid query has a valid self key
    with pytest.raises(ValueError, match='max_selected_keys'):
        numpy_plan(50000, 4, 'dense', max_selected_keys=256)
    with pytest.raises(ValueError, match='prefix'):
        compact_provisional_plan(3, 3, 1, kind='local', window=3,
            valid_block_mask=xp.asarray([[True, False, True]]))
    with pytest.raises(NotImplementedError):
        numpy_plan(4, 2, 'retrieval')


def sparse_layer(dtype='float64'):
    return GQAAttention(
        d_model=8, n_q_heads=4, n_kv_heads=2, d_head=2,
        input_std=0.18, output_std=0.1, rng=RandomStream(120),
        dtype=dtype, attention_config=AttentionLayerConfig(heads=(
            LocalAttentionConfig(4), DilatedAttentionConfig(8, 3, 1),
            GlobalSparseAttentionConfig(3, 1), DenseAttentionConfig(),
        )),
    )


def reference_provisional(attn, cache, context_length, valid):
    q, k, v = (asnumpy(cache[key]) for key in ('q', 'k', 'v'))
    b, total, h, dh = q.shape
    c, z = context_length, total - context_length
    out = np.zeros((b, z, h, dh), dtype=np.float64)
    for head, cfg in enumerate(attn.attention_config.heads):
        args = dict(kind=cfg.kind)
        for name in ('window', 'dilation', 'offset', 'stride'):
            if hasattr(cfg, name):
                args[name] = getattr(cfg, name)
        plan = compact_provisional_plan(c, z, b, valid_block_mask=xp.asarray(valid), **args)
        indices = asnumpy(plan.key_indices)
        visible = asnumpy(plan.valid_mask)
        for sample in range(b):
            for j in range(z):
                positions = indices[sample, 0, j][visible[sample, 0, j]]
                if not len(positions):
                    continue
                kv_head = head // attn.group_size
                scores = q[sample, c + j, head] @ k[sample, positions, kv_head].T * attn.scale
                p = np.exp(scores - max(scores))
                p /= p.sum()
                out[sample, j, head] = p @ v[sample, positions, kv_head]
    return out.reshape(b, z, attn.d_model) @ asnumpy(attn.Wo.data)


def test_static_groups_match_direct_fp64_reference_and_context_invariance():
    layer = sparse_layer()
    x = xp.asarray(np.random.default_rng(121).normal(size=(2, 10, 8)), dtype='float64')
    valid = xp.asarray([[1, 1, 1, 1], [1, 1, 0, 0]], dtype=bool)
    result, cache = layer.forward(x, block_context_length=6, block_valid_mask=valid)
    expected = reference_provisional(layer, cache, 6, asnumpy(valid))
    np.testing.assert_allclose(asnumpy(result[:, 6:]), expected, atol=2e-12, rtol=2e-12)
    causal = layer.forward(x[:, :6], return_cache=False)
    np.testing.assert_allclose(asnumpy(result[:, :6]), asnumpy(causal), atol=2e-12, rtol=2e-12)
    perturbed = x.copy()
    perturbed[:, 9] += 1.0
    changed = layer.forward(perturbed, block_context_length=6, block_valid_mask=valid,
                            return_cache=False)
    np.testing.assert_allclose(asnumpy(changed[:, :6]), asnumpy(causal), atol=2e-12, rtol=2e-12)
    np.testing.assert_allclose(asnumpy(changed[1, 6]), asnumpy(result[1, 6]), atol=2e-12, rtol=2e-12)
    assert np.linalg.norm(asnumpy(changed[0, 6] - result[0, 6])) > 1e-10


def test_complete_attention_gradients_match_finite_differences():
    layer = sparse_layer()
    rng = np.random.default_rng(122)
    x = xp.asarray(rng.normal(size=(1, 8, 8)), dtype='float64')
    direction = xp.asarray(rng.normal(size=x.shape), dtype='float64')
    dy = xp.asarray(rng.normal(size=x.shape), dtype='float64')
    direction /= xp.sqrt(xp.sum(direction * direction))
    y, cache = layer.forward(x, block_context_length=5)
    layer.zero_grad()
    dx = layer.backward(dy, cache)
    exact = float(xp.sum(dx * direction))
    def loss(inp):
        out = layer.forward(inp, block_context_length=5, return_cache=False)
        return float(xp.sum(out * dy))
    eps = 1e-6
    fd = (loss(x + eps * direction) - loss(x - eps * direction)) / (2 * eps)
    np.testing.assert_allclose(exact, fd, rtol=2e-5, atol=5e-8)
    # Projection weight also includes gradients from both streams.
    wd = xp.asarray(rng.normal(size=layer.Wq.data.shape))
    weight = layer.Wq.data.copy()
    def weight_loss(displacement):
        layer.Wq.data[...] = weight + displacement * wd
        return loss(x)
    analytic = float(xp.sum(layer.Wq.grad * wd))
    fd_weight = (weight_loss(eps) - weight_loss(-eps)) / (2 * eps)
    layer.Wq.data[...] = weight
    np.testing.assert_allclose(analytic, fd_weight, rtol=2e-5, atol=5e-8)


def compact_model():
    config = replace(ModelConfig.tiny_inspection(), attention_layers=(
        AttentionLayerConfig(heads=(LocalAttentionConfig(5), DilatedAttentionConfig(8, 2))),
    ))
    return DecoderLanguageModel(config, rng_seed=124, dtype='float64')


def test_end_to_end_refinement_with_sparse_block_gradient():
    model = compact_model()
    ref = DenseRefinementTrainingReference(model, RefinementReferenceConfig(
        block_size=2, enable_compact_static_attention=True,
    ), rng_seed=126)
    ctx = xp.asarray([[3, 4, 5, 6]], dtype=xp.int64)
    targets = xp.asarray([[7, 8]], dtype=xp.int64)
    loss, tape = ref.forward(ctx, targets)
    ref.zero_grad()
    ref.backward(tape)
    assert np.isfinite(float(loss['commitment_ce']))
    assert np.linalg.norm(asnumpy(ref.states.soft.grad)) > 1e-12
    original = float(model.embedding.W.data[7, 0])
    analytic = float(model.embedding.W.grad[7, 0])
    eps = 1e-5
    model.embedding.W.data[7, 0] = original + eps
    positive = float(ref.forward(ctx, targets)[0]['total_loss'])
    model.embedding.W.data[7, 0] = original - eps
    negative = float(ref.forward(ctx, targets)[0]['total_loss'])
    model.embedding.W.data[7, 0] = original
    np.testing.assert_allclose(analytic, (positive - negative)/(2*eps), rtol=3e-3, atol=3e-5)


def test_reject_unconfigured_mode_and_preserve_original_path():
    model = compact_model()
    with pytest.raises(NotImplementedError):
        DenseRefinementTrainingReference(model)
    x = xp.asarray(np.random.default_rng(127).normal(size=(1, 5, 16)), dtype='float64')
    normal = model.forward(None, input_embeddings=x, return_cache=False)
    assert normal.shape == (1, 5, 256)
    with pytest.raises(ValueError):
        model.forward(None, input_embeddings=x, block_context_length=5)
    with pytest.raises(ValueError):
        model.forward(None, input_embeddings=x, block_context_length=3,
                      attention_allow_mask=xp.ones((5,5),dtype=bool))


def test_64k_history_plan_remains_k_rows_not_quadratic():
    c, k = 65_536, 4
    for kind, settings, limit in (
        ('local', dict(window=128), 128 + k),
        ('dilated', dict(window=32_768, dilation=64, offset=1), 512 + k),
        ('global_sparse', dict(stride=256, offset=0), 256 + k),
    ):
        plan = compact_provisional_plan(c, k, 2, kind=kind, **settings)
        assert plan.key_indices.shape[2] == k
        assert plan.key_indices.shape[3] <= limit
        assert plan.key_indices.size < 2 * k * (limit + 1)
        assert asnumpy(plan.key_indices).max() < c + k


def test_empty_context_and_short_padded_block_full_backward():
    model = compact_model()
    ref = DenseRefinementTrainingReference(model, RefinementReferenceConfig(
        block_size=3, enable_compact_static_attention=True,
    ), rng_seed=130)
    context = xp.empty((1, 0), dtype=xp.int64)
    target = xp.asarray([[5, -100, -100]], dtype=xp.int64)
    valid = xp.asarray([[True, False, False]])
    loss, tape = ref.forward(context, target, valid_target_mask=valid)
    assert loss['supervised_tokens'] == 1
    ref.zero_grad()
    ref.backward(tape)
    assert np.isfinite(float(loss['total_loss']))
    assert all(np.isfinite(asnumpy(p.grad)).all() for p in ref.parameters())


def test_activation_checkpoint_replay_matches_direct_sparse_block_gradients():
    model = compact_model()
    rng = np.random.default_rng(134)
    x = xp.asarray(rng.normal(size=(1, 7, 16)), dtype='float64')
    upstream = xp.asarray(rng.normal(size=(1, 7, 256)), dtype='float64')
    pos = xp.arange(100, 107, dtype=xp.int64)
    model.zero_grad()
    normal, tape = model.forward(None, input_embeddings=x,
        block_context_length=4, position_ids=pos)
    dx_normal = model.backward(upstream, tape)
    grad_normal = [p.grad.copy() for p in model.parameters()]
    model.zero_grad()
    replay, tape = model.forward(None, input_embeddings=x,
        block_context_length=4, position_ids=pos, activation_checkpoint=True)
    dx_replay = model.backward(upstream, tape)
    grad_replay = [p.grad.copy() for p in model.parameters()]
    np.testing.assert_allclose(asnumpy(replay), asnumpy(normal), atol=1e-12)
    np.testing.assert_allclose(asnumpy(dx_replay), asnumpy(dx_normal), atol=1e-12)
    for a, b in zip(grad_replay, grad_normal):
        np.testing.assert_allclose(asnumpy(a), asnumpy(b), atol=1e-12)
