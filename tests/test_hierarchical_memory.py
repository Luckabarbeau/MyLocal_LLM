import dataclasses
import numpy as np

from mini_llm.backend import RandomStream, asnumpy, xp
from mini_llm.checkpoint import load_checkpoint, save_checkpoint
from mini_llm.config import MemoryContextConfig, ModelConfig
from mini_llm.data.token_shards import (
    build_packed_document_index,
    create_hierarchical_memory_minibatch,
)
from mini_llm.model.decoder_lm import DecoderLanguageModel
from mini_llm.ops.context_router import CausalQueryPooler
from mini_llm.ops.hierarchical_memory import (
    ExternalMemoryReader,
    HierarchicalMemoryRouter,
    build_external_memory_plan,
)


def _tiny_config(*, top_k=2, router_weight_scale=1.0):
    memory = MemoryContextConfig(
        enabled=True,
        memory_length=8,
        recent_length=4,
        target_length=3,
        block_size=2,
        top_k_blocks=top_k,
        router_query_length=4,
        router_dim=4,
        query_pooling="mean",
        history_pooling="mean",
        router_weight_scale=router_weight_scale,
        read_heads=2,
        read_kv_heads=1,
        read_query_chunk=2,
    )
    return ModelConfig(
        tokenizer_vocab_size=32,
        context_length=memory.active_length,
        n_layers=1,
        d_model=8,
        n_q_heads=2,
        n_kv_heads=1,
        d_head=4,
        n_experts=2,
        top_k=1,
        d_ff=16,
        dtype="float32",
        memory_context=memory,
    )


def test_memory_presets_keep_deep_active_length_fixed():
    configs = [
        ModelConfig.wide_500m_memory_16k(),
        ModelConfig.wide_500m_memory_32k(),
        ModelConfig.wide_500m_memory_64k(),
    ]
    assert {cfg.memory_context.active_length for cfg in configs} == {4096}
    assert {cfg.context_length for cfg in configs} == {4096}
    assert [cfg.memory_context.memory_length for cfg in configs] == [16384, 32768, 65536]
    assert [cfg.memory_context.searchable_blocks for cfg in configs] == [128, 256, 512]
    assert {cfg.memory_context.target_length for cfg in configs} == {4096}
    assert {cfg.memory_context.router_query_length for cfg in configs} == {4096}
    assert len({cfg.estimated_parameter_count() for cfg in configs}) == 1


def test_hierarchical_minibatch_returns_dense_training_window_labels():
    data = np.arange(200, dtype=np.uint16)
    source, labels = create_hierarchical_memory_minibatch(
        data,
        batch_size=3,
        memory_length=8,
        target_length=3,
        rng=np.random.default_rng(3),
    )
    assert source.shape == (3, 11)
    assert labels.shape == (3, 3)
    np.testing.assert_array_equal(labels[:, 0], source[:, 9])
    np.testing.assert_array_equal(labels[:, 1], source[:, 10])
    np.testing.assert_array_equal(labels[:, 2], source[:, 10] + 1)


def test_future_target_tokens_cannot_change_earlier_memory_selection():
    model = DecoderLanguageModel(_tiny_config(top_k=1), rng_seed=11)
    source = xp.asarray([[1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11]], dtype=xp.int64)
    active_a, _ = model._hierarchical_context_forward(source)

    # Route 0 predicts the token after source[8].  Changing source[9:] is future
    # information for that prediction and therefore cannot alter route 0.
    changed = source.copy()
    changed[:, 9:] = xp.asarray([[21, 22]], dtype=xp.int64)
    active_b, _ = model._hierarchical_context_forward(changed)
    np.testing.assert_array_equal(
        asnumpy(active_a.selected_blocks[:, :1]),
        asnumpy(active_b.selected_blocks[:, :1]),
    )
    np.testing.assert_allclose(
        asnumpy(active_a.route_weights[:, :1]),
        asnumpy(active_b.route_weights[:, :1]),
        rtol=0,
        atol=0,
    )


def test_per_position_candidate_mask_excludes_recent_history():
    model = DecoderLanguageModel(_tiny_config(top_k=2), rng_seed=5)
    source = xp.asarray([[1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11]], dtype=xp.int64)
    active, _ = model._hierarchical_context_forward(source)

    selected = asnumpy(active.selected_blocks)[0]
    # Prediction rows correspond to current source positions 8,9,10.  With a
    # recent horizon of four, complete block ends must be <= 5,6,7.
    cutoffs = np.array([5, 6, 7])
    selected_ends = (selected + 1) * 2
    assert np.all(selected_ends <= cutoffs[:, None])

    np.testing.assert_array_equal(asnumpy(active.token_ids), asnumpy(source[:, 8:11]))
    np.testing.assert_array_equal(asnumpy(active.position_ids)[0], np.arange(8, 11))
    assert active.target_start == 0
    assert active.target_end == 3


def test_short_history_route_falls_back_to_current_window_without_retrieval():
    memory = MemoryContextConfig(
        enabled=True,
        memory_length=4,
        recent_length=4,
        target_length=3,
        block_size=2,
        top_k_blocks=1,
        router_query_length=4,
        router_dim=4,
        query_pooling="mean",
        history_pooling="mean",
        read_heads=2,
        read_kv_heads=1,
        read_query_chunk=2,
    )
    cfg = dataclasses.replace(_tiny_config(top_k=1), context_length=3, memory_context=memory)
    model = DecoderLanguageModel(cfg, rng_seed=17)
    source = xp.asarray([[1, 4, 7, 2, 3, 8, 5]], dtype=xp.int64)
    active, _ = model._hierarchical_context_forward(source)
    assert not bool(asnumpy(active.route_valid)[0, 0])
    # Empty memory read must be exactly zero, so the first deep row is the raw
    # current-token embedding.
    expected = model.embedding.W.data[source[:, 4:5]]
    np.testing.assert_allclose(
        asnumpy(active.embeddings[:, :1]), asnumpy(expected), rtol=1e-6, atol=1e-7
    )


def test_router_uses_available_old_blocks_before_full_topk_budget_exists():
    memory = MemoryContextConfig(
        enabled=True,
        memory_length=8,
        recent_length=6,
        target_length=2,
        block_size=2,
        top_k_blocks=2,
        router_query_length=6,
        router_dim=4,
        query_pooling="mean",
        history_pooling="mean",
        read_heads=1,
        read_kv_heads=1,
        read_query_chunk=1,
    )
    cfg = dataclasses.replace(
        _tiny_config(top_k=1), context_length=2, memory_context=memory
    )
    model = DecoderLanguageModel(cfg, rng_seed=23)
    source = xp.asarray([[1, 2, 3, 4, 5, 6, 7, 8, 9, 10]], dtype=xp.int64)
    active, _ = model._hierarchical_context_forward(source)

    # First route cutoff is source position 3, so only block [0:2] is a
    # complete eligible block even though top_k_blocks=2.  It must be used
    # rather than disabling retrieval until two old blocks exist.
    valid = asnumpy(active.selected_valid)[0, 0]
    weights = asnumpy(active.route_weights)[0, 0]
    assert int(valid.sum()) == 1
    np.testing.assert_allclose(weights[valid].sum(), 1.0, rtol=1e-7, atol=1e-8)
    np.testing.assert_allclose(weights[~valid], 0.0, rtol=0, atol=0)



def test_router_uses_available_old_blocks_before_topk_capacity_is_full():
    config = MemoryContextConfig(
        enabled=True,
        memory_length=8,
        recent_length=4,
        target_length=2,
        block_size=2,
        top_k_blocks=2,
        router_query_length=4,
        router_dim=3,
        query_pooling="mean",
        history_pooling="mean",
        read_heads=1,
        read_kv_heads=1,
        read_query_chunk=1,
    )
    router = HierarchicalMemoryRouter(
        d_model=4,
        config=config,
        rng=RandomStream(23),
        input_std=0.05,
        dtype="float64",
    )
    rng = np.random.default_rng(29)
    history = xp.asarray(rng.normal(size=(1, 8, 4)), dtype="float64")
    query_source = xp.asarray(rng.normal(size=(1, 5, 4)), dtype="float64")
    starts = xp.asarray([4, 5], dtype=xp.int64)
    # First route has no complete old block; second has exactly block 0.
    cutoffs = xp.asarray([0, 2], dtype=xp.int64)
    weights, selected, selected_valid, route_valid, _, _ = router.forward(
        history, query_source, starts, cutoffs
    )
    np.testing.assert_array_equal(
        asnumpy(route_valid), np.array([[False, True]])
    )
    np.testing.assert_array_equal(
        asnumpy(selected_valid[0]), np.array([[False, False], [True, False]])
    )
    np.testing.assert_allclose(asnumpy(weights[0, 0]), 0.0, atol=0.0)
    np.testing.assert_allclose(asnumpy(weights[0, 1]), np.array([1.0, 0.0]), atol=1e-12)
    assert int(asnumpy(selected[0, 1, 0])) == 0

def test_external_plan_keeps_different_per_position_blocks_without_union_context():
    selected = xp.asarray([[[0, 2], [1, 3], [0, 3]]], dtype=xp.int64)
    weights = xp.asarray([[[0.6, 0.4], [0.2, 0.8], [0.5, 0.5]]], dtype="float32")
    selected_valid = xp.ones(selected.shape, dtype=bool)
    plan = build_external_memory_plan(
        selected, weights, selected_valid,
        block_size=2, weight_scale=1.0, weight_eps=1e-8
    )
    assert plan.key_indices.shape == (1, 1, 3, 4)
    np.testing.assert_array_equal(
        asnumpy(plan.key_indices[0, 0]),
        np.array([[0, 1, 4, 5], [2, 3, 6, 7], [0, 1, 6, 7]]),
    )


def test_external_memory_reader_backward_matches_finite_difference():
    config = MemoryContextConfig(
        enabled=True,
        memory_length=8,
        recent_length=4,
        target_length=2,
        block_size=2,
        top_k_blocks=2,
        router_query_length=4,
        router_dim=3,
        query_pooling="mean",
        history_pooling="mean",
        router_weight_scale=0.7,
        read_heads=1,
        read_kv_heads=1,
        read_query_chunk=1,
    )
    reader = ExternalMemoryReader(
        d_model=4,
        d_head=2,
        config=config,
        rng=RandomStream(13),
        input_std=0.05,
        output_std=0.05,
        dtype="float64",
    )
    rng = np.random.default_rng(19)
    history = xp.asarray(rng.normal(size=(1, 8, 4)), dtype="float64")
    query = xp.asarray(rng.normal(size=(1, 2, 4)), dtype="float64")
    selected = xp.asarray([[[0, 2], [1, 3]]], dtype=xp.int64)
    weights = xp.asarray([[[0.65, 0.35], [0.3, 0.7]]], dtype="float64")
    selected_valid = xp.ones(selected.shape, dtype=bool)
    coeff = xp.asarray(rng.normal(size=(1, 2, 4)), dtype="float64")

    history_pos = xp.arange(8, dtype=xp.int64)
    query_pos = xp.asarray([8, 9], dtype=xp.int64)
    output, cache = reader.forward(
        history, query, selected, weights, selected_valid,
        history_position_ids=history_pos, query_position_ids=query_pos,
        return_cache=True,
    )
    reader.zero_grad()
    _, _, dweights = reader.backward(coeff, cache, history)
    analytic_wq = float(reader.W_q.grad[0, 0])
    analytic_weight = float(dweights[0, 0, 0])

    eps = 1e-6
    original_wq = float(reader.W_q.data[0, 0])

    def objective_wq(value):
        reader.W_q.data[0, 0] = value
        out = reader.forward(
            history, query, selected, weights, selected_valid,
            history_position_ids=history_pos, query_position_ids=query_pos,
            return_cache=False,
        )
        return float(xp.sum(out * coeff))

    plus = objective_wq(original_wq + eps)
    minus = objective_wq(original_wq - eps)
    reader.W_q.data[0, 0] = original_wq
    fd_wq = (plus - minus) / (2 * eps)
    assert abs(fd_wq - analytic_wq) < 2e-6

    weights_work = weights.copy()
    original_weight = float(weights_work[0, 0, 0])

    def objective_weight(value):
        weights_work[0, 0, 0] = value
        out = reader.forward(
            history, query, selected, weights_work, selected_valid,
            history_position_ids=history_pos, query_position_ids=query_pos,
            return_cache=False,
        )
        return float(xp.sum(out * coeff))

    plus = objective_weight(original_weight + eps)
    minus = objective_weight(original_weight - eps)
    weights_work[0, 0, 0] = original_weight
    fd_weight = (plus - minus) / (2 * eps)
    assert abs(fd_weight - analytic_weight) < 2e-6


def test_memory_reader_query_chunk_changes_execution_not_semantics():
    cfg_a = _tiny_config(top_k=2)
    memory_b = dataclasses.replace(cfg_a.memory_context, read_query_chunk=1)
    cfg_b = dataclasses.replace(cfg_a, memory_context=memory_b)
    model_a = DecoderLanguageModel(cfg_a, rng_seed=27)
    model_b = DecoderLanguageModel(cfg_b, rng_seed=27)
    source = xp.asarray([[1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11]], dtype=xp.int64)

    active_a, _ = model_a._hierarchical_context_forward(source)
    active_b, _ = model_b._hierarchical_context_forward(source)
    np.testing.assert_array_equal(
        asnumpy(active_a.selected_blocks), asnumpy(active_b.selected_blocks)
    )
    np.testing.assert_allclose(
        asnumpy(active_a.route_weights), asnumpy(active_b.route_weights),
        rtol=1e-7, atol=1e-8,
    )
    np.testing.assert_allclose(
        asnumpy(active_a.embeddings), asnumpy(active_b.embeddings),
        rtol=1e-6, atol=1e-7,
    )


def test_chunked_lm_head_projects_dense_window_and_scatters_backward():
    model = DecoderLanguageModel(_tiny_config(), rng_seed=9)
    rng = np.random.default_rng(4)
    hidden = xp.asarray(rng.normal(size=(2, 3, 8)), dtype="float32")
    targets = xp.asarray(rng.integers(0, 32, size=(2, 3)), dtype=xp.int32)

    loss, cache = model.chunked_lm_head_loss_forward(
        hidden,
        targets,
        chunk_tokens=2,
        target_slice=(0, 3),
        return_device_loss=False,
    )
    assert np.isfinite(loss)
    dx = model.chunked_lm_head_backward(cache)
    assert dx.shape == hidden.shape
    assert np.max(np.abs(asnumpy(dx))) > 0.0


def test_sliding_mean_query_pool_matches_explicit_windows_and_backward():
    rng = np.random.default_rng(7)
    x = xp.asarray(rng.normal(size=(1, 9, 4)), dtype="float64")
    starts = xp.asarray([4, 5, 7, 9], dtype=xp.int64)
    pool = CausalQueryPooler(4, 4, strategy="mean")
    pooled, cache = pool.forward(x, starts)
    expected = np.stack([
        asnumpy(x)[0, max(0, int(s) - 4):int(s)].mean(axis=0)
        for s in asnumpy(starts)
    ])[None, :, :]
    np.testing.assert_allclose(asnumpy(pooled), expected, rtol=1e-12, atol=1e-12)

    dy = xp.asarray(rng.normal(size=pooled.shape), dtype="float64")
    dx = pool.backward(dy, cache)
    expected_dx = np.zeros((1, 9, 4), dtype=np.float64)
    for r, s in enumerate(asnumpy(starts)):
        lo = max(0, int(s) - 4)
        expected_dx[0, lo:int(s)] += asnumpy(dy)[0, r] / (int(s) - lo)
    np.testing.assert_allclose(asnumpy(dx), expected_dx, rtol=1e-12, atol=1e-12)


def test_hierarchical_router_backward_matches_finite_difference():
    config = MemoryContextConfig(
        enabled=True,
        memory_length=10,
        recent_length=4,
        target_length=2,
        block_size=2,
        top_k_blocks=2,
        router_query_length=4,
        router_dim=3,
        query_pooling="mean",
        history_pooling="mean",
        read_heads=1,
        read_kv_heads=1,
        read_query_chunk=2,
    )
    router = HierarchicalMemoryRouter(
        d_model=4,
        config=config,
        rng=RandomStream(3),
        input_std=0.05,
        dtype="float64",
    )
    rng = np.random.default_rng(8)
    history = xp.asarray(rng.normal(size=(1, 10, 4)), dtype="float64")
    query_source = xp.asarray(rng.normal(size=(1, 6, 4)), dtype="float64")
    starts = xp.asarray([5, 6], dtype=xp.int64)
    cutoffs = xp.asarray([6, 8], dtype=xp.int64)
    coeff = xp.asarray([[[0.3, -0.7], [-0.2, 0.4]]], dtype="float64")

    weights, _, _, _, _, cache = router.forward(history, query_source, starts, cutoffs)
    router.zero_grad()
    router.backward(coeff, cache)
    analytic = float(router.W_query.grad[0, 0])

    original = float(router.W_query.data[0, 0])
    eps = 1e-6

    def objective(value):
        router.W_query.data[0, 0] = value
        w, _, _, _, _, _ = router.forward(history, query_source, starts, cutoffs)
        return float(xp.sum(w * coeff))

    plus = objective(original + eps)
    minus = objective(original - eps)
    router.W_query.data[0, 0] = original
    fd = (plus - minus) / (2 * eps)
    assert abs(fd - analytic) < 1e-6


def _run_tiny_training_backward(model, source, targets, checkpoint):
    model.zero_grad()
    hidden, cache = model.forward_body(
        source, return_cache=True, activation_checkpoint=checkpoint
    )
    loss, loss_cache = model.chunked_lm_head_loss_forward(
        hidden,
        targets,
        chunk_tokens=2,
        target_slice=cache["target_slice"],
        return_device_loss=False,
    )
    dx = model.chunked_lm_head_backward(loss_cache)
    model.backward_body(dx, cache)
    grads = {p.name: asnumpy(p.grad).copy() for p in model.parameters()}
    return loss, grads


def test_tiny_end_to_end_memory_reader_router_gradients_and_checkpoint_replay_match():
    cfg = _tiny_config(top_k=2)
    source = xp.asarray([[1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11]], dtype=xp.int64)
    targets = xp.asarray([[10, 11, 12]], dtype=xp.int32)
    reference = DecoderLanguageModel(cfg, rng_seed=21)
    checkpointed = DecoderLanguageModel(cfg, rng_seed=21)

    loss_a, grads_a = _run_tiny_training_backward(reference, source, targets, False)
    loss_b, grads_b = _run_tiny_training_backward(checkpointed, source, targets, True)
    assert abs(loss_a - loss_b) < 1e-7
    for name in (
        "memory_router.W_query",
        "memory_router.W_history",
        "memory_reader.W_q",
        "memory_reader.W_k",
        "memory_reader.W_v",
        "memory_reader.W_o",
    ):
        assert np.linalg.norm(grads_a[name]) > 0.0
    for name in grads_a:
        np.testing.assert_allclose(grads_a[name], grads_b[name], rtol=3e-5, atol=3e-6)


def test_memory_parameters_roundtrip_checkpoint(tmp_path):
    model = DecoderLanguageModel(_tiny_config(), rng_seed=31)
    params = {p.name: p.data for p in model.parameters()}
    save_checkpoint(tmp_path, params)
    loaded, _, _ = load_checkpoint(
        tmp_path, param_names=[p.name for p in model.parameters()], skip_optimizer=True
    )
    for name in (
        "memory_router.W_query",
        "memory_router.W_history",
        "memory_reader.W_q",
        "memory_reader.W_k",
        "memory_reader.W_v",
        "memory_reader.W_o",
    ):
        assert name in loaded
        np.testing.assert_allclose(asnumpy(loaded[name]), asnumpy(params[name]))



def test_document_aware_sampler_keeps_history_and_targets_inside_one_document():
    eos = 0
    # Distinct value ranges make accidental cross-document contamination obvious.
    packed = np.asarray(
        [10, 11, 12, 13, 14, eos,
         20, 21, 22, 23, 24, 25, 26, eos,
         30, 31, 32, 33, 34, 35, 36, 37, eos],
        dtype=np.uint16,
    )
    index = build_packed_document_index(packed, eos)
    source, labels, metadata = create_hierarchical_memory_minibatch(
        packed,
        batch_size=32,
        memory_length=8,
        target_length=4,
        rng=np.random.default_rng(123),
        eos_token_id=eos,
        document_index=index,
        document_aware=True,
        return_metadata=True,
    )

    for b in range(source.shape[0]):
        doc_start = int(metadata["document_starts"][b])
        doc_end = int(metadata["document_ends"][b])
        target_start = int(metadata["target_source_starts"][b])
        valid_targets = int(metadata["target_valid_lengths"][b])
        history_valid_start = int(metadata["history_valid_starts"][b])
        expected_history_start = max(doc_start, target_start - 8)
        expected_history = packed[expected_history_start:target_start]

        np.testing.assert_array_equal(
            source[b, history_valid_start:8], expected_history
        )
        np.testing.assert_array_equal(
            source[b, 8:8 + valid_targets],
            packed[target_start:target_start + valid_targets],
        )
        np.testing.assert_array_equal(
            labels[b, :valid_targets],
            packed[target_start + 1:target_start + valid_targets + 1],
        )
        assert doc_start <= target_start < doc_end
        assert target_start + valid_targets < doc_end + 1
        assert np.all(metadata["target_loss_mask"][b, :valid_targets] == 1.0)
        assert np.all(metadata["target_loss_mask"][b, valid_targets:] == 0.0)


def test_short_document_uses_only_available_history_and_masks_padding_loss():
    eos = 0
    packed = np.asarray([41, 42, 43, eos], dtype=np.uint16)
    index = build_packed_document_index(packed, eos)
    source, labels, metadata = create_hierarchical_memory_minibatch(
        packed,
        batch_size=1,
        memory_length=8,
        target_length=4,
        rng=np.random.default_rng(7),
        eos_token_id=eos,
        document_index=index,
        document_aware=True,
        return_metadata=True,
    )
    assert int(metadata["history_valid_starts"][0]) == 8
    assert int(metadata["target_valid_lengths"][0]) == 3
    np.testing.assert_array_equal(source[0, 8:12], np.array([41, 42, 43, eos]))
    np.testing.assert_array_equal(labels[0], np.array([42, 43, eos, eos]))
    np.testing.assert_array_equal(
        metadata["target_loss_mask"][0], np.array([1.0, 1.0, 1.0, 0.0])
    )


def test_document_start_mask_prevents_padded_history_from_being_retrieved():
    memory = MemoryContextConfig(
        enabled=True,
        memory_length=8,
        recent_length=4,
        target_length=4,
        block_size=2,
        top_k_blocks=2,
        router_query_length=4,
        router_dim=4,
        query_pooling="mean",
        history_pooling="mean",
        read_heads=2,
        read_kv_heads=1,
        read_query_chunk=2,
    )
    cfg = dataclasses.replace(
        _tiny_config(top_k=1), context_length=4, memory_context=memory
    )
    model = DecoderLanguageModel(cfg, rng_seed=31)

    # Only two real pre-target tokens exist; the first six history slots are
    # left padding and must never become memory candidates.
    source = xp.asarray([[0, 0, 0, 0, 0, 0, 9, 10, 11, 12, 13, 14]], dtype=xp.int64)
    metadata = {"history_valid_starts": np.asarray([6], dtype=np.int64)}
    active, _ = model._hierarchical_context_forward(
        source, memory_metadata=metadata
    )
    valid = asnumpy(active.selected_valid)[0]
    selected = asnumpy(active.selected_blocks)[0]
    # The real block [6:8] may become old enough by the final route, but no
    # block touching the left-padded region [0:6] is ever eligible.
    assert not bool(np.any(valid[:3]))
    assert bool(np.any(valid[3]))
    assert np.all(selected[valid] == 3)


def test_causal_query_pool_mean_ignores_left_padding_per_batch():
    pool = CausalQueryPooler(
        d_model=1,
        query_window=4,
        strategy="mean",
        rng=RandomStream(1),
        dtype="float64",
    )
    x = xp.asarray(
        [
            [[100.0], [100.0], [2.0], [4.0], [6.0]],
            [[100.0], [1.0], [3.0], [5.0], [7.0]],
        ],
        dtype="float64",
    )
    route_starts = xp.asarray([3, 5], dtype=xp.int64)
    pooled, cache = pool.forward(
        x,
        route_starts,
        valid_starts=xp.asarray([2, 1], dtype=xp.int64),
    )
    expected = np.array(
        [
            [[2.0], [4.0]],          # [2], then mean([2,4,6])
            [[2.0], [4.0]],          # mean([1,3]), then mean([1,3,5,7])
        ]
    )
    np.testing.assert_allclose(asnumpy(pooled), expected, rtol=0, atol=1e-12)

    dpooled = xp.ones_like(pooled)
    dx = asnumpy(pool.backward(dpooled, cache))
    # Padding rows receive no query-pool gradient.
    np.testing.assert_allclose(dx[0, :2], 0.0, rtol=0, atol=0)
    np.testing.assert_allclose(dx[1, :1], 0.0, rtol=0, atol=0)
