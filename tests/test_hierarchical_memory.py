import dataclasses
import numpy as np

from mini_llm.backend import RandomStream, asnumpy, xp
from mini_llm.checkpoint import (
    initialize_matching_parameters, load_checkpoint, save_checkpoint,
)
from mini_llm.config import (
    MemoryContextConfig, ModelConfig, AttentionLayerConfig,
    LocalAttentionConfig, RetrievalAttentionConfig, ContextRouterConfig,
)
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
    assert {cfg.memory_context.integration_mode for cfg in configs} == {"routed_prefix"}
    assert {cfg.memory_context.active_length for cfg in configs} == {6144}
    assert {cfg.context_length for cfg in configs} == {6144}
    assert [cfg.memory_context.memory_length for cfg in configs] == [16384, 32768, 65536]
    assert [cfg.memory_context.searchable_blocks for cfg in configs] == [96, 224, 480]
    assert {cfg.memory_context.target_length for cfg in configs} == {4096}
    assert {cfg.memory_context.retrieved_length for cfg in configs} == {2048}
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



def _tiny_routed_prefix_config(top_k_blocks=1, surrogate_scale=1.0):
    memory = MemoryContextConfig(
        enabled=True,
        memory_length=8,
        recent_length=4,
        target_length=4,
        block_size=2,
        top_k_blocks=top_k_blocks,
        router_query_length=4,
        router_dim=3,
        query_pooling="mean",
        history_pooling="mean",
        integration_mode="routed_prefix",
        memory_training="joint",
        router_surrogate_scale=surrogate_scale,
        router_gumbel_noise=True,
        retrieval_batch_probability=1.0,
        read_heads=1,
        read_kv_heads=1,
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
        dtype="float64",
        memory_context=memory,
    )


def test_routed_prefix_reopens_all_history_directly_when_it_fits_budget():
    model = DecoderLanguageModel(
        _tiny_routed_prefix_config(top_k_blocks=2), rng_seed=41
    )
    source = xp.asarray([[1, 2, 3, 4, 5, 6, 7, 8]], dtype=xp.int64)
    active, cache = model._hierarchical_context_forward(
        source, memory_metadata=_terminal_metadata()
    )
    assert cache["mode"] == "routed_prefix"
    assert cache["retrieval_active"] is True
    assert cache["prefix_mode"] == "direct_history"
    assert cache["router_cache"] is None
    np.testing.assert_array_equal(asnumpy(active.token_ids), asnumpy(source))
    np.testing.assert_array_equal(asnumpy(active.position_ids)[0], np.arange(8))
    assert active.embeddings.shape == (1, 8, 8)
    assert active.target_start == 4
    assert active.target_end == 8


def test_routed_prefix_can_train_plain_dense_window_without_retrieval():
    model = DecoderLanguageModel(_tiny_routed_prefix_config(), rng_seed=43)
    source = xp.asarray([[1, 2, 3, 4, 5, 6, 7, 8]], dtype=xp.int64)
    metadata = _terminal_metadata()
    metadata["use_retrieval"] = False
    active, cache = model._hierarchical_context_forward(source, memory_metadata=metadata)
    assert cache["retrieval_active"] is False
    assert active.embeddings.shape == (1, 4, 8)
    np.testing.assert_array_equal(asnumpy(active.token_ids), np.array([[5, 6, 7, 8]]))
    np.testing.assert_array_equal(asnumpy(active.position_ids), np.array([[4, 5, 6, 7]]))
    assert active.target_start == 0
    assert active.target_end == 4


def test_routed_prefix_gumbel_selection_is_reproducible_from_seed():
    model = DecoderLanguageModel(_tiny_routed_prefix_config(), rng_seed=47)
    # Equalize router scores so selection is controlled entirely by Gumbel noise.
    model.memory_router.W_query.data[...] = 0.0
    model.memory_router.W_history.data[...] = 0.0
    source = xp.asarray([[1, 2, 3, 4, 5, 6, 7, 8]], dtype=xp.int64)
    meta = _terminal_metadata()
    meta.update({
        "router_stochastic": True,
        "router_gumbel_seed": 12345,
        "router_temperature": 0.7,
    })
    a, _ = model._hierarchical_context_forward(source, memory_metadata=meta)
    b, _ = model._hierarchical_context_forward(source, memory_metadata=meta)
    np.testing.assert_array_equal(asnumpy(a.selected_blocks), asnumpy(b.selected_blocks))


def test_routed_prefix_st_surrogate_reaches_unselected_history_blocks():
    model = DecoderLanguageModel(_tiny_routed_prefix_config(), rng_seed=53)
    router = model.memory_router
    history = xp.asarray(
        np.arange(32, dtype=np.float64).reshape(1, 4, 8) / 17.0
    )
    recent = xp.asarray(
        np.arange(32, 64, dtype=np.float64).reshape(1, 4, 8) / 19.0
    )
    _, selected, _, _, cache = router.forward_routed_prefix(
        history, recent, stochastic=False, temperature=1.0
    )
    block = int(asnumpy(selected)[0, 0])
    selected_embeddings = history[:, block * 2:block * 2 + 2, :]
    dselected = xp.ones_like(selected_embeddings)
    router.zero_grad()
    dhistory, _ = router.backward_routed_prefix(
        dselected, selected_embeddings, cache
    )
    block_norms = [
        np.linalg.norm(asnumpy(dhistory[:, i * 2:(i + 1) * 2, :]))
        for i in range(2)
    ]
    assert block_norms[block] > 0.0
    assert block_norms[1 - block] > 0.0


def test_routed_prefix_surrogate_is_forward_identity_but_lm_trains_router():
    source = xp.asarray([[1, 2, 3, 4, 5, 6, 7, 8]], dtype=xp.int64)
    targets = xp.asarray([[6, 7, 8, 9]], dtype=xp.int64)
    meta = _terminal_metadata()
    with_st = DecoderLanguageModel(
        _tiny_routed_prefix_config(surrogate_scale=1.0), rng_seed=59
    )
    no_st = DecoderLanguageModel(
        _tiny_routed_prefix_config(surrogate_scale=0.0), rng_seed=59
    )
    hidden_a, _ = with_st.forward_body(source, memory_metadata=meta, return_cache=True)
    hidden_b, _ = no_st.forward_body(source, memory_metadata=meta, return_cache=True)
    np.testing.assert_allclose(asnumpy(hidden_a), asnumpy(hidden_b), rtol=0, atol=0)

    logits, cache = with_st.forward(source, memory_metadata=meta, return_cache=True)
    loss, loss_cache = with_st.compute_loss(logits, targets)
    with_st.zero_grad()
    dlogits = with_st.backward_loss(loss_cache)
    with_st.backward(dlogits, cache)
    assert np.linalg.norm(asnumpy(with_st.memory_router.W_query.grad)) > 0.0
    assert np.linalg.norm(asnumpy(with_st.memory_router.W_history.grad)) > 0.0

def _tiny_terminal_landmark_config(memory_training="joint", top_k_blocks=1):
    layer_router = ContextRouterConfig(
        history_block_size=2,
        routing_stride=1,
        query_window=2,
        router_dim=2,
        top_k_blocks=1,
        exclude_recent_tokens=2,
        query_pooling="mean",
        history_pooling="mean",
        num_queries=1,
    )
    attention = AttentionLayerConfig(
        heads=(
            LocalAttentionConfig(window=4),
            RetrievalAttentionConfig(layer_router, group="far"),
        )
    )
    memory = MemoryContextConfig(
        enabled=True,
        memory_length=8,
        recent_length=4,
        target_length=4,
        block_size=2,
        top_k_blocks=top_k_blocks,
        router_query_length=4,
        router_dim=3,
        query_pooling="mean",
        history_pooling="mean",
        integration_mode="terminal_landmark",
        memory_attention_layer=-1,
        memory_training=memory_training,
        read_heads=1,
        read_kv_heads=1,
    )
    return ModelConfig(
        tokenizer_vocab_size=32,
        context_length=4,
        n_layers=1,
        d_model=8,
        n_q_heads=2,
        n_kv_heads=1,
        d_head=4,
        n_experts=2,
        top_k=1,
        d_ff=16,
        dtype="float64",
        attention_layers=(attention,),
        memory_context=memory,
    )


def _terminal_metadata(history_start=0, valid=4):
    return {
        "history_valid_starts": np.asarray([history_start], dtype=np.int64),
        "target_valid_lengths": np.asarray([valid], dtype=np.int64),
        "target_loss_mask": np.asarray([[1.0] * valid + [0.0] * (4 - valid)], dtype=np.float32),
    }


def test_routed_prefix_preset_memory_length_is_total_horizon():
    cfg = ModelConfig.wide_500m_memory_64k().memory_context
    assert cfg.integration_mode == "routed_prefix"
    assert cfg.memory_length == 65536
    assert cfg.distant_memory_length == 61440
    assert cfg.source_input_length == 65536
    assert cfg.retrieved_length == 2048
    assert cfg.active_length == 6144
    assert cfg.searchable_blocks == 480


def test_terminal_landmark_router_runs_once_and_reopens_exact_blocks():
    model = DecoderLanguageModel(_tiny_terminal_landmark_config(), rng_seed=7)
    source = xp.asarray([[1, 2, 3, 4, 5, 6, 7, 8]], dtype=xp.int64)
    active, cache = model._hierarchical_context_forward(
        source, memory_metadata=_terminal_metadata()
    )
    mem = active.terminal_memory
    assert active.embeddings.shape == (1, 4, 8)
    assert mem.selected_blocks.shape == (1, 1)
    assert mem.gate_scores.shape == (1, 1)
    assert mem.selected_embeddings.shape == (1, 2, 8)
    block = int(asnumpy(mem.selected_blocks)[0, 0])
    expected_ids = asnumpy(source)[0, block * 2:block * 2 + 2]
    np.testing.assert_array_equal(asnumpy(mem.selected_token_ids)[0], expected_ids)
    assert cache["mode"] == "terminal_landmark"


def test_terminal_landmark_only_terminal_loss_trains_memory_system():
    cfg = _tiny_terminal_landmark_config()
    model = DecoderLanguageModel(cfg, rng_seed=9)
    source = xp.asarray([[1, 2, 3, 4, 5, 6, 7, 8]], dtype=xp.int64)
    targets = xp.asarray([[6, 7, 8, 9]], dtype=xp.int32)

    hidden, cache = model.forward_body(
        source, memory_metadata=_terminal_metadata(), return_cache=True
    )
    # Exclude the terminal next-token label: the routed memory has no path to
    # any earlier prediction and every 0058C memory gradient must be zero.
    loss, loss_cache = model.chunked_lm_head_loss_forward(
        hidden, targets, chunk_tokens=2,
        loss_mask=xp.asarray([[1, 1, 1, 0]], dtype=xp.float32),
        target_slice=cache["target_slice"],
    )
    model.zero_grad()
    dx = model.chunked_lm_head_backward(loss_cache)
    model.backward_body(dx, cache)
    memory_grads = [
        asnumpy(p.grad) for p in model.parameters()
        if p.name.startswith("memory_router.") or ".terminal_memory." in p.name
    ]
    assert memory_grads
    assert all(np.count_nonzero(g) == 0 for g in memory_grads)

    # Restoring only the terminal target produces a non-zero retrieval gradient.
    hidden, cache = model.forward_body(
        source, memory_metadata=_terminal_metadata(), return_cache=True
    )
    loss, loss_cache = model.chunked_lm_head_loss_forward(
        hidden, targets, chunk_tokens=2,
        loss_mask=xp.asarray([[0, 0, 0, 1]], dtype=xp.float32),
        target_slice=cache["target_slice"],
    )
    model.zero_grad()
    dx = model.chunked_lm_head_backward(loss_cache)
    model.backward_body(dx, cache)
    norms = {
        p.name: np.linalg.norm(asnumpy(p.grad)) for p in model.parameters()
        if p.name.startswith("memory_router.") or ".terminal_memory." in p.name
    }
    assert norms["memory_router.W_query"] > 0.0
    assert norms["blocks.0.attention.terminal_memory.W_v"] > 0.0


def test_terminal_landmark_unselected_history_token_does_not_change_output():
    cfg = _tiny_terminal_landmark_config()
    model = DecoderLanguageModel(cfg, rng_seed=13)
    source = xp.asarray([[1, 2, 3, 4, 5, 6, 7, 8]], dtype=xp.int64)
    meta = _terminal_metadata()
    hidden_a, cache_a = model.forward_body(source, memory_metadata=meta, return_cache=True)
    selected = int(asnumpy(cache_a["memory_context_cache"]["terminal_memory"].selected_blocks)[0, 0])
    unselected = 1 - selected
    changed = source.copy()
    changed[:, unselected * 2] = 15
    hidden_b, _ = model.forward_body(changed, memory_metadata=meta, return_cache=True)
    # Only the router block summaries could alter the selected identity. Choose
    # a tiny perturbation test only when top-1 remains stable.
    _, cache_b = model.forward_body(changed, memory_metadata=meta, return_cache=True)
    selected_b = int(asnumpy(cache_b["memory_context_cache"]["terminal_memory"].selected_blocks)[0, 0])
    if selected_b == selected:
        np.testing.assert_allclose(
            asnumpy(hidden_a[:, -1]), asnumpy(hidden_b[:, -1]), rtol=1e-12, atol=1e-12
        )


def test_terminal_landmark_checkpoint_replay_matches_full_cache():
    cfg = _tiny_terminal_landmark_config()
    source = xp.asarray([[1, 2, 3, 4, 5, 6, 7, 8]], dtype=xp.int64)
    targets = xp.asarray([[6, 7, 8, 9]], dtype=xp.int32)
    meta = _terminal_metadata()
    a = DecoderLanguageModel(cfg, rng_seed=21)
    b = DecoderLanguageModel(cfg, rng_seed=21)

    def run(model, checkpoint):
        model.zero_grad()
        h, cache = model.forward_body(
            source, memory_metadata=meta, return_cache=True,
            activation_checkpoint=checkpoint,
        )
        loss, lc = model.chunked_lm_head_loss_forward(
            h, targets, chunk_tokens=2, target_slice=cache["target_slice"]
        )
        dx = model.chunked_lm_head_backward(lc)
        model.backward_body(dx, cache)
        return loss, {p.name: asnumpy(p.grad).copy() for p in model.parameters()}

    la, ga = run(a, False)
    lb, gb = run(b, True)
    assert abs(la - lb) < 1e-10
    for name in ga:
        np.testing.assert_allclose(ga[name], gb[name], rtol=2e-8, atol=2e-9)



def test_terminal_landmark_selected_history_token_changes_terminal_output():
    cfg = _tiny_terminal_landmark_config()
    model = DecoderLanguageModel(cfg, rng_seed=17)
    source = xp.asarray([[1, 2, 3, 4, 5, 6, 7, 8]], dtype=xp.int64)
    meta = _terminal_metadata()
    hidden_a, cache_a = model.forward_body(source, memory_metadata=meta, return_cache=True)
    selected = int(asnumpy(cache_a["memory_context_cache"]["terminal_memory"].selected_blocks)[0, 0])
    changed = source.copy()
    changed[:, selected * 2] = 15
    hidden_b, cache_b = model.forward_body(changed, memory_metadata=meta, return_cache=True)
    selected_b = int(asnumpy(cache_b["memory_context_cache"]["terminal_memory"].selected_blocks)[0, 0])
    if selected_b == selected:
        assert not np.allclose(
            asnumpy(hidden_a[:, -1]), asnumpy(hidden_b[:, -1]), rtol=1e-10, atol=1e-12
        )


def test_terminal_landmark_router_gate_gradient_matches_finite_difference():
    # Select every old block so the finite difference never crosses a discrete
    # top-k identity boundary. This isolates the differentiable Landmark gate.
    cfg = _tiny_terminal_landmark_config(top_k_blocks=2)
    model = DecoderLanguageModel(cfg, rng_seed=23)
    source = xp.asarray([[1, 2, 3, 4, 5, 6, 7, 8]], dtype=xp.int64)
    targets = xp.asarray([[6, 7, 8, 9]], dtype=xp.int32)
    mask = xp.asarray([[0, 0, 0, 1]], dtype=xp.float32)
    meta = _terminal_metadata()

    model.zero_grad()
    logits, cache = model.forward(source, memory_metadata=meta, return_cache=True)
    loss, lc = model.compute_loss(logits, targets, loss_mask=mask)
    model.backward(model.backward_loss(lc), cache)
    p = next(p for p in model.parameters() if p.name == "memory_router.W_query")
    analytic = float(asnumpy(p.grad)[0, 0])

    original = float(asnumpy(p.data)[0, 0])
    eps = 1e-6
    losses = []
    for delta in (+eps, -eps):
        p.data[0, 0] = original + delta
        logits = model.forward(source, memory_metadata=meta, return_cache=False)
        value, _ = model.compute_loss(logits, targets, loss_mask=mask)
        losses.append(float(value))
    p.data[0, 0] = original
    numerical = (losses[0] - losses[1]) / (2.0 * eps)
    np.testing.assert_allclose(analytic, numerical, rtol=2e-3, atol=2e-7)


def test_router_only_projects_one_terminal_row_and_freezes_lm_head_gradient():
    cfg = _tiny_terminal_landmark_config(memory_training="router_only")
    model = DecoderLanguageModel(cfg, rng_seed=29)
    source = xp.asarray([[1, 2, 3, 4, 5, 6, 7, 8]], dtype=xp.int64)
    meta = _terminal_metadata()
    logits, cache = model.forward(source, memory_metadata=meta, return_cache=True)
    assert logits.shape == (1, 1, cfg.tokenizer_vocab_size)
    targets = xp.asarray([[9]], dtype=xp.int32)
    model.zero_grad()
    _, lc = model.compute_loss(logits, targets)
    model.backward(model.backward_loss(lc), cache)
    assert np.count_nonzero(asnumpy(model.output_proj.W.grad)) == 0
    optimized_names = {p.name for p in model.optimization_parameters()}
    assert optimized_names
    assert all(
        name.startswith("memory_router.") or ".terminal_memory." in name
        for name in optimized_names
    )


def test_document_sampler_can_require_real_old_history_for_router_only():
    eos = 99
    # One document has enough room for a 4-token working window plus >=4 old
    # tokens. The short document must not be chosen when min_history_tokens=4.
    shard = np.asarray([1, 2, 3, eos, 10, 11, 12, 13, 14, 15, 16, 17, 18, eos])
    index = build_packed_document_index(shard, eos)
    inputs, labels, meta = create_hierarchical_memory_minibatch(
        shard, batch_size=4, memory_length=6, target_length=4,
        rng=np.random.default_rng(5), eos_token_id=eos, document_index=index,
        document_aware=True, return_metadata=True, min_history_tokens=4,
    )
    assert np.all(meta["target_source_starts"] >= 8)
    assert np.all(meta["target_valid_lengths"] == 4)
    assert np.all(meta["history_valid_starts"] <= 2)



def test_terminal_landmark_memory_value_gradient_matches_finite_difference():
    cfg = _tiny_terminal_landmark_config(top_k_blocks=2)
    model = DecoderLanguageModel(cfg, rng_seed=37)
    source = xp.asarray([[1, 2, 3, 4, 5, 6, 7, 8]], dtype=xp.int64)
    targets = xp.asarray([[6, 7, 8, 9]], dtype=xp.int32)
    mask = xp.asarray([[0, 0, 0, 1]], dtype=xp.float32)
    meta = _terminal_metadata()

    model.zero_grad()
    logits, cache = model.forward(source, memory_metadata=meta, return_cache=True)
    _, lc = model.compute_loss(logits, targets, loss_mask=mask)
    model.backward(model.backward_loss(lc), cache)
    p = next(p for p in model.parameters() if p.name.endswith("terminal_memory.W_v"))
    analytic = float(asnumpy(p.grad)[0, 0])

    original = float(asnumpy(p.data)[0, 0])
    eps = 1e-6
    losses = []
    for delta in (+eps, -eps):
        p.data[0, 0] = original + delta
        logits = model.forward(source, memory_metadata=meta, return_cache=False)
        value, _ = model.compute_loss(logits, targets, loss_mask=mask)
        losses.append(float(value))
    p.data[0, 0] = original
    numerical = (losses[0] - losses[1]) / (2.0 * eps)
    np.testing.assert_allclose(analytic, numerical, rtol=2e-3, atol=2e-7)



def test_terminal_landmark_parameters_roundtrip_checkpoint(tmp_path):
    model = DecoderLanguageModel(_tiny_terminal_landmark_config(), rng_seed=41)
    params = {p.name: p.data for p in model.parameters()}
    save_checkpoint(tmp_path, params)
    loaded, _, _ = load_checkpoint(
        tmp_path, param_names=[p.name for p in model.parameters()], skip_optimizer=True
    )
    names = [
        "memory_router.W_query",
        "memory_router.W_history",
        "blocks.0.attention.terminal_memory.norm.gamma",
        "blocks.0.attention.terminal_memory.W_k",
        "blocks.0.attention.terminal_memory.W_v",
    ]
    for name in names:
        assert name in loaded
        np.testing.assert_allclose(asnumpy(loaded[name]), asnumpy(params[name]))



def test_terminal_memory_initialization_preserves_backbone_rng_sequence():
    mem_cfg = _tiny_terminal_landmark_config()
    disabled = dataclasses.replace(mem_cfg.memory_context, enabled=False)
    base_cfg = dataclasses.replace(mem_cfg, memory_context=disabled)
    memory_model = DecoderLanguageModel(mem_cfg, rng_seed=53)
    base_model = DecoderLanguageModel(base_cfg, rng_seed=53)
    base_params = {p.name: asnumpy(p.data) for p in base_model.parameters()}
    for p in memory_model.parameters():
        if p.name in base_params:
            np.testing.assert_array_equal(asnumpy(p.data), base_params[p.name])


def test_router_only_retains_backward_state_only_from_memory_layer():
    one = _tiny_terminal_landmark_config(memory_training="router_only")
    two = dataclasses.replace(
        one,
        n_layers=2,
        attention_layers=(one.attention_layers[0], one.attention_layers[0]),
        memory_context=dataclasses.replace(
            one.memory_context, memory_attention_layer=-1
        ),
    )
    model = DecoderLanguageModel(two, rng_seed=59)
    source = xp.asarray([[1, 2, 3, 4, 5, 6, 7, 8]], dtype=xp.int64)

    _, cache = model.forward_body(
        source, memory_metadata=_terminal_metadata(), return_cache=True,
        activation_checkpoint=False,
    )
    assert cache["block_cache_start_layer"] == 1
    assert len(cache["block_caches"]) == 1

    _, cache = model.forward_body(
        source, memory_metadata=_terminal_metadata(), return_cache=True,
        activation_checkpoint=True,
    )
    assert cache["block_cache_start_layer"] == 1
    assert len(cache["block_checkpoints"]) == 1


def test_terminal_landmark_grouped_probabilities_are_normalized_and_diagnostic():
    cfg = _tiny_terminal_landmark_config(top_k_blocks=2)
    model = DecoderLanguageModel(cfg, rng_seed=61)
    source = xp.asarray([[1, 2, 3, 4, 5, 6, 7, 8]], dtype=xp.int64)
    _, cache = model.forward_body(
        source, memory_metadata=_terminal_metadata(), return_cache=True
    )
    mem = cache["memory_context_cache"]["terminal_memory"]
    assert mem is not None
    block_probs = asnumpy(mem.attention_block_probs)
    history_mass = asnumpy(mem.attention_history_mass)
    assert np.all(history_mass >= 0.0)
    assert np.all(history_mass <= 1.0 + 1e-12)
    np.testing.assert_allclose(
        block_probs.sum(axis=-1), history_mass, rtol=1e-12, atol=1e-12
    )
    diagnostics = model.memory_routing_diagnostics(cache)
    assert "history_attention_mass_mean" in diagnostics
    assert "max_block_attention_mean" in diagnostics
    assert "max_within_block_token_prob_mean" in diagnostics


def test_router_only_two_layer_backward_works_with_truncated_cache():
    one = _tiny_terminal_landmark_config(memory_training="router_only", top_k_blocks=2)
    two = dataclasses.replace(
        one,
        n_layers=2,
        attention_layers=(one.attention_layers[0], one.attention_layers[0]),
        memory_context=dataclasses.replace(one.memory_context, memory_attention_layer=-1),
    )
    model = DecoderLanguageModel(two, rng_seed=67)
    source = xp.asarray([[1, 2, 3, 4, 5, 6, 7, 8]], dtype=xp.int64)
    target = xp.asarray([[9]], dtype=xp.int32)
    model.zero_grad()
    logits, cache = model.forward(
        source, memory_metadata=_terminal_metadata(), return_cache=True
    )
    assert cache["block_cache_start_layer"] == 1
    assert len(cache["block_caches"]) == 1
    loss, loss_cache = model.compute_loss(logits, target)
    model.backward(model.backward_loss(loss_cache), cache)
    assert np.isfinite(float(loss))
    assert np.linalg.norm(asnumpy(next(
        p for p in model.parameters() if p.name == "memory_router.W_query"
    ).grad)) > 0.0


def test_backbone_checkpoint_streams_into_terminal_memory_model(tmp_path):
    mem_cfg = _tiny_terminal_landmark_config()
    base_cfg = dataclasses.replace(
        mem_cfg, memory_context=dataclasses.replace(mem_cfg.memory_context, enabled=False)
    )
    base = DecoderLanguageModel(base_cfg, rng_seed=71)
    save_checkpoint(tmp_path, {p.name: p.data for p in base.parameters()})
    memory = DecoderLanguageModel(mem_cfg, rng_seed=72)
    loaded, missing, _ = initialize_matching_parameters(tmp_path, memory.parameters())
    expected_missing = {
        p.name for p in memory.parameters()
        if p.name.startswith("memory_router.") or ".terminal_memory." in p.name
    }
    assert set(missing) == expected_missing
    assert set(loaded) == {p.name for p in base.parameters()}
    base_values = {p.name: asnumpy(p.data) for p in base.parameters()}
    for p in memory.parameters():
        if p.name in base_values:
            np.testing.assert_array_equal(asnumpy(p.data), base_values[p.name])


def test_document_sampler_returns_shorter_same_document_history_when_needed():
    """Short packed shards keep their genuine history instead of aborting."""
    eos = 99
    # Longest document has six prediction pairs. A 4-token target is valid,
    # but target(4) + requested history(4) cannot both be satisfied.
    shard = np.asarray([
        1, 2, 3, eos,
        10, 11, 12, 13, 14, 15, eos,
        20, 21, eos,
    ])
    index = build_packed_document_index(shard, eos)
    inputs, labels, meta = create_hierarchical_memory_minibatch(
        shard,
        batch_size=2,
        memory_length=8,
        target_length=4,
        rng=np.random.default_rng(17),
        eos_token_id=eos,
        document_index=index,
        document_aware=True,
        return_metadata=True,
        min_history_tokens=4,
    )

    assert inputs.shape == (2, 12)
    assert labels.shape == (2, 4)
    assert np.all(meta["target_valid_lengths"] == 4)
    assert not np.any(meta["min_history_satisfied"])
    # The fallback still stays inside one EOS-bounded document.
    assert np.all(meta["document_starts"] == 4)
    assert np.all(meta["document_ends"] == 11)


def test_routed_prefix_uses_all_available_short_history_without_router():
    """A 4k+epsilon document should prepend every real old token directly."""
    model = DecoderLanguageModel(
        _tiny_routed_prefix_config(top_k_blocks=2), rng_seed=67
    )
    # External store is four rows wide, but only the last two are real history.
    source = xp.asarray([[0, 0, 3, 4, 5, 6, 7, 8]], dtype=xp.int64)
    meta = _terminal_metadata(history_start=2, valid=4)
    meta["use_retrieval"] = True
    active, cache = model._hierarchical_context_forward(source, memory_metadata=meta)

    assert cache["retrieval_active"] is True
    assert cache["prefix_mode"] == "direct_history"
    assert cache["prefix_length"] == 2
    assert cache["router_cache"] is None
    np.testing.assert_array_equal(
        asnumpy(active.token_ids), np.asarray([[3, 4, 5, 6, 7, 8]])
    )
    np.testing.assert_array_equal(
        asnumpy(active.position_ids), np.asarray([[2, 3, 4, 5, 6, 7]])
    )
    assert active.target_start == 2
    assert active.target_end == 6


def test_routed_prefix_direct_history_backward_does_not_train_router():
    model = DecoderLanguageModel(
        _tiny_routed_prefix_config(top_k_blocks=2), rng_seed=71
    )
    source = xp.asarray([[0, 0, 3, 4, 5, 6, 7, 8]], dtype=xp.int64)
    targets = xp.asarray([[6, 7, 8, 9]], dtype=xp.int64)
    meta = _terminal_metadata(history_start=2, valid=4)
    meta["use_retrieval"] = True
    logits, cache = model.forward(source, memory_metadata=meta, return_cache=True)
    _, loss_cache = model.compute_loss(logits, targets)
    model.zero_grad()
    model.backward(model.backward_loss(loss_cache), cache)
    np.testing.assert_allclose(asnumpy(model.memory_router.W_query.grad), 0.0, rtol=0, atol=0)
    np.testing.assert_allclose(asnumpy(model.memory_router.W_history.grad), 0.0, rtol=0, atol=0)
