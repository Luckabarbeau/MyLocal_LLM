import dataclasses
import numpy as np

from mini_llm.backend import RandomStream, asnumpy, xp
from mini_llm.checkpoint import load_checkpoint, save_checkpoint
from mini_llm.config import MemoryContextConfig, ModelConfig
from mini_llm.data.token_shards import create_hierarchical_memory_minibatch
from mini_llm.model.decoder_lm import DecoderLanguageModel
from mini_llm.ops.hierarchical_memory import HierarchicalMemoryRouter


def _tiny_config(*, top_k=2, router_weight_scale=1.0):
    memory = MemoryContextConfig(
        enabled=True,
        memory_length=8,
        recent_length=4,
        target_length=3,
        block_size=2,
        top_k_blocks=top_k,
        router_query_length=2,
        router_dim=4,
        query_pooling="learned",
        history_pooling="mean",
        router_weight_scale=router_weight_scale,
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
    assert {cfg.memory_context.active_length for cfg in configs} == {7168}
    assert {cfg.context_length for cfg in configs} == {7168}
    assert [cfg.memory_context.memory_length for cfg in configs] == [16384, 32768, 65536]
    assert [cfg.memory_context.searchable_blocks for cfg in configs] == [96, 224, 480]
    assert len({cfg.estimated_parameter_count() for cfg in configs}) == 1


def test_hierarchical_minibatch_returns_only_target_labels():
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
    np.testing.assert_array_equal(labels[:, :-1], source[:, 9:11])
    np.testing.assert_array_equal(labels[:, 0], source[:, 9])
    np.testing.assert_array_equal(labels[:, 1], source[:, 10])
    np.testing.assert_array_equal(labels[:, 2], source[:, 10] + 1)


def test_target_tokens_cannot_change_top_level_memory_selection():
    model = DecoderLanguageModel(_tiny_config(top_k=1), rng_seed=11)
    source = xp.asarray([[1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11]], dtype=xp.int64)
    active_a, _ = model._hierarchical_context_forward(source)
    changed = source.copy()
    changed[:, 8:] = xp.asarray([[21, 22, 23]], dtype=xp.int64)
    active_b, _ = model._hierarchical_context_forward(changed)
    np.testing.assert_array_equal(
        asnumpy(active_a.selected_blocks), asnumpy(active_b.selected_blocks)
    )
    np.testing.assert_allclose(
        asnumpy(active_a.route_weights), asnumpy(active_b.route_weights), rtol=0, atol=0
    )


def test_active_context_is_chronological_and_preserves_source_positions():
    model = DecoderLanguageModel(_tiny_config(top_k=2), rng_seed=5)
    source = xp.asarray([[1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11]], dtype=xp.int64)
    active, _ = model._hierarchical_context_forward(source)

    selected = asnumpy(active.selected_blocks)[0]
    assert np.all(selected[:-1] < selected[1:])
    expected_retrieved = np.concatenate(
        [np.arange(block * 2, block * 2 + 2) for block in selected]
    )
    expected_positions = np.concatenate(
        [expected_retrieved, np.arange(4, 8), np.arange(8, 11)]
    )
    np.testing.assert_array_equal(asnumpy(active.position_ids)[0], expected_positions)
    np.testing.assert_array_equal(asnumpy(active.source_indices)[0], expected_positions)
    assert active.target_start == 8
    assert active.target_end == 11


def test_all_blocks_selected_neutral_gate_matches_direct_model():
    memory_cfg = _tiny_config(top_k=2, router_weight_scale=0.0)
    direct_cfg = dataclasses.replace(
        memory_cfg,
        context_length=11,
        memory_context=MemoryContextConfig(enabled=False),
    )
    memory_model = DecoderLanguageModel(memory_cfg, rng_seed=17)
    direct_model = DecoderLanguageModel(direct_cfg, rng_seed=17)
    source = xp.asarray([[1, 4, 7, 2, 3, 8, 5, 9, 6, 10, 11]], dtype=xp.int64)

    memory_hidden = memory_model.forward_body(source, return_cache=False)
    direct_hidden = direct_model.forward_body(source, return_cache=False)
    np.testing.assert_allclose(
        asnumpy(memory_hidden), asnumpy(direct_hidden), rtol=1e-5, atol=1e-6
    )


def test_chunked_lm_head_projects_only_target_rows_and_scatters_backward():
    model = DecoderLanguageModel(_tiny_config(), rng_seed=9)
    rng = np.random.default_rng(4)
    hidden = xp.asarray(rng.normal(size=(2, 11, 8)), dtype="float32")
    targets = xp.asarray(rng.integers(0, 32, size=(2, 3)), dtype=xp.int32)

    loss, cache = model.chunked_lm_head_loss_forward(
        hidden,
        targets,
        chunk_tokens=2,
        target_slice=(8, 11),
        return_device_loss=False,
    )
    assert np.isfinite(loss)
    dx = model.chunked_lm_head_backward(cache)
    assert dx.shape == hidden.shape
    assert np.max(np.abs(asnumpy(dx[:, :8, :]))) == 0.0
    assert np.max(np.abs(asnumpy(dx[:, 8:, :]))) > 0.0


def test_hierarchical_router_backward_matches_finite_difference():
    config = MemoryContextConfig(
        enabled=True,
        memory_length=10,
        recent_length=4,
        target_length=2,
        block_size=2,
        top_k_blocks=2,
        router_query_length=2,
        router_dim=3,
        query_pooling="mean",
        history_pooling="mean",
    )
    router = HierarchicalMemoryRouter(
        d_model=4,
        config=config,
        rng=RandomStream(3),
        input_std=0.05,
        dtype="float64",
    )
    rng = np.random.default_rng(8)
    history = xp.asarray(rng.normal(size=(1, 6, 4)), dtype="float64")
    recent = xp.asarray(rng.normal(size=(1, 4, 4)), dtype="float64")
    coeff = xp.asarray([[0.3, -0.7]], dtype="float64")

    weights, _, cache = router.forward(history, recent)
    router.zero_grad()
    router.backward(coeff, cache)
    analytic = float(router.W_query.grad[0, 0])

    original = float(router.W_query.data[0, 0])
    eps = 1e-6

    def objective(value):
        router.W_query.data[0, 0] = value
        w, _, _ = router.forward(history, recent)
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


def test_tiny_end_to_end_router_gradients_and_checkpoint_replay_match():
    cfg = _tiny_config(top_k=2)
    source = xp.asarray([[1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11]], dtype=xp.int64)
    targets = xp.asarray([[10, 11, 12]], dtype=xp.int32)
    reference = DecoderLanguageModel(cfg, rng_seed=21)
    checkpointed = DecoderLanguageModel(cfg, rng_seed=21)

    loss_a, grads_a = _run_tiny_training_backward(reference, source, targets, False)
    loss_b, grads_b = _run_tiny_training_backward(checkpointed, source, targets, True)
    assert abs(loss_a - loss_b) < 1e-7
    assert np.linalg.norm(grads_a["memory_router.W_query"]) > 0.0
    assert np.linalg.norm(grads_a["memory_router.W_history"]) > 0.0
    for name in grads_a:
        np.testing.assert_allclose(grads_a[name], grads_b[name], rtol=2e-5, atol=2e-6)


def test_router_parameters_roundtrip_checkpoint(tmp_path):
    model = DecoderLanguageModel(_tiny_config(), rng_seed=31)
    params = {p.name: p.data for p in model.parameters()}
    save_checkpoint(tmp_path, params)
    loaded, _, _ = load_checkpoint(
        tmp_path, param_names=[p.name for p in model.parameters()], skip_optimizer=True
    )
    assert "memory_router.W_query" in loaded
    assert "memory_router.W_history" in loaded
    np.testing.assert_allclose(
        asnumpy(loaded["memory_router.W_query"]),
        asnumpy(params["memory_router.W_query"]),
    )
