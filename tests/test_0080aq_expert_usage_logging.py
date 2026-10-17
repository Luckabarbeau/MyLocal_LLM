"""0080AQ: training-only, per-layer, per-expert utilization monitoring."""

import csv
import numpy as np
import pytest

from mini_llm.backend import BACKEND_NAME, RandomStream
from mini_llm.blocks.moe_block import MoE
from mini_llm.diagnostics.moe_usage import ExpertUsageCollector, summarize_usage
from mini_llm.ops.router import Router

pytestmark = pytest.mark.skipif(BACKEND_NAME != "numpy", reason="CPU statistics checks")


def test_complete_two_expert_collapse_detected_and_probabilities_preserved():
    collector = ExpertUsageCollector()
    probs = np.array([[[.9, .1], [.8, .2], [.7, .3], [.6, .4]]], dtype=np.float64)
    indices = np.zeros((1, 4, 1), dtype=np.int64)
    collector.observe(3, indices, probs)
    collector.sampled_microbatches = 1
    rows = collector.rows(step=10)
    assert len(rows) == 2
    r0, r1 = rows
    assert (r0["layer"], r0["expert"], r0["positions"], r0["selections"]) == (3, 0, 4, 4)
    assert (r1["expert"], r1["selections"], r1["selection_share"]) == (1, 0, 0)
    np.testing.assert_allclose([r0["mean_router_probability"], r1["mean_router_probability"]], [.75, .25])
    np.testing.assert_allclose(r0["mean_selected_probability_mass"], .75)
    s = summarize_usage(rows)
    assert s["under_one_percent"] == 1
    assert s["min_expert_share"] == 0


def test_topk_selection_share_accounts_for_two_assignments_per_token():
    collector = ExpertUsageCollector()
    p = np.array([[[.6, .3, .1], [.1, .6, .3]]], dtype=np.float64)
    collector.observe(0, np.array([[[0, 1], [1, 2]]]), p)
    rows = collector.rows(2)
    np.testing.assert_allclose([r["selection_share"] for r in rows], [.25, .5, .25])
    np.testing.assert_allclose([r["token_hit_rate"] for r in rows], [.5, 1., .5])
    np.testing.assert_allclose(rows[0]["mean_selected_probability_mass"], .9)


def test_router_observation_does_not_change_outputs_or_gradients():
    router = Router(6, 2, 1, .05, RandomStream(9), dtype="float64")
    x = np.random.default_rng(7).normal(size=(2, 7, 6))
    reference, indices, cache = router.forward(x)
    upstream = np.random.default_rng(8).normal(size=reference.shape)
    router.zero_grad()
    dx_ref = router.backward(upstream, cache)
    grad_ref = router.W_router_param.grad.copy()
    router.zero_grad()
    collector = ExpertUsageCollector()
    router.usage_collector = collector
    router.usage_layer = 4
    observed, indices_obs, observed_cache = router.forward(x)
    dx_obs = router.backward(upstream, observed_cache)
    np.testing.assert_array_equal(observed, reference)
    np.testing.assert_array_equal(indices_obs, indices)
    np.testing.assert_array_equal(dx_obs, dx_ref)
    np.testing.assert_array_equal(router.W_router_param.grad, grad_ref)
    assert len(collector.rows(1)) == 2
    router.usage_collector = None
    router.forward(x)
    assert collector.rows(1)[0]["positions"] == 14


def test_model_backward_checkpoint_recompute_does_not_double_count():
    from mini_llm.config import ModelConfig
    from mini_llm.model.decoder_lm import DecoderLanguageModel
    model = DecoderLanguageModel(ModelConfig.tiny_inspection(), rng_seed=18, dtype="float32")
    router = model.blocks[0].moe.router
    collector = ExpertUsageCollector()
    router.usage_collector = collector
    router.usage_layer = 0
    ids = np.arange(16, dtype=np.int64).reshape(1, 16)
    hidden, cache = model.forward_body(ids, activation_checkpoint=True)
    observed = collector.rows(1)[0]["positions"]
    model.backward_body(np.ones_like(hidden), cache)
    assert collector.rows(1)[0]["positions"] == observed
    assert router.usage_collector is collector
    router.usage_collector = None


def test_all_32_deep_layers_are_observed_with_one_primary_forward():
    from dataclasses import replace
    from mini_llm.config import ModelConfig
    from mini_llm.model.decoder_lm import DecoderLanguageModel
    cfg = replace(ModelConfig.tiny_inspection(), n_layers=32, n_experts=2, top_k=1)
    model = DecoderLanguageModel(cfg, rng_seed=4, dtype="float32")
    collector = ExpertUsageCollector()
    for idx, block in enumerate(model.blocks):
        block.moe.router.usage_collector = collector
        block.moe.router.usage_layer = idx
    ids = np.arange(16, dtype=np.int64).reshape(1, 16)
    hidden, cache = model.forward_body(ids)
    model.backward_body(np.ones_like(hidden), cache)
    rows = collector.rows(1)
    assert len(rows) == 32 * 2
    assert {r["layer"] for r in rows} == set(range(32))
    assert all(r["positions"] == 16 for r in rows)
    for idx in range(32):
        assert sum(r["selections"] for r in rows if r["layer"] == idx) == 16
    for block in model.blocks:
        block.moe.router.usage_collector = None


def test_extended_trainer_collects_and_appends_training_rows(monkeypatch, tmp_path):
    from mini_llm.config import ModelConfig
    from mini_llm.model.decoder_lm import DecoderLanguageModel
    from mini_llm.train_extended import ExtendedTrainer

    monkeypatch.setenv("MINI_LLM_EXPERT_USAGE_INTERVAL", "1")
    monkeypatch.setenv("MINI_LLM_EXPERT_USAGE_MICROBATCHES", "1")
    config = ModelConfig.tiny_inspection()
    model = DecoderLanguageModel(config, rng_seed=14, dtype="float32")
    shard = tmp_path / "shard.bin"
    np.random.default_rng(4).integers(0, 256, 2048, dtype=np.uint16).tofile(shard)
    log = tmp_path / "training.csv"
    trainer = ExtendedTrainer(
        model, [str(shard)], [str(shard)], batch_size=2,
        seq_length=16, grad_accum_steps=2, total_steps=2,
        warmup_steps=1, val_interval=100, save_interval=100,
        log_file=str(log), rng_seed=7,
    )
    trainer.train(num_steps=2, log_interval=1)
    csv_file = tmp_path / "training.expert_usage.csv"
    assert csv_file.exists()
    with csv_file.open(newline="") as f:
        rows = list(csv.DictReader(f))
    assert len(rows) == 2 * config.n_layers * config.n_experts
    assert sorted(set(r["step"] for r in rows)) == ["1", "2"]
    assert all(int(r["sampled_microbatches"]) == 1 for r in rows)
    assert all(int(r["positions"]) == 32 for r in rows)
    assert all(block.moe.router.usage_collector is None for block in model.blocks)
