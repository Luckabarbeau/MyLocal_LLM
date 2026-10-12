"""0078F observational routing diagnostics and fixed validation invariants."""
import copy
import csv
import hashlib
import numpy as np
import pytest

from mini_llm.config import MemoryContextConfig, ModelConfig
from mini_llm.model.decoder_lm import DecoderLanguageModel
from mini_llm.train_extended import ExtendedTrainer


def make_trainer(tmp_path, monkeypatch, *, accum=1, mixed_steps=4):
    monkeypatch.setenv("MINI_LLM_LM_HEAD_CHUNK_TOKENS", "4")
    memory = MemoryContextConfig(
        enabled=True, memory_length=16, recent_length=4, target_length=4,
        block_size=2, top_k_blocks=1, router_query_length=4, router_dim=4,
        read_heads=2, integration_mode="routed_prefix", memory_training="joint",
        retrieval_batch_probability=1., read_kv_heads=1, read_query_chunk=2,
    )
    config = ModelConfig(
        tokenizer_vocab_size=32, context_length=memory.active_length,
        n_layers=1, d_model=8, n_q_heads=2, n_kv_heads=1,
        d_head=4, n_experts=2, top_k=1, d_ff=16,
        dtype="float32", memory_context=memory,
    )
    # Repeated short documents plus one long document, EOS-bounded.
    short_data = np.asarray([4, 5, 2] * 60 + [6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 2], dtype=np.uint16)
    long_data = np.asarray([4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 2] * 35, dtype=np.uint16)
    short_path = tmp_path / "short.bin"
    long_path = tmp_path / "long.bin"
    short_data.tofile(short_path)
    long_data.tofile(long_path)
    sources = {"short": [str(short_path)], "long": [str(long_path)]}
    trainer = ExtendedTrainer(
        model=DecoderLanguageModel(config, rng_seed=1),
        train_shard_paths=[str(short_path), str(long_path)],
        val_shard_paths=[str(short_path), str(long_path)],
        train_source_shards=sources, val_source_shards=sources,
        source_weights={"short": 0.5, "long": 0.5},
        batch_size=2, seq_length=memory.source_input_length,
        grad_accum_steps=accum, total_steps=2, warmup_steps=1,
        val_steps=1, val_interval=100, save_interval=100,
        val_fixed_mixed_steps=mixed_steps, val_fixed_mixed_seed=789,
        val_representative_steps=3, val_per_source_steps=3,
        eos_token_id=2, log_file=str(tmp_path / "train.csv"),
    )
    return trainer


def test_routing_mode_cache_accounting_is_actual_and_unweighted(tmp_path, monkeypatch):
    t = make_trainer(tmp_path, monkeypatch)
    metadata = {"use_retrieval": True, "history_valid_starts": np.asarray([6, 8])}
    dense = t._routing_mode_observation({"memory_context_cache": {"prefix_mode": "none", "prefix_length": 0}}, metadata, 3.0, 2)
    direct = t._routing_mode_observation({"memory_context_cache": {"prefix_mode": "direct_history", "prefix_length": 3}}, metadata, 5.0, 2)
    routed = t._routing_mode_observation({"memory_context_cache": {"prefix_mode": "routed_blocks", "prefix_length": 4}}, metadata, 7.0, 2)
    total = t._new_routing_mode_stats()
    for item in (dense, direct, routed):
        t._merge_routing_mode_stats(total, item)
    assert total["available_history_tokens"] == 3 * (6 + 4)
    assert total["used_history_tokens"] == 0 + 6 + 8
    assert total["microbatches"] == 3
    assert total["requested_microbatches"] == 3
    assert total["dense_microbatches"] == total["direct_microbatches"] == total["routed_microbatches"] == 1
    assert total["routed_supervised_tokens"] == 7
    assert total["supervised_token_weight"] == 15
    with pytest.raises(ValueError, match="nonzero prefix"):
        t._routing_mode_observation({"memory_context_cache": {"prefix_mode": "direct_history", "prefix_length": 0}}, metadata, 1., 2)


def test_all_accumulation_microbatches_counted_and_checkpointed(tmp_path, monkeypatch):
    trainer = make_trainer(tmp_path, monkeypatch, accum=3)
    trainer.train(num_steps=1, log_interval=1)
    stats = trainer._last_step_routing_modes
    assert stats["microbatches"] == 3
    assert stats["samples"] == 6
    assert sum(stats[name + "_microbatches"] for name in ("dense", "direct", "routed")) == 3
    assert sum(stats[name + "_supervised_tokens"] for name in ("dense", "direct", "routed")) == pytest.approx(stats["supervised_token_weight"])
    assert stats["used_history_tokens"] <= stats["available_history_tokens"]
    with (tmp_path / "train.routing_modes.csv").open() as stream:
        rows = list(csv.DictReader(stream))
    assert [row["scope"] for row in rows] == ["interval", "cumulative_since_patch"]
    assert int(rows[0]["microbatches"]) == 3
    old = trainer.routing_modes_total.copy()
    restored = make_trainer(tmp_path, monkeypatch, accum=3)
    restored.restore_routing_mode_stats({"version": 1, "since_step": trainer.routing_modes_since_step, "total": old})
    assert restored.routing_modes_total == old
    assert restored.routing_modes_since_step == trainer.routing_modes_since_step
    restored.restore_routing_mode_stats(None)
    assert restored.routing_modes_total["microbatches"] == 0


def test_fixed_mixed_reuses_exact_samples_without_mutating_rng(tmp_path, monkeypatch):
    trainer = make_trainer(tmp_path, monkeypatch, mixed_steps=12)
    train_before = copy.deepcopy(trainer.train_rng.bit_generator.state)
    val_before = copy.deepcopy(trainer.val_rng.bit_generator.state)
    shards_before = dict(trainer.current_val_source_shard_idx)
    source_old = trainer.compute_val_loss_by_source()
    r1 = trainer.compute_val_loss_fixed_mixed()
    r2 = trainer.compute_val_loss_fixed_mixed()
    assert r1["sample_sha256"] == r2["sample_sha256"]
    assert r1["ce_nats_per_target"] == pytest.approx(r2["ce_nats_per_target"], abs=1e-7)
    assert 0 < r1["supervised_token_weight"] <= r1["target_capacity_tokens"]
    assert trainer.train_rng.bit_generator.state == train_before
    assert trainer.val_rng.bit_generator.state == val_before
    assert trainer.current_val_source_shard_idx == shards_before
    source_new = trainer.compute_val_loss_by_source()
    assert {s: x["sample_sha256"] for s, x in source_old.items()} == {s: x["sample_sha256"] for s, x in source_new.items()}
    trainer._log_fixed_mixed_loss(r1)
    with (tmp_path / "train.val_fixed_mixed.csv").open() as stream:
        assert next(csv.DictReader(stream))["sample_sha256"] == r1["sample_sha256"]


def test_representative_uses_document_first_but_retains_history_threshold(tmp_path, monkeypatch):
    import mini_llm.train_extended as module
    trainer = make_trainer(tmp_path, monkeypatch)
    captured = []
    real = module.create_hierarchical_memory_minibatch
    def capture(*args, **kwargs):
        captured.append((kwargs["sampling_policy"], kwargs["min_history_tokens"]))
        return real(*args, **kwargs)
    monkeypatch.setattr(module, "create_hierarchical_memory_minibatch", capture)
    trainer.compute_val_loss_by_source_representative()
    assert captured and all(mode == "document_first" and threshold == 3 for mode, threshold in captured)
    captured.clear()
    trainer.compute_val_loss_by_source()
    assert captured and all(mode == "legacy" and threshold == 3 for mode, threshold in captured)
    captured.clear()
    trainer.compute_val_loss_fixed_mixed()
    assert captured and all(mode == "document_first" and threshold == 3 for mode, threshold in captured)
    with pytest.raises(ValueError, match="invalid validation sampling_policy_override"):
        trainer.get_val_batch(source_override="long", sampling_policy_override="wrong")
