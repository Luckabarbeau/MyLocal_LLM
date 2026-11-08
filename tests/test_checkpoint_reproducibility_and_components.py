"""Checkpoint reproducibility and read-only component update diagnostics."""

import csv
import json
import os

import numpy as np
import pytest

import train_model
from mini_llm.checkpoint import load_checkpoint, resolve_checkpoint_path, save_checkpoint
from mini_llm.config import DualResolutionConfig, ModelConfig
from mini_llm.model.decoder_lm import DecoderLanguageModel
from mini_llm.train_extended import ExtendedTrainer


def _model(seed=211):
    config = ModelConfig(
        tokenizer_vocab_size=128, context_length=16, n_layers=1,
        d_model=16, d_ff=32, n_q_heads=2, n_kv_heads=1, d_head=8,
        n_experts=1, top_k=1, ffn_mode="dense", tie_embeddings=True,
        dtype="float32", dual_resolution=DualResolutionConfig(
            enabled=True, block_size=4, local_window=8,
            compression_mode="matrix_pool", summaries_per_block=2,
            global_heads=1, global_head_dim=8, max_blocks=16,
            global_query_chunk=4,
        ),
    )
    return DecoderLanguageModel(config, rng_seed=seed, dtype="float32")


def _trainer(model, shard, checkpoint, **updates):
    options = dict(
        model=model, train_shard_paths=[str(shard)], val_shard_paths=[str(shard)],
        eos_token_id=127, batch_size=1, seq_length=16, grad_accum_steps=2,
        gradient_accumulation_weighting="supervised_tokens",
        optimizer_kind="hybrid-muon", muon_matmul_precision="fp32",
        total_steps=8, warmup_steps=2, peak_lr=1.5e-4,
        grad_clip=1.0, weight_decay=0.1, val_interval=100,
        save_interval=100, val_steps=1, rng_seed=42,
        checkpoint_dir=str(checkpoint),
    )
    options.update(updates)
    return ExtendedTrainer(**options)


def _shard(path):
    np.array([
        1, 2, 3, 4, 127, 6, 7, 8, 9, 10, 127,
        15, 16, 17, 18, 19, 20, 21, 22, 127,
    ], dtype=np.uint16).tofile(path)
    return path


def test_resume_bootstrap_restores_saved_runtime_before_backend_import(tmp_path, monkeypatch):
    checkpoint = tmp_path / "model"
    checkpoint.mkdir()
    (checkpoint / "training_state.json").write_text(json.dumps({
        "training_resume_contract": {
            "runtime_environment": {
                "MINI_LLM_BACKEND": "cupy",
                "MINI_LLM_SAVED_FAST_PATH": "1",
            },
        },
    }))
    environment = {"MINI_LLM_STALE": "1", "UNRELATED": "preserved"}
    monkeypatch.setattr(train_model.os, "environ", environment)
    monkeypatch.setattr(train_model.sys, "argv", [
        "train_model.py", "--resume-from", str(checkpoint),
        "--backend", "numpy",
    ])

    assert train_model._bootstrap_resume_environment_from_cli()
    train_model._bootstrap_backend_from_cli()

    assert environment == {
        "MINI_LLM_BACKEND": "numpy",
        "MINI_LLM_SAVED_FAST_PATH": "1",
        "UNRELATED": "preserved",
    }


def test_nonexact_resume_bootstrap_preserves_explicit_environment(tmp_path, monkeypatch):
    checkpoint = tmp_path / "model"
    checkpoint.mkdir()
    (checkpoint / "training_state.json").write_text(json.dumps({
        "training_resume_contract": {
            "runtime_environment": {"MINI_LLM_SAVED": "1"},
        },
    }))
    environment = {"MINI_LLM_EXPLICIT": "1"}
    monkeypatch.setattr(train_model.os, "environ", environment)
    monkeypatch.setattr(train_model.sys, "argv", [
        "train_model.py", "--resume-from", str(checkpoint),
        "--allow-nonexact-resume",
    ])

    assert not train_model._bootstrap_resume_environment_from_cli()
    assert environment == {"MINI_LLM_EXPLICIT": "1"}


def test_public_checkpoint_resolution_recovers_previous(tmp_path):
    checkpoint = tmp_path / "model"
    save_checkpoint(checkpoint, {"w": np.array([3], dtype=np.float32)})
    previous = tmp_path / "model.previous"
    os.replace(checkpoint, previous)
    assert resolve_checkpoint_path(checkpoint) == previous
    loaded, _, _ = load_checkpoint(checkpoint, skip_optimizer=True)
    np.testing.assert_array_equal(loaded["w"], [3])


def test_step_coherence_preserves_historical_offset_and_rejects_missing_state(capsys):
    assert train_model.validate_checkpoint_step_coherence(
        {"step": 17}, {"step": 17},
    ) == 17
    assert train_model.validate_checkpoint_step_coherence(
        {"step": 17}, {"step": 67},
    ) == 17
    assert "preserving checkpoint trainer/optimizer step offset" in capsys.readouterr().out
    with pytest.raises(ValueError, match="missing"):
        train_model.validate_checkpoint_step_coherence({"step": 17}, {})


def test_versioned_resume_contract_persists_and_rejects_drift(tmp_path, monkeypatch, capsys):
    monkeypatch.setenv("MINI_LLM_DUAL_RESOLUTION_SHAPE_BUCKET", "fine")
    shard = _shard(tmp_path / "train.bin")
    checkpoint = tmp_path / "checkpoint"
    original = _trainer(_model(), shard, checkpoint)
    original.save()
    _, _, state = load_checkpoint(checkpoint, skip_optimizer=True)
    assert state is not None
    contract = state["training_resume_contract"]
    assert contract["grad_accum_steps"] == 2
    assert contract["scheduler"]["peak_lr"] == 1.5e-4
    assert contract["scheduler"]["warmup_steps"] == 2
    assert contract["scheduler"]["total_steps"] == 8
    assert contract["scheduler"]["min_lr"] == pytest.approx(1.5e-5)
    assert contract["optimizer_kind"] == "hybrid-muon"
    assert contract["runtime_environment"][
        "MINI_LLM_DUAL_RESOLUTION_SHAPE_BUCKET"
    ] == "fine"

    exact = _trainer(_model(), shard, tmp_path / "exact")
    assert exact.validate_training_resume_contract(state) == []

    changed = _trainer(
        _model(), shard, tmp_path / "changed",
        grad_accum_steps=4, weight_decay=0.2,
    )
    with pytest.raises(ValueError, match="allow-nonexact-resume"):
        changed.validate_training_resume_contract(state)
    mismatches = changed.validate_training_resume_contract(
        state, allow_changes=True,
    )
    fields = {row["field"] for row in mismatches}
    assert {"grad_accum_steps", "weight_decay"} <= fields
    assert "intentional non-exact" in capsys.readouterr().out

    legacy = dict(state)
    legacy.pop("training_resume_contract")
    assert exact.validate_training_resume_contract(legacy) == []
    assert "legacy checkpoint" in capsys.readouterr().out


def test_component_monitor_is_bitwise_observational(tmp_path, monkeypatch):
    monkeypatch.setenv("MINI_LLM_DUAL_RESOLUTION_SHAPE_BUCKET", "fine")
    shard = _shard(tmp_path / "train.bin")
    baseline = _trainer(_model(), shard, tmp_path / "baseline")
    output = tmp_path / "component_updates.csv"
    observed = _trainer(
        _model(), shard, tmp_path / "observed",
        component_update_monitor=True,
        component_update_interval=1,
        component_update_output=str(output),
        component_update_sample_elements=64,
    )

    baseline_result = baseline.train_step()
    observed_result = observed.train_step()
    np.testing.assert_array_equal(baseline_result, observed_result)
    for left, right in zip(baseline.model.parameters(), observed.model.parameters()):
        assert left.name == right.name
        left_data = left.data.get() if hasattr(left.data, "get") else left.data
        right_data = right.data.get() if hasattr(right.data, "get") else right.data
        np.testing.assert_array_equal(left_data, right_data)
    for group in ("master_weights", "m", "v", "muon_m"):
        left = baseline.optimizer.checkpoint_state()[group]
        right = observed.optimizer.checkpoint_state()[group]
        assert set(left) == set(right)
        for name in left:
            np.testing.assert_array_equal(left[name], right[name])

    with output.open(newline="") as handle:
        rows = list(csv.DictReader(handle))
    assert rows
    components = {row["component"] for row in rows}
    assert {
        "tied_embedding_unembedding", "block_norm", "local_attention_qkv",
        "local_attention_output", "matrix_pool_score",
        "matrix_pool_position_bias", "compressed_global_attention",
        "ffn_gate_up", "ffn_down", "final_norm",
    } <= components
    assert {row["optimizer"] for row in rows} == {"adamw", "muon"}
    assert all(float(row["sample_count"]) > 0 for row in rows)
    assert any(float(row["update_rms"]) > 0 for row in rows)
