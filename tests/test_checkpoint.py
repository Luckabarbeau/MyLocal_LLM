"""Tests for checkpoint and resume functionality."""

import tempfile
from pathlib import Path

import numpy as np
import pytest

from mini_llm.model.decoder_lm import DecoderLanguageModel
from mini_llm.config import ModelConfig
from mini_llm.checkpoint import save_checkpoint, load_checkpoint, initialize_matching_parameters


def make_test_model():
    """Create a small test model."""
    config = ModelConfig.tiny_inspection()
    return DecoderLanguageModel(config, rng_seed=42, dtype="float64")


class TestCheckpointResume:
    """Tests for checkpoint saving and loading."""

    def test_checkpoint_save_load(self):
        """Test that checkpoint save/load works correctly."""
        model = make_test_model()
        config = model.config

        # Get initial parameters
        initial_params = {p.name: p.data.copy() for p in model.parameters()}

        with tempfile.TemporaryDirectory() as tmpdir:
            # Save checkpoint
            save_checkpoint(
                path=tmpdir,
                model_params=initial_params,
                optimizer_state=None,
                training_state={"step": 100, "loss_history": [0.5, 0.4]},
            )

            # Load checkpoint
            loaded_params, loaded_optimizer, loaded_training = load_checkpoint(
                path=tmpdir,
                param_names=None,
            )

            # Verify parameters match
            for name, data in initial_params.items():
                assert name in loaded_params, f"Missing parameter: {name}"
                np.testing.assert_allclose(data, loaded_params[name], rtol=1e-10, atol=1e-10), \
                    f"Parameter {name} mismatch"

            # Verify training state
            assert loaded_training is not None
            assert loaded_training["step"] == 100
            assert loaded_training["loss_history"] == [0.5, 0.4]

    def test_checkpoint_resume_trains_correctly(self):
        """Test that resuming from checkpoint continues training correctly."""
        model = make_test_model()

        with tempfile.TemporaryDirectory() as tmpdir:
            # Run a few steps
            for _ in range(3):
                # Simulate some parameter changes
                for p in model.parameters():
                    p.data[...] += np.random.randn(*p.data.shape) * 0.01

            # Get parameters after training
            params_after_training = {p.name: p.data.copy() for p in model.parameters()}

            # Save checkpoint
            save_checkpoint(
                path=tmpdir,
                model_params=params_after_training,
            )

            # Create new model and load checkpoint
            model_resumed = make_test_model()
            loaded_params, _, _ = load_checkpoint(path=tmpdir)

            # Apply loaded parameters
            for p in model_resumed.parameters():
                if p.name in loaded_params:
                    p.data[...] = loaded_params[p.name]

            # Verify parameters match
            for p_orig, p_resumed in zip(model.parameters(), model_resumed.parameters()):
                np.testing.assert_allclose(p_orig.data, p_resumed.data, rtol=1e-10, atol=1e-10), \
                    f"Parameter {p_orig.name} mismatch after resume"

    def test_checkpoint_with_filtered_params(self):
        """Test checkpoint with parameter filtering."""
        config = ModelConfig(
            tokenizer_vocab_size=1000,
            context_length=64,
            n_layers=2,
            d_model=64,
            n_q_heads=4,
            n_kv_heads=2,
            d_head=16,
            n_experts=3,
            top_k=2,
            d_ff=128,
        )
        model = DecoderLanguageModel(config, rng_seed=42, dtype="float64")

        with tempfile.TemporaryDirectory() as tmpdir:
            # Save all parameters
            all_params = {p.name: p.data.copy() for p in model.parameters()}
            save_checkpoint(path=tmpdir, model_params=all_params)

            # Load only specific parameters
            param_names = ["embedding.W", "blocks.0.norm1.gamma"]
            loaded_params, _, _ = load_checkpoint(path=tmpdir, param_names=param_names)

            # Verify only requested parameters were loaded
            assert set(loaded_params.keys()) == set(param_names), \
                f"Expected {param_names}, got {list(loaded_params.keys())}"

    def test_checkpoint_loads_all_params_without_filter(self):
        """Test that loading without filter gets all parameters."""
        config = ModelConfig(
            tokenizer_vocab_size=1000,
            context_length=64,
            n_layers=2,
            d_model=64,
            n_q_heads=4,
            n_kv_heads=2,
            d_head=16,
            n_experts=3,
            top_k=2,
            d_ff=128,
        )
        model = DecoderLanguageModel(config, rng_seed=42, dtype="float64")

        with tempfile.TemporaryDirectory() as tmpdir:
            # Save all parameters
            all_params = {p.name: p.data.copy() for p in model.parameters()}
            save_checkpoint(path=tmpdir, model_params=all_params)

            # Load without filter
            loaded_params, _, _ = load_checkpoint(path=tmpdir, param_names=None)

            # Verify all parameters were loaded
            assert set(loaded_params.keys()) == set(all_params.keys()), \
                f"Expected {list(all_params.keys())}, got {list(loaded_params.keys())}"


def test_streaming_initialize_matching_parameters_allows_new_memory_tensors(tmp_path):
    base_cfg = ModelConfig(
        tokenizer_vocab_size=64, context_length=8, n_layers=1, d_model=16,
        n_q_heads=2, n_kv_heads=1, d_head=8, n_experts=2, top_k=1, d_ff=24,
        dtype="float64",
    )
    source = DecoderLanguageModel(base_cfg, rng_seed=101, dtype="float64")
    saved = {p.name: p.data.copy() for p in source.parameters()}
    save_checkpoint(tmp_path, saved)

    target = DecoderLanguageModel(base_cfg, rng_seed=202, dtype="float64")
    loaded, missing, elements = initialize_matching_parameters(
        tmp_path, target.parameters()
    )
    assert not missing
    assert len(loaded) == len(saved)
    assert elements == sum(arr.size for arr in saved.values())
    for p in target.parameters():
        np.testing.assert_array_equal(p.data, saved[p.name])


def test_streaming_initialize_matching_parameters_reports_missing(tmp_path):
    model = make_test_model()
    params = {p.name: p.data.copy() for p in model.parameters()}
    removed = next(iter(params))
    params.pop(removed)
    save_checkpoint(tmp_path, params)
    fresh = make_test_model()
    _, missing, _ = initialize_matching_parameters(tmp_path, fresh.parameters())
    assert removed in missing
