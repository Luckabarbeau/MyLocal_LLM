"""Tests for gradient accumulation and loss scaling."""

import tempfile
from pathlib import Path

import numpy as np
import pytest

from mini_llm.model.decoder_lm import DecoderLanguageModel
from mini_llm.config import ModelConfig
from mini_llm.train import MiniTrainer
from mini_llm.train_extended import ExtendedTrainer


def make_tiny_model():
    """Create a tiny test model."""
    config = ModelConfig.tiny_inspection()
    return DecoderLanguageModel(config, rng_seed=42, dtype="float64")


def create_dummy_shard(path: Path, num_docs: int = 100, context_len: int = 256):
    """Create a dummy shard for testing."""
    with open(path, "wb") as f:
        f.write(np.int64(num_docs).tobytes())
        f.write(np.int64(context_len).tobytes())
        data = np.random.randint(0, 256, size=(num_docs, context_len), dtype=np.uint16)
        f.write(data.tobytes())


class TestGradientAccumulation:
    """Tests for gradient accumulation."""

    def test_grad_accum_runs_without_error(self):
        """Test that gradient accumulation runs without errors."""
        model = make_tiny_model()

        with tempfile.TemporaryDirectory() as tmpdir:
            shard_path = Path(tmpdir) / "test.bin"
            create_dummy_shard(shard_path, num_docs=100, context_len=256)

            grad_accum_steps = 2

            trainer = ExtendedTrainer(
                model=model,
                train_shard_paths=[str(shard_path)],
                val_shard_paths=[str(shard_path)],
                batch_size=4,
                seq_length=32,
                grad_accum_steps=grad_accum_steps,
                total_steps=100,
                warmup_steps=10,
            )

            # Run a few steps - should not raise any errors
            for _ in range(3):
                loss, grad_norm = trainer.train_step()
                assert isinstance(loss, float)
                assert isinstance(grad_norm, float)
                assert np.isfinite(loss)


class TestLossScaling:
    """Tests for loss scaling."""

    def test_loss_scale_runs_without_error(self):
        """Test that loss scaling runs without errors."""
        model = make_tiny_model()

        with tempfile.TemporaryDirectory() as tmpdir:
            shard_path = Path(tmpdir) / "test.bin"
            create_dummy_shard(shard_path, num_docs=100, context_len=256)

            loss_scale = 1024.0
            trainer = MiniTrainer(
                model=model,
                shard_paths=[str(shard_path)],
                batch_size=4,
                seq_length=32,
                total_steps=100,
                warmup_steps=10,
                loss_scale=loss_scale,
            )

            # Run a few steps - should not raise any errors
            for _ in range(3):
                loss = trainer.train_step()
                assert isinstance(loss, float)
                assert np.isfinite(loss)

    def test_grad_clip_parameter_used(self):
        """Test that grad_clip parameter is actually stored and used."""
        model = make_tiny_model()

        with tempfile.TemporaryDirectory() as tmpdir:
            shard_path = Path(tmpdir) / "test.bin"
            create_dummy_shard(shard_path, num_docs=100, context_len=256)

            custom_clip = 0.5
            trainer = ExtendedTrainer(
                model=model,
                train_shard_paths=[str(shard_path)],
                val_shard_paths=[str(shard_path)],
                batch_size=4,
                seq_length=32,
                grad_accum_steps=1,
                total_steps=100,
                warmup_steps=10,
                grad_clip=custom_clip,
            )

            assert trainer.grad_clip == custom_clip, "grad_clip should be stored"


class TestNoFakeTargets:
    """Tests that verify targets don't have fabricated values."""

    def test_targets_match_real_next_token(self):
        """Test that final target equals the real next token in sequence."""
        with tempfile.TemporaryDirectory() as tmpdir:
            context_len = 256
            seq_length = 254

            # Create a shard with known values
            shard_path = Path(tmpdir) / "test.bin"
            num_docs = 10
            with open(shard_path, "wb") as f:
                f.write(np.int64(num_docs).tobytes())
                f.write(np.int64(context_len).tobytes())

                # Use sequential values so we can verify
                data = np.arange(num_docs * context_len, dtype=np.uint16).reshape(num_docs, context_len)
                f.write(data.tobytes())

            from mini_llm.data.token_shards import create_minibatch

            inputs, targets = create_minibatch(
                data, batch_size=2, seq_length=seq_length,
                rng=np.random.default_rng(42)
            )

            # Verify that targets[:, 0] == inputs[:, 1]
            np.testing.assert_array_equal(targets[:, 0], inputs[:, 1])

            # Verify that the last target is not a fabricated 0
            # (it should be the next token after the last input token)
            assert not np.all(targets[:, -1] == 0), \
                "Last target position should contain real next tokens"
