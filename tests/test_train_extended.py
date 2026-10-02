"""Tests for ExtendedTrainer functionality."""

import tempfile
from pathlib import Path

import numpy as np
import pytest

from mini_llm.model.decoder_lm import DecoderLanguageModel
from mini_llm.config import ModelConfig
from mini_llm.train_extended import ExtendedTrainer


def make_model():
    """Create a tiny test model."""
    config = ModelConfig.tiny_inspection()
    return DecoderLanguageModel(config, rng_seed=42, dtype="float32")


def create_dummy_shard(path: Path, num_tokens: int):
    """Create a dummy packed token shard."""
    tokens = np.random.randint(0, 256, size=num_tokens, dtype=np.uint16)
    tokens.tofile(path)


class TestExtendedTrainerInit:
    """Tests for ExtendedTrainer initialization."""
    
    def test_initialization(self):
        """Test trainer initialization."""
        model = make_model()
        
        with tempfile.TemporaryDirectory() as tmpdir:
            # Create dummy shards
            train_shard = Path(tmpdir) / "train_shard_00000.bin"
            val_shard = Path(tmpdir) / "val_shard_00000.bin"
            
            create_dummy_shard(train_shard, num_tokens=1000)
            create_dummy_shard(val_shard, num_tokens=100)
            
            trainer = ExtendedTrainer(
                model=model,
                train_shard_paths=[str(train_shard)],
                val_shard_paths=[str(val_shard)],
                batch_size=2,
                seq_length=32,
                total_steps=100,
                warmup_steps=10,
                rng_seed=42,
            )
            
            assert trainer.step == 0
            assert trainer.tokens_processed == 0
            assert trainer.train_rng is not None
            assert trainer.val_rng is not None
    
    def test_persistent_rngs(self):
        """Test that RNGs are persistent across batches."""
        model = make_model()
        
        with tempfile.TemporaryDirectory() as tmpdir:
            train_shard = Path(tmpdir) / "train_shard_00000.bin"
            val_shard = Path(tmpdir) / "val_shard_00000.bin"
            create_dummy_shard(train_shard, num_tokens=1000)
            create_dummy_shard(val_shard, num_tokens=100)
            
            trainer = ExtendedTrainer(
                model=model,
                train_shard_paths=[str(train_shard)],
                val_shard_paths=[str(val_shard)],
                batch_size=2,
                seq_length=32,
                total_steps=100,
                warmup_steps=10,
                rng_seed=42,
            )
            
            # Get initial RNG state
            train_state_0 = trainer.train_rng.bit_generator.state
            
            # Sample some blocks (simulating batch generation)
            for _ in range(5):
                inputs, targets = trainer.get_train_batch()
            
            # RNG state should have advanced
            train_state_1 = trainer.train_rng.bit_generator.state
            
            # States should be different after sampling
            assert train_state_0 != train_state_1


class TestExtendedTrainerTokensProcessed:
    """Tests for tokens_processed tracking."""
    
    def test_tokens_tracked_per_step(self):
        """Test that tokens_processed is tracked correctly."""
        model = make_model()
        
        with tempfile.TemporaryDirectory() as tmpdir:
            train_shard = Path(tmpdir) / "train_shard_00000.bin"
            val_shard = Path(tmpdir) / "val_shard_00000.bin"
            create_dummy_shard(train_shard, num_tokens=10000)
            create_dummy_shard(val_shard, num_tokens=100)
            
            batch_size = 4
            seq_length = 64
            
            trainer = ExtendedTrainer(
                model=model,
                train_shard_paths=[str(train_shard)],
                val_shard_paths=[str(val_shard)],
                batch_size=batch_size,
                seq_length=seq_length,
                total_steps=100,
                warmup_steps=10,
                rng_seed=42,
            )
            
            # Initial state
            assert trainer.tokens_processed == 0
            
            # After one step
            inputs, targets = trainer.get_train_batch()
            # tokens_processed should be updated in get_train_batch
            expected_tokens = batch_size * seq_length
            assert trainer.tokens_processed == expected_tokens
    
    def test_tokens_processed_accumulates(self):
        """Test that tokens_processed accumulates across steps."""
        model = make_model()
        
        with tempfile.TemporaryDirectory() as tmpdir:
            train_shard = Path(tmpdir) / "train_shard_00000.bin"
            val_shard = Path(tmpdir) / "val_shard_00000.bin"
            create_dummy_shard(train_shard, num_tokens=10000)
            create_dummy_shard(val_shard, num_tokens=100)
            
            batch_size = 2
            seq_length = 32
            
            trainer = ExtendedTrainer(
                model=model,
                train_shard_paths=[str(train_shard)],
                val_shard_paths=[str(val_shard)],
                batch_size=batch_size,
                seq_length=seq_length,
                total_steps=100,
                warmup_steps=10,
                rng_seed=42,
            )
            
            # Simulate multiple steps
            for _ in range(5):
                inputs, targets = trainer.get_train_batch()
            
            expected_tokens = 5 * batch_size * seq_length
            assert trainer.tokens_processed == expected_tokens


class TestExtendedTrainerCheckpoint:
    """Tests for ExtendedTrainer checkpoint functionality."""
    
    def test_save_load_rng_states(self):
        """Test that RNG states are saved and loaded correctly."""
        model = make_model()
        
        with tempfile.TemporaryDirectory() as tmpdir:
            train_shard = Path(tmpdir) / "train_shard_00000.bin"
            val_shard = Path(tmpdir) / "val_shard_00000.bin"
            
            create_dummy_shard(train_shard, num_tokens=1000)
            create_dummy_shard(val_shard, num_tokens=100)
            
            trainer1 = ExtendedTrainer(
                model=model,
                train_shard_paths=[str(train_shard)],
                val_shard_paths=[str(val_shard)],
                batch_size=2,
                seq_length=32,
                total_steps=100,
                warmup_steps=10,
                rng_seed=42,
                checkpoint_dir=tmpdir,
            )
            
            # Sample some blocks to advance RNG
            for _ in range(3):
                inputs, targets = trainer1.get_train_batch()
            
            # Get RNG states before save
            train_state_before = trainer1.train_rng.bit_generator.state
            val_state_before = trainer1.val_rng.bit_generator.state
            
            # Save
            trainer1.save()
            
            # Create new trainer with same initial config but different seed (to be overwritten)
            trainer2 = ExtendedTrainer(
                model=model,
                train_shard_paths=[str(train_shard)],
                val_shard_paths=[str(val_shard)],
                batch_size=2,
                seq_length=32,
                total_steps=100,
                warmup_steps=10,
                rng_seed=999,  # Different initial seed
                checkpoint_dir=tmpdir,
            )
            
            # Load checkpoint to restore RNG state
            trainer2.step = trainer1.step
            trainer2.tokens_processed = trainer1.tokens_processed
            trainer2.current_train_shard_idx = trainer1.current_train_shard_idx
            trainer2.current_val_shard_idx = trainer1.current_val_shard_idx
            
            # Load and restore RNG states from checkpoint
            from mini_llm.checkpoint import load_checkpoint
            param_names = [p.name for p in model.parameters()]
            loaded_params, optimizer_state, training_state = load_checkpoint(
                Path(tmpdir),
                param_names=param_names,
            )
            
            # Restore RNG states
            if training_state and "train_rng_state" in training_state:
                trainer2.train_rng.bit_generator.state = training_state["train_rng_state"]
            if training_state and "val_rng_state" in training_state:
                trainer2.val_rng.bit_generator.state = training_state["val_rng_state"]
            
            # Get next batch with same RNG state
            inputs1, targets1 = trainer1.get_train_batch()
            inputs2, targets2 = trainer2.get_train_batch()
            
            # Should produce identical batches (deterministic with same RNG state)
            np.testing.assert_array_equal(inputs1, inputs2)
            np.testing.assert_array_equal(targets1, targets2)


class TestExtendedTrainerLogging:
    """Tests for ExtendedTrainer logging."""
    
    def test_tokens_processed_in_log(self):
        """Test that tokens_processed is included in log entries."""
        model = make_model()
        
        with tempfile.TemporaryDirectory() as tmpdir:
            train_shard = Path(tmpdir) / "train_shard_00000.bin"
            val_shard = Path(tmpdir) / "val_shard_00000.bin"
            
            # Create a larger shard for multiple steps
            create_dummy_shard(train_shard, num_tokens=100000)
            create_dummy_shard(val_shard, num_tokens=100)
            
            log_file = Path(tmpdir) / "train.csv"
            
            trainer = ExtendedTrainer(
                model=model,
                train_shard_paths=[str(train_shard)],
                val_shard_paths=[str(val_shard)],
                batch_size=4,
                seq_length=64,
                total_steps=100,
                warmup_steps=10,
                log_file=str(log_file),
                rng_seed=42,
            )
            
            # Run a few steps
            for _ in range(5):
                trainer.train_step()
            
            # Check log file header
            with open(log_file, "r") as f:
                header = f.readline().strip()
            
            assert "tokens_processed" in header
    
    def test_log_file_created(self):
        """Test that log file is created."""
        model = make_model()
        
        with tempfile.TemporaryDirectory() as tmpdir:
            train_shard = Path(tmpdir) / "train_shard_00000.bin"
            val_shard = Path(tmpdir) / "val_shard_00000.bin"
            create_dummy_shard(train_shard, num_tokens=1000)
            create_dummy_shard(val_shard, num_tokens=100)
            
            log_file = Path(tmpdir) / "train.csv"
            
            trainer = ExtendedTrainer(
                model=model,
                train_shard_paths=[str(train_shard)],
                val_shard_paths=[str(val_shard)],
                batch_size=2,
                seq_length=32,
                total_steps=100,
                warmup_steps=10,
                log_file=str(log_file),
                rng_seed=42,
            )
            
            # Log file should exist after init (header written)
            assert log_file.exists()


class TestExtendedTrainerMixedSources:
    def test_weighted_source_selection(self):
        model = make_model()
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            paths = {}
            for name, token in (("a", 11), ("b", 22)):
                train = root / f"{name}_train.bin"
                val = root / f"{name}_val.bin"
                np.full(2048, token, dtype=np.uint16).tofile(train)
                np.full(1024, token, dtype=np.uint16).tofile(val)
                paths[name] = (train, val)

            trainer = ExtendedTrainer(
                model=model,
                train_shard_paths=[str(paths["a"][0]), str(paths["b"][0])],
                val_shard_paths=[str(paths["a"][1]), str(paths["b"][1])],
                train_source_shards={
                    name: [str(pair[0])] for name, pair in paths.items()
                },
                val_source_shards={
                    name: [str(pair[1])] for name, pair in paths.items()
                },
                source_weights={"a": 1.0, "b": 0.0},
                batch_size=2,
                seq_length=32,
                total_steps=10,
                warmup_steps=1,
                rng_seed=42,
            )

            inputs, targets = trainer.get_train_batch()
            assert np.all(inputs == 11)
            assert np.all(targets == 11)
            assert trainer.train_source_batch_counts == {"a": 1, "b": 0}

            val_inputs, val_targets = trainer.get_val_batch()
            assert np.all(val_inputs == 11)
            assert np.all(val_targets == 11)
            assert trainer.val_source_batch_counts == {"a": 1, "b": 0}

    def test_mixed_source_weights_are_normalized(self):
        model = make_model()
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            paths = {}
            for name in ("a", "b"):
                train = root / f"{name}_train.bin"
                val = root / f"{name}_val.bin"
                np.ones(512, dtype=np.uint16).tofile(train)
                np.ones(512, dtype=np.uint16).tofile(val)
                paths[name] = (train, val)

            trainer = ExtendedTrainer(
                model=model,
                train_shard_paths=[str(paths["a"][0]), str(paths["b"][0])],
                val_shard_paths=[str(paths["a"][1]), str(paths["b"][1])],
                train_source_shards={
                    name: [str(pair[0])] for name, pair in paths.items()
                },
                val_source_shards={
                    name: [str(pair[1])] for name, pair in paths.items()
                },
                source_weights={"a": 7.0, "b": 3.0},
                batch_size=1,
                seq_length=16,
                total_steps=10,
                warmup_steps=1,
            )
            assert trainer.source_weights["a"] == pytest.approx(0.7)
            assert trainer.source_weights["b"] == pytest.approx(0.3)
            assert trainer.shard_cache_size >= 2
