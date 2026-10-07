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

class TestExtendedTrainerStepAccounting:
    """Regression tests for completed optimizer-step numbering."""

    def test_train_uses_completed_step_for_logging_validation_and_save(self, capsys):
        model = make_model()

        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            train_shard = root / "train_shard_00000.bin"
            val_shard = root / "val_shard_00000.bin"
            create_dummy_shard(train_shard, num_tokens=1024)
            create_dummy_shard(val_shard, num_tokens=1024)
            log_file = root / "train.csv"

            trainer = ExtendedTrainer(
                model=model,
                train_shard_paths=[str(train_shard)],
                val_shard_paths=[str(val_shard)],
                batch_size=1,
                seq_length=16,
                total_steps=10,
                warmup_steps=2,
                log_file=str(log_file),
                val_interval=5,
                val_steps=1,
                save_interval=10,
                rng_seed=42,
            )

            validation_steps = []
            save_steps = []

            def fake_train_step():
                trainer.step += 1
                trainer.optimizer.lr = 1.0e-4
                return float(trainer.step), 1.0

            def fake_compute_val_loss():
                validation_steps.append(trainer.step)
                return 1.25

            def fake_save():
                save_steps.append(trainer.step)

            trainer.train_step = fake_train_step
            trainer.compute_val_loss = fake_compute_val_loss
            trainer.save = fake_save

            losses = trainer.train(num_steps=10, log_interval=1)

            assert len(losses) == 10
            assert trainer.step == 10
            assert validation_steps == [5, 10]
            assert save_steps == [10]

            output = capsys.readouterr().out
            assert "Step 10/10" in output
            assert "Step 11/10" not in output

            rows = log_file.read_text().strip().splitlines()
            logged_steps = [int(row.split(",", 1)[0]) for row in rows[1:]]
            assert logged_steps == list(range(1, 11))


def test_0056_chunked_lm_head_training_step(monkeypatch):
    """The training-only 0056 path should run without materialized full logits."""
    monkeypatch.setenv("MINI_LLM_LM_HEAD_CHUNK_TOKENS", "4")
    model = make_model()
    with tempfile.TemporaryDirectory() as tmpdir:
        train_shard = Path(tmpdir) / "train_shard_00000.bin"
        val_shard = Path(tmpdir) / "val_shard_00000.bin"
        create_dummy_shard(train_shard, num_tokens=4096)
        create_dummy_shard(val_shard, num_tokens=1024)
        trainer = ExtendedTrainer(
            model=model,
            train_shard_paths=[str(train_shard)],
            val_shard_paths=[str(val_shard)],
            batch_size=2,
            seq_length=8,
            grad_accum_steps=1,
            total_steps=10,
            warmup_steps=1,
            rng_seed=44,
        )
        loss, grad_norm = trainer.train_step()
        assert np.isfinite(loss)
        assert np.isfinite(grad_norm)
        assert trainer.lm_head_chunk_tokens == 4


def test_block_activation_checkpoint_training_step(monkeypatch):
    """0057 trainer wiring should execute one checkpointed training step."""
    monkeypatch.setenv("MINI_LLM_ACTIVATION_CHECKPOINT", "block")
    monkeypatch.setenv("MINI_LLM_LM_HEAD_CHUNK_TOKENS", "8")
    model = make_model()

    with tempfile.TemporaryDirectory() as tmpdir:
        train_shard = Path(tmpdir) / "train_shard_00000.bin"
        val_shard = Path(tmpdir) / "val_shard_00000.bin"
        create_dummy_shard(train_shard, num_tokens=4096)
        create_dummy_shard(val_shard, num_tokens=512)
        trainer = ExtendedTrainer(
            model=model,
            train_shard_paths=[str(train_shard)],
            val_shard_paths=[str(val_shard)],
            batch_size=1,
            seq_length=16,
            grad_accum_steps=1,
            total_steps=4,
            warmup_steps=1,
            rng_seed=73,
        )
        assert trainer.activation_checkpoint == "block"
        loss, grad_norm = trainer.train_step()
        assert np.isfinite(loss)
        assert np.isfinite(grad_norm)


class TestMemoryCoverageDiagnostics:
    """CPU-only accounting for real same-document memory usage."""

    @staticmethod
    def make_accounting_trainer(tmp_path=None):
        from types import SimpleNamespace

        trainer = object.__new__(ExtendedTrainer)
        trainer.batch_size = 2
        trainer.seq_length = 65_536
        trainer.log_file = None if tmp_path is None else Path(tmp_path) / "train.csv"
        trainer.step = 10
        trainer.memory_context = SimpleNamespace(
            distant_memory_length=61_440,
            target_length=4_096,
            active_length=4_096,
        )
        return trainer

    def test_real_history_and_padding_are_counted_separately(self):
        trainer = self.make_accounting_trainer()
        stats = trainer._empty_memory_token_stats()
        metadata = {
            # Real history lengths are 2,048 and 50,000 tokens.
            "history_valid_starts": np.asarray([59_392, 11_440], dtype=np.int64),
            "target_valid_lengths": np.asarray([4_096, 3_000], dtype=np.int64),
        }
        mask = np.zeros((2, 4_096), dtype=np.float32)
        mask[0, :4_096] = 1.0
        mask[1, :3_000] = 1.0

        trainer._accumulate_memory_batch_token_stats(stats, metadata, mask)

        assert stats["samples"] == 2
        assert stats["valid_history_tokens"] == 52_048
        assert stats["valid_active_tokens"] == 7_096
        assert stats["valid_source_tokens"] == 59_144
        assert stats["supervised_tokens"] == 7_096
        assert stats["history_sum"] == 52_048
        assert stats["history_min"] == 2_048
        assert stats["history_max"] == 50_000
        assert sum(stats["history_hist"]) == 2
        # One sample is in [0, 4k), the other in the final [48k, 60k] bin.
        assert stats["history_hist"][0] == 1
        assert stats["history_hist"][-1] == 1

    def test_memory_stats_merge_preserves_distribution(self):
        trainer = self.make_accounting_trainer()
        first = trainer._empty_memory_token_stats()
        second = trainer._empty_memory_token_stats()
        mask = np.ones((2, 4_096), dtype=np.float32)

        trainer._accumulate_memory_batch_token_stats(
            first,
            {
                "history_valid_starts": np.asarray([61_440, 57_344]),
                "target_valid_lengths": np.asarray([4_096, 4_096]),
            },
            mask,
        )
        trainer._accumulate_memory_batch_token_stats(
            second,
            {
                "history_valid_starts": np.asarray([45_056, 0]),
                "target_valid_lengths": np.asarray([4_096, 4_096]),
            },
            mask,
        )
        trainer._merge_memory_token_stats(first, second)

        assert first["samples"] == 4
        assert first["history_min"] == 0
        assert first["history_max"] == 61_440
        assert first["history_sum"] == 81_920
        assert sum(first["history_hist"]) == 4

    def test_memory_metrics_use_companion_csv(self, tmp_path):
        trainer = self.make_accounting_trainer(tmp_path)
        trainer._setup_memory_metrics_logging()
        stats = trainer._empty_memory_token_stats()
        trainer._accumulate_memory_batch_token_stats(
            stats,
            {
                "history_valid_starts": np.asarray([57_344, 49_152]),
                "target_valid_lengths": np.asarray([4_096, 4_096]),
            },
            np.ones((2, 4_096), dtype=np.float32),
        )
        trainer._log_memory_metrics(stats, steps_in_interval=1, elapsed_seconds=2.0)

        path = Path(tmp_path) / "train.memory.csv"
        assert path.exists()
        rows = path.read_text().strip().splitlines()
        assert len(rows) == 2
        assert "valid_history_tokens" in rows[0]
        assert "history_mean_per_sample" in rows[0]
        assert "history_bin_0_4096" in rows[0]


def test_0060b3_chunked_lm_head_validation_matches_full_logits(monkeypatch):
    """Validation should reuse the bounded LM-head path and never call full forward."""
    monkeypatch.setenv("MINI_LLM_LM_HEAD_CHUNK_TOKENS", "4")
    model = make_model()

    with tempfile.TemporaryDirectory() as tmpdir:
        train_shard = Path(tmpdir) / "train_shard_00000.bin"
        val_shard = Path(tmpdir) / "val_shard_00000.bin"
        create_dummy_shard(train_shard, num_tokens=4096)
        create_dummy_shard(val_shard, num_tokens=4096)
        trainer = ExtendedTrainer(
            model=model,
            train_shard_paths=[str(train_shard)],
            val_shard_paths=[str(val_shard)],
            batch_size=2,
            seq_length=8,
            grad_accum_steps=1,
            total_steps=10,
            warmup_steps=1,
            val_steps=1,
            rng_seed=91,
        )

        inputs, targets = trainer.get_val_batch()
        reference_logits = model.forward(inputs, return_cache=False)
        reference_loss, _ = model.compute_loss(reference_logits, targets)

        def fixed_val_batch(return_metadata=False):
            if return_metadata:
                return inputs, targets, None
            return inputs, targets

        trainer.get_val_batch = fixed_val_batch

        def full_forward_must_not_run(*args, **kwargs):
            raise AssertionError(
                "chunked validation must not materialize full vocabulary logits"
            )

        monkeypatch.setattr(model, "forward", full_forward_must_not_run)
        chunked_loss = trainer.compute_val_loss()

        assert trainer.lm_head_chunk_tokens == 4
        assert chunked_loss == pytest.approx(reference_loss, rel=1e-6, abs=1e-7)
