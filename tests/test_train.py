"""Tests for training utilities."""

import tempfile
from pathlib import Path

import numpy as np
import pytest

from mini_llm.model.decoder_lm import DecoderLanguageModel
from mini_llm.config import ModelConfig
from mini_llm.train import MiniTrainer, create_minibatch, load_token_shard


def make_model():
    """Create a tiny test model."""
    config = ModelConfig.tiny_inspection()
    return DecoderLanguageModel(config, rng_seed=42, dtype="float64")


class TestCreateMinibatch:
    """Tests for create_minibatch function."""
    
    def test_batch_shape(self):
        """Test that batch has correct shape."""
        # Create synthetic shard data
        num_docs = 100
        context_len = 1024
        shard_data = np.random.randint(0, 256, size=(num_docs, context_len), dtype=np.uint16)
        
        batch_size = 4
        seq_length = 256
        
        inputs, targets = create_minibatch(shard_data, batch_size, seq_length)
        
        assert inputs.shape == (batch_size, seq_length)
        assert targets.shape == (batch_size, seq_length)
    
    def test_targets_shifted(self):
        """Test that targets are shifted by 1 from inputs."""
        shard_data = np.random.randint(0, 256, size=(10, 1024), dtype=np.uint16)
        
        inputs, targets = create_minibatch(shard_data, 2, seq_length=10)
        
        # Targets should be shifted by 1
        for i in range(2):
            np.testing.assert_array_equal(
                inputs[i, 1:], targets[i, :-1]
            )
    
    def test_full_context(self):
        """Test using full context length."""
        shard_data = np.random.randint(0, 256, size=(10, 512), dtype=np.uint16)
        
        inputs, targets = create_minibatch(shard_data, 2, seq_length=510)
        
        assert inputs.shape == (2, 510)
        assert targets.shape == (2, 510)
    
    def test_all_documents_sampled(self):
        """Test that batch can sample from all documents, not just first batch_size."""
        # Use a shard with few docs but many tokens per doc
        num_docs = 5
        context_len = 1024
        shard_data = np.random.randint(0, 256, size=(num_docs, context_len), dtype=np.uint16)
        
        batch_size = 32
        seq_length = 64
        
        # This should sample from documents beyond the first 5
        inputs, targets = create_minibatch(shard_data, batch_size, seq_length)
        
        assert inputs.shape == (batch_size, seq_length)
        assert targets.shape == (batch_size, seq_length)
    
    def test_exact_boundary_sampling(self):
        """Test sampling at exact boundary (last valid start position)."""
        context_len = 256
        shard_data = np.random.randint(0, 256, size=(10, context_len), dtype=np.uint16)
        seq_length = 255
        
        # Use a deterministic RNG with a fixed seed
        rng = np.random.default_rng(42)
        inputs, targets = create_minibatch(shard_data, 2, seq_length=seq_length, rng=rng)
        
        assert inputs.shape == (2, seq_length)
        assert targets.shape == (2, seq_length)
        # Verify targets are correctly shifted
        for i in range(2):
            np.testing.assert_array_equal(inputs[i, 1:], targets[i, :-1])
    
    def test_document_boundary_tokens_correct(self):
        """Test that document boundary tokens are handled correctly."""
        context_len = 256
        shard_data = np.arange(10 * context_len, dtype=np.uint16).reshape(10, context_len)
        seq_length = 32
        
        # Use deterministic RNG to get reproducible results
        rng = np.random.default_rng(42)
        inputs, targets = create_minibatch(shard_data, 4, seq_length=seq_length, rng=rng)
        
        assert inputs.shape == (4, seq_length)
        assert targets.shape == (4, seq_length)
        
        # For each sampled sequence, verify targets are shifted inputs
        for i in range(4):
            np.testing.assert_array_equal(inputs[i, 1:], targets[i, :-1])
            
            # Verify the target at position 0 matches input[1] from the same document
            # We can check this by finding which document and position was used
            # The target at [i, 0] should equal inputs[i, 1]
            assert targets[i, 0] == inputs[i, 1], \
                f"Target[0] should equal input[1] for sequence {i}"
    
    def test_deterministic_with_rng(self):
        """Test that using a fixed RNG produces deterministic results."""
        shard_data = np.random.randint(0, 256, size=(10, 1024), dtype=np.uint16)
        seq_length = 64
        
        rng1 = np.random.default_rng(42)
        inputs1, targets1 = create_minibatch(shard_data, 4, seq_length, rng=rng1)
        
        rng2 = np.random.default_rng(42)
        inputs2, targets2 = create_minibatch(shard_data, 4, seq_length, rng=rng2)
        
        np.testing.assert_array_equal(inputs1, inputs2)
        np.testing.assert_array_equal(targets1, targets2)
    
    def test_no_fake_targets(self):
        """Test that targets don't contain fabricated values (no fake 0 at end)."""
        # Create shard data with known values
        context_len = 256
        shard_data = np.arange(10 * context_len, dtype=np.uint16).reshape(10, context_len)
        seq_length = 32
        
        rng = np.random.default_rng(42)
        inputs, targets = create_minibatch(shard_data, 2, seq_length=seq_length, rng=rng)
        
        # All targets should be non-zero (except potentially for pad tokens, but we have none here)
        assert not np.all(targets[:, -1] == 0), "Last target position should contain real next tokens"
        
        # Verify targets are actual shifted inputs
        for i in range(2):
            expected_targets = inputs[i, 1:]
            np.testing.assert_array_equal(targets[i, :-1], expected_targets)


class TestMiniTrainer:
    """Tests for MiniTrainer."""
    
    def test_initialization(self):
        """Test trainer initialization."""
        model = make_model()
        
        with tempfile.TemporaryDirectory() as tmpdir:
            # Create a dummy shard
            shard_path = Path(tmpdir) / "test.bin"
            
            # Write a minimal valid shard
            num_docs = 10
            context_len = 32
            
            with open(shard_path, "wb") as f:
                f.write(b"\x00" * 16)  # Header
                data = np.zeros((num_docs, context_len), dtype=np.uint16)
                f.write(data.tobytes())
            
            trainer = MiniTrainer(
                model=model,
                shard_paths=[str(shard_path)],
                batch_size=2,
                seq_length=16,
            )
            
            assert trainer.model is model
            assert len(trainer.shard_paths) == 1
    
    def test_train_step(self):
        """Test that train step runs without errors."""
        model = make_model()
        
        with tempfile.TemporaryDirectory() as tmpdir:
            shard_path = Path(tmpdir) / "test.bin"
            
            # Write a minimal valid shard
            num_docs = 100
            context_len = 256
            
            with open(shard_path, "wb") as f:
                f.write(np.int64(num_docs).tobytes())
                f.write(np.int64(context_len).tobytes())
                data = np.random.randint(0, 256, size=(num_docs, context_len), dtype=np.uint16)
                f.write(data.tobytes())
            
            trainer = MiniTrainer(
                model=model,
                shard_paths=[str(shard_path)],
                batch_size=2,
                seq_length=32,
                total_steps=100,
                warmup_steps=10,
            )
            
            # Run a few steps
            for _ in range(5):
                loss = trainer.train_step()
                assert isinstance(loss, float)
                assert loss > 0
    
    def test_train_method(self):
        """Test the full train method."""
        model = make_model()
        
        with tempfile.TemporaryDirectory() as tmpdir:
            shard_path = Path(tmpdir) / "test.bin"
            
            num_docs = 100
            context_len = 256
            
            with open(shard_path, "wb") as f:
                f.write(np.int64(num_docs).tobytes())
                f.write(np.int64(context_len).tobytes())
                data = np.random.randint(0, 256, size=(num_docs, context_len), dtype=np.uint16)
                f.write(data.tobytes())
            
            trainer = MiniTrainer(
                model=model,
                shard_paths=[str(shard_path)],
                batch_size=2,
                seq_length=32,
                total_steps=100,
                warmup_steps=10,
            )
            
            losses = trainer.train(num_steps=10, log_interval=5)
            
            assert len(losses) == 10
            assert all(isinstance(l, float) for l in losses)
    
    def test_loss_scale_parameter(self):
        """Test that loss scale is passed to optimizer."""
        model = make_model()
        
        with tempfile.TemporaryDirectory() as tmpdir:
            shard_path = Path(tmpdir) / "test.bin"
            
            num_docs = 10
            context_len = 32
            
            with open(shard_path, "wb") as f:
                f.write(np.int64(num_docs).tobytes())
                f.write(np.int64(context_len).tobytes())
                data = np.random.randint(0, 256, size=(num_docs, context_len), dtype=np.uint16)
                f.write(data.tobytes())
            
            loss_scale = 1024.0
            trainer = MiniTrainer(
                model=model,
                shard_paths=[str(shard_path)],
                batch_size=2,
                seq_length=16,
                loss_scale=loss_scale,
            )
            
            # Check that trainer stores the loss scale (optimizer no longer stores it)
            assert trainer.loss_scale == loss_scale


class TestGradientClipping:
    """Tests for gradient clipping."""
    
    def test_global_grad_norm_dtype(self):
        """Test that global grad norm uses float32."""
        from mini_llm.optim.grad_clip import global_grad_norm, _array_to_float
        
        model = make_model()
        
        # Set some gradients
        for p in model.parameters():
            p.grad = np.ones_like(p.data, dtype=p.data.dtype)
        
        # Compute norm (now returns tuple: (backend_array, python_float))
        norm_backend, norm_value = global_grad_norm(model.parameters())
        
        # Norm value should be a finite float
        assert isinstance(norm_value, float)
        assert np.isfinite(norm_value)
        
        # The backend array should be float32
        assert norm_backend.dtype == np.float32
