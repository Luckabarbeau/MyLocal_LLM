"""Tests for packed token dataset functionality."""

import tempfile
from pathlib import Path

import numpy as np
import pytest

from mini_llm.data.packed_dataset import (
    PackedTokenDataset,
    PackedDatasetGenerator,
    DatasetManifest,
    load_packed_dataset,
    verify_manifest,
)


class TestDatasetManifest:
    """Tests for DatasetManifest."""
    
    def test_to_dict_roundtrip(self):
        """Test that to_dict and from_dict are inverses."""
        manifest = DatasetManifest(
            format_version="1.0",
            tokenizer_hash="abc123",
            vocab_size=32000,
            token_dtype="uint16",
            eos_token_id=2,
            total_train_tokens=1000000,
            total_val_tokens=100000,
            train_document_count=50000,
            val_document_count=5000,
            train_shard_count=4,
            val_shard_count=1,
            source_dataset_name="Cosmopedia-v2",
            preprocessing_seed=42,
            context_length=512,
        )
        
        d = manifest.to_dict()
        manifest2 = DatasetManifest.from_dict(d)
        
        assert manifest.format_version == manifest2.format_version
        assert manifest.tokenizer_hash == manifest2.tokenizer_hash
        assert manifest.vocab_size == manifest2.vocab_size
    
    def test_invalid_dtype(self):
        """Test that invalid token dtype raises error."""
        manifest = DatasetManifest(
            format_version="1.0",
            tokenizer_hash="abc123",
            vocab_size=32000,
            token_dtype="float64",  # Invalid
            eos_token_id=2,
            total_train_tokens=1000000,
            total_val_tokens=100000,
            train_document_count=50000,
            val_document_count=5000,
            train_shard_count=4,
            val_shard_count=1,
            source_dataset_name="Cosmopedia-v2",
            preprocessing_seed=42,
            context_length=512,
        )
        
        with pytest.raises(ValueError):
            verify_manifest(manifest)
    
    def test_vocab_exceeds_uint16(self):
        """Test that vocab > 65535 with uint16 raises error."""
        manifest = DatasetManifest(
            format_version="1.0",
            tokenizer_hash="abc123",
            vocab_size=70000,  # Exceeds uint16
            token_dtype="uint16",
            eos_token_id=2,
            total_train_tokens=1000000,
            total_val_tokens=100000,
            train_document_count=50000,
            val_document_count=5000,
            train_shard_count=4,
            val_shard_count=1,
            source_dataset_name="Cosmopedia-v2",
            preprocessing_seed=42,
            context_length=512,
        )
        
        with pytest.raises(ValueError):
            verify_manifest(manifest)


class TestPackedDatasetGenerator:
    """Tests for PackedDatasetGenerator."""
    
    def test_initialization(self):
        """Test generator initialization."""
        gen = PackedDatasetGenerator(
            tokenizer_hash="abc123",
            vocab_size=16384,
            eos_token_id=2,
            context_length=512,
            shard_size_mb=256.0,
            preprocessing_seed=42,
        )
        
        assert gen.vocab_size == 16384
        assert gen.eos_token_id == 2
    
    def test_initialization_exceeds_uint16(self):
        """Test that vocab > 65535 with uint16 raises error."""
        with pytest.raises(ValueError, match="exceeds uint16 limit"):
            PackedDatasetGenerator(
                tokenizer_hash="abc123",
                vocab_size=70000,
                eos_token_id=2,
                dtype="uint16",  # This should fail
            )
    
    def test_document_hash_deterministic(self):
        """Test that document hashing is deterministic."""
        gen = PackedDatasetGenerator(
            tokenizer_hash="abc123",
            vocab_size=16384,
            eos_token_id=2,
        )
        
        doc1 = "This is a test document."
        doc2 = "This is another test document."
        
        hash1_a = gen.compute_document_hash(doc1)
        hash1_b = gen.compute_document_hash(doc1)
        hash2 = gen.compute_document_hash(doc2)
        
        # Same document should have same hash
        assert hash1_a == hash1_b
        # Different documents should (with high probability) have different hashes
        assert hash1_a != hash2
    
    def test_train_val_split_deterministic(self):
        """Test that train/val split is deterministic."""
        documents = [f"Document {i}" for i in range(100)]
        
        gen1 = PackedDatasetGenerator(
            tokenizer_hash="abc123",
            vocab_size=16384,
            eos_token_id=2,
            preprocessing_seed=42,
        )
        
        gen2 = PackedDatasetGenerator(
            tokenizer_hash="abc123",
            vocab_size=16384,
            eos_token_id=2,
            preprocessing_seed=42,
        )
        
        train1, val1 = gen1.split_documents(documents, val_ratio=0.1)
        train2, val2 = gen2.split_documents(documents, val_ratio=0.1)
        
        # Same seed should produce same split
        assert set(train1) == set(train2)
        assert set(val1) == set(val2)
        
        # No overlap between train and val
        assert len(set(train1) & set(val1)) == 0
        
        # Counts should be roughly correct
        assert len(val1) == 10  # 10% of 100


class TestPackedTokenDataset:
    """Tests for PackedTokenDataset."""
    
    def test_get_block_ranges(self):
        """Test block range calculation."""
        # Create a minimal mock dataset with correct shard count
        manifest = DatasetManifest(
            format_version="1.0",
            tokenizer_hash="abc123",
            vocab_size=16384,
            token_dtype="uint16",
            eos_token_id=2,
            total_train_tokens=10000,
            total_val_tokens=1000,
            train_document_count=50000,
            val_document_count=5000,
            train_shard_count=1,  # Fixed: match actual shard count
            val_shard_count=1,
            source_dataset_name="Cosmopedia-v2",
            preprocessing_seed=42,
            context_length=512,
        )
        
        # Create temp files
        with tempfile.TemporaryDirectory() as tmpdir:
            shard_path = Path(tmpdir) / "train_shard_00000.bin"
            
            # Write 10000 tokens
            tokens = np.arange(10000, dtype=np.uint16)
            tokens.tofile(shard_path)
            
            dataset = PackedTokenDataset(
                shard_paths=[shard_path],
                manifest=manifest,
            )
            
            # Get block ranges for seq_length=512
            # Each block needs 513 tokens (512 + 1)
            ranges = dataset.get_block_ranges(seq_length=512)
            
            # Should have 10000 // 513 = 19 blocks
            assert len(ranges) == 19
            
            # First block starts at 0, ends at 513
            assert ranges[0] == (0, 513)
            
            # Last block: valid starts are 0, 513, 1026, ..., max_start where max_start+513 <= 10000
            # max_start = 10000 - 513 = 9487
            # Number of blocks = floor(10000 / 513) = 19
            # Last start = (19-1) * 513 = 18 * 513 = 9234
            assert ranges[-1][0] == 18 * 513
    
    def test_get_block(self):
        """Test block extraction."""
        with tempfile.TemporaryDirectory() as tmpdir:
            shard_path = Path(tmpdir) / "train_shard_00000.bin"
            
            # Write tokens: [0, 1, 2, ..., 99]
            tokens = np.arange(100, dtype=np.uint16)
            tokens.tofile(shard_path)
            
            manifest = DatasetManifest(
                format_version="1.0",
                tokenizer_hash="abc123",
                vocab_size=16384,
                token_dtype="uint16",
                eos_token_id=2,
                total_train_tokens=100,
                total_val_tokens=0,
                train_document_count=1,
                val_document_count=0,
                train_shard_count=1,
                val_shard_count=0,
                source_dataset_name="Cosmopedia-v2",
                preprocessing_seed=42,
                context_length=512,
            )
            
            dataset = PackedTokenDataset(
                shard_paths=[shard_path],
                manifest=manifest,
            )
            
            # Get block with seq_length=10 (needs 11 tokens)
            inputs, targets = dataset.get_block(start=0, seq_length=10)
            
            assert inputs.shape == (10,)
            assert targets.shape == (10,)
            
            # Verify next-token relationship
            np.testing.assert_array_equal(inputs, np.arange(0, 10))
            np.testing.assert_array_equal(targets, np.arange(1, 11))
    
    def test_sample_block(self):
        """Test random block sampling."""
        with tempfile.TemporaryDirectory() as tmpdir:
            shard_path = Path(tmpdir) / "train_shard_00000.bin"
            
            # Write tokens: [0, 1, 2, ..., 99]
            tokens = np.arange(100, dtype=np.uint16)
            tokens.tofile(shard_path)
            
            manifest = DatasetManifest(
                format_version="1.0",
                tokenizer_hash="abc123",
                vocab_size=16384,
                token_dtype="uint16",
                eos_token_id=2,
                total_train_tokens=100,
                total_val_tokens=0,
                train_document_count=1,
                val_document_count=0,
                train_shard_count=1,
                val_shard_count=0,
                source_dataset_name="Cosmopedia-v2",
                preprocessing_seed=42,
                context_length=512,
            )
            
            dataset = PackedTokenDataset(
                shard_paths=[shard_path],
                manifest=manifest,
            )
            
            rng = np.random.default_rng(42)
            inputs, targets = dataset.sample_block(seq_length=10, rng=rng)
            
            assert inputs.shape == (10,)
            assert targets.shape == (10,)
            
            # Verify next-token relationship
            np.testing.assert_array_equal(targets, inputs + 1)


class TestLoadPackedDataset:
    """Tests for load_packed_dataset."""
    
    def test_load_dataset(self):
        """Test loading a packed dataset."""
        with tempfile.TemporaryDirectory() as tmpdir:
            # Create a simple shard
            shard_path = Path(tmpdir) / "train_shard_00000.bin"
            tokens = np.arange(100, dtype=np.uint16)
            tokens.tofile(shard_path)
            
            # Create manifest
            manifest = DatasetManifest(
                format_version="1.0",
                tokenizer_hash="abc123",
                vocab_size=16384,
                token_dtype="uint16",
                eos_token_id=2,
                total_train_tokens=100,
                total_val_tokens=0,
                train_document_count=1,
                val_document_count=0,
                train_shard_count=1,
                val_shard_count=0,
                source_dataset_name="Cosmopedia-v2",
                preprocessing_seed=42,
                context_length=512,
            )
            
            manifest_path = Path(tmpdir) / "manifest.json"
            with open(manifest_path, "w") as f:
                import json
                json.dump(manifest.to_dict(), f)
            
            dataset, loaded_manifest = load_packed_dataset(tmpdir, is_train=True)
            
            assert dataset.manifest.format_version == "1.0"
            assert dataset.total_tokens == 100


class TestCreateMinibatchPacked:
    """Tests for create_minibatch with packed format."""
    
    def test_packed_format_detection(self):
        """Test that packed format (1D) is detected correctly."""
        from mini_llm.data.token_shards import create_minibatch
        
        # 1D array = packed format
        packed_tokens = np.arange(1000, dtype=np.uint16)
        
        inputs, targets = create_minibatch(
            packed_tokens,
            batch_size=4,
            seq_length=64,
            rng=np.random.default_rng(42),
        )
        
        assert inputs.shape == (4, 64)
        assert targets.shape == (4, 64)
    
    def test_packed_format_next_token(self):
        """Test that packed format correctly produces next-token targets."""
        from mini_llm.data.token_shards import create_minibatch
        
        # Create specific packed data
        packed_tokens = np.arange(1000, dtype=np.uint16)
        
        inputs, targets = create_minibatch(
            packed_tokens,
            batch_size=1,
            seq_length=10,
            rng=np.random.default_rng(42),
        )
        
        # For the one sequence, verify next-token relationship
        # (Note: random start position, so we just check the pattern)
        assert targets.shape == (1, 10)
        assert inputs.shape == (1, 10)
        
        # Each target[i] should equal input[i+1] within the sequence
        np.testing.assert_array_equal(targets[0, :-1], inputs[0, 1:])
    
    def test_packed_format_requires_seq_length(self):
        """Test that packed format requires seq_length."""
        from mini_llm.data.token_shards import create_minibatch
        
        packed_tokens = np.arange(1000, dtype=np.uint16)
        
        with pytest.raises(ValueError, match="seq_length must be specified"):
            create_minibatch(packed_tokens, batch_size=4, seq_length=None)


class TestOldFormatBackwardCompatibility:
    """Tests for backward compatibility with old rectangular format."""
    
    def test_old_format_detection(self):
        """Test that old format (2D) is still supported."""
        from mini_llm.data.token_shards import create_minibatch
        
        # 2D array = old format
        shard_data = np.random.randint(0, 100, size=(10, 256), dtype=np.uint16)
        
        inputs, targets = create_minibatch(
            shard_data,
            batch_size=4,
            seq_length=64,
            rng=np.random.default_rng(42),
        )
        
        assert inputs.shape == (4, 64)
        assert targets.shape == (4, 64)
