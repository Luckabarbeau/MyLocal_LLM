"""Tests for the SimpleBPETokenizer."""

import tempfile
from pathlib import Path

import numpy as np
import pytest

from mini_llm.tokenizer.tokenizer import SimpleBPETokenizer


class TestSimpleBPETokenizer:
    """Tests for SimpleBPETokenizer."""
    
    def test_basic_encoding_decoding(self):
        """Test basic encode/decode roundtrip after training."""
        tokenizer = SimpleBPETokenizer(vocab_size=1000)
        
        # Train on the text first
        tokenizer.train(["hello world"])
        
        text = "hello world"
        ids = tokenizer.encode(text)
        decoded = tokenizer.decode(ids)
        
        assert decoded == text
    
    def test_special_tokens(self):
        """Test special tokens are handled."""
        tokenizer = SimpleBPETokenizer(vocab_size=1000)
        
        # Check special tokens exist in initial vocab (before training)
        assert tokenizer.pad_token in tokenizer.token_to_id
        assert tokenizer.eos_token in tokenizer.token_to_id
        assert tokenizer.unk_token in tokenizer.token_to_id
    
    def test_unknown_character_uses_unk(self):
        """Test unknown characters map to unk token."""
        tokenizer = SimpleBPETokenizer(vocab_size=1000)
        
        # Train on specific text that doesn't include 'x', 'y', 'z'
        tokenizer.train(["hello world"])
        
        # Use characters not in training data
        text = "xyz"
        ids = tokenizer.encode(text)
        
        # Should contain unk token for unknown chars
        unk_id = tokenizer.token_to_id[tokenizer.unk_token]
        assert unk_id in ids
    
    def test_vocabulary_size(self):
        """Test vocabulary size is correct."""
        tokenizer = SimpleBPETokenizer(vocab_size=16_384)
        
        assert len(tokenizer) <= 16_384
    
    def test_train_on_samples(self):
        """Test training on sample texts."""
        texts = [
            "hello world",
            "hello there",
            "world peace",
            "hello world peace",
        ]
        
        tokenizer = SimpleBPETokenizer(vocab_size=500)
        tokenizer.train(texts)
        
        # Should be able to encode
        ids = tokenizer.encode("hello")
        assert len(ids) > 0
        
        # Should be able to decode
        decoded = tokenizer.decode(ids)
        assert isinstance(decoded, str)
    
    def test_long_text_encoding(self):
        """Test long text encoding."""
        tokenizer = SimpleBPETokenizer(vocab_size=1000)
        
        # Train first
        tokenizer.train(["hello "])
        
        long_text = "hello " * 1000
        ids = tokenizer.encode(long_text)
        
        assert len(ids) > 0
    
    def test_save_and_load(self):
        """Test saving and loading tokenizer."""
        tokenizer = SimpleBPETokenizer(vocab_size=1000)
        
        texts = ["hello world", "test text", "another example"]
        tokenizer.train(texts)
        
        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "tokenizer.json"
            tokenizer.save(str(path))
            
            loaded = SimpleBPETokenizer.load(str(path))
            
            # Check vocab sizes match
            assert len(tokenizer) == len(loaded)
            
            # Check encoding matches
            text = "hello world"
            assert tokenizer.encode(text) == loaded.encode(text)
            assert tokenizer.decode(tokenizer.encode(text)) == loaded.decode(loaded.encode(text))


class TestTokenShardWriter:
    """Tests for TokenShardWriter."""
    
    def test_write_shard(self, tmp_path):
        """Test writing a shard file."""
        from mini_llm.data.token_shards import TokenShardWriter
        
        writer = TokenShardWriter(str(tmp_path), context_length=10)
        
        # Create test data
        token_ids = [
            [1, 2, 3],
            [4, 5, 6, 7],
            [8, 9],
        ]
        
        shard_path = writer.write_shard(0, token_ids)
        
        assert shard_path is not None
        assert shard_path.exists()
        assert shard_path.suffix == ".bin"
    
    def test_empty_token_ids(self, tmp_path):
        """Test handling empty token IDs."""
        from mini_llm.data.token_shards import TokenShardWriter
        
        writer = TokenShardWriter(str(tmp_path), context_length=10)
        
        shard_path = writer.write_shard(0, [])
        
        assert shard_path is None
    
    def test_dtype_validation(self, tmp_path):
        """Test dtype validation."""
        from mini_llm.data.token_shards import TokenShardWriter
        
        with pytest.raises(ValueError, match="Unsupported dtype"):
            TokenShardWriter(str(tmp_path), dtype="float32")


class TestTokenShardGenerator:
    """Tests for TokenShardGenerator."""
    
    def test_initialization(self):
        """Test generator initialization."""
        from mini_llm.data.parquet_reader import CosmopediaParquetReader
        from mini_llm.data.token_shards import TokenShardGenerator
        
        tokenizer = SimpleBPETokenizer(vocab_size=1000)
        
        # Use a temp directory for dataset path to avoid FileNotFoundError
        with tempfile.TemporaryDirectory() as tmpdir:
            # Create empty parquet file to satisfy validation
            import pyarrow as pa
            import pyarrow.parquet as pq
            table = pa.table({"text": []})
            pq.write_table(table, Path(tmpdir) / "train-00000-of-00010.parquet")
            
            parquet_reader = CosmopediaParquetReader(
                dataset_path=tmpdir,
                columns=["text"],
            )
            
            # This should work - just verify it can be instantiated
            generator = TokenShardGenerator(
                tokenizer,
                parquet_reader,
                output_dir="/tmp/test_shards",
            )
            
            assert generator is not None
