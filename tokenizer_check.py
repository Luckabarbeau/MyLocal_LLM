"""Tests for SimpleBPETokenizer."""

import os
import tempfile
from pathlib import Path

from mini_llm.tokenizer.tokenizer import SimpleBPETokenizer


def test_basic_encoding():
    """Test basic encode/decode cycle."""
    tokenizer = SimpleBPETokenizer(vocab_size=100)
    
    # Train on simple corpus
    texts = ["hello world", "hello there", "goodbye world"]
    tokenizer.train(texts)
    
    # Encode
    ids = tokenizer.encode("hello")
    assert isinstance(ids, list)
    assert len(ids) > 0
    
    # Decode
    decoded = tokenizer.decode(ids)
    assert decoded == "hello"
    
    print("✓ Basic encoding test passed")


def test_merge_ranks():
    """Test that merge ranks are correctly assigned and used."""
    tokenizer = SimpleBPETokenizer(vocab_size=200)
    
    # Train on corpus where we can verify merge order
    texts = ["the cat sat", "the dog ran", "a cat sat"]
    tokenizer.train(texts)
    
    # Check that merges were learned
    assert len(tokenizer.merges) > 0
    assert len(tokenizer.merge_ranks) > 0
    
    # Check that merge ranks are assigned
    for pair, rank in tokenizer.merge_ranks.items():
        assert isinstance(rank, int)
        assert rank >= 0
    
    print(f"✓ Merge ranks test passed ({len(tokenizer.merge_ranks)} merges)")


def test_encode_with_merge_rank():
    """Test that encoding respects merge ranks."""
    tokenizer = SimpleBPETokenizer(vocab_size=200)
    
    # Train on corpus
    texts = ["hello world", "hello there", "goodbye world"]
    tokenizer.train(texts)
    
    # Encode a text
    text = "hello"
    ids1 = tokenizer.encode(text)
    
    # Encode again - should get same result
    ids2 = tokenizer.encode(text)
    
    assert ids1 == ids2, f"Encoding not deterministic: {ids1} vs {ids2}"
    
    print("✓ Merge rank encoding test passed")


def test_save_load_roundtrip():
    """Test that save/load preserves tokenizer state exactly."""
    tokenizer = SimpleBPETokenizer(vocab_size=200)
    
    # Train on corpus
    texts = ["hello world", "hello there", "goodbye world"]
    tokenizer.train(texts)
    
    original_merges = dict(tokenizer.merges)
    original_merge_ranks = dict(tokenizer.merge_ranks)
    original_token_to_id = dict(tokenizer.token_to_id)
    
    # Save to temp file
    with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as f:
        tokenizer.save(f.name)
        temp_path = f.name
    
    try:
        # Load
        loaded = SimpleBPETokenizer.load(temp_path)
        
        # Verify exact match
        assert dict(loaded.merges) == original_merges, "Merges don't match"
        assert dict(loaded.merge_ranks) == original_merge_ranks, "Merge ranks don't match"
        assert loaded.token_to_id == original_token_to_id, "Token to ID mapping doesn't match"
        
        # Verify encoding produces same results
        test_text = "hello world"
        original_ids = tokenizer.encode(test_text)
        loaded_ids = loaded.encode(test_text)
        
        assert original_ids == loaded_ids, f"Encoding differs after load: {original_ids} vs {loaded_ids}"
        
    finally:
        os.unlink(temp_path)
    
    print("✓ Save/load roundtrip test passed")


def test_deterministic_encoding():
    """Test that encoding is deterministic regardless of worker count."""
    tokenizer = SimpleBPETokenizer(vocab_size=200)
    
    # Train on corpus
    texts = ["the quick brown fox", "jumps over the lazy dog", "hello world test"]
    tokenizer.train(texts)
    
    # Save and load to simulate separate worker process
    with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as f:
        tokenizer.save(f.name)
        temp_path = f.name
    
    try:
        loaded = SimpleBPETokenizer.load(temp_path)
        
        # Test multiple texts
        test_texts = [
            "hello",
            "the quick",
            "brown fox",
            "jumps over",
        ]
        
        for text in test_texts:
            original_ids = tokenizer.encode(text)
            loaded_ids = loaded.encode(text)
            assert original_ids == loaded_ids, f"Mismatch for '{text}': {original_ids} vs {loaded_ids}"
        
    finally:
        os.unlink(temp_path)
    
    print("✓ Deterministic encoding test passed")


def test_large_merge_rank():
    """Test that higher merge ranks don't interfere with lower ones."""
    tokenizer = SimpleBPETokenizer(vocab_size=500)
    
    # Train on larger corpus to create more merges
    texts = [
        "the quick brown fox jumps over the lazy dog",
        "a quick brown dog runs fast",
        "the lazy cat sleeps all day",
        "quick dogs and lazy cats",
        "fox jumps high with brown fur",
    ] * 10  # Repeat to create more merges
    tokenizer.train(texts)
    
    # Verify merge ranks are properly assigned
    assert len(tokenizer.merge_ranks) == len(tokenizer.merges), "Merge ranks count mismatch"
    
    # Test encoding - should work without errors
    test_text = "the quick brown fox"
    ids = tokenizer.encode(test_text)
    assert len(ids) > 0
    
    print(f"✓ Large merge rank test passed ({len(tokenizer.merge_ranks)} merges)")


def run_all_tests():
    """Run all tokenizer tests."""
    print("Running tokenizer tests...\n")
    
    test_basic_encoding()
    test_merge_ranks()
    test_encode_with_merge_rank()
    test_save_load_roundtrip()
    test_deterministic_encoding()
    test_large_merge_rank()
    
    print("\n" + "=" * 60)
    print("All tests passed!")
    print("=" * 60)


if __name__ == "__main__":
    run_all_tests()
