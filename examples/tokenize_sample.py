"""Example: Tokenize sample text using the SimpleBPETokenizer."""

from mini_llm.tokenizer.tokenizer import SimpleBPETokenizer


def main():
    print("=" * 60)
    print("SimpleBPETokenizer Example")
    print("=" * 60)
    
    # Create tokenizer
    tokenizer = SimpleBPETokenizer(vocab_size=1000)
    
    # Sample training data
    texts = [
        "Hello world!",
        "This is a test.",
        "Learning to tokenize text.",
        "The quick brown fox jumps over the lazy dog.",
    ]
    
    print("\nTraining tokenizer on sample texts...")
    tokenizer.train(texts)
    print(f"Vocabulary size: {len(tokenizer)}")
    
    # Test encoding
    test_text = "Hello, this is a new sentence!"
    print(f"\nTest text: \"{test_text}\"")
    
    # Encode
    token_ids = tokenizer.encode(test_text)
    print(f"Token IDs: {token_ids}")
    
    # Decode
    decoded = tokenizer.decode(token_ids)
    print(f"Decoded: \"{decoded}\"")
    
    # Show special tokens
    print(f"\nSpecial tokens:")
    print(f"  pad_token: {tokenizer.pad_token} (ID: {tokenizer.token_to_id[tokenizer.pad_token]})")
    print(f"  eos_token: {tokenizer.eos_token} (ID: {tokenizer.token_to_id[tokenizer.eos_token]})")
    print(f"  unk_token: {tokenizer.unk_token} (ID: {tokenizer.token_to_id[tokenizer.unk_token]})")
    
    # Show first few vocabulary entries
    print(f"\nFirst 10 vocabulary entries:")
    for i in range(min(10, len(tokenizer))):
        token = tokenizer.id_to_token[i]
        print(f"  {i}: '{token}'")
    
    print("\n" + "=" * 60)
    print("Done!")
    print("=" * 60)
    
    return 0


if __name__ == "__main__":
    import sys
    sys.exit(main())
