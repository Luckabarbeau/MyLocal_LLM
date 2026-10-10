#!/usr/bin/env python3
"""Tokenizer benchmarks comparing Simple and Fast implementations.

Usage:
    python benchmarks/benchmark_tokenizer.py --smoke-test
    python benchmarks/benchmark_tokenizer.py --vocab-size 16384 --corpus-mb 100
"""

import argparse
import json
import time
from pathlib import Path

from mini_llm.tokenizer.tokenizer import (
    SimpleBPETokenizer,
    FastBPETokenizer,
    _HF_AVAILABLE,
)


def generate_sample_texts(num_samples=1000) -> list:
    """Generate synthetic sample data for benchmarking."""
    import random
    random.seed(42)

    # Code samples
    code_snippets = [
        "def hello(name):\n    return f'Hello {name}!'\n",
        "class MyClass:\n    def __init__(self, x):\n        self.x = x\n",
        "import numpy as np\ndef process(data):\n    return np.mean(data)\n",
        "# Python code example\nasync def fetch(url):\n    return await http.get(url)\n",
    ] * 25

    # English prose
    prose_samples = [
        "The quick brown fox jumps over the lazy dog. This sentence contains every letter of the alphabet.",
        "Machine learning has revolutionized artificial intelligence in recent years.",
        "Neural networks are computing systems inspired by biological neural networks.",
        "Deep learning architectures include convolutional, recurrent, and transformer models.",
    ] * 25

    # Mathematical symbols
    math_samples = [
        "∫₀^∞ e^(-x²) dx = √π / 2",
        "E = mc²",
        "∇·E = 4πρ",
        "∑ₙ₌₀^∞ xⁿ/n! = eˣ",
    ] * 25

    # Mix them together
    all_samples = code_snippets + prose_samples + math_samples
    random.shuffle(all_samples)

    return all_samples[:num_samples]


def benchmark_training():
    """Benchmark training speed for both tokenizers."""
    print("=" * 70)
    print("TRAINING BENCHMARK")
    print("=" * 70)

    # Generate sample texts (~5MB)
    print("\nGenerating sample data...")
    sample_texts = generate_sample_texts(num_samples=1000)
    total_bytes = sum(len(t.encode('utf-8')) for t in sample_texts)
    print(f"Sample: {len(sample_texts)} documents, {total_bytes / (1024*1024):.2f} MB")

    # Train with different vocab sizes
    vocab_sizes = [256, 512, 1024] if '--smoke-test' in ' '.join(__import__('sys').argv) else [256, 1024, 8192]

    results = {
        "training": [],
    }

    for vocab_size in vocab_sizes:
        print(f"\n--- Vocabulary Size: {vocab_size} ---")

        # Simple tokenizer
        if '--smoke-test' in ' '.join(__import__('sys').argv) or vocab_size <= 1024:
            print("\n  SimpleBPETokenizer:")
            tokenizer = SimpleBPETokenizer(vocab_size=vocab_size)
            start = time.time()
            tokenizer.train(sample_texts)
            elapsed = time.time() - start

            print(f"    Training time: {elapsed:.2f}s")
            print(f"    Final vocab size: {len(tokenizer)}")
            print(f"    Number of merges: {len(tokenizer.merges)}")

            results["training"].append({
                "backend": "simple_bpe",
                "vocab_size": vocab_size,
                "corpus_mb": total_bytes / (1024 * 1024),
                "time_seconds": elapsed,
                "merges": len(tokenizer.merges),
            })

        # Fast tokenizer (if available)
        if _HF_AVAILABLE and ( '--smoke-test' in ' '.join(__import__('sys').argv) or vocab_size <= 8192):
            print("\n  FastBPETokenizer:")
            tokenizer = FastBPETokenizer(vocab_size=vocab_size, threads=4)
            start = time.time()
            tokenizer.train(sample_texts)
            elapsed = time.time() - start

            print(f"    Training time: {elapsed:.2f}s")
            print(f"    Final vocab size: {len(tokenizer)}")

            results["training"].append({
                "backend": "fast_bpe",
                "vocab_size": vocab_size,
                "corpus_mb": total_bytes / (1024 * 1024),
                "time_seconds": elapsed,
                "merges": len(tokenizer),
            })

    return results


def benchmark_encoding():
    """Benchmark encoding speed for both tokenizers."""
    print("\n" + "=" * 70)
    print("ENCODING BENCHMARK")
    print("=" * 70)

    # Train a tokenizer first
    print("\nTraining tokenizer...")
    sample_texts = generate_sample_texts(num_samples=500)
    vocab_size = 1024

    simple_tokenizer = SimpleBPETokenizer(vocab_size=vocab_size)
    simple_tokenizer.train(sample_texts)

    fast_tokenizer = None
    if _HF_AVAILABLE:
        fast_tokenizer = FastBPETokenizer(vocab_size=vocab_size, threads=4)
        fast_tokenizer.train(sample_texts)

    # Get test texts
    encode_texts = sample_texts[:200]
    total_chars = sum(len(t) for t in encode_texts)

    print(f"\nEncoding {len(encode_texts)} test documents ({total_chars} chars)...")

    results = {
        "encoding": [],
    }

    # Simple tokenizer
    print("\n  SimpleBPETokenizer:")
    start = time.time()
    for text in encode_texts:
        simple_tokenizer.encode(text)
    elapsed = time.time() - start

    print(f"    Single-thread encoding: {elapsed:.2f}s")
    print(f"    Throughput: {total_chars / 1024 / elapsed:.2f} KB/s")

    results["encoding"].append({
        "backend": "simple_bpe",
        "mode": "single",
        "threads": 1,
        "time_seconds": elapsed,
        "chars_processed": total_chars,
        "kb_per_second": total_chars / 1024 / elapsed,
        "docs_per_second": len(encode_texts) / elapsed,
    })

    # Fast tokenizer (if available)
    if fast_tokenizer:
        print("\n  FastBPETokenizer:")

        # Single encoding
        start = time.time()
        for text in encode_texts:
            fast_tokenizer.encode(text)
        single_elapsed = time.time() - start

        # Batch encoding
        start = time.time()
        fast_tokenizer.encode_batch(encode_texts)
        batch_elapsed = time.time() - start

        print(f"    Single: {single_elapsed:.2f}s")
        print(f"    Batch:  {batch_elapsed:.2f}s")
        print(f"    MB/s (single): {total_chars / 1024 / single_elapsed:.2f}")
        print(f"    MB/s (batch):  {total_chars / 1024 / batch_elapsed:.2f}")

        results["encoding"].append({
            "backend": "fast_bpe",
            "mode": "single",
            "threads": 1,
            "time_seconds": single_elapsed,
            "chars_processed": total_chars,
            "kb_per_second": total_chars / 1024 / single_elapsed,
            "docs_per_second": len(encode_texts) / single_elapsed,
        })

        results["encoding"].append({
            "backend": "fast_bpe",
            "mode": "batch",
            "threads": 4,
            "time_seconds": batch_elapsed,
            "chars_processed": total_chars,
            "kb_per_second": total_chars / 1024 / batch_elapsed,
            "docs_per_second": len(encode_texts) / batch_elapsed,
        })

    return results


def main():
    parser = argparse.ArgumentParser(description="Tokenizer Benchmark Suite")
    parser.add_argument("--smoke-test", action="store_true", help="Run quick smoke test")
    parser.add_argument("--vocab-size", type=int, default=16384, help="Vocabulary size")
    parser.add_argument("--corpus-mb", type=float, default=5.0, help="Corpus size in MB")

    args = parser.parse_args()

    print("\n" + "#" * 70)
    print("# TOKENIZER BENCHMARK SUITE")
    print("#" * 70)

    results = {
        "smoke_test": args.smoke_test,
        "vocab_size": args.vocab_size,
        "corpus_mb": args.corpus_mb,
        "hf_available": _HF_AVAILABLE,
    }

    training_results = benchmark_training()
    encoding_results = benchmark_encoding()
    
    # Extract actual lists
    results["training"] = training_results.get("training", [])
    results["encoding"] = encoding_results.get("encoding", [])

    # Print summary
    print("\n" + "=" * 70)
    print("BENCHMARK SUMMARY")
    print("=" * 70)

    # Training summary
    print("\n--- TRAINING SPEED (seconds) ---")
    for r in results["training"]:
        backend = r["backend"]
        vocab = r["vocab_size"]
        time_val = r["time_seconds"]
        merges = r.get("merges", "N/A")

        if isinstance(merges, int):
            print(f"{backend:15s} | vocab={vocab:5d} | time={time_val:6.2f}s | merges={merges:5d}")
        else:
            print(f"{backend:15s} | vocab={vocab:5d} | time={time_val:6.2f}s")

    # Encoding summary
    print("\n--- ENCODING SPEED (KB/s) ---")
    for r in results["encoding"]:
        backend = r["backend"]
        mode = r["mode"]
        threads = r["threads"]
        kbps = r.get("kb_per_second", 0)

        print(f"{backend:15s} | {mode:6s} | threads={threads} | {kbps:8.2f} KB/s")

    print("\n" + "=" * 70)
    print("BENCHMARK COMPLETE")
    print("=" * 70)

    return 0


if __name__ == "__main__":
    import sys
    sys.exit(main())
