"""Tokenizer configuration for MyLocalLLM.

This module provides configuration classes and utilities for managing
tokenizer settings across the project.
"""

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

from mini_llm.tokenizer.tokenizer import (
    SimpleBPETokenizer,
    FastBPETokenizer,
    TokenizerProtocol,
)


@dataclass
class TokenizerConfig:
    """
    Configuration for tokenizer settings.

    This configuration is used to:
    - Control tokenizer training parameters
    - Ensure consistency between training and inference
    - Provide sensible defaults for different use cases
    """

    # Basic settings
    vocab_size: int = 16_384
    backend: str = "fast"  # "simple" or "fast"
    threads: int = 8

    # Training parameters
    max_documents: Optional[int] = None
    max_bytes: Optional[int] = None
    sample_mb: Optional[float] = None  # Alternative to max_bytes

    # Special tokens
    pad_token: str = "<|pad|>"
    eos_token: str = "<|eos|>"
    unk_token: str = "<|unk|>"

    # Path settings
    tokenizer_path: Optional[str] = None  # Existing tokenizer to load
    output_path: Optional[str] = None  # Where to save trained tokenizer

    # Smoke test mode
    smoke_test: bool = False

    def __post_init__(self):
        """Validate configuration after initialization."""
        # Ensure vocab_size is reasonable for uint16
        if self.vocab_size > 65_536:
            raise ValueError(
                f"Vocabulary size {self.vocab_size} exceeds uint16 limit (65536). "
                "Consider reducing vocab_size or using uint32."
            )

        # Convert sample_mb to max_bytes if provided
        if self.sample_mb is not None:
            self.max_bytes = int(self.sample_mb * 1024 * 1024)

    @classmethod
    def from_dict(cls, config_dict: dict) -> "TokenizerConfig":
        """Create a TokenizerConfig from a dictionary."""
        return cls(**config_dict)

    def to_dict(self) -> dict:
        """Convert to dictionary for serialization."""
        return {
            "vocab_size": self.vocab_size,
            "backend": self.backend,
            "threads": self.threads,
            "max_documents": self.max_documents,
            "max_bytes": self.max_bytes,
            "sample_mb": self.sample_mb,
            "pad_token": self.pad_token,
            "eos_token": self.eos_token,
            "unk_token": self.unk_token,
            "tokenizer_path": self.tokenizer_path,
            "output_path": self.output_path,
            "smoke_test": self.smoke_test,
        }

    @classmethod
    def smoke_test(cls) -> "TokenizerConfig":
        """Create configuration for smoke test."""
        return cls(
            vocab_size=512,
            backend="fast",
            threads=4,
            sample_mb=5.0,
            smoke_test=True,
        )

    @classmethod
    def development(cls) -> "TokenizerConfig":
        """Create configuration for development/testing."""
        return cls(
            vocab_size=8192,
            backend="fast",
            threads=8,
            sample_mb=100.0,
        )

    @classmethod
    def production(cls) -> "TokenizerConfig":
        """Create configuration for production training."""
        return cls(
            vocab_size=16_384,
            backend="fast",
            threads=8,
            sample_mb=750.0,
        )


class TokenizerManager:
    """
    Manager for tokenizer lifecycle.

    This class handles:
    - Loading existing tokenizers
    - Training new tokenizers when needed
    - Saving tokenizers with metadata
    - Validating tokenizer compatibility
    """

    def __init__(self, config: TokenizerConfig):
        """
        Initialize the tokenizer manager.

        Args:
            config: TokenizerConfig instance
        """
        self.config = config
        self.tokenizer: Optional[TokenizerProtocol] = None

    def load_or_train(self) -> TokenizerProtocol:
        """
        Load existing tokenizer or train a new one.

        Returns:
            Tokenizer instance (loaded or newly trained)
        """
        # Check if we should reuse an existing tokenizer
        if self.config.tokenizer_path and Path(self.config.tokenizer_path).exists():
            print(f"Loading existing tokenizer from: {self.config.tokenizer_path}")
            self.tokenizer = self._load_tokenizer(self.config.tokenizer_path)
            return self.tokenizer

        # Check if output exists and we shouldn't retrain
        if self.config.output_path and Path(self.config.output_path).exists():
            print(f"Using existing tokenizer at: {self.config.output_path}")
            self.tokenizer = self._load_tokenizer(self.config.output_path)
            return self.tokenizer

        # Train new tokenizer
        print("No existing tokenizer found. Training new tokenizer...")
        self.tokenizer = self._train_tokenizer()
        return self.tokenizer

    def _load_tokenizer(self, path: str) -> TokenizerProtocol:
        """Load tokenizer from file."""
        path = Path(path)

        # Try to detect backend from file format
        if path.suffix == '.json':
            try:
                # Try SimpleBPETokenizer first
                return SimpleBPETokenizer.load(str(path))
            except Exception:
                # Try FastBPETokenizer
                return FastBPETokenizer.load(str(path))
        else:
            # HF format (not JSON)
            return FastBPETokenizer.load(str(path))

    def _train_tokenizer(self) -> TokenizerProtocol:
        """Train a new tokenizer."""
        from mini_llm.data.parquet_reader import CosmopediaParquetReader

        # Create parquet reader
        dataset_path = Path(self.config.dataset_path if hasattr(self.config, 'dataset_path') else "../cosmopedia-v2/cosmopedia-v2")
        parquet_reader = CosmopediaParquetReader(dataset_path=dataset_path)

        # Sample texts
        max_docs = self.config.max_documents
        max_bytes = self.config.max_bytes

        if self.config.smoke_test:
            max_bytes = 5_000_000
            max_docs = 500

        sample_texts = []
        total_chars = 0

        for record in parquet_reader.iter_records():
            text = record.get("text", "")
            if text:
                if max_docs and len(sample_texts) >= max_docs:
                    break
                if max_bytes and total_chars + len(text) > max_bytes:
                    break

                sample_texts.append(text)
                total_chars += len(text)

        print(f"Sampled {len(sample_texts)} documents ({total_chars / (1024*1024):.1f} MB)")

        # Train tokenizer
        if self.config.backend == "simple":
            tokenizer = SimpleBPETokenizer(vocab_size=self.config.vocab_size)
        else:
            tokenizer = FastBPETokenizer(vocab_size=self.config.vocab_size, threads=self.config.threads)

        tokenizer.train(sample_texts)

        # Save tokenizer if output path specified
        if self.config.output_path:
            Path(self.config.output_path).parent.mkdir(parents=True, exist_ok=True)
            tokenizer.save(self.config.output_path)
            print(f"Tokenizer saved to: {self.config.output_path}")

        return tokenizer
