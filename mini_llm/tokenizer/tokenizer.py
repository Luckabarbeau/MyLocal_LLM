"""Tokenizer module for MyLocalLLM.

This module provides multiple tokenizer backends:
- SimpleBPETokenizer: Reference Python implementation (educational)
- FastBPETokenizer: Production BPE using Hugging Face tokenizers (fast)

Special tokens are consistently defined as:
- <|pad|>: Padding token (ID 0)
- <|eos|>: End of sequence token (ID 1)
- <|unk|>: Unknown token (ID 2)
"""

import json
import os
from collections import Counter
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np


# Try to import Hugging Face tokenizers for the fast implementation
_HF_AVAILABLE = False
try:
    from tokenizers import Tokenizer as HFTokenizer, models, pre_tokenizers, decoders, trainers
    _HF_AVAILABLE = True
except ImportError:
    pass


class TokenizerProtocol:
    """Abstract interface for all tokenizer implementations."""

    def train(self, texts: List[str], vocab_size: int) -> None:
        """Train the tokenizer on text data."""
        raise NotImplementedError

    def encode(self, text: str) -> List[int]:
        """Encode a single text string to token IDs."""
        raise NotImplementedError

    def encode_batch(self, texts: List[str]) -> List[List[int]]:
        """Encode multiple texts efficiently."""
        raise NotImplementedError

    def decode(self, ids: List[int]) -> str:
        """Decode token IDs to text."""
        raise NotImplementedError

    def save(self, path: str) -> None:
        """Save tokenizer to file."""
        raise NotImplementedError

    @classmethod
    def load(cls, path: str) -> "TokenizerProtocol":
        """Load tokenizer from file."""
        raise NotImplementedError

    def __len__(self) -> int:
        """Return vocabulary size."""
        raise NotImplementedError


class SimpleBPETokenizer(TokenizerProtocol):
    """
    Reference Python BPE tokenizer for educational purposes.

    This implementation:
    - Uses byte-pair encoding for subword tokenization
    - Maintains deterministic merge order
    - Has explicit, transparent code suitable for learning
    - Is intentionally NOT optimized for performance

    For production use, see FastBPETokenizer.
    """

    def __init__(self, vocab_size: int = 16_384):
        """
        Initialize the tokenizer.

        Args:
            vocab_size: Target vocabulary size (including special tokens)
        """
        self.vocab_size = vocab_size

        # Special tokens
        self.pad_token = "<|pad|>"
        self.eos_token = "<|eos|>"
        self.unk_token = "<|unk|>"

        # Vocabulary maps - initialize with special tokens
        self.token_to_id: Dict[str, int] = {}
        self.id_to_token: Dict[int, str] = {}

        # BPE merge rules: (left, right) -> merged_token
        self.merges: Dict[Tuple[str, str], str] = {}

        # Merge ranks: rank of each merge (lower = learned earlier, higher priority)
        self.merge_ranks: Dict[Tuple[str, str], int] = {}

        # Pre-add special tokens to vocabulary (to reserve their IDs)
        for i, token in enumerate([self.pad_token, self.eos_token, self.unk_token]):
            self.token_to_id[token] = i
            self.id_to_token[i] = token

    def _get_stats(self, tokens: List[str]) -> Counter:
        """Get frequency of adjacent token pairs."""
        pairs = Counter()
        for i in range(len(tokens) - 1):
            pairs[(tokens[i], tokens[i + 1])] += 1
        return pairs

    def _merge_pair(self, tokens: List[str], pair: Tuple[str, str]) -> List[str]:
        """Merge one pair of tokens using a sliding window approach."""
        if not tokens:
            return tokens

        result = [tokens[0]]

        for i in range(1, len(tokens)):
            prev_token = result[-1]
            curr_token = tokens[i]

            if (prev_token, curr_token) == pair:
                result[-1] = self.merges[pair]
            else:
                result.append(curr_token)

        return result

    def train(self, texts: List[str], vocab_size: int = None) -> None:
        """
        Train the tokenizer on a list of texts.

        Args:
            texts: List of text samples for training
            vocab_size: Optional target vocabulary size (uses instance default if None)
        """
        if vocab_size is not None:
            self.vocab_size = vocab_size

        # Start with character-level vocabulary
        vocab = set()
        for text in texts:
            vocab.update(text)

        # Add special tokens
        special_tokens = [self.pad_token, self.eos_token, self.unk_token]
        for token in special_tokens:
            if token not in vocab:
                vocab.add(token)

        vocab_with_special = set(vocab)
        for token in special_tokens:
            vocab_with_special.add(token)

        # Assign IDs: special tokens first, then character tokens, then BPE merges
        current_id = 0

        # Special tokens (already in token_to_id from __init__, just make sure)
        for token in special_tokens:
            if token not in self.token_to_id:
                self.token_to_id[token] = current_id
                self.id_to_token[current_id] = token
            current_id = max(current_id, self.token_to_id[token]) + 1

        # Character tokens (sorted for determinism)
        for char in sorted(vocab_with_special):
            if char not in self.token_to_id:
                self.token_to_id[char] = current_id
                self.id_to_token[current_id] = char
                current_id += 1

        # Initialize tokens with individual characters
        tokens = [list(text) for text in texts]

        # Build BPE merges up to target vocab size
        max_merges = self.vocab_size - len(self.token_to_id)

        for merge_rank in range(max_merges):
            # Get all current token pairs
            all_pairs = Counter()
            for text_tokens in tokens:
                pairs = self._get_stats(text_tokens)
                all_pairs.update(pairs)

            if not all_pairs:
                break

            # Find most frequent pair
            best_pair = max(all_pairs.items(), key=lambda x: x[1])

            # Create new merged token
            merged_token = f"{best_pair[0][0]}{best_pair[0][1]}"

            # Check if we already have this token
            if merged_token in self.token_to_id:
                continue

            # Add merge rule with rank
            self.merges[best_pair[0]] = merged_token
            self.merge_ranks[best_pair[0]] = merge_rank

            # Add to vocabulary
            self.token_to_id[merged_token] = current_id
            self.id_to_token[current_id] = merged_token
            current_id += 1

            # Apply merge to all texts
            tokens = [self._merge_pair(t, best_pair[0]) for t in tokens]

    def encode(self, text: str) -> List[int]:
        """
        Encode text as token IDs.

        Args:
            text: Input text string

        Returns:
            List of token IDs
        """
        if not self.token_to_id:
            raise ValueError("Tokenizer not trained. Call train() first.")

        # Start with character-level tokens
        tokens = list(text)

        # Apply BPE merges in rank order (lowest rank = highest priority)
        while True:
            # Find all valid pairs and their ranks
            valid_pairs = []
            for i in range(len(tokens) - 1):
                pair = (tokens[i], tokens[i + 1])
                if pair in self.merges:
                    rank = self.merge_ranks.get(pair, float('inf'))
                    valid_pairs.append((i, pair, rank))

            if not valid_pairs:
                break

            # Choose the pair with lowest rank
            valid_pairs.sort(key=lambda x: x[2])
            best_idx, best_pair, _ = valid_pairs[0]

            # Apply the merge
            tokens = self._merge_pair(tokens, best_pair)

        # Convert to IDs
        unk_token_id = self.token_to_id.get(self.unk_token, 0)
        ids = []
        for token in tokens:
            if token in self.token_to_id:
                ids.append(self.token_to_id[token])
            else:
                ids.append(unk_token_id)

        return ids

    def encode_batch(self, texts: List[str]) -> List[List[int]]:
        """Encode multiple texts using the single text encoder."""
        return [self.encode(text) for text in texts]

    def decode(self, ids: List[int]) -> str:
        """
        Decode token IDs to text.

        Args:
            ids: List of token IDs

        Returns:
            Decoded text string
        """
        tokens = []
        for id_ in ids:
            if id_ in self.id_to_token:
                tokens.append(self.id_to_token[id_])

        return "".join(tokens)

    def save(self, path: str) -> None:
        """
        Save tokenizer to JSON file.

        Args:
            path: Path to save tokenizer
        """
        config = {
            "vocab_size": self.vocab_size,
            "pad_token": self.pad_token,
            "eos_token": self.eos_token,
            "unk_token": self.unk_token,
            "token_to_id": {k: v for k, v in self.token_to_id.items()},
            "id_to_token": {str(k): v for k, v in self.id_to_token.items()},
            "merges": [[left, right, token] for (left, right), token in self.merges.items()],
            "merge_ranks": [[left, right, rank] for (left, right), rank in self.merge_ranks.items()],
        }

        with open(path, "w") as f:
            json.dump(config, f, indent=2)

    @classmethod
    def load(cls, path: str) -> "SimpleBPETokenizer":
        """
        Load tokenizer from JSON file.

        Args:
            path: Path to saved tokenizer

        Returns:
            Loaded tokenizer
        """
        with open(path) as f:
            config = json.load(f)

        tokenizer = cls(vocab_size=config["vocab_size"])
        tokenizer.pad_token = config["pad_token"]
        tokenizer.eos_token = config["eos_token"]
        tokenizer.unk_token = config["unk_token"]
        tokenizer.token_to_id = {k: v for k, v in config["token_to_id"].items()}
        tokenizer.id_to_token = {int(k): v for k, v in config["id_to_token"].items()}

        # Load merges as tuples from JSON arrays
        tokenizer.merges = {
            (item[0], item[1]): item[2] for item in config.get("merges", [])
        }
        tokenizer.merge_ranks = {
            (item[0], item[1]): item[2] for item in config.get("merge_ranks", [])
        }

        return tokenizer

    def __len__(self) -> int:
        """Return vocabulary size."""
        return len(self.token_to_id)


class FastBPETokenizer(TokenizerProtocol):
    """
    Production BPE tokenizer using Hugging Face tokenizers library.

    This implementation provides:
    - Fast training using Rust backend
    - Efficient batch encoding
    - Multi-threaded processing
    - Deterministic output (when using same seed/config)

    For educational purposes, see SimpleBPETokenizer.
    """

    def __init__(self, vocab_size: int = 16_384, threads: int = 8):
        """
        Initialize the tokenizer.

        Args:
            vocab_size: Target vocabulary size
            threads: Number of threads for parallel processing
        """
        self.vocab_size = vocab_size
        self.threads = threads
        self._tokenizer = None
        self._hf_module = None

        # Special tokens (must be added first for consistent IDs)
        self.pad_token = "<|pad|>"
        self.eos_token = "<|eos|>"
        self.unk_token = "<|unk|>"

    def _get_hf_module(self):
        """Lazy import of Hugging Face tokenizers."""
        if self._hf_module is None:
            if not _HF_AVAILABLE:
                raise ImportError(
                    "Hugging Face tokenizers not available. Install with: pip install tokenizers"
                )
            from tokenizers import Tokenizer as HFTokenizer, models, pre_tokenizers, decoders, trainers
            self._hf_module = {
                'HFTokenizer': HFTokenizer,
                'models': models,
                'pre_tokenizers': pre_tokenizers,
                'decoders': decoders,
                'trainers': trainers,
            }
        return self._hf_module

    def _init_hf_tokenizer(self) -> None:
        """Initialize the underlying HF tokenizer."""
        if self._tokenizer is not None:
            return

        hf = self._get_hf_module()

        # Create BPE model with byte-level preprocessing
        # Byte-level BPE handles arbitrary Unicode and code well
        self._tokenizer = hf['HFTokenizer'](hf['models'].BPE(
            unk_token=self.unk_token,
            dropout=None,
        ))

        # Pre-tokenizer: byte-level pre-tokenization
        self._tokenizer.pre_tokenizer = hf['pre_tokenizers'].ByteLevel(
            use_prefix_space=False,
            add_prefix_space=False,
        )

        # Decoder: byte-level decoder
        self._tokenizer.decoder = hf['decoders'].ByteLevel()

        # Set special tokens
        special_tokens = [self.pad_token, self.eos_token, self.unk_token]
        self._tokenizer.add_special_tokens(special_tokens)

    def train(self, texts: List[str], vocab_size: int = None) -> None:
        """
        Train the tokenizer on text data.

        Args:
            texts: List of text samples for training
            vocab_size: Optional target vocabulary size
        """
        if vocab_size is not None:
            self.vocab_size = vocab_size

        if not _HF_AVAILABLE:
            raise RuntimeError(
                "Hugging Face tokenizers not available. Install with: pip install tokenizers"
            )

        self._init_hf_tokenizer()

        # Configure trainer
        hf = self._get_hf_module()
        trainer = hf['trainers'].BpeTrainer(
            vocab_size=self.vocab_size,
            min_frequency=1,  # Lower frequency to capture more patterns
            special_tokens=[self.pad_token, self.eos_token, self.unk_token],
            show_progress=False,
            initial_alphabet=hf['pre_tokenizers'].ByteLevel.alphabet(),
        )

        # Hugging Face tokenizers accepts any Python iterator.  Do not
        # materialize large pretraining samples here: mixed-corpus tokenizer
        # training can intentionally stream several GB of text.
        texts_iterable = texts if not isinstance(texts, tuple) else list(texts)

        # Train from the iterable so corpus sampling remains memory bounded.
        self._tokenizer.train_from_iterator(texts_iterable, trainer=trainer)

    def encode(self, text: str) -> List[int]:
        """Encode a single text."""
        if self._tokenizer is None:
            raise ValueError("Tokenizer not trained. Call train() first.")

        result = self._tokenizer.encode(text)
        # HF tokenizer encode() returns Encoding object with .ids attribute
        # or may return a list directly in some configurations
        if isinstance(result, list):
            return result
        elif hasattr(result, 'ids'):
            ids = result.ids
            # ids might already be a list or numpy array
            if isinstance(ids, list):
                return ids
            else:
                return ids.tolist()
        else:
            raise TypeError(f"Unexpected encoding result type: {type(result)}")

    def encode_batch(self, texts: List[str]) -> List[List[int]]:
        """Encode multiple texts efficiently."""
        if self._tokenizer is None:
            raise ValueError("Tokenizer not trained. Call train() first.")

        result = self._tokenizer.encode_batch(texts)
        # Handle both Encoding objects and raw lists
        if not result:
            return []
        
        first = result[0]
        if isinstance(first, list):
            return result
        elif hasattr(first, 'ids'):
            encoded_list = []
            for enc in result:
                ids = enc.ids
                # ids might already be a list or numpy array
                if isinstance(ids, list):
                    encoded_list.append(ids)
                else:
                    encoded_list.append(ids.tolist())
            return encoded_list
        else:
            raise TypeError(f"Unexpected encoding batch result type: {type(first)}")

    def decode(self, ids: List[int]) -> str:
        """Decode token IDs to text."""
        if self._tokenizer is None:
            raise ValueError("Tokenizer not trained. Call train() first.")

        return self._tokenizer.decode(ids)

    def save(self, path: str) -> None:
        """Save tokenizer in HF format."""
        if self._tokenizer is None:
            raise ValueError("Tokenizer not trained. Call train() first.")

        # Save in HF format (not JSON - use .json suffix for compatibility)
        self._tokenizer.save(path)

    @classmethod
    def load(cls, path: str) -> "FastBPETokenizer":
        """Load tokenizer from HF format file."""
        if not _HF_AVAILABLE:
            raise RuntimeError(
                "Hugging Face tokenizers not available. Install with: pip install tokenizers"
            )

        from tokenizers import Tokenizer as HFTokenizer

        tokenizer = cls()
        tokenizer._tokenizer = HFTokenizer.from_file(path)
        tokenizer._hf_module = None  # Will be lazily loaded on next use

        return tokenizer

    def __len__(self) -> int:
        """Return vocabulary size."""
        if self._tokenizer is None:
            return 0
        try:
            return len(self._tokenizer.get_vocab())
        except Exception:
            return self.vocab_size


def get_tokenizer(backend: str = "simple", **kwargs) -> TokenizerProtocol:
    """
    Factory function to create tokenizer instances.

    Args:
        backend: "simple" for Python reference, "fast" for Hugging Face production
        **kwargs: Arguments passed to tokenizer constructor

    Returns:
        Tokenizer instance
    """
    if backend == "simple":
        return SimpleBPETokenizer(**kwargs)
    elif backend == "fast":
        return FastBPETokenizer(**kwargs)
    else:
        raise ValueError(f"Unknown tokenizer backend: {backend}")


# Special token IDs (consistent across all implementations)
SPECIAL_TOKEN_IDS = {
    "pad_token_id": 0,
    "eos_token_id": 1,
    "unk_token_id": 2,
}
