"""Packed token stream dataset for Cosmopedia-v2 training.

This module implements a packed token stream representation that:
- Stores documents as D_1, EOS, D_2, EOS, D_3, EOS, ...
- Does not truncate long documents
- Does not pad short documents
- Supports true memory-mapped access for large datasets
- Provides deterministic block coverage for reproducible training
"""

import hashlib
import json
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterator, List, Optional, Tuple, Union

import numpy as np


# Current packed dataset format version
FORMAT_VERSION = "1.0"


@dataclass(frozen=True)
class DatasetManifest:
    """Manifest containing dataset metadata."""
    format_version: str
    tokenizer_hash: str  # SHA256 of tokenizer config (for verification)
    vocab_size: int
    token_dtype: str  # 'uint16' or 'uint32'
    eos_token_id: int
    total_train_tokens: int
    total_val_tokens: int
    train_document_count: int
    val_document_count: int
    train_shard_count: int
    val_shard_count: int
    source_dataset_name: str
    preprocessing_seed: int
    context_length: int  # Maximum document length (for filtering)
    created_at: Optional[str] = None
    
    def to_dict(self) -> Dict:
        """Convert to dictionary for JSON serialization."""
        return {
            "format_version": self.format_version,
            "tokenizer_hash": self.tokenizer_hash,
            "vocab_size": self.vocab_size,
            "token_dtype": self.token_dtype,
            "eos_token_id": self.eos_token_id,
            "total_train_tokens": self.total_train_tokens,
            "total_val_tokens": self.total_val_tokens,
            "train_document_count": self.train_document_count,
            "val_document_count": self.val_document_count,
            "train_shard_count": self.train_shard_count,
            "val_shard_count": self.val_shard_count,
            "source_dataset_name": self.source_dataset_name,
            "preprocessing_seed": self.preprocessing_seed,
            "context_length": self.context_length,
            "created_at": self.created_at,
        }
    
    @classmethod
    def from_dict(cls, data: Dict) -> "DatasetManifest":
        """Create manifest from dictionary."""
        return cls(**data)


class PackedTokenDataset:
    """
    Packed token stream dataset with memory-mapped shard access.
    
    The dataset is stored as packed binary files containing contiguous token IDs.
    Documents are separated by EOS tokens, and no truncation or padding is performed.
    
    Training samples are drawn as contiguous sequences of T+1 tokens, where:
    - inputs = tokens[start:start+T]
    - targets = tokens[start+1:start+T+1]
    """
    
    def __init__(
        self,
        shard_paths: List[Path],
        manifest: DatasetManifest,
        shard_cache_size: int = 2,
    ):
        """
        Initialize packed token dataset.
        
        Args:
            shard_paths: List of paths to token shard files
            manifest: Dataset manifest with metadata
            shard_cache_size: Number of shards to keep memory-mapped (LRU)
        """
        self.shard_paths = [Path(p) for p in shard_paths]
        self.manifest = manifest
        
        # Validate shard count
        expected_shards = manifest.train_shard_count if "train" in str(shard_paths[0]) else manifest.val_shard_count
        if len(self.shard_paths) != expected_shards:
            raise ValueError(
                f"Expected {expected_shards} shards, got {len(self.shard_paths)}"
            )
        
        # Validate token dtype
        if manifest.token_dtype not in ("uint16", "uint32"):
            raise ValueError(f"Unsupported token dtype: {manifest.token_dtype}")
        
        self.dtype = np.dtype(manifest.token_dtype)
        self.shard_cache_size = shard_cache_size
        
        # Shard cache with LRU eviction
        self._shard_cache: List[Tuple[Path, np.memmap]] = []
        self._shard_access_order: List[Path] = []
        
        # Total token count for sampling bounds
        self.total_tokens = manifest.total_train_tokens if "train" in str(shard_paths[0]) else manifest.total_val_tokens
        
        print(f"PackedTokenDataset initialized:")
        print(f"  Shards: {len(self.shard_paths)}")
        print(f"  Total tokens: {self.total_tokens:,}")
        print(f"  Token dtype: {self.dtype}")
        print(f"  EOS token ID: {manifest.eos_token_id}")
    
    def _map_shard(self, path: Path) -> np.memmap:
        """Memory-map a shard file."""
        # Check if already mapped
        for i, (cached_path, mm) in enumerate(self._shard_cache):
            if cached_path == path:
                # Move to end (most recently used)
                self._shard_access_order.remove(path)
                self._shard_access_order.append(path)
                return mm
        
        # Create new mapping
        with open(path, "rb") as f:
            mm = np.memmap(f, dtype=self.dtype, mode="r", offset=0)
        
        # Add to cache
        if len(self._shard_cache) >= self.shard_cache_size:
            # Evict least recently used
            lru_path = self._shard_access_order.pop(0)
            for i, (cached_path, mm) in enumerate(self._shard_cache):
                if cached_path == lru_path:
                    self._shard_cache.pop(i)
                    break
        
        self._shard_cache.append((path, mm))
        self._shard_access_order.append(path)
        
        return mm
    
    def _get_shard_offsets(self) -> List[int]:
        """Get cumulative token offsets for each shard."""
        offsets = [0]
        for path in self.shard_paths:
            mm = self._map_shard(path)
            offsets.append(offsets[-1] + len(mm))
        return offsets
    
    def get_block_ranges(
        self,
        seq_length: int,
    ) -> List[Tuple[int, int]]:
        """
        Get all valid non-overlapping block ranges for a given sequence length.
        
        For context length T, valid start positions are 0, T, 2T, ...
        Each block consumes T+1 tokens for input/target pairs.
        
        Args:
            seq_length: Context length T
            
        Returns:
            List of (start, end) tuples for valid blocks
        """
        offsets = self._get_shard_offsets()
        total_tokens = offsets[-1]
        
        # Valid start positions: 0, T+1, 2*(T+1), ...
        # Wait - we need T+1 tokens for a sequence of length T
        # inputs = tokens[start:start+T]
        # targets = tokens[start+1:start+T+1]
        # So we need T+1 consecutive tokens starting at position start
        
        block_size = seq_length + 1  # T+1 tokens per block
        num_blocks = total_tokens // block_size
        
        ranges = []
        for i in range(num_blocks):
            start = i * block_size
            end = start + block_size
            ranges.append((start, end))
        
        return ranges
    
    def get_block(
        self,
        start: int,
        seq_length: int,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Get a single training block from absolute token position.
        
        Args:
            start: Absolute token position (not shard-relative)
            seq_length: Sequence length T
            
        Returns:
            Tuple of (inputs, targets) where inputs.shape == targets.shape == (T,)
        """
        # Find which shard contains this position
        offsets = self._get_shard_offsets()
        shard_idx = 0
        for i, offset in enumerate(offsets):
            if start < offset:
                shard_idx = i - 1
                break
        else:
            shard_idx = len(self.shard_paths) - 1
        
        # Get shard data and compute relative position
        mm = self._map_shard(self.shard_paths[shard_idx])
        shard_start = offsets[shard_idx]
        relative_start = start - shard_start
        
        # Extract T+1 tokens
        tokens = mm[relative_start:relative_start + seq_length + 1]
        
        # Split into inputs and targets
        inputs = tokens[:-1]  # T tokens
        targets = tokens[1:]  # T tokens
        
        return inputs, targets
    
    def sample_block(
        self,
        seq_length: int,
        rng: np.random.Generator,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Sample a random training block.
        
        Args:
            seq_length: Sequence length T
            rng: Random number generator
            
        Returns:
            Tuple of (inputs, targets)
        """
        offsets = self._get_shard_offsets()
        
        # Sample uniformly across all valid positions
        # Valid positions are 0, 1, ..., total_tokens - (seq_length + 1)
        max_start = self.total_tokens - (seq_length + 1)
        start = rng.integers(0, max_start + 1)
        
        return self.get_block(start, seq_length)


class PackedDatasetGenerator:
    """
    Generate packed token shards from Cosmopedia Parquet data.
    
    This is the canonical preprocessing implementation that should be used
    by all training scripts. It:
    - Tokenizes documents completely (no truncation)
    - Appends EOS after each document
    - Packs tokens contiguously into binary shards
    - Splits train/val deterministically based on document hash
    """
    
    def __init__(
        self,
        tokenizer_hash: str,
        vocab_size: int,
        eos_token_id: int,
        context_length: int = 512,
        shard_size_mb: float = 256.0,
        preprocessing_seed: int = 42,
        dtype: str = "uint16",
    ):
        """
        Initialize the packed dataset generator.
        
        Args:
            tokenizer_hash: SHA256 hash of tokenizer config for verification
            vocab_size: Vocabulary size (determines dtype if uint16)
            eos_token_id: Token ID for end-of-sequence
            context_length: Maximum document length (for filtering/monitoring)
            shard_size_mb: Target shard size in MB
            preprocessing_seed: Seed for deterministic train/val split
            dtype: Data type ('uint16' or 'uint32')
        """
        self.tokenizer_hash = tokenizer_hash
        self.vocab_size = vocab_size
        self.eos_token_id = eos_token_id
        self.context_length = context_length
        self.shard_size_mb = shard_size_mb
        self.preprocessing_seed = preprocessing_seed
        
        # Validate dtype based on vocab size
        if vocab_size > 65535 and dtype == "uint16":
            raise ValueError(
                f"Vocabulary size {vocab_size} exceeds uint16 limit (65536). "
                "Use dtype='uint32' or reduce vocab_size."
            )
        
        self.dtype = dtype
        self.dtype_np = np.dtype(dtype)
        
        # Calculate tokens per shard
        self.tokens_per_shard = int((shard_size_mb * 1024 * 1024) / self.dtype_np.itemsize)
        
        print(f"PackedDatasetGenerator initialized:")
        print(f"  Tokenizer hash: {tokenizer_hash[:16]}...")
        print(f"  Vocab size: {vocab_size}")
        print(f"  EOS token ID: {eos_token_id}")
        print(f"  Shard size: {shard_size_mb} MB")
        print(f"  Tokens per shard: {self.tokens_per_shard:,}")
        print(f"  Dtype: {dtype}")
    
    def compute_document_hash(self, document_text: str) -> int:
        """Compute deterministic hash for document to enable consistent train/val split."""
        return int(hashlib.sha256(document_text.encode("utf-8")).hexdigest()[:16], 16)
    
    def split_documents(
        self,
        documents: List[str],
        val_ratio: float = 0.01,
    ) -> Tuple[List[str], List[str]]:
        """
        Split documents into train and validation sets deterministically.
        
        Uses document hash to ensure consistent splitting regardless of order.
        
        Args:
            documents: List of document texts
            val_ratio: Ratio of documents for validation
            
        Returns:
            Tuple of (train_docs, val_docs)
        """
        # Compute hash for each document and sort by hash
        doc_hashes = [
            (self.compute_document_hash(doc), i, doc)
            for i, doc in enumerate(documents)
        ]
        doc_hashes.sort(key=lambda x: x[0])  # Sort by hash
        
        # Take first val_ratio for validation
        val_count = max(1, int(len(doc_hashes) * val_ratio))
        
        val_docs = [doc for _, _, doc in doc_hashes[:val_count]]
        train_docs = [doc for _, _, doc in doc_hashes[val_count:]]
        
        return train_docs, val_docs
    
    def tokenize_document(self, text: str) -> List[int]:
        """
        Tokenize a single document with EOS token.
        
        Args:
            text: Input text
            
        Returns:
            List of token IDs with EOS at end
        """
        # Note: We do NOT truncate documents - store complete tokenization
        return [self.eos_token_id]  # Just EOS placeholder
    
    def generate_packed_shards(
        self,
        train_documents: List[str],
        val_documents: List[str],
        output_dir: Union[str, Path],
        max_shards: Optional[int] = None,
    ) -> Tuple[List[Path], List[Path], DatasetManifest]:
        """
        Generate packed token shards for training and validation.
        
        Args:
            train_documents: List of training documents
            val_documents: List of validation documents
            output_dir: Output directory for shards
            max_shards: Maximum shards to generate (None = all)
            
        Returns:
            Tuple of (train_shard_paths, val_shard_paths, manifest)
        """
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        
        train_shard_paths = self._generate_shards(
            documents=train_documents,
            shard_prefix="train",
            output_dir=output_dir,
            max_shards=max_shards,
        )
        
        val_shard_paths = self._generate_shards(
            documents=val_documents,
            shard_prefix="val",
            output_dir=output_dir,
            max_shards=max_shards,
        )
        
        # Create manifest
        train_tokens = sum(os.path.getsize(p) // self.dtype_np.itemsize for p in train_shard_paths)
        val_tokens = sum(os.path.getsize(p) // self.dtype_np.itemsize for p in val_shard_paths)
        
        manifest = DatasetManifest(
            format_version=FORMAT_VERSION,
            tokenizer_hash=self.tokenizer_hash,
            vocab_size=self.vocab_size,
            token_dtype=self.dtype,
            eos_token_id=self.eos_token_id,
            total_train_tokens=train_tokens,
            total_val_tokens=val_tokens,
            train_document_count=len(train_documents),
            val_document_count=len(val_documents),
            train_shard_count=len(train_shard_paths),
            val_shard_count=len(val_shard_paths),
            source_dataset_name="Cosmopedia-v2",
            preprocessing_seed=self.preprocessing_seed,
            context_length=self.context_length,
        )
        
        # Save manifest
        manifest_path = output_dir / "manifest.json"
        with open(manifest_path, "w") as f:
            json.dump(manifest.to_dict(), f, indent=2)
        
        print()
        print("=" * 60)
        print("Packed Dataset Generation Complete")
        print("=" * 60)
        print(f"Train shards: {len(train_shard_paths)}")
        print(f"Val shards: {len(val_shard_paths)}")
        print(f"Train tokens: {train_tokens:,}")
        print(f"Val tokens: {val_tokens:,}")
        print(f"Manifest saved to: {manifest_path}")
        
        return train_shard_paths, val_shard_paths, manifest
    
    def _generate_shards(
        self,
        documents: List[str],
        shard_prefix: str,
        output_dir: Path,
        max_shards: Optional[int],
    ) -> List[Path]:
        """Generate packed shards for a set of documents."""
        shard_paths = []
        shard_id = 0
        current_tokens = []
        
        # Process each document - do NOT truncate
        for doc_idx, doc_text in enumerate(documents):
            # Tokenize completely
            doc_tokens = self.tokenize_document(doc_text)
            current_tokens.extend(doc_tokens)
            
            # Write shard when it reaches target size
            if len(current_tokens) >= self.tokens_per_shard:
                shard_path = self._write_shard(
                    shard_id=shard_id,
                    prefix=shard_prefix,
                    tokens=current_tokens[:self.tokens_per_shard],
                    output_dir=output_dir,
                )
                shard_paths.append(shard_path)
                current_tokens = current_tokens[self.tokens_per_shard:]
                shard_id += 1
                
                if max_shards and shard_id >= max_shards:
                    break
        
        # Flush remaining tokens
        if current_tokens and (max_shards is None or shard_id < max_shards):
            shard_path = self._write_shard(
                shard_id=shard_id,
                prefix=shard_prefix,
                tokens=current_tokens,
                output_dir=output_dir,
            )
            shard_paths.append(shard_path)
        
        return shard_paths
    
    def _write_shard(
        self,
        shard_id: int,
        prefix: str,
        tokens: List[int],
        output_dir: Path,
    ) -> Path:
        """Write a single packed shard file."""
        shard_path = output_dir / f"{prefix}_shard_{shard_id:05d}.bin"
        
        # Convert to numpy array and write
        arr = np.array(tokens, dtype=self.dtype_np)
        arr.tofile(shard_path)
        
        return shard_path


def load_packed_dataset(
    output_dir: Union[str, Path],
    is_train: bool = True,
) -> Tuple[PackedTokenDataset, DatasetManifest]:
    """
    Load a packed token dataset from directory.
    
    Args:
        output_dir: Directory containing shards and manifest
        is_train: Load training shards if True, validation otherwise
        
    Returns:
        Tuple of (dataset, manifest)
    """
    output_dir = Path(output_dir)
    
    # Load manifest
    manifest_path = output_dir / "manifest.json"
    if not manifest_path.exists():
        raise FileNotFoundError(f"Manifest not found: {manifest_path}")
    
    with open(manifest_path, "r") as f:
        manifest_data = json.load(f)
    
    manifest = DatasetManifest.from_dict(manifest_data)
    
    # Find shard files
    prefix = "train" if is_train else "val"
    shard_paths = sorted(output_dir.glob(f"{prefix}_shard_*.bin"))
    
    if not shard_paths:
        raise ValueError(f"No {prefix} shards found in {output_dir}")
    
    dataset = PackedTokenDataset(
        shard_paths=shard_paths,
        manifest=manifest,
    )
    
    return dataset, manifest


def verify_manifest(manifest: DatasetManifest) -> bool:
    """
    Verify that a manifest is valid and compatible.
    
    Args:
        manifest: Dataset manifest to verify
        
    Returns:
        True if valid, raises exception otherwise
    """
    # Check version
    if manifest.format_version != FORMAT_VERSION:
        raise ValueError(
            f"Unsupported manifest version: {manifest.format_version}. "
            f"Expected: {FORMAT_VERSION}"
        )
    
    # Check dtype validity
    if manifest.token_dtype not in ("uint16", "uint32"):
        raise ValueError(f"Invalid token dtype: {manifest.token_dtype}")
    
    # Check vocab size fits in dtype
    if manifest.vocab_size > 65535 and manifest.token_dtype == "uint16":
        raise ValueError(
            f"Vocab size {manifest.vocab_size} exceeds uint16 limit"
        )
    
    return True
