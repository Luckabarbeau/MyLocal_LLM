"""Generate binary token shards from Cosmopedia Parquet data.

This module converts text documents to token IDs and writes them as
memory-mapped binary files for efficient training.
"""

import mmap
import os
from pathlib import Path
from typing import Iterator, List, Optional, Tuple

import numpy as np
import pyarrow.parquet as pq

from mini_llm.data.parquet_reader import CosmopediaParquetReader
from mini_llm.tokenizer.tokenizer import SimpleBPETokenizer


class TokenShardWriter:
    """
    Writes tokenized documents to binary shards.
    
    Each shard is a memory-mapped file containing:
    - Header: (num_documents, context_length) as int64
    - Data: uint16 token IDs (packed 2 tokens per byte for efficiency)
    
    For very large vocabularies (> 65536), use uint32 instead.
    """
    
    def __init__(
        self,
        output_dir: str,
        context_length: int = 1024,
        dtype: str = "uint16",
    ):
        """
        Initialize the shard writer.
        
        Args:
            output_dir: Directory to write shards
            context_length: Maximum sequence length
            dtype: Data type for tokens ('uint16' or 'uint32')
        """
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.context_length = context_length
        self.dtype = np.dtype(dtype)
        
        # Vocabulary size determines dtype
        if dtype == "uint16":
            self.max_vocab_size = 65_536
        elif dtype == "uint32":
            self.max_vocab_size = 4_294_967_296
        else:
            raise ValueError(f"Unsupported dtype: {dtype}")
    
    def write_shard(
        self,
        shard_id: int,
        token_ids: List[List[int]],
        documents_per_shard: int = 10_000,
    ) -> Path:
        """
        Write a shard of tokenized documents.
        
        Args:
            shard_id: Shard index
            token_ids: List of document token ID lists
            documents_per_shard: Target documents per shard
            
        Returns:
            Path to written shard file
        """
        if not token_ids:
            return None
        
        # Trim to context length and filter empty documents
        trimmed = []
        for ids in token_ids:
            if ids:
                trimmed.append(ids[: self.context_length])
        
        if not trimmed:
            return None
        
        # Calculate file size
        num_documents = len(trimmed)
        header_size = 2 * 8  # Two int64 values
        tokens_per_doc = self.context_length
        data_size = num_documents * tokens_per_doc * self.dtype.itemsize
        total_size = header_size + data_size
        
        # Write file
        shard_path = self.output_dir / f"shard_{shard_id:05d}.bin"
        
        with open(shard_path, "wb") as f:
            f.write(b"\x00" * total_size)
        
        # Memory-map and write
        with open(shard_path, "r+b") as f:
            mm = mmap.mmap(f.fileno(), total_size)
            
            # Write header: (num_documents, context_length)
            mm[0:8] = np.int64(num_documents).tobytes()
            mm[8:16] = np.int64(self.context_length).tobytes()
            
            # Write token data
            offset = header_size
            for doc_tokens in trimmed:
                # Pad or truncate to context length
                padded = doc_tokens + [0] * (tokens_per_doc - len(doc_tokens))
                arr = np.array(padded, dtype=self.dtype)
                mm[offset : offset + arr.nbytes] = arr.tobytes()
                offset += arr.nbytes
            
            mm.flush()
            mm.close()
        
        return shard_path
    
    def estimate_documents_per_shard(
        self, text_length_mb: float = 100.0
    ) -> int:
        """
        Estimate documents per shard for target file size.
        
        Args:
            text_length_mb: Target shard size in MB
            
        Returns:
            Estimated documents per shard
        """
        # Rough estimate: ~2 bytes per token, context_length tokens per doc
        bytes_per_doc = self.context_length * 2
        target_bytes = text_length_mb * 1024 * 1024
        return int(target_bytes / bytes_per_doc)


class TokenShardGenerator:
    """
    Generates tokenized binary shards from Cosmopedia Parquet data.
    """
    
    def __init__(
        self,
        tokenizer: SimpleBPETokenizer,
        parquet_reader: CosmopediaParquetReader,
        output_dir: str = "token_shards",
        context_length: int = 1024,
        documents_per_shard: int = 10_000,
        batch_size: int = 100,
    ):
        """
        Initialize the shard generator.
        
        Args:
            tokenizer: Tokenizer instance
            parquet_reader: CosmopediaParquetReader instance
            output_dir: Directory for output shards
            context_length: Maximum sequence length
            documents_per_shard: Target documents per shard
            batch_size: Documents to process at once
        """
        self.tokenizer = tokenizer
        self.parquet_reader = parquet_reader
        self.output_dir = Path(output_dir)
        self.context_length = context_length
        self.documents_per_shard = documents_per_shard
        self.batch_size = batch_size
        
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # Validate vocab size
        if len(tokenizer) > 65_536:
            raise ValueError(
                f"Vocabulary size {len(tokenizer)} exceeds uint16 limit (65536). "
                "Consider reducing vocab_size or using uint32."
            )
    
    def tokenize_document(self, text: str) -> List[int]:
        """
        Tokenize a single document with EOS token.
        
        IMPORTANT: We do NOT truncate documents - store complete tokenization.
        The packed dataset format handles context length at sampling time.
        
        Args:
            text: Input text
            
        Returns:
            List of token IDs with EOS at end
        """
        ids = self.tokenizer.encode(text)
        # Add EOS token
        if ids and ids[-1] != self.tokenizer.token_to_id.get(self.tokenizer.eos_token):
            eos_id = self.tokenizer.token_to_id.get(self.tokenizer.eos_token, 0)
            ids.append(eos_id)
        return ids
    
    def _process_batch(
        self, documents: List[str]
    ) -> tuple[List[List[int]], int]:
        """
        Process a batch of documents.
        
        Args:
            documents: List of text documents
            
        Returns:
            Tuple of (tokenized documents, dropped count)
        """
        tokenized = []
        dropped = 0
        
        for doc in documents:
            try:
                ids = self.tokenize_document(doc)
                if ids:  # Skip empty documents
                    tokenized.append(ids)
                else:
                    dropped += 1
            except Exception as e:
                dropped += 1
                continue
        
        return tokenized, dropped
    
    def generate_shards(self, max_shards: Optional[int] = None) -> List[Path]:
        """
        Generate all token shards.
        
        Args:
            max_shards: Maximum shards to generate (None for all)
            
        Returns:
            List of paths to generated shards
        """
        from tqdm import tqdm
        
        shard_paths = []
        shard_id = 0
        current_batch_docs: List[str] = []
        current_batch_tokens: List[List[int]] = []
        
        print(f"Generating token shards from {self.parquet_reader.num_shards} shards")
        print(f"Target: {self.documents_per_shard} documents per shard")
        print(f"Output directory: {self.output_dir}")
        
        total_docs = 0
        total_dropped = 0
        processed_indices = set()  # Track which documents we've processed (Issue #6)
        
        for record_idx, record in enumerate(tqdm(
            self.parquet_reader.iter_records(),
            desc="Processing documents",
            unit="doc",
        )):
            text = record.get("text", "")
            if text:
                current_batch_docs.append(text)
                
                # Process batch when full
                if len(current_batch_docs) >= self.batch_size:
                    tokenized, dropped = self._process_batch(current_batch_docs)
                    current_batch_tokens.extend(tokenized)
                    total_dropped += dropped
                    total_docs += len(tokenized)
                    
                    # Track which documents we processed (Issue #6 - no duplication)
                    for i in range(len(current_batch_docs)):
                        if current_batch_docs[i]:  # Non-empty docs get a tokenized entry
                            processed_indices.add(len(processed_indices))
                    
                    # Always clear the docs batch after processing
                    current_batch_docs = []
                    
                    # Check if we should flush to shard
                    if len(current_batch_tokens) >= self.documents_per_shard:
                        shard_path = self._flush_shard(
                            shard_id, current_batch_tokens[: self.documents_per_shard]
                        )
                        if shard_path:
                            shard_paths.append(shard_path)
                            shard_id += 1
                        
                        # Keep remaining tokens
                        current_batch_tokens = current_batch_tokens[self.documents_per_shard :]
                        
                        if max_shards and shard_id >= max_shards:
                            break
            
            if max_shards and shard_id >= max_shards:
                break
        
        # Flush remaining documents - FIX FOR ISSUE #5: This was being skipped!
        if current_batch_docs:
            tokenized, dropped = self._process_batch(current_batch_docs)
            current_batch_tokens.extend(tokenized)
            total_docs += len(tokenized)
            total_dropped += dropped
        
        if current_batch_tokens and (not max_shards or shard_id < max_shards):
            if current_batch_tokens:
                shard_path = self._flush_shard(shard_id, current_batch_tokens)
                if shard_path:
                    shard_paths.append(shard_path)
            total_docs += len(current_batch_tokens)
        
        print(f"\nGeneration complete!")
        print(f"  Shards written: {len(shard_paths)}")
        print(f"  Total documents processed: {total_docs}")
        print(f"  Documents dropped: {total_dropped}")
        
        return shard_paths
    
    def _flush_shard(
        self, shard_id: int, token_data: List[List[int]]
    ) -> Optional[Path]:
        """Write a single shard file."""
        if not token_data:
            return None
        
        writer = TokenShardWriter(
            str(self.output_dir),
            context_length=self.context_length,
            dtype="uint16" if len(self.tokenizer) <= 65_536 else "uint32",
        )
        
        return writer.write_shard(shard_id, token_data)


def generate_token_shards(
    dataset_path: str = "../cosmopedia-v2/cosmopedia-v2",
    output_dir: str = "token_shards",
    context_length: int = 1024,
    documents_per_shard: int = 10_000,
    max_shards: Optional[int] = None,
) -> List[Path]:
    """
    Convenience function to generate token shards.
    
    Args:
        dataset_path: Path to Cosmopedia Parquet directory
        output_dir: Output directory for shards
        context_length: Maximum sequence length
        documents_per_shard: Documents per shard
        max_shards: Maximum shards to generate
        
    Returns:
        List of paths to generated shard files
    """
    # Create parquet reader
    parquet_reader = CosmopediaParquetReader(dataset_path=dataset_path)
    
    # Train tokenizer on a sample (limited to avoid memory issues)
    print("Training tokenizer on sample...")
    sample_texts = []
    count = 0
    for record in parquet_reader.iter_records():
        sample_texts.append(record.get("text", ""))
        count += 1
        if count >= 500:  # Limit to 500 documents for tokenizer training
            break
    
    tokenizer = SimpleBPETokenizer(vocab_size=16_384)
    tokenizer.train(sample_texts)
    
    # Save tokenizer
    tokenizer_path = Path(output_dir) / "tokenizer.json"
    tokenizer.save(str(tokenizer_path))
    print(f"Tokenizer saved to {tokenizer_path}")
    
    # Generate shards
    generator = TokenShardGenerator(
        tokenizer=tokenizer,
        parquet_reader=parquet_reader,
        output_dir=output_dir,
        context_length=context_length,
        documents_per_shard=documents_per_shard,
    )
    
    return generator.generate_shards(max_shards=max_shards)


def map_token_shard(
    shard_path: str,
    seq_length: Optional[int] = None,
) -> np.memmap:
    """Memory-map packed or legacy token shards without a full RAM copy.

    When a legacy rectangular row is shorter than ``seq_length + 1``, map
    only its token payload as a flat stream.  This supports existing
    context-512 shards at training context 512+ while excluding the header.
    """
    path = Path(shard_path)
    file_size = path.stat().st_size
    itemsize = np.dtype(np.uint16).itemsize

    if file_size >= 16:
        with open(path, "rb") as f:
            header = f.read(16)

        num_docs = int.from_bytes(header[0:8], byteorder="little")
        context_len = int.from_bytes(header[8:16], byteorder="little")
        expected_data_size = num_docs * context_len * itemsize
        is_legacy = (
            num_docs > 0
            and context_len > 0
            and expected_data_size == file_size - 16
        )

        if is_legacy:
            required = None if seq_length is None else int(seq_length) + 1
            if required is not None and context_len < required:
                return np.memmap(
                    path, dtype=np.uint16, mode="r", offset=16,
                    shape=(num_docs * context_len,),
                )

            return np.memmap(
                path, dtype=np.uint16, mode="r", offset=16,
                shape=(num_docs, context_len),
            )

    return np.memmap(path, dtype=np.uint16, mode="r")

def load_token_shard(shard_path: str) -> np.ndarray:
    """
    Load a token shard from binary file.
    
    Supports both formats:
    1. OLD FORMAT: Has header (num_documents, context_length) as int64
    2. NEW FORMAT: Direct packed token stream (1D array)
    
    Args:
        shard_path: Path to .bin file
        
    Returns:
        Either (num_docs, context_length) array or 1D packed token array
    """
    import mmap
    
    with open(shard_path, "rb") as f:
        # Check file size - if small, likely old format with header
        f.seek(0, 2)  # Seek to end
        file_size = f.tell()
        f.seek(0)  # Reset
        
        if file_size > 100:  # Heuristic: likely has header
            try:
                header = f.read(16)
                num_docs = int.from_bytes(header[0:8], byteorder="little")
                context_len = int.from_bytes(header[8:16], byteorder="little")
                
                # Verify this looks like valid old-format data
                expected_data_size = num_docs * context_len * 2
                if num_docs > 0 and context_len > 0 and expected_data_size < file_size - 16:
                    # This looks like old format
                    f.seek(16)
                    total_bytes = min(expected_data_size, file_size - 16)
                    data = f.read(total_bytes)
                    
                    arr = np.frombuffer(data, dtype=np.uint16)
                    return arr.reshape(num_docs, context_len)
            except:
                pass
        
        # New format: packed token stream (1D array)
        f.seek(0)
        data = f.read()
        arr = np.frombuffer(data, dtype=np.uint16)
        return arr


def create_minibatch(
    shard_data: np.ndarray,
    batch_size: int,
    seq_length: Optional[int] = None,
    rng: Optional[np.random.Generator] = None,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Create a random minibatch from shard data.
    
    This function works with two types of shard data:
    1. OLD FORMAT (num_docs, context_length): Fixed-length rectangular documents
    2. NEW FORMAT (packed token stream): 1D array of contiguous tokens with EOS separators
    
    For packed data, documents are D_1, EOS, D_2, EOS, ... stored in a flat array.
    Training samples extract T+1 consecutive tokens where:
        inputs = tokens[start:start+T]
        targets = tokens[start+1:start+T+1]
    
    Args:
        shard_data: Either (num_docs, context_length) or 1D packed token stream
        batch_size: Number of sequences per batch
        seq_length: Length of each sequence T (defaults to full context for old format)
        rng: Random generator for reproducibility (uses global by default)
        
    Returns:
        Tuple of (inputs, targets) where targets are shifted by 1 position
    """
    if rng is None:
        rng = np.random.default_rng()
    
    # Detect format: old (2D) or new (1D packed)
    if shard_data.ndim == 1:
        # NEW FORMAT: Packed token stream
        return _create_minibatch_packed(shard_data, batch_size, seq_length, rng)
    else:
        # OLD FORMAT: Rectangular documents with padding
        return _create_minibatch_old(shard_data, batch_size, seq_length, rng)


def _create_minibatch_old(
    shard_data: np.ndarray,
    batch_size: int,
    seq_length: Optional[int] = None,
    rng: Optional[np.random.Generator] = None,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Create minibatch from old rectangular format (num_docs, context_length).
    
    Note: This format truncates long documents and pads short ones.
    New code should use packed format instead.
    """
    if rng is None:
        rng = np.random.default_rng()
    
    if seq_length is None:
        seq_length = shard_data.shape[1]
    
    num_docs, context_len = shard_data.shape
    
    # Ensure we can sample valid sequences
    # We need seq_length + 1 tokens to produce seq_length input/target pairs
    required_length = seq_length + 1
    max_start = context_len - required_length
    if max_start < 0:
        raise ValueError(
            f"context_len ({context_len}) must be >= seq_length + 1 ({required_length})."
        )
    
    # Sample random document IDs (not just first batch_size documents)
    doc_ids = rng.integers(0, num_docs, batch_size)
    
    # Sample random starting positions within each document
    start_pos = rng.integers(0, max_start + 1, batch_size)
    
    # Extract sequences using vectorized indexing
    # Create offsets [0, 1, ..., seq_length]
    offsets = np.arange(required_length)
    
    # Compute indices for each position in the sequence
    # indices has shape (batch_size, required_length)
    indices = doc_ids[:, None] * context_len + (start_pos[:, None] + offsets)
    
    # Flatten shard_data and gather
    flat_data = shard_data.ravel()
    all_tokens = flat_data[indices]
    
    # Split into inputs and targets
    inputs = all_tokens[:, :-1]  # shape (batch_size, seq_length)
    targets = all_tokens[:, 1:]  # shape (batch_size, seq_length)
    
    return inputs, targets


def _create_minibatch_packed(
    packed_tokens: np.ndarray,
    batch_size: int,
    seq_length: Optional[int] = None,
    rng: Optional[np.random.Generator] = None,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Create minibatch from packed token stream format.
    
    Documents are stored as: D_1, EOS, D_2, EOS, D_3, EOS, ...
    Training samples draw T+1 consecutive tokens from anywhere in the stream.
    
    Args:
        packed_tokens: 1D array of contiguous token IDs with EOS separators
        batch_size: Number of sequences per batch
        seq_length: Length of each sequence T
        rng: Random generator for reproducibility
        
    Returns:
        Tuple of (inputs, targets) where inputs.shape == targets.shape == (batch_size, T)
    """
    if rng is None:
        rng = np.random.default_rng()
    
    if seq_length is None:
        raise ValueError("seq_length must be specified for packed token format")
    
    # For packed format, we can sample from anywhere in the stream
    # We need seq_length + 1 consecutive tokens
    total_tokens = len(packed_tokens)
    required_length = seq_length + 1
    
    if total_tokens < required_length:
        raise ValueError(
            f"Packed token stream has {total_tokens} tokens, "
            f"but need {required_length} for seq_length={seq_length}"
        )
    
    # Sample random starting positions
    max_start = total_tokens - required_length
    start_positions = rng.integers(0, max_start + 1, batch_size)
    
    # Create offsets [0, 1, ..., seq_length]
    offsets = np.arange(required_length)
    
    # Compute indices for each position in the sequence
    # indices has shape (batch_size, required_length)
    indices = start_positions[:, None] + offsets[None, :]
    
    # Gather tokens
    all_tokens = packed_tokens[indices]
    
    # Split into inputs and targets
    inputs = all_tokens[:, :-1]  # shape (batch_size, seq_length)
    targets = all_tokens[:, 1:]  # shape (batch_size, seq_length)
    
    return inputs, targets
