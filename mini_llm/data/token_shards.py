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
        shard_paths = []
        shard_id = 0
        current_batch_docs: List[str] = []
        current_batch_tokens: List[List[int]] = []
        
        print(f"Generating token shards from {self.parquet_reader.num_shards} shards")
        print(f"Target: {self.documents_per_shard} documents per shard")
        print(f"Output directory: {self.output_dir}")
        
        total_docs = 0
        total_dropped = 0
        
        for record in tqdm(
            self.parquet_reader.iter_records(),
            desc="Processing documents",
            unit="doc",
        ):
            text = record.get("text", "")
            if text:
                current_batch_docs.append(text)
                
                # Process batch when full
                if len(current_batch_docs) >= self.batch_size:
                    tokenized, dropped = self._process_batch(current_batch_docs)
                    current_batch_tokens.extend(tokenized)
                    total_dropped += dropped
                    
                    # Check if we should flush to shard
                    if len(current_batch_tokens) >= self.documents_per_shard:
                        shard_path = self._flush_shard(
                            shard_id, current_batch_tokens[: self.documents_per_shard]
                        )
                        if shard_path:
                            shard_paths.append(shard_path)
                            shard_id += 1
                        
                        # Keep remaining
                        current_batch_tokens = current_batch_tokens[self.documents_per_shard :]
                        current_batch_docs = []
                        total_docs += self.documents_per_shard
                        
                        if max_shards and shard_id >= max_shards:
                            break
            
            if max_shards and shard_id >= max_shards:
                break
        
        # Flush remaining
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


def load_token_shard(shard_path: str) -> np.ndarray:
    """
    Load a token shard from binary file.
    
    Args:
        shard_path: Path to .bin file
        
    Returns:
        Array of shape (num_documents, context_length)
    """
    import mmap
    
    with open(shard_path, "rb") as f:
        # Read header: (num_documents, context_length) as int64
        header = f.read(16)
        num_docs = int.from_bytes(header[0:8], byteorder="little")
        context_len = int.from_bytes(header[8:16], byteorder="little")
        
        # Read data using mmap for large files or direct read for small ones
        f.seek(16)
        total_bytes = num_docs * context_len * 2  # uint16 = 2 bytes
        data = f.read(total_bytes)
        
        arr = np.frombuffer(data, dtype=np.uint16)
        return arr.reshape(num_docs, context_len)


def create_minibatch(
    shard_data: np.ndarray,
    batch_size: int,
    seq_length: Optional[int] = None,
    rng: Optional[np.random.Generator] = None,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Create a random minibatch from shard data.
    
    Args:
        shard_data: Array of shape (num_docs, context_length)
        batch_size: Number of sequences per batch
        seq_length: Length of each sequence (defaults to full context)
        rng: Random generator for reproducibility (uses global by default)
        
    Returns:
        Tuple of (inputs, targets) where targets are shifted by 1 position
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
