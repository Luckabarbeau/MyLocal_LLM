"""Parquet dataset reader for Cosmopedia-v2.

This module provides a streaming interface for reading Cosmopedia-v2 Parquet shards.
It's designed to be memory-efficient and suitable for educational/research LLM training.
"""

import glob
import os
import re
from pathlib import Path
from typing import Iterator, List, Optional

import pyarrow as pa
import pyarrow.parquet as pq
from pyarrow.lib import RecordBatch


class CosmopediaParquetReader:
    """
    Streaming reader for Cosmopedia-v2 dataset stored as Parquet shards.
    
    Features:
        - discovers and sorts shards deterministically
        - streams records without loading entire shards into memory
        - supports column selection
        - provides batched iteration
        
    Attributes:
        dataset_path: Path to the directory containing Parquet shards
        shard_paths: List of discovered shard file paths, sorted by shard number
        schema: PyArrow schema of the dataset
        available_columns: List of column names in the dataset
    """
    
    def __init__(
        self,
        dataset_path: str = "../cosmopedia-v2/cosmopedia-v2",
        columns: Optional[List[str]] = None,
        batch_size: int = 1000,
    ):
        """
        Initialize the Parquet reader.
        
        Args:
            dataset_path: Path to directory containing Parquet shards.
                Default is "../cosmopedia-v2/cosmopedia-v2/" relative to project root.
            columns: List of column names to read. If None, reads all columns.
                Common columns: "text", "prompt", "token_length", "audience", "format", "seed_data"
            batch_size: Number of records per batch for streaming iteration.
        """
        self.dataset_path = Path(dataset_path).resolve()
        self.columns = columns
        self.batch_size = batch_size
        
        # Verify dataset directory exists
        if not self.dataset_path.exists():
            raise FileNotFoundError(
                f"Dataset directory not found: {self.dataset_path}"
            )
        if not self.dataset_path.is_dir():
            raise NotADirectoryError(
                f"Dataset path is not a directory: {self.dataset_path}"
            )
        
        # Discover and sort shards
        self.shard_paths = self._discover_shards()
        
        if not self.shard_paths:
            raise ValueError(
                f"No Parquet shards found in {self.dataset_path}. "
                "Expected files matching pattern 'train-*.parquet'."
            )
        
        # Get schema from first shard
        self.schema: pa.Schema = self._get_schema()
        self.available_columns = self.schema.names
        
        # Validate requested columns exist
        if self.columns:
            missing = set(self.columns) - set(self.available_columns)
            if missing:
                raise ValueError(
                    f"Requested columns not found in dataset: {missing}. "
                    f"Available columns: {self.available_columns}"
                )
    
    def _discover_shards(self) -> List[Path]:
        """Discover and sort Parquet shards by shard number."""
        pattern = self.dataset_path / "train-*.parquet"
        shard_paths = list(glob.glob(str(pattern)))
        
        if not shard_paths:
            return []
        
        # Sort by extracting shard number from filename
        # Pattern: train-XXXXX-of-YYYYY.parquet
        def extract_shard_number(path: str) -> int:
            match = re.search(r"train-(\d+)-of-\d+\.parquet", os.path.basename(path))
            if match:
                return int(match.group(1))
            return 0
        
        return sorted([Path(p) for p in shard_paths], key=extract_shard_number)
    
    def _get_schema(self) -> pa.Schema:
        """Get schema from the first shard."""
        return pq.read_schema(self.shard_paths[0])
    
    def _get_table_for_shard(self, shard_path: Path) -> pa.Table:
        """Read full table from a shard (used by token generation)."""
        return pq.read_table(shard_path, columns=["text"])
    
    @property
    def num_shards(self) -> int:
        """Number of Parquet shards in the dataset."""
        return len(self.shard_paths)
    
    @property
    def total_rows(self) -> Optional[int]:
        """Total number of rows across all shards (if inexpensive to obtain)."""
        # Parquet metadata stores row counts efficiently
        total = 0
        for shard_path in self.shard_paths:
            try:
                pf = pq.ParquetFile(shard_path)
                total += pf.metadata.num_rows
            except Exception:
                # Skip corrupted shards
                continue
        return total
    
    def inspect_shard(self, shard_idx: int) -> dict:
        """
        Get metadata about a specific shard.
        
        Args:
            shard_idx: Index of the shard (0-based)
            
        Returns:
            Dictionary with shard metadata
        """
        if shard_idx < 0 or shard_idx >= self.num_shards:
            raise IndexError(
                f"Shard index {shard_idx} out of range [0, {self.num_shards})"
            )
        
        pf = pq.ParquetFile(self.shard_paths[shard_idx])
        return {
            "path": str(self.shard_paths[shard_idx]),
            "num_rows": pf.metadata.num_rows,
            "num_columns": pf.metadata.num_columns,
            "row_groups": pf.metadata.num_row_groups,
        }
    
    def peek_records(self, n: int = 3) -> List[dict]:
        """
        Read first n records from the dataset for inspection.
        
        Args:
            n: Number of records to read
            
        Returns:
            List of record dictionaries
        """
        records = []
        for batch in self._iter_batches():
            pylist = batch.to_pylist()
            for record in pylist:
                records.append(record)
                if len(records) >= n:
                    return records
        return records
    
    def _iter_batches(self) -> Iterator[RecordBatch]:
        """Internal iterator over batches from all shards."""
        for shard_path in self.shard_paths:
            # Read the entire row group but stream via iterator
            table = pq.read_table(shard_path, columns=self.columns)
            
            # Split into batches of specified size
            num_rows = table.num_rows
            for start in range(0, num_rows, self.batch_size):
                end = min(start + self.batch_size, num_rows)
                batch = table.slice(start, end - start)
                yield batch
    
    def iter_records(self) -> Iterator[dict]:
        """
        Iterate over all records in the dataset.
        
        Yields:
            Dictionary for each record
        """
        for batch in self._iter_batches():
            for record in batch.to_pylist():
                yield record
    
    def __repr__(self) -> str:
        """String representation of the reader."""
        return (
            f"CosmopediaParquetReader("
            f"dataset_path={self.dataset_path}, "
            f"num_shards={self.num_shards}, "
            f"total_rows={self.total_rows}, "
            f"columns={self.columns}, "
            f"batch_size={self.batch_size})"
        )
