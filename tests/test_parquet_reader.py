"""Tests for the Parquet reader."""

import tempfile
import shutil
from pathlib import Path

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
import pytest

from mini_llm.data.parquet_reader import CosmopediaParquetReader


def create_synthetic_parquet_file(
    path: Path, num_rows: int, columns: dict
) -> None:
    """
    Create a synthetic Parquet file for testing.
    
    Args:
        path: Path to write the Parquet file
        num_rows: Number of rows to generate
        columns: Dict mapping column names to values (or value generators)
    """
    data = {}
    for col_name, col_value in columns.items():
        if callable(col_value):
            data[col_name] = [col_value() for _ in range(num_rows)]
        else:
            data[col_name] = [col_value] * num_rows
    
    table = pa.table(data)
    pq.write_table(table, path)


def create_multishard_dataset(
    temp_dir: Path, num_shards: int, rows_per_shard: int
) -> list:
    """
    Create multiple synthetic Parquet shards.
    
    Args:
        temp_dir: Directory to create shards in
        num_shards: Number of shards to create
        rows_per_shard: Rows per shard
        
    Returns:
        List of created shard paths
    """
    shard_paths = []
    for i in range(num_shards):
        filename = f"train-{i:05d}-of-{num_shards:05d}.parquet"
        path = temp_dir / filename
        
        data = {
            "text": [f"Sample text {i}-{j}" for j in range(rows_per_shard)],
            "prompt": [f"Prompt {i}-{j}" for j in range(rows_per_shard)],
            "token_length": [100 + j for j in range(rows_per_shard)],
        }
        
        table = pa.table(data)
        pq.write_table(table, path)
        shard_paths.append(path)
    
    return shard_paths


class TestCosmopediaParquetReader:
    """Tests for CosmopediaParquetReader."""
    
    def test_init_with_nonexistent_directory(self):
        """Should raise FileNotFoundError for nonexistent directory."""
        with pytest.raises(FileNotFoundError):
            CosmopediaParquetReader(dataset_path="/nonexistent/path")
    
    def test_total_rows_handles_corrupted_shards(self, tmp_path):
        """Should handle corrupted shards gracefully."""
        # Create a valid shard
        data = {"text": ["sample"] * 10}
        table = pa.table(data)
        pq.write_table(table, tmp_path / "train-00000-of-00010.parquet")
        
        # Create a corrupted shard
        corrupted_path = tmp_path / "train-00001-of-00010.parquet"
        with open(corrupted_path, "w") as f:
            f.write("not a parquet file")
        
        reader = CosmopediaParquetReader(dataset_path=str(tmp_path))
        
        # Should still return count for valid shards
        assert reader.total_rows == 10
    
    def test_init_with_file_instead_of_directory(self, tmp_path):
        """Should raise NotADirectoryError when path is a file."""
        file_path = tmp_path / "test.parquet"
        file_path.touch()
        
        with pytest.raises(NotADirectoryError):
            CosmopediaParquetReader(dataset_path=str(file_path))
    
    def test_init_with_no_shards(self, tmp_path):
        """Should raise ValueError when no shards are found."""
        with pytest.raises(ValueError, match="No Parquet shards found"):
            CosmopediaParquetReader(dataset_path=str(tmp_path))
    
    def test_init_discovers_and_sorts_shards(self, tmp_path):
        """Should discover shards and sort them by shard number."""
        # Create shards in non-sequential order
        for i in [5, 2, 8, 1]:
            filename = f"train-{i:05d}-of-10.parquet"
            path = tmp_path / filename
            data = {"text": ["sample"] * 10}
            pq.write_table(pa.table(data), path)
        
        reader = CosmopediaParquetReader(dataset_path=str(tmp_path))
        
        # Check that shards are sorted by number
        shard_numbers = []
        for p in reader.shard_paths:
            match = __import__('re').search(r"train-(\d+)-of-", p.name)
            if match:
                shard_numbers.append(int(match.group(1)))
        
        assert shard_numbers == [1, 2, 5, 8]
    
    def test_init_reads_schema(self, tmp_path):
        """Should read schema from first shard."""
        data = {
            "text": ["sample"] * 10,
            "prompt": ["prompt"] * 10,
            "token_length": [100] * 10,
        }
        table = pa.table(data)
        pq.write_table(table, tmp_path / "train-00000-of-00010.parquet")
        
        reader = CosmopediaParquetReader(dataset_path=str(tmp_path))
        
        assert "text" in reader.available_columns
        assert "prompt" in reader.available_columns
        assert "token_length" in reader.available_columns
    
    def test_init_with_missing_columns(self, tmp_path):
        """Should raise ValueError for requested missing columns."""
        data = {"text": ["sample"] * 10}
        table = pa.table(data)
        pq.write_table(table, tmp_path / "train-00000-of-00010.parquet")
        
        with pytest.raises(ValueError, match="Requested columns not found"):
            CosmopediaParquetReader(
                dataset_path=str(tmp_path),
                columns=["text", "missing_column"]
            )
    
    def test_iter_records_returns_all_records(self, tmp_path):
        """Should iterate over all records from all shards."""
        create_multishard_dataset(tmp_path, num_shards=3, rows_per_shard=5)
        
        reader = CosmopediaParquetReader(dataset_path=str(tmp_path))
        
        records = list(reader.iter_records())
        
        assert len(records) == 15  # 3 shards * 5 rows
    
    def test_iter_records_returns_correct_data(self, tmp_path):
        """Should return correct record data."""
        create_multishard_dataset(tmp_path, num_shards=2, rows_per_shard=3)
        
        reader = CosmopediaParquetReader(dataset_path=str(tmp_path))
        
        records = list(reader.iter_records())
        
        # Check first shard
        assert records[0]["text"] == "Sample text 0-0"
        assert records[1]["text"] == "Sample text 0-1"
        assert records[3]["text"] == "Sample text 1-0"
    
    def test_iter_records_with_column_selection(self, tmp_path):
        """Should only return selected columns."""
        data = {
            "text": ["sample"] * 5,
            "prompt": ["prompt"] * 5,
            "token_length": [100] * 5,
        }
        table = pa.table(data)
        pq.write_table(table, tmp_path / "train-00000-of-00010.parquet")
        
        reader = CosmopediaParquetReader(
            dataset_path=str(tmp_path),
            columns=["text", "token_length"]
        )
        
        record = next(reader.iter_records())
        
        assert "text" in record
        assert "token_length" in record
        assert "prompt" not in record
    
    def test_total_rows(self, tmp_path):
        """Should correctly count total rows across shards."""
        create_multishard_dataset(tmp_path, num_shards=3, rows_per_shard=10)
        
        reader = CosmopediaParquetReader(dataset_path=str(tmp_path))
        
        assert reader.total_rows == 30
    
    def test_num_shards(self, tmp_path):
        """Should correctly count number of shards."""
        create_multishard_dataset(tmp_path, num_shards=7, rows_per_shard=5)
        
        reader = CosmopediaParquetReader(dataset_path=str(tmp_path))
        
        assert reader.num_shards == 7
    
    def test_inspect_shard(self, tmp_path):
        """Should return shard metadata."""
        data = {"text": ["sample"] * 100}
        table = pa.table(data)
        pq.write_table(table, tmp_path / "train-00000-of-00010.parquet")
        
        reader = CosmopediaParquetReader(dataset_path=str(tmp_path))
        
        metadata = reader.inspect_shard(0)
        
        assert "path" in metadata
        assert "num_rows" in metadata
        assert metadata["num_rows"] == 100
    
    def test_inspect_shard_out_of_range(self, tmp_path):
        """Should raise IndexError for invalid shard index."""
        data = {"text": ["sample"] * 10}
        table = pa.table(data)
        pq.write_table(table, tmp_path / "train-00000-of-00010.parquet")
        
        reader = CosmopediaParquetReader(dataset_path=str(tmp_path))
        
        with pytest.raises(IndexError):
            reader.inspect_shard(5)
    
    def test_peek_records(self, tmp_path):
        """Should return first n records."""
        data = {"text": [f"record_{i}" for i in range(100)]}
        table = pa.table(data)
        pq.write_table(table, tmp_path / "train-00000-of-00010.parquet")
        
        reader = CosmopediaParquetReader(dataset_path=str(tmp_path))
        
        records = reader.peek_records(n=5)
        
        assert len(records) == 5
        assert records[0]["text"] == "record_0"
        assert records[4]["text"] == "record_4"
    
    def test_repr(self, tmp_path):
        """Should produce informative string representation."""
        data = {"text": ["sample"] * 10}
        table = pa.table(data)
        pq.write_table(table, tmp_path / "train-00000-of-00010.parquet")
        
        reader = CosmopediaParquetReader(
            dataset_path=str(tmp_path),
            columns=["text"],
            batch_size=50
        )
        
        repr_str = repr(reader)
        
        assert "CosmopediaParquetReader" in repr_str
        assert "num_shards=1" in repr_str
        assert "total_rows=10" in repr_str


class TestSyntheticDatasetProcessing:
    """Tests that process data through the reader."""
    
    def test_process_all_text(self, tmp_path):
        """Should be able to read and process all text."""
        # Create dataset with varying text lengths
        def gen_text():
            return "x" * np.random.randint(50, 200)
        
        data = {
            "text": [gen_text() for _ in range(100)],
            "prompt": ["prompt"] * 100,
        }
        table = pa.table(data)
        pq.write_table(table, tmp_path / "train-00000-of-00010.parquet")
        
        reader = CosmopediaParquetReader(
            dataset_path=str(tmp_path),
            columns=["text"]
        )
        
        all_text = list(reader.iter_records())
        
        assert len(all_text) == 100
        for record in all_text:
            assert isinstance(record["text"], str)
            assert len(record["text"]) >= 50
    
    def test_column_exclusion_preserves_selected(self, tmp_path):
        """Should preserve selected columns while excluding others."""
        data = {
            "text": ["text_value"] * 5,
            "prompt": ["prompt_value"] * 5,
            "token_length": [100] * 5,
        }
        table = pa.table(data)
        pq.write_table(table, tmp_path / "train-00000-of-00010.parquet")
        
        reader = CosmopediaParquetReader(
            dataset_path=str(tmp_path),
            columns=["text", "token_length"]
        )
        
        for record in reader.iter_records():
            assert "text" in record
            assert "token_length" in record
            assert record["text"] == "text_value"
            assert record["token_length"] == 100
