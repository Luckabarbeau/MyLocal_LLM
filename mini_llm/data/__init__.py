"""Data pipeline modules.

Imports are intentionally lazy.  Some corpus formats do not require PyArrow,
and importing :mod:`mini_llm.data` should not force the Parquet dependency to
load before a caller actually requests a Parquet-backed component.
"""

__all__ = [
    "CosmopediaParquetReader",
    "PackedTokenDataset",
    "PackedDatasetGenerator",
    "DatasetManifest",
    "load_packed_dataset",
    "verify_manifest",
    "TokenShardGenerator",
    "TokenShardWriter",
    "generate_token_shards",
    "load_token_shard",
    "create_minibatch",
]


def __getattr__(name):
    if name == "CosmopediaParquetReader":
        from mini_llm.data.parquet_reader import CosmopediaParquetReader

        return CosmopediaParquetReader
    if name in {
        "PackedTokenDataset",
        "PackedDatasetGenerator",
        "DatasetManifest",
        "load_packed_dataset",
        "verify_manifest",
    }:
        from importlib import import_module

        return getattr(import_module("mini_llm.data.packed_dataset"), name)
    if name in {
        "TokenShardGenerator",
        "TokenShardWriter",
        "generate_token_shards",
        "load_token_shard",
        "create_minibatch",
    }:
        from importlib import import_module

        return getattr(import_module("mini_llm.data.token_shards"), name)
    raise AttributeError(name)
