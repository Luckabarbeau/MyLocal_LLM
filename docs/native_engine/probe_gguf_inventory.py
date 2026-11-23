"""Header-only native/Python checkpoint parity; run as a module from repo root."""
import argparse
from collections import Counter, defaultdict
import hashlib
import importlib.metadata
import json
from pathlib import Path
import subprocess
import time

from mini_llm.native import Runtime
from mini_llm.external.gguf_store import MultiShardTensorStore
import gguf.constants


def sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def identity(path):
    st = path.stat()
    return dict(path=str(path), device=st.st_dev, inode=st.st_ino, size=st.st_size,
                mtime_ns=st.st_mtime_ns, ctime_ns=st.st_ctime_ns)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("checkpoint", type=Path)
    parser.add_argument("output", type=Path)
    args = parser.parse_args()
    started = time.monotonic()
    runtime = Runtime()
    fields = ("name", "shard_index", "shape", "ggml_shape", "relative_offset",
              "data_offset", "nbytes", "row_bytes", "byte_shape", "byte_strides")
    report = {"scope": "header-only, no tensor decode or GPU uploads",
              "command_checkpoint": str(args.checkpoint.resolve()),
              "gguf_version": importlib.metadata.version("gguf"),
              "constants_path": gguf.constants.__file__,
              "constants_sha256": sha256(gguf.constants.__file__),
              "git_head": subprocess.check_output(["git", "--no-pager", "rev-parse", "HEAD"], text=True).strip()}
    source_paths = ["mini_llm/native_cpp/gguf_loader.cpp", "mini_llm/native_cpp/gguf_loader.h",
                    "mini_llm/native_cpp/runtime.cpp", "mini_llm/native_cpp/runtime.h",
                    "mini_llm/native_cpp/CMakeLists.txt", "mini_llm/native/__init__.py",
                    "mini_llm/external/gguf_store.py", "mini_llm/ops/quantized_weight.py",
                    "tests/test_native_engine_gguf.py", __file__]
    report["source_sha256"] = {str(p): sha256(p) for p in source_paths}
    report["library"] = {"path": runtime._lib._name, "sha256": sha256(runtime._lib._name)}
    # Discover/stat before native open to detect changes during the whole probe.
    from mini_llm.external.gguf_store import _discover
    paths, _ = _discover(args.checkpoint)
    before = [identity(p) for p in paths]
    with runtime.gguf(args.checkpoint) as native, MultiShardTensorStore.open(args.checkpoint) as python:
        assert native.paths == python.paths == paths
        assert native.tensors.keys() == python.tensors.keys()
        metadata_counts = []
        for actual, expected in zip(native.shard_metadata, python.shard_metadata, strict=True):
            # Python adds synthetic GGUF.* fields not stored in the file.
            expected = {k: v for k, v in expected.items() if not k.startswith("GGUF.")}
            assert actual == expected, "per-shard metadata mismatch"
            metadata_counts.append(len(actual))
        by_type = defaultdict(lambda: {"count": 0, "nbytes": 0})
        shards = []
        for index, path in enumerate(paths):
            ds = [d for d in native.tensors.values() if d.shard_index == index]
            assert ds, "probe requires at least one tensor per shard"
            base = ds[0].data_offset - ds[0].relative_offset
            assert base <= 64 * 1024 * 1024, "header hash exceeds 64 MiB bound"
            with path.open("rb") as stream:
                header = stream.read(base)
            assert len(header) == base
            shards.append({**before[index], "tensor_count": len(ds),
                           "metadata_count": metadata_counts[index], "data_offset": base,
                           "tensor_bytes": sum(d.nbytes for d in ds),
                           "header_sha256": hashlib.sha256(header).hexdigest()})
        for name, actual in native.tensors.items():
            expected = python.tensors[name]
            for field in fields:
                assert getattr(actual, field) == getattr(expected, field), (name, field)
            assert (actual.type_id, actual.block_elements, actual.block_bytes) == (
                expected.format.type_id, expected.format.block_elements, expected.format.block_bytes)
            entry = by_type[expected.format.name]
            entry.update(type_id=actual.type_id, block_elements=actual.block_elements,
                         block_bytes=actual.block_bytes)
            entry["count"] += 1
            entry["nbytes"] += actual.nbytes
        report.update(tensor_count=len(native.tensors),
                      tensor_bytes=sum(d.nbytes for d in native.tensors.values()),
                      total_file_bytes=sum(p.stat().st_size for p in paths),
                      shards=shards, formats=dict(sorted(by_type.items())),
                      descriptor_fields_compared=list(fields) + ["type_id", "block_elements", "block_bytes"])
    assert before == [identity(p) for p in paths], "checkpoint changed during probe"
    report.update(status="passed", elapsed_seconds=time.monotonic() - started,
                  full_payload_hashes="not attempted: header-only IO scope; 83+ GB checkpoint",
                  checkpoint_revision="unknown; stat and header hashes are not payload identity")
    args.output.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
