"""Safe, shape-first loading of native HF Llama Safetensors checkpoints.

No PyTorch/Transformers dependency. Sharded and single-file checkpoints share
an indexed *lazy* mapping: inspecting 300 tensor shapes does not materialize
300 arrays. Each required tensor is read once when attached to the model.
"""

import hashlib
import json
from collections.abc import Mapping
from pathlib import Path

from .config import ExternalLlamaConfig
from .llama_model import ExternalLlamaModel


def _safe_filename(directory, filename):
    relative = Path(filename)
    if relative.is_absolute() or any(part in {"..", "."} for part in relative.parts):
        raise ValueError(f"Unsafe Safetensors shard path: {filename!r}")
    path = (directory / relative).resolve()
    if not path.is_relative_to(directory.resolve()):
        raise ValueError(f"Safetensors shard escapes model directory: {filename!r}")
    if not path.is_file():
        raise FileNotFoundError(f"Safetensors shard missing: {path}")
    return path


class SafetensorsWeightMap(Mapping):
    """A filename/tensor directory with metadata-only shape inspection.

    Index contents are validated against actual file keys; a malformed index
    cannot silently rename, omit or duplicate tensors. No file data is read
    until `mapping[tensor_name]` is requested.
    """

    def __init__(self, directory):
        try:
            from safetensors import safe_open
        except ImportError as exc:
            raise RuntimeError("Install optional dependency: pip install safetensors") from exc
        directory = Path(directory).expanduser().resolve()
        if not directory.is_dir():
            raise FileNotFoundError(f"External checkpoint folder not found: {directory}")
        index = directory / "model.safetensors.index.json"
        single = directory / "model.safetensors"
        if index.is_file():
            with index.open("r", encoding="utf-8") as stream:
                manifest = json.load(stream)
            if not isinstance(manifest, dict) or not isinstance(manifest.get("weight_map"), dict):
                raise ValueError("Invalid Safetensors index: expected weight_map object")
            locations = manifest["weight_map"]
            if not locations:
                raise ValueError("Safetensors index contains no tensors")
        elif single.is_file():
            with safe_open(str(single), framework="np", device="cpu") as stream:
                locations = {key: "model.safetensors" for key in stream.keys()}
        else:
            raise FileNotFoundError("Expected model.safetensors or model.safetensors.index.json")

        self.directory = directory
        self._files = {}
        for name, shard_name in locations.items():
            if not isinstance(name, str) or not isinstance(shard_name, str):
                raise ValueError("Safetensors weight_map must map tensor names to paths")
            self._files[name] = _safe_filename(directory, shard_name)

        self._shapes = {}
        self._dtypes = {}
        # Validate all indexed tensors and reject undocumented tensors on disk.
        # This also gives strict shape validation without large host allocations.
        for shard in sorted(set(self._files.values())):
            assigned = {name for name, filename in self._files.items() if filename == shard}
            with safe_open(str(shard), framework="np", device="cpu") as stream:
                actual = set(stream.keys())
                missing = assigned - actual
                extra = actual - assigned
                if missing or extra:
                    raise ValueError(
                        f"Safetensors index disagrees with {shard.name}: "
                        f"missing={sorted(missing)[:5]}, extra={sorted(extra)[:5]}"
                    )
                for name in assigned:
                    tensor = stream.get_slice(name)
                    self._shapes[name] = tuple(tensor.get_shape())
                    self._dtypes[name] = tensor.get_dtype()

    def __iter__(self):
        return iter(self._files)

    def __len__(self):
        return len(self._files)

    def __getitem__(self, name):
        if name not in self._files:
            raise KeyError(name)
        from safetensors import safe_open
        with safe_open(str(self._files[name]), framework="np", device="cpu") as stream:
            return stream.get_tensor(name)

    def shape_of(self, name):
        return self._shapes[name]

    def dtype_of(self, name):
        return self._dtypes[name]

    @property
    def weight_files(self):
        return tuple(sorted(set(self._files.values())))


def _sha256(path):
    hasher = hashlib.sha256()
    with open(path, "rb") as stream:
        for chunk in iter(lambda: stream.read(2 * 1024 * 1024), b""):
            hasher.update(chunk)
    return hasher.hexdigest()


def load_safetensors_model(directory, *, dtype="float32", strict=True, hash_weights=False):
    """Instantiate native inference and lazily bind official HF weights.

    Returns (model, provenance). Does not initialize a native train model,
    allocate optimizer/gradients, or use Transformers during inference.
    Weight hashes may be expensive for large models, so full hashing is opt-in;
    the config hash and exact shard names are always retained.
    """
    directory = Path(directory).expanduser().resolve()
    config_file = directory / "config.json"
    config = ExternalLlamaConfig.from_json(config_file)
    reader = SafetensorsWeightMap(directory)
    model = ExternalLlamaModel(config, dtype=dtype)
    model.set_weights(reader, strict=strict)
    provenance = {
        "source_directory": str(directory),
        "architecture": "llama",
        "format": "safetensors",
        "config_sha256": _sha256(config_file),
        "weight_files": [path.name for path in reader.weight_files],
        "original_dtypes": sorted(set(reader._dtypes.values())),
        "runtime_dtype": str(dtype),
        "weight_sha256": ({path.name: _sha256(path) for path in reader.weight_files}
                          if hash_weights else None),
    }
    return model, provenance
