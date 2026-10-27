"""Safely migrate a historical-read checkpoint to independent history heads.

The source is hashed before and after migration, copied into a temporary sibling,
strictly audited, and atomically published only after validation succeeds.
"""

import argparse
import hashlib
import json
import os
import shutil
import uuid
from pathlib import Path
from types import SimpleNamespace

import numpy as np

from mini_llm.optim.hybrid_muon import muon_eligible


DEFAULT_LAYERS = (1, 2, 3, 4, 6, 8, 10, 12)  # one-based CLI notation


def _json(path):
    with path.open() as stream:
        return json.load(stream)


def _write_json(path, value):
    with path.open("w") as stream:
        json.dump(value, stream, indent=2, sort_keys=True)


def _array_names(path):
    return {item.stem for item in path.glob("*.npy")} if path.is_dir() else set()


def _manifest(root):
    """Content manifest used to prove migration did not alter the source."""
    result = {}
    for path in sorted(item for item in root.rglob("*") if item.is_file()):
        digest = hashlib.sha256()
        with path.open("rb") as stream:
            for chunk in iter(lambda: stream.read(8 * 1024 * 1024), b""):
                digest.update(chunk)
        result[str(path.relative_to(root))] = digest.hexdigest()
    return result


def _eligible(name, shape):
    parameter = SimpleNamespace(name=name, data=SimpleNamespace(shape=tuple(shape)))
    return bool(muon_eligible(parameter))


def _expected_additions(config, layers):
    width, head_dim = int(config["d_model"]), int(config["d_head"])
    additions = {}
    for layer in layers:
        index = layer - 1
        additions.update({
            f"blocks.{index}.full_history.Wq": (width, head_dim),
            f"blocks.{index}.full_history.Wk": (width, head_dim),
            f"blocks.{index}.full_history.Wv": (width, head_dim),
            f"blocks.{index}.full_history.Wo": (head_dim, width),
        })
    return additions


def audit_migration(source, destination, *, source_manifest=None):
    """Strict static checkpoint/optimizer audit without allocating a full model."""
    source, destination = Path(source).resolve(), Path(destination).resolve()
    if source_manifest is not None and _manifest(source) != source_manifest:
        raise RuntimeError("source checkpoint changed during migration")
    source_config, config = _json(source / "config.json"), _json(destination / "config.json")
    layers = tuple(int(i) + 1 for i in config.get("full_history_layers", ()))
    additions = _expected_additions(config, layers)
    source_names = _array_names(source / "model_params")
    destination_names = _array_names(destination / "model_params")
    if destination_names != source_names | set(additions):
        raise ValueError("migrated model parameter set is incomplete or contains extras")
    for name in source_names:
        old = np.load(source / "model_params" / f"{name}.npy", mmap_mode="r")
        new = np.load(destination / "model_params" / f"{name}.npy", mmap_mode="r")
        if old.shape != new.shape or old.dtype != new.dtype or not np.array_equal(old, new):
            raise ValueError(f"original model tensor changed: {name}")
    for name, shape in additions.items():
        value = np.load(destination / "model_params" / f"{name}.npy", mmap_mode="r")
        if tuple(value.shape) != tuple(shape):
            raise ValueError(f"new parameter shape mismatch: {name}")
        if name.endswith(".Wo") and np.any(value != 0):
            raise ValueError(f"new output projection is not exactly zero: {name}")

    source_state = _json(source / "training_state.json")
    state = _json(destination / "training_state.json")
    preserved = dict(state)
    preserved.pop("muon_hyperparameters", None)
    original = dict(source_state)
    original.pop("muon_hyperparameters", None)
    original_birth = original.pop("optimizer_parameter_birth_steps", None)
    migrated_birth = preserved.pop("optimizer_parameter_birth_steps", None)
    if preserved != original:
        raise ValueError("migration changed unrelated training-state metadata")
    optimizer_kind = state.get("optimizer_kind")
    optimizer = destination / "optimizer_state"
    expected = destination_names
    masters = _array_names(optimizer / "master_weights")
    if masters != expected:
        raise ValueError("optimizer master-weight set does not match model parameters")
    if optimizer_kind == "hybrid-muon":
        settings = state.get("muon_hyperparameters") or {}
        muon = set(settings.get("group_names", ()))
        if _array_names(optimizer / "muon_m") != muon:
            raise ValueError("Muon momentum tensors do not match saved group_names")
        adam = expected - muon
        if _array_names(optimizer / "m") != adam or _array_names(optimizer / "v") != adam:
            raise ValueError("Hybrid-Muon AdamW tensor sets are incomplete")
        for name in additions:
            shape = additions[name]
            if _eligible(name, shape) != (name in muon):
                raise ValueError(f"new optimizer group is inconsistent with muon_eligible: {name}")
        if migrated_birth != original_birth:
            raise ValueError("Hybrid-Muon migration changed Adam birth-step metadata")
    elif optimizer_kind == "adamw":
        if _array_names(optimizer / "m") != expected or _array_names(optimizer / "v") != expected:
            raise ValueError("AdamW tensor sets are incomplete")
        birth = migrated_birth or {}
        step = int(state.get("step", 0))
        for name in additions:
            if int(birth.get(name, -1)) != step:
                raise ValueError(f"new AdamW parameter has wrong birth step: {name}")
        for name, value in (original_birth or {}).items():
            if birth.get(name) != value:
                raise ValueError(f"existing AdamW birth step changed: {name}")
    else:
        raise ValueError(f"unsupported or missing optimizer_kind: {optimizer_kind!r}")
    if source_config.get("full_history_layers"):
        raise ValueError("audit source unexpectedly already has full-history heads")
    return {
        "source_files": len(source_manifest or _manifest(source)),
        "original_parameters": len(source_names),
        "added_tensors": len(additions),
        "added_parameters": sum(int(np.prod(shape)) for shape in additions.values()),
        "optimizer_kind": optimizer_kind,
        "training_step": int(state.get("step", 0)),
    }


def migrate(source, destination, *, layers=DEFAULT_LAYERS):
    source, destination = Path(source).resolve(), Path(destination).resolve()
    if source == destination or source in destination.parents or destination in source.parents:
        raise ValueError("source and destination must be distinct, nonnested directories")
    required = ("model_params", "optimizer_state", "config.json", "training_state.json")
    if not source.is_dir() or any(not (source / item).exists() for item in required):
        raise FileNotFoundError("source must contain model, optimizer, config, and training state")
    if destination.exists():
        raise FileExistsError("destination must not exist; migration will not overwrite files")
    config = _json(source / "config.json")
    training_state = _json(source / "training_state.json")
    memory = config.get("memory_context", {})
    if not (memory.get("enabled", False) and memory.get("integration_mode") == "historical_read"):
        raise ValueError("full-history upgrade requires the historical_read ensemble family")
    if config.get("full_history_layers"):
        raise ValueError("source checkpoint already includes full-history heads")
    depth, width = int(config["n_layers"]), int(config["d_model"])
    if (not layers or len(layers) != len(set(layers))
            or any(i < 1 or i > depth for i in layers)):
        raise ValueError("layer positions must be unique and within checkpoint depth")
    if config.get("ffn_mode") != "dense":
        raise ValueError("this migration is intended for dense ensemble foundations")
    additions = _expected_additions(config, layers)
    original_names = _array_names(source / "model_params")
    for layer in layers:
        original = source / "model_params" / f"blocks.{layer-1}.attention.Wq.npy"
        if not original.exists() or tuple(np.load(original, mmap_mode="r").shape) != (width, width):
            raise ValueError(f"original attention projection is missing or incompatible at layer {layer}")
    reference = np.load(source / "model_params" / "embedding.W.npy", mmap_mode="r")
    storage_dtype = reference.dtype
    if storage_dtype.kind == "V" and storage_dtype.itemsize == 2:
        import ml_dtypes
        storage_dtype = np.dtype(ml_dtypes.bfloat16)
    rng = np.random.default_rng(1000100)
    staged_arrays = {}
    for name, shape in additions.items():
        if name in original_names:
            raise ValueError(f"new parameter already present: {name}")
        staged_arrays[name] = (np.zeros(shape, dtype=storage_dtype) if name.endswith(".Wo")
                               else (rng.standard_normal(shape, dtype=np.float32)
                                     * float(config.get("init_std", 0.02))).astype(storage_dtype))

    source_manifest = _manifest(source)
    temporary = destination.parent / f".{destination.name}.tmp-{uuid.uuid4().hex}"
    try:
        shutil.copytree(source, temporary, symlinks=False)
        for name, data in staged_arrays.items():
            np.save(temporary / "model_params" / f"{name}.npy", data)
        optimizer = temporary / "optimizer_state"
        master_dir = optimizer / "master_weights"
        if not master_dir.is_dir():
            raise ValueError("migration requires complete FP32 master weights")
        for name, data in staged_arrays.items():
            np.save(master_dir / f"{name}.npy", data.astype(np.float32))

        optimizer_kind = training_state.get("optimizer_kind")
        if optimizer_kind == "hybrid-muon":
            settings = training_state.get("muon_hyperparameters") or {}
            if not settings.get("group_names"):
                raise ValueError("Hybrid-Muon checkpoint lacks saved group_names")
            muon_names = set(settings["group_names"])
            for name, data in staged_arrays.items():
                target = "muon_m" if _eligible(name, data.shape) else "m"
                (optimizer / target).mkdir(exist_ok=True)
                np.save(optimizer / target / f"{name}.npy", np.zeros(data.shape, np.float32))
                if target == "m":
                    (optimizer / "v").mkdir(exist_ok=True)
                    np.save(optimizer / "v" / f"{name}.npy", np.zeros(data.shape, np.float32))
                else:
                    muon_names.add(name)
            settings["group_names"] = sorted(muon_names)
            training_state["muon_hyperparameters"] = settings
        elif optimizer_kind == "adamw":
            for component in ("m", "v"):
                if not (optimizer / component).is_dir():
                    raise ValueError(f"AdamW checkpoint lacks optimizer_state/{component}")
                for name, data in staged_arrays.items():
                    np.save(optimizer / component / f"{name}.npy",
                            np.zeros(data.shape, np.float32))
            birth = dict(training_state.get("optimizer_parameter_birth_steps") or {})
            for name in staged_arrays:
                birth[name] = int(training_state.get("step", 0))
            training_state["optimizer_parameter_birth_steps"] = birth
        else:
            raise ValueError(f"unsupported or missing optimizer_kind: {optimizer_kind!r}")

        config["full_history_layers"] = [i - 1 for i in layers]
        _write_json(temporary / "config.json", config)
        _write_json(temporary / "training_state.json", training_state)
        report = audit_migration(source, temporary, source_manifest=source_manifest)
        marker = {
            "format": "0100B",
            "source_checkpoint": str(source),
            "source_manifest_sha256": hashlib.sha256(
                json.dumps(source_manifest, sort_keys=True).encode("utf8")
            ).hexdigest(),
            "layers_one_based": list(layers),
            "layers_zero_based": [i - 1 for i in layers],
            "head_width": int(config["d_head"]),
            "initialization": "zero additive output, independent random Q/K/V",
            **report,
        }
        _write_json(temporary / "full_history_migration.json", marker)
        if _manifest(source) != source_manifest:
            raise RuntimeError("source checkpoint changed before publication")
        os.replace(temporary, destination)
    except BaseException:
        if temporary.exists():
            shutil.rmtree(temporary)
        raise
    print(f"Migrated {source} -> {destination}")
    print(f"Added {report['added_tensors']} arrays / {report['added_parameters']:,} parameters")
    print(f"Audit passed: optimizer={report['optimizer_kind']} step={report['training_step']}")
    return marker


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", required=True)
    parser.add_argument("--destination", required=True)
    parser.add_argument("--layers", nargs="+", type=int, default=list(DEFAULT_LAYERS),
                        help="one-based historical attention layers")
    parser.add_argument("--audit-only", action="store_true",
                        help="audit an already migrated source/destination pair")
    args = parser.parse_args()
    if args.audit_only:
        print(json.dumps(audit_migration(args.source, args.destination), indent=2))
    else:
        migrate(args.source, args.destination, layers=tuple(args.layers))


if __name__ == "__main__":
    main()
