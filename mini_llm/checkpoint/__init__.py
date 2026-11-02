"""Checkpoint utilities for saving and loading model state."""

import json
import os
import shutil
import uuid
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union

import numpy as np

from mini_llm.backend import xp, BACKEND_NAME, asnumpy


def _convert_to_numpy(data):
    """Convert backend or host-offloaded arrays to NumPy without assumptions."""
    if BACKEND_NAME == "cupy" and hasattr(data, "get"):
        return data.get()
    return np.asarray(data)


def _optimizer_offload_mode():
    """Return the active optimizer offload policy for checkpoint loading."""
    if BACKEND_NAME != "cupy":
        return "none"
    raw = os.environ.get("MINI_LLM_OPTIMIZER_OFFLOAD", "none").strip().lower()
    return raw if raw in {"moments", "full"} else "none"


def _write_payload(path, model_params, optimizer_state, training_state):
    """Write an isolated checkpoint payload; never mutate the active snapshot."""
    params_path = path / "model_params"
    params_path.mkdir()
    for name, data in model_params.items():
        np.save(params_path / f"{name}.npy", _convert_to_numpy(data))

    if optimizer_state is not None:
        opt_path = path / "optimizer_state"
        opt_path.mkdir()
        scalars = {}
        for name, value in optimizer_state.items():
            if isinstance(value, dict):
                nested = opt_path / name
                nested.mkdir()
                nested_scalars = {}
                for key, item in value.items():
                    if hasattr(item, "__array__"):
                        np.save(nested / f"{key}.npy", _convert_to_numpy(item))
                    else:
                        nested_scalars[key] = item
                if nested_scalars:
                    (nested / "scalars.json").write_text(
                        json.dumps(nested_scalars), encoding="utf-8")
            elif hasattr(value, "__array__"):
                np.save(opt_path / f"{name}.npy", _convert_to_numpy(value))
            else:
                scalars[name] = value
        if scalars:
            (opt_path / "scalars.json").write_text(
                json.dumps(scalars), encoding="utf-8")

    if training_state is not None:
        (path / "training_state.json").write_text(
            json.dumps(training_state, indent=2), encoding="utf-8")


def _checkpoint_previous(path):
    return path.with_name(path.name + ".previous")


def _resolve_checkpoint_path(path):
    """Recover reads after interruption between old and new directory renames."""
    path = Path(path)
    if path.is_dir():
        return path
    backup = _checkpoint_previous(path)
    if not path.exists() and backup.is_dir():
        return backup
    return path


def save_checkpoint(
    path: Union[str, Path],
    model_params: Dict[str, np.ndarray],
    optimizer_state: Optional[Dict] = None,
    training_state: Optional[Dict] = None,
):
    """Publish a complete checkpoint using a staged sibling directory.

    Legacy paths/layout stay unchanged. The previous complete checkpoint is
    retained at ``<path>.previous`` for interruption recovery. The active
    directory is *never* modified while parameter and optimizer files are
    being written. A crash during the final pair of directory renames can
    leave the active path absent; load_checkpoint then reads .previous.

    This operation requires the staging and destination to share a filesystem.
    Consumers must not write ancillary files into the checkpoint directory
    concurrently with checkpoint publication (the normal training writer is
    serialized). A transactional checkpoint requires additional disk space.
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists() and not path.is_dir():
        raise NotADirectoryError(str(path))
    previous = _checkpoint_previous(path)
    if not path.exists() and previous.is_dir():
        # Recover a committed predecessor before attempting another save.
        os.replace(previous, path)
    stage = path.with_name(path.name + ".pending-" + uuid.uuid4().hex)
    stage.mkdir()
    activated = False
    moved_old = False
    try:
        # The training launcher stores config/tokenizer and supporting files
        # alongside the payload. Preserve them while replacing managed paths.
        if path.is_dir():
            for entry in path.iterdir():
                if entry.name in {"model_params", "optimizer_state", "training_state.json"}:
                    continue
                destination = stage / entry.name
                if entry.is_symlink():
                    destination.symlink_to(os.readlink(entry))
                elif entry.is_dir():
                    shutil.copytree(entry, destination, copy_function=os.link,
                                    symlinks=True)
                else:
                    os.link(entry, destination)
        _write_payload(stage, model_params, optimizer_state, training_state)
        # Force payload visibility before switching checkpoint directory names.
        # fsync regular files is intentionally not done here for the large
        # optimizer tensors: atomic replacement, not power-loss durability,
        # is the specific contract of this patch.
        if previous.exists():
            shutil.rmtree(previous)
        if path.exists():
            os.replace(path, previous)
            moved_old = True
        os.replace(stage, path)
        activated = True
    except BaseException:
        if moved_old and not path.exists() and previous.is_dir():
            os.replace(previous, path)
        raise
    finally:
        if not activated and stage.exists():
            shutil.rmtree(stage)
    print(f"Checkpoint saved to {path}")


def load_checkpoint(
    path: Union[str, Path],
    param_names: Optional[List[str]] = None,
    skip_optimizer: bool = False,
    skip_training: bool = False,
) -> Tuple[Dict[str, np.ndarray], Optional[Dict], Optional[Dict]]:
    """
    Load model, optimizer, and training state from disk.
    
    Args:
        path: Directory containing checkpoint
        param_names: Optional list of specific parameter names to load
        skip_optimizer: If True, skip loading optimizer state (useful for inference)
        skip_training: If True, skip loading training state
        
    Returns:
        Tuple of (model_params, optimizer_state, training_state)
    """
    path = _resolve_checkpoint_path(path)
    
    # Load model parameters
    params_path = path / "model_params"
    model_params = {}
    
    for npy_file in params_path.glob("*.npy"):
        name = npy_file.stem
        if param_names is None or name in param_names:
            model_params[name] = _load_parameter_array(npy_file)
    
    # Load optimizer state (convert NumPy back to appropriate type if needed)
    optimizer_state = None
    if not skip_optimizer:
        optimizer_dir = path / "optimizer_state"
        if optimizer_dir.exists():
            optimizer_state = {}
            
            # Load step scalar from scalars.json
            scalars_path = optimizer_dir / "scalars.json"
            if scalars_path.exists():
                with open(scalars_path, "r") as f:
                    optimizer_state.update(json.load(f))
            
            # Load master weights (if they exist)
            master_weights_path = optimizer_dir / "master_weights"
            keep_optimizer_host = _optimizer_offload_mode() != "none"
            if master_weights_path.exists() and master_weights_path.is_dir():
                optimizer_state["master_weights"] = {}
                for npy_file in master_weights_path.glob("*.npy"):
                    arr = np.load(npy_file)
                    if BACKEND_NAME == "cupy" and not keep_optimizer_host:
                        import cupy
                        arr = cupy.asarray(arr)
                    optimizer_state["master_weights"][npy_file.stem] = arr
            
            # Load moment arrays
            m_path = optimizer_dir / "m"
            v_path = optimizer_dir / "v"
            
            keep_moments_host = _optimizer_offload_mode() in {"moments", "full"}
            if m_path.exists() and m_path.is_dir():
                optimizer_state["m"] = {}
                for npy_file in m_path.glob("*.npy"):
                    arr = np.load(npy_file)
                    if BACKEND_NAME == "cupy" and not keep_moments_host:
                        import cupy
                        arr = cupy.asarray(arr)
                    optimizer_state["m"][npy_file.stem] = arr
            
            if v_path.exists() and v_path.is_dir():
                optimizer_state["v"] = {}
                for npy_file in v_path.glob("*.npy"):
                    arr = np.load(npy_file)
                    if BACKEND_NAME == "cupy" and not keep_moments_host:
                        import cupy
                        arr = cupy.asarray(arr)
                    optimizer_state["v"][npy_file.stem] = arr
            # Preserve future optimizer extensions and nested scalar state,
            # without changing the established host-offload rules for m/v.
            for nested in optimizer_dir.iterdir():
                if not nested.is_dir() or nested.name in {"m", "v", "muon_m", "master_weights"}:
                    continue
                contents = {}
                for array_file in nested.glob("*.npy"):
                    arr = np.load(array_file)
                    contents[array_file.stem] = (xp.asarray(arr) if BACKEND_NAME == "cupy" else arr)
                if (nested / "scalars.json").is_file():
                    contents.update(json.loads((nested / "scalars.json").read_text(encoding="utf-8")))
                optimizer_state[nested.name] = contents
            for scalar_dir in ("m", "v", "muon_m", "master_weights"):
                scalar_path = optimizer_dir / scalar_dir / "scalars.json"
                if scalar_path.is_file():
                    optimizer_state.setdefault(scalar_dir, {}).update(
                        json.loads(scalar_path.read_text(encoding="utf-8")))
            for array_file in optimizer_dir.glob("*.npy"):
                arr = np.load(array_file)
                optimizer_state[array_file.stem] = (xp.asarray(arr) if BACKEND_NAME == "cupy" else arr)
    
    # 0076A: Muon has only one momentum tensor; never synthesize AdamW v.
    if optimizer_state is not None:
        muon_path = optimizer_dir / "muon_m"
        if muon_path.is_dir():
            optimizer_state["muon_m"] = {}
            for npy_file in muon_path.glob("*.npy"):
                arr = np.load(npy_file)
                if BACKEND_NAME == "cupy" and not keep_moments_host:
                    import cupy
                    arr = cupy.asarray(arr)
                optimizer_state["muon_m"][npy_file.stem] = arr

    # Load training state
    training_state = None
    if not skip_training:
        training_path = path / "training_state.json"
        if training_path.exists():
            with open(training_path, "r") as f:
                training_state = json.load(f)
    
    print(f"Checkpoint loaded from {path}")
    return model_params, optimizer_state, training_state



def _load_parameter_array(npy_file: Path):
    """Load one serialized parameter and restore BF16/backend representation."""
    arr = np.load(npy_file)
    # NumPy serializes ml_dtypes.bfloat16 as a two-byte void dtype.
    if arr.dtype.kind == "V" and arr.dtype.itemsize == 2:
        try:
            import ml_dtypes
        except ImportError as exc:
            raise RuntimeError(
                "Checkpoint contains BF16 parameters but ml-dtypes is not installed. "
                "Install it with: python -m pip install ml-dtypes"
            ) from exc
        arr = arr.view(ml_dtypes.bfloat16)
    if BACKEND_NAME == "cupy":
        import cupy
        arr = cupy.asarray(arr)
    return arr


def initialize_matching_parameters(path: Union[str, Path], parameters):
    """Stream matching checkpoint tensors directly into existing Parameters.

    Unlike :func:`load_checkpoint`, this function never materializes the whole
    checkpoint in a dictionary.  It is intended for 0058C ``--init-from`` where
    a ~500M 4k backbone initializes a slightly larger memory model and keeping a
    second full set of device arrays would be wasteful.

    Returns ``(loaded_names, missing_names, loaded_elements)``. Shape mismatches
    fail immediately because a same-name/different-shape tensor cannot be a safe
    weights-only initialization.
    """
    params_path = Path(path) / "model_params"
    if not params_path.is_dir():
        raise FileNotFoundError(f"checkpoint parameter directory not found: {params_path}")

    loaded = []
    missing = []
    loaded_elements = 0
    for parameter in parameters:
        npy_file = params_path / f"{parameter.name}.npy"
        if not npy_file.exists():
            missing.append(parameter.name)
            continue
        arr = _load_parameter_array(npy_file)
        if tuple(arr.shape) != tuple(parameter.data.shape):
            raise ValueError(
                f"checkpoint shape mismatch for {parameter.name}: "
                f"checkpoint={arr.shape}, model={parameter.data.shape}"
            )
        parameter.data[...] = arr
        loaded.append(parameter.name)
        loaded_elements += int(parameter.data.size)
        del arr
    return loaded, missing, loaded_elements

def count_parameters(model_params: Dict[str, np.ndarray]) -> int:
    """Count total trainable parameters."""
    total = 0
    for name, data in model_params.items():
        total += data.size
    return total


def get_param_shapes(model_params: Dict[str, np.ndarray]) -> Dict[str, tuple]:
    """Get shapes of all parameters."""
    return {name: data.shape for name, data in model_params.items()}
