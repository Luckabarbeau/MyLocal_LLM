"""Checkpoint utilities for saving and loading model state."""

import json
import os
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


def save_checkpoint(
    path: Union[str, Path],
    model_params: Dict[str, np.ndarray],
    optimizer_state: Optional[Dict] = None,
    training_state: Optional[Dict] = None,
):
    """
    Save model, optimizer, and training state to disk.
    
    Handles both NumPy and CuPy arrays correctly by converting to NumPy
    before saving. This ensures checkpoint compatibility across backends.
    
    Args:
        path: Directory to save checkpoint
        model_params: Dictionary mapping parameter names to numpy arrays
        optimizer_state: Optional optimizer state dictionary
        training_state: Optional training state (step, loss history, etc.)
    """
    path = Path(path)
    path.mkdir(parents=True, exist_ok=True)
    
    # Save model parameters (convert CuPy to NumPy if needed)
    params_path = path / "model_params"
    params_path.mkdir(exist_ok=True)
    
    for name, data in model_params.items():
        data_np = _convert_to_numpy(data)
        np.save(params_path / f"{name}.npy", data_np)
    
    # Save optimizer state if provided (convert CuPy to NumPy if needed)
    if optimizer_state is not None:
        # Save optimizer state arrays directly as npy files instead of JSON lists
        # This avoids corruption and memory issues with large arrays
        opt_state_dir = path / "optimizer_state"
        opt_state_dir.mkdir(exist_ok=True)
        
        for key, value in optimizer_state.items():
            if isinstance(value, dict):
                key_dir = opt_state_dir / key
                key_dir.mkdir(exist_ok=True)
                for k, v in value.items():
                    if hasattr(v, '__array__'):
                        arr_np = _convert_to_numpy(v)
                        np.save(key_dir / f"{k}.npy", arr_np)
                    else:
                        # Save scalars/other values as JSON
                        with open(key_dir / "scalars.json", "w") as f:
                            json.dump({k: v}, f)
            elif hasattr(value, '__array__'):
                arr_np = _convert_to_numpy(value)
                np.save(opt_state_dir / f"{key}.npy", arr_np)
            else:
                with open(opt_state_dir / "scalars.json", "w") as f:
                    json.dump({key: value}, f)
    
    # Save training state if provided
    if training_state is not None:
        with open(path / "training_state.json", "w") as f:
            json.dump(training_state, f, indent=2)
    
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
    path = Path(path)
    
    # Load model parameters
    params_path = path / "model_params"
    model_params = {}
    
    for npy_file in params_path.glob("*.npy"):
        name = npy_file.stem
        if param_names is None or name in param_names:
            arr = np.load(npy_file)
            # NumPy serializes ml_dtypes.bfloat16 as a 2-byte void dtype
            # (|V2). Recover the BF16 interpretation before transferring to
            # CuPy; otherwise CuPy sees UnstructuredVoid<2> and cannot use the
            # values in arithmetic or assignments. Model parameters are never
            # intentionally stored as raw void arrays, so |V2 is unambiguous
            # here.
            if arr.dtype.kind == "V" and arr.dtype.itemsize == 2:
                try:
                    import ml_dtypes
                except ImportError as exc:
                    raise RuntimeError(
                        "Checkpoint contains BF16 parameters but ml-dtypes is "
                        "not installed. Install it with: python -m pip install ml-dtypes"
                    ) from exc
                arr = arr.view(ml_dtypes.bfloat16)
            # Convert to CuPy array if using CuPy backend
            if BACKEND_NAME == "cupy":
                import cupy
                arr = cupy.asarray(arr)
            model_params[name] = arr
    
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
    
    # Load training state
    training_state = None
    if not skip_training:
        training_path = path / "training_state.json"
        if training_path.exists():
            with open(training_path, "r") as f:
                training_state = json.load(f)
    
    print(f"Checkpoint loaded from {path}")
    return model_params, optimizer_state, training_state


def count_parameters(model_params: Dict[str, np.ndarray]) -> int:
    """Count total trainable parameters."""
    total = 0
    for name, data in model_params.items():
        total += data.size
    return total


def get_param_shapes(model_params: Dict[str, np.ndarray]) -> Dict[str, tuple]:
    """Get shapes of all parameters."""
    return {name: data.shape for name, data in model_params.items()}
