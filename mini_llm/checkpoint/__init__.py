"""Checkpoint utilities for saving and loading model state."""

import json
import os
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union

import numpy as np

from mini_llm.backend import xp, BACKEND_NAME, asnumpy


def _convert_to_numpy(data):
    """Convert data to NumPy array, handling CuPy arrays."""
    if BACKEND_NAME == "cupy":
        return asnumpy(data)
    return np.asarray(data)


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
        # Convert any CuPy/NumPy arrays in optimizer state to Python lists/floats
        opt_state_np = {}
        for key, value in optimizer_state.items():
            if isinstance(value, dict):
                opt_state_np[key] = {}
                for k, v in value.items():
                    if hasattr(v, '__array__'):
                        # Convert numpy/cupy array to list for JSON
                        opt_state_np[key][k] = _convert_to_numpy(v).tolist()
                    else:
                        opt_state_np[key][k] = v
            elif hasattr(value, '__array__'):
                opt_state_np[key] = _convert_to_numpy(value).tolist()
            else:
                opt_state_np[key] = value
        with open(path / "optimizer_state.json", "w") as f:
            json.dump(opt_state_np, f, indent=2)
    
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
            # Convert to CuPy array if using CuPy backend
            if BACKEND_NAME == "cupy":
                import cupy
                arr = cupy.asarray(arr)
            model_params[name] = arr
    
    # Load optimizer state (convert NumPy back to appropriate type if needed)
    optimizer_state = None
    if not skip_optimizer:
        optimizer_path = path / "optimizer_state.json"
        if optimizer_path.exists():
            with open(optimizer_path, "r") as f:
                optimizer_state = json.load(f)
    
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
