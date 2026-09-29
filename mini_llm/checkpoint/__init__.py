"""Checkpoint utilities for saving and loading model state."""

import json
import os
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union

import numpy as np


def save_checkpoint(
    path: Union[str, Path],
    model_params: Dict[str, np.ndarray],
    optimizer_state: Optional[Dict] = None,
    training_state: Optional[Dict] = None,
):
    """
    Save model, optimizer, and training state to disk.
    
    Args:
        path: Directory to save checkpoint
        model_params: Dictionary mapping parameter names to numpy arrays
        optimizer_state: Optional optimizer state dictionary
        training_state: Optional training state (step, loss history, etc.)
    """
    path = Path(path)
    path.mkdir(parents=True, exist_ok=True)
    
    # Save model parameters
    params_path = path / "model_params"
    params_path.mkdir(exist_ok=True)
    
    for name, data in model_params.items():
        np.save(params_path / f"{name}.npy", data)
    
    # Save optimizer state if provided
    if optimizer_state is not None:
        with open(path / "optimizer_state.json", "w") as f:
            json.dump(optimizer_state, f, indent=2)
    
    # Save training state if provided
    if training_state is not None:
        with open(path / "training_state.json", "w") as f:
            json.dump(training_state, f, indent=2)
    
    print(f"Checkpoint saved to {path}")


def load_checkpoint(
    path: Union[str, Path],
    param_names: Optional[List[str]] = None,
) -> Tuple[Dict[str, np.ndarray], Optional[Dict], Optional[Dict]]:
    """
    Load model, optimizer, and training state from disk.
    
    Args:
        path: Directory containing checkpoint
        param_names: Optional list of specific parameter names to load
        
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
            model_params[name] = np.load(npy_file)
    
    # Load optimizer state
    optimizer_path = path / "optimizer_state.json"
    optimizer_state = None
    if optimizer_path.exists():
        with open(optimizer_path, "r") as f:
            optimizer_state = json.load(f)
    
    # Load training state
    training_path = path / "training_state.json"
    training_state = None
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
