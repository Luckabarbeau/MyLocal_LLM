"""Strict interoperability between native checkpoints and PyTorch models.

PyTorch remains an optional backend.  Importing this module does not import
``torch`` or the process-global NumPy/CuPy backend.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, is_dataclass
import json
from pathlib import Path
from typing import Any, Callable, Mapping

import numpy as np


@dataclass(frozen=True)
class LoadedTorch:
    """A strictly loaded native checkpoint backed by a PyTorch model."""

    model: Any
    tokenizer: Any
    context_length: int
    optimizer_state: dict[str, Any] | None
    training_state: dict[str, Any]
    tokenizer_path: Path
    checkpoint: Path

    @property
    def config(self):
        return self.model.config


def _torch():
    try:
        import torch
    except ImportError as exc:
        raise RuntimeError(
            "PyTorch checkpoint support requires torch to be installed"
        ) from exc
    return torch


def _resolve_dtype(torch, dtype: Any):
    if isinstance(dtype, torch.dtype):
        return dtype
    if not isinstance(dtype, str):
        raise TypeError("dtype must be a torch.dtype or an explicit dtype name")
    aliases = {
        "bfloat16": torch.bfloat16,
        "bf16": torch.bfloat16,
        "float16": torch.float16,
        "fp16": torch.float16,
        "float32": torch.float32,
        "fp32": torch.float32,
        "float64": torch.float64,
        "fp64": torch.float64,
    }
    try:
        return aliases[dtype.lower()]
    except KeyError as exc:
        raise ValueError(f"unsupported PyTorch checkpoint dtype: {dtype!r}") from exc


def _load_numpy(path: Path) -> np.ndarray:
    array = np.load(path, allow_pickle=False)
    if array.dtype.kind == "V":
        if array.dtype.itemsize != 2:
            raise TypeError(f"unsupported void parameter dtype in {path}: {array.dtype}")
        try:
            import ml_dtypes
        except ImportError as exc:
            raise RuntimeError(
                "Checkpoint contains BF16 arrays but ml-dtypes is not installed"
            ) from exc
        array = array.view(ml_dtypes.bfloat16)
    return array


def _numpy_to_torch(array: np.ndarray, *, device: Any, dtype: Any | None = None):
    torch = _torch()
    if array.dtype.kind == "V" and array.dtype.itemsize == 2:
        try:
            import ml_dtypes
        except ImportError as exc:
            raise RuntimeError(
                "Checkpoint contains BF16 arrays but ml-dtypes is not installed"
            ) from exc
        array = array.view(ml_dtypes.bfloat16)
    if str(array.dtype) == "bfloat16":
        tensor = torch.from_numpy(array.astype(np.float32, copy=False))
        target_dtype = torch.bfloat16 if dtype is None else dtype
        return tensor.to(device=device, dtype=target_dtype)
    tensor = torch.from_numpy(np.ascontiguousarray(array))
    return tensor.to(device=device, dtype=dtype) if dtype is not None else tensor.to(device=device)


def _canonical_parameters(model: Any) -> list[tuple[str, Any]]:
    torch = _torch()
    method = getattr(model, "canonical_named_parameters", None)
    if not callable(method):
        raise TypeError("PyTorch model must define canonical_named_parameters()")
    parameters = list(method())
    names: list[str] = []
    identities: list[int] = []
    for item in parameters:
        if not isinstance(item, tuple) or len(item) != 2:
            raise TypeError("canonical_named_parameters() must yield (name, nn.Parameter)")
        name, parameter = item
        if not isinstance(name, str) or not name:
            raise TypeError("canonical parameter names must be non-empty strings")
        if not isinstance(parameter, torch.nn.Parameter):
            raise TypeError(f"canonical parameter {name!r} is not torch.nn.Parameter")
        names.append(name)
        identities.append(id(parameter))
    if len(names) != len(set(names)):
        raise ValueError("Model has duplicate canonical parameter names")
    if len(identities) != len(set(identities)):
        raise ValueError("Model canonical enumeration contains a tied parameter more than once")
    return parameters


def _load_state_directory(directory: Path, *, device: Any) -> dict[str, Any] | None:
    if not directory.exists():
        return None
    if not directory.is_dir():
        raise NotADirectoryError(str(directory))
    state: dict[str, Any] = {}
    scalar_file = directory / "scalars.json"
    if scalar_file.is_file():
        scalars = json.loads(scalar_file.read_text(encoding="utf-8"))
        if not isinstance(scalars, dict):
            raise ValueError(f"optimizer scalar file must contain an object: {scalar_file}")
        state.update(scalars)
    for array_file in sorted(directory.glob("*.npy")):
        if array_file.stem in state:
            raise ValueError(f"duplicate optimizer state key: {array_file.stem}")
        state[array_file.stem] = _numpy_to_torch(
            _load_numpy(array_file), device=device
        )
    for nested in sorted(entry for entry in directory.iterdir() if entry.is_dir()):
        if nested.name in state:
            raise ValueError(f"duplicate optimizer state key: {nested.name}")
        state[nested.name] = _load_state_directory(nested, device=device) or {}
    return state


def _restore_active_layers(model: Any, config: Any, training_state: Mapping[str, Any]) -> None:
    if "progressive_active_layers" in training_state:
        active = int(training_state["progressive_active_layers"])
    elif getattr(config, "progressive_depth", False):
        active = int(getattr(config, "progressive_initial_layers"))
    else:
        return
    setter = getattr(model, "set_active_layers", None)
    if callable(setter):
        setter(active)
    elif hasattr(model, "active_layers"):
        model.active_layers = active


def load_checkpoint(
    checkpoint: str | Path,
    model: Any | None = None,
    *,
    model_factory: Callable[[Any], Any] | None = None,
    device: Any,
    dtype: Any,
    requested_context: int | None = None,
    skip_optimizer: bool = False,
    skip_training: bool = False,
) -> LoadedTorch:
    """Load a native checkpoint into a PyTorch model with exact schema checks.

    Exactly one of ``model`` and ``model_factory`` must be supplied.  All names
    and shapes are validated before checkpoint values are copied into the model.
    ``device`` and ``dtype`` are mandatory so no implicit placement or precision
    policy can leak into checkpoint loading.
    """
    torch = _torch()
    target_device = torch.device(device)
    target_dtype = _resolve_dtype(torch, dtype)

    from mini_llm.checkpoint import resolve_checkpoint_path
    from mini_llm.model_loading import (
        _load_native_tokenizer,
        read_native_config,
        resolve_native_context,
    )

    path = Path(resolve_checkpoint_path(Path(checkpoint).expanduser().resolve()))
    config = read_native_config(path)
    if (model is None) == (model_factory is None):
        raise ValueError("supply exactly one of model or model_factory")
    if model is None:
        model = model_factory(config)
    if not hasattr(model, "config") or model.config != config:
        raise ValueError("PyTorch model config does not exactly match config.json")

    parameters = _canonical_parameters(model)
    expected_names = {name for name, _ in parameters}
    params_path = path / "model_params"
    if not params_path.is_dir():
        raise FileNotFoundError(f"checkpoint parameter directory not found: {params_path}")
    files = sorted(params_path.glob("*.npy"))
    saved_names = {file.stem for file in files}
    if saved_names != expected_names:
        missing = sorted(expected_names - saved_names)
        unexpected = sorted(saved_names - expected_names)
        raise ValueError(
            "Checkpoint parameter names do not exactly match the model; "
            f"missing={missing[:5]}, unexpected={unexpected[:5]}"
        )

    arrays = {file.stem: _load_numpy(file) for file in files}
    for name, parameter in parameters:
        if tuple(arrays[name].shape) != tuple(parameter.shape):
            raise ValueError(
                f"Checkpoint parameter shape mismatch for {name}: "
                f"checkpoint={tuple(arrays[name].shape)}, model={tuple(parameter.shape)}"
            )

    training_state: dict[str, Any] = {}
    if not skip_training:
        training_file = path / "training_state.json"
        if training_file.is_file():
            loaded_training = json.loads(training_file.read_text(encoding="utf-8"))
            if not isinstance(loaded_training, dict):
                raise ValueError("training_state.json must contain a JSON object")
            training_state = loaded_training
    optimizer_state = None
    if not skip_optimizer:
        optimizer_state = _load_state_directory(path / "optimizer_state", device=target_device)
    tokenizer, tokenizer_path = _load_native_tokenizer(path)
    context_length = resolve_native_context(config, requested_context)

    moved = model.to(device=target_device, dtype=target_dtype)
    if moved is not model:
        raise TypeError("PyTorch model.to() must update and return the same model")
    moved_parameters = _canonical_parameters(model)
    if [name for name, _ in moved_parameters] != [name for name, _ in parameters]:
        raise ValueError("canonical parameter names changed during model.to()")
    converted: dict[str, Any] = {}
    for name, parameter in moved_parameters:
        if parameter.device != target_device or parameter.dtype != target_dtype:
            raise ValueError(
                f"model.to() did not place {name} on {target_device} with {target_dtype}"
            )
        converted[name] = _numpy_to_torch(
            arrays[name], device=target_device, dtype=target_dtype
        )

    with torch.no_grad():
        for name, parameter in moved_parameters:
            parameter.copy_(converted[name])
    _restore_active_layers(model, config, training_state)
    return LoadedTorch(
        model=model,
        tokenizer=tokenizer,
        context_length=context_length,
        optimizer_state=optimizer_state,
        training_state=training_state,
        tokenizer_path=tokenizer_path,
        checkpoint=path,
    )


def _torch_to_numpy(value: Any):
    torch = _torch()
    if isinstance(value, torch.Tensor):
        tensor = value.detach().cpu()
        if tensor.dtype == torch.bfloat16:
            try:
                import ml_dtypes
            except ImportError as exc:
                raise RuntimeError(
                    "Saving BF16 tensors requires ml-dtypes"
                ) from exc
            return tensor.float().numpy().astype(ml_dtypes.bfloat16)
        return tensor.numpy()
    if isinstance(value, Mapping):
        return {key: _torch_to_numpy(item) for key, item in value.items()}
    if isinstance(value, tuple):
        return tuple(_torch_to_numpy(item) for item in value)
    if isinstance(value, list):
        return [_torch_to_numpy(item) for item in value]
    return value


def _json_config(config: Any) -> dict[str, Any]:
    if is_dataclass(config):
        value = asdict(config)
    elif isinstance(config, Mapping):
        value = dict(config)
    elif callable(getattr(config, "to_dict", None)):
        value = config.to_dict()
    else:
        raise TypeError("model.config must be a dataclass, mapping, or expose to_dict()")
    if not isinstance(value, dict):
        raise TypeError("serialized model config must be a JSON object")
    json.dumps(value)
    return value


def _prepare_ancillary_files(path: Path, model: Any, tokenizer: Any | None) -> None:
    path.mkdir(parents=True, exist_ok=True)
    config_value = _json_config(model.config)
    config_file = path / "config.json"
    if config_file.is_file():
        current = json.loads(config_file.read_text(encoding="utf-8"))
        if current != config_value:
            raise ValueError("existing config.json does not exactly match model.config")
    else:
        config_file.write_text(
            json.dumps(config_value, indent=2), encoding="utf-8"
        )
    if tokenizer is not None:
        tokenizer_file = path / "tokenizer.json"
        if tokenizer_file.exists():
            raise FileExistsError(f"refusing to overwrite existing tokenizer: {tokenizer_file}")
        tokenizer.save(str(tokenizer_file))


def save_checkpoint(
    path: str | Path,
    model: Any,
    optimizer_state: Mapping[str, Any] | None = None,
    training_state: Mapping[str, Any] | None = None,
    *,
    tokenizer: Any | None = None,
) -> Path:
    """Export a PyTorch model in the atomic native checkpoint layout.

    Torch tensors are always detached and recursively converted to host NumPy
    arrays before calling the native atomic publisher.  This avoids its unsafe
    generic ``np.asarray`` path for CUDA, gradient-tracking, and BF16 tensors.
    """
    from mini_llm.checkpoint import save_checkpoint as atomic_save_checkpoint

    destination = Path(path).expanduser().resolve()
    parameters = _canonical_parameters(model)
    _prepare_ancillary_files(destination, model, tokenizer)
    model_params = {
        name: _torch_to_numpy(parameter) for name, parameter in parameters
    }
    converted_optimizer = (
        None if optimizer_state is None else _torch_to_numpy(optimizer_state)
    )
    converted_training = (
        None if training_state is None else _torch_to_numpy(dict(training_state))
    )
    if converted_training is not None:
        try:
            json.dumps(converted_training)
        except TypeError as exc:
            raise TypeError(
                "training_state must remain JSON-serializable after Torch conversion"
            ) from exc
    atomic_save_checkpoint(
        destination,
        model_params,
        optimizer_state=converted_optimizer,
        training_state=converted_training,
    )
    return destination


load_pytorch_checkpoint = load_checkpoint
save_pytorch_checkpoint = save_checkpoint

__all__ = [
    "LoadedTorch",
    "load_checkpoint",
    "load_pytorch_checkpoint",
    "save_checkpoint",
    "save_pytorch_checkpoint",
]
