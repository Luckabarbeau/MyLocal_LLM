"""Synchronous host-side native diagnostics; timings never imply GPU completion.

No device arrays are read. Instantiate once per process with a unique run_id;
roles are supplied by the caller (normally ``owner-cuda`` / ``peer-hip``).
Rotation is record-atomic: a single oversized record may exceed the byte limit.
"""
from __future__ import annotations

from contextlib import contextmanager
from dataclasses import asdict, is_dataclass
from functools import wraps
import hashlib
import importlib
from importlib import metadata as package_metadata
import json
import math
import os
from pathlib import Path
import platform
import time
import traceback

import numpy as np


_METADATA = frozenset({
    "document_starts", "document_ends", "target_source_starts",
    "target_valid_lengths", "history_valid_starts", "historyvalidstarts",
    "router_gumbel_seed", "executed_sequence_length", "shape_bucket",
})
_TIMING = "host_return_not_gpu_completion"


def _descriptor(value):
    result = {"type": type(value).__module__ + "." + type(value).__qualname__}
    for name in ("shape", "dtype", "strides"):
        attr = getattr(value, name, None)
        if attr is not None:
            result[name] = str(attr) if name == "dtype" else list(attr)
    return result


def _safe(value, *, arrays=False, full=False):
    """Only explicit NumPy CPU types may expose values; never coerce tensors."""
    if value is None or isinstance(value, (str, bool, int)):
        return value
    if isinstance(value, float):
        return value if math.isfinite(value) else str(value)
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, np.ndarray):
        result = _descriptor(value)
        if (arrays or full) and value.dtype.kind in "biufUS":
            flat = value.reshape(-1)
            values = flat if full else flat[:32]
            if full:
                return _safe(values.tolist(), full=True)
            result.update(values=_safe(values.tolist()), count=int(value.size),
                          truncated=value.size > 32)
        return result
    if isinstance(value, np.bool_):
        return bool(value)
    if isinstance(value, np.integer):
        return int(value)
    if isinstance(value, np.floating):
        return _safe(float(value))
    if isinstance(value, dict):
        return {str(key): _safe(item, arrays=arrays, full=full)
                for key, item in value.items()}
    if isinstance(value, (tuple, list)):
        items = value if full or not arrays else value[:32]
        return [_safe(item, arrays=arrays, full=full) for item in items]
    if isinstance(value, bytes):
        return value.decode("utf-8", errors="replace")
    return _descriptor(value)


def _cpu_summary(value, *, hash_values=False):
    result = _descriptor(value)
    if isinstance(value, np.ndarray) and value.dtype.kind in "biuf":
        if hash_values:
            result["sha256"] = hashlib.sha256(value.tobytes(order="C")).hexdigest()
        if value.size:
            result["min"] = _safe(value.min())
            result["max"] = _safe(value.max())
    return result


def _integer_env(name, default, minimum):
    text = os.environ.get(name, str(default))
    if not text.isascii() or not text.isdecimal() or int(text) < minimum:
        raise ValueError(f"{name} must be an integer >= {minimum}")
    return int(text)


def _backend(trainer):
    model = getattr(trainer, "model", None)
    for owner in (trainer, model):
        for name in ("backend", "xp"):
            candidate = getattr(owner, name, None)
            if candidate is not None:
                if isinstance(candidate, str):
                    if candidate not in {"numpy", "cupy"}:
                        raise ValueError("GpuEventLog requires the native NumPy/CuPy backend")
                    return importlib.import_module(candidate)
                return getattr(candidate, "xp", candidate)
    # Existing native models use the process backend rather than an instance field.
    if os.environ.get("MINI_LLM_BACKEND", "numpy").lower() not in {"numpy", "cupy"}:
        raise ValueError("GpuEventLog requires the native NumPy/CuPy backend")
    return importlib.import_module("mini_llm.backend").xp


class GpuEventLog:
    """Direct JSONL writer with reversible model hooks and no GPU synchronization."""

    def __init__(self, trainer, *, role, directory, run_id):
        for label, value in (("role", role), ("run_id", run_id)):
            if not isinstance(value, str) or not value or value in {".", ".."} or any(
                char in value for char in ("/", "\\", "\x00")
            ):
                raise ValueError(f"{label} must be a nonempty filename component")
        self._limit = _integer_env("NATIVE_GPU_LOG_MAX_MB", 64, 1) * 1024 * 1024
        self._backups = _integer_env("NATIVE_GPU_LOG_BACKUPS", 4, 0)
        layers = os.environ.get("NATIVE_GPU_LOG_LAYERS", "1")
        if layers not in {"0", "1"}:
            raise ValueError("NATIVE_GPU_LOG_LAYERS must be 0 or 1")
        self._layers = layers == "1"
        self.role, self.run_id = role, run_id
        self.phase = "train"
        self._context = {}
        self._stages = []
        self._originals = []
        self._closed = False
        self._bytes = 0
        xp = _backend(trainer)
        backend_name = getattr(xp, "__name__", type(xp).__name__)
        runtime = {"pid": os.getpid(), "backend": backend_name, "python": platform.python_version(),
                   "platform": platform.platform(), "kernel": platform.release()}
        self._pool = None
        if backend_name == "cupy":
            runtime["is_hip"] = bool(xp.cuda.runtime.is_hip)
            runtime["device_properties"] = _safe(
                xp.cuda.runtime.getDeviceProperties(xp.cuda.runtime.getDevice()), arrays=True)
            self._pool = xp.get_default_memory_pool()
        versions = {}
        for name in ("numpy", "cupy", "cupy-cuda12x", "cupy-cuda13x", "cupy-rocm-7-0", "triton", "ml-dtypes"):
            try:
                versions[name] = package_metadata.version(name)
            except package_metadata.PackageNotFoundError:
                pass
        model = getattr(trainer, "model", None)
        config = getattr(model, "config", None)
        config = asdict(config) if is_dataclass(config) and not isinstance(config, type) else config
        settings = {name: getattr(trainer, name, None) for name in (
            "grad_accum_steps", "gradient_accumulation_weighting", "precision",
            "mixed_precision", "eos_token_id", "batch_size", "seq_length",
            "target_seq_length", "activation_checkpoint", "source_weights",
            "train_shard_paths", "val_shard_paths", "train_source_shards", "val_source_shards",
        )}
        settings["model_dtype"] = str(getattr(model, "dtype", None))
        environment = {key: value for key, value in os.environ.items()
                       if key.startswith(("MINI_LLM_", "AMD_SERIALIZE", "TRITON", "ROCM"))
                       and not any(word in key.upper() for word in ("SECRET", "PASSWORD", "API_KEY", "CREDENTIAL"))
                                              and "TOKEN" not in key.upper().replace("TOKENS", "")}
        manifest = _safe(dict(role=role, run_id=run_id, runtime=runtime, versions=versions,
                              model_config=config, training=settings, environment=environment,
                              timing_semantics=_TIMING, max_bytes=self._limit, backups=self._backups))
        directory = Path(directory)
        directory.mkdir(parents=True, exist_ok=True)
        self.path = directory / f"{role}.{run_id}.jsonl"
        self.manifest_path = directory / f"{role}.{run_id}.manifest.json"
        self._file = self.path.open("xb", buffering=0)
        try:
            with self.manifest_path.open("x", encoding="utf-8") as handle:
                json.dump(manifest, handle, allow_nan=False, indent=2)
                handle.write("\n")
        except BaseException:
            self._file.close()
            self._closed = True
            raise

    def _rotate(self):
        self._file.close()
        if self._backups:
            oldest = Path(f"{self.path}.{self._backups}")
            oldest.unlink(missing_ok=True)
            for index in range(self._backups - 1, 0, -1):
                source = Path(f"{self.path}.{index}")
                if source.exists():
                    source.replace(f"{self.path}.{index + 1}")
            self.path.replace(f"{self.path}.1")
        else:
            self.path.unlink()
        self._file = self.path.open("xb", buffering=0)
        self._bytes = 0

    def emit(self, event, **fields):
        if self._closed:
            raise ValueError("GpuEventLog is closed")
        record = dict(fields)
        record.update(self._context)
        record.update(event=event, role=self.role, run_id=self.run_id, phase=self.phase,
                      pid=os.getpid(), time_ns=time.time_ns(), monotonic_ns=time.monotonic_ns(),
                                            stages=list(self._stages))
        payload = (json.dumps(_safe(record), allow_nan=False, separators=(",", ":")) + "\n").encode("utf-8")
        if self._bytes and self._bytes + len(payload) > self._limit:
            self._rotate()
        remaining = memoryview(payload)
        while remaining:
            written = self._file.write(remaining)
            if not written:
                raise OSError("GpuEventLog write made no progress")
            self._bytes += written
            remaining = remaining[written:]

    @contextmanager
    def stage(self, name, **fields):
        self._stages.append(name)
        try:
            self.emit("stage_begin", name=name, **fields)
            start = time.perf_counter()
            try:
                yield
            except BaseException as exc:
                self.emit("error", **{**fields, "name": name,
                          "exception_type": type(exc).__name__, "message": str(exc),
                          "traceback": traceback.format_exc()})
                raise
            finally:
                self.emit("stage_end", **{**fields, "name": name,
                          "host_elapsed_s": time.perf_counter() - start,
                          "semantics": _TIMING})
        finally:
            self._stages.pop()

    def microbatch_begin(self, trainer, microbatch):
        self._context = dict(weight_version=trainer.step, update_step=trainer.step + 1,
                             microbatch=microbatch, phase=self.phase)
        fields = {name: getattr(trainer, name, None) for name in (
            "current_train_shard_idx", "current_train_source_shard_idx", "tokens_processed",
            "train_source_batch_counts", "train_source_supervised_tokens",
        )}
        fields["train_rng_state"] = _safe(trainer.train_rng.bit_generator.state, full=True)
        if self._pool is not None:
            fields["memory_pool"] = dict(used_bytes=self._pool.used_bytes(),
                                         total_bytes=self._pool.total_bytes())
        self.emit("microbatch_begin", **fields)

    def batch_ready(self, trainer, inputs, targets, metadata, loss_mask, valid_tokens,
                    microbatch_weight, activation_checkpoint):
        metadata = {} if metadata is None else metadata
        selected = {key: _safe(value, arrays=True) for key, value in metadata.items()
                    if key in _METADATA}
        # Unknown block metadata is summarized, never materialized or serialized raw.
        blocks = {str(key): self._block_summary(value) for key, value in metadata.items()
                  if "block" in str(key).lower()}
        source = getattr(trainer, "_last_train_batch_source", "local")
        source_shards = getattr(trainer, "train_source_shards", None)
        if source_shards is not None:
            paths = source_shards.get(source, ())
            cursor = (getattr(trainer, "current_train_source_shard_idx", None) or {}).get(source, 0)
        else:
            paths = getattr(trainer, "train_shard_paths", ())
            cursor = getattr(trainer, "current_train_shard_idx", 0)
        shard = str(paths[(cursor - 1) % len(paths)]) if paths else None
        self.emit("batch_ready", source=source, shard_path=shard,
                  inputs=_cpu_summary(inputs, hash_values=True),
                  targets=_cpu_summary(targets, hash_values=True),
                  metadata=selected, block_metadata=blocks,
                  loss_mask=None if loss_mask is None else _cpu_summary(loss_mask),
                  valid_tokens=valid_tokens, microbatch_weight=microbatch_weight,
                  activation_checkpoint=activation_checkpoint)

    @staticmethod
    def _block_summary(value):
        if isinstance(value, dict):
            return {str(key): GpuEventLog._block_summary(item) for key, item in value.items()}
        if isinstance(value, (list, tuple)):
            return {"count": len(value), "descriptors": [_descriptor(item) for item in value[:32]]}
        result = _descriptor(value)
        if isinstance(value, np.ndarray):
            result["count"] = int(value.size)
        return result

    def install_model_hooks(self, model):
        if self._closed:
            raise ValueError("GpuEventLog is closed")
        if not self._layers:
            return
        for index, block in enumerate(getattr(model, "blocks", ())):
            for target, prefix in ((block, "layer"), (getattr(block, "attention", None), "attention")):
                if target is None:
                    continue
                for method in ("forward", "backward"):
                    if any(owner is target and name == method for owner, name, _, _ in self._originals):
                        continue
                    original = getattr(target, method, None)
                    if original is None:
                        continue
                    had_instance = method in vars(target)
                    saved = vars(target).get(method)
                    wrapped = self._wrap(original, f"{prefix}.{method}", index)
                    setattr(target, method, wrapped)
                    self._originals.append((target, method, had_instance, saved))

    def _wrap(self, original, name, index):
        @wraps(original)
        def wrapped(*args, **kwargs):
            tensors = {"args": [_descriptor(arg) for arg in args],
                       "kwargs": {key: _descriptor(value) for key, value in kwargs.items()}}
            with self.stage(name, layerindex=index, tensors=tensors):
                return original(*args, **kwargs)
        return wrapped

    def close(self):
        if self._closed:
            return
        try:
            for owner, name, had_instance, saved in reversed(self._originals):
                if had_instance:
                    setattr(owner, name, saved)
                else:
                    delattr(owner, name)
            self._originals.clear()
        finally:
            self._closed = True
            self._file.close()
