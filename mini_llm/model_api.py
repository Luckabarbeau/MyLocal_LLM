"""0103-0: additive public model facade; NO new numerical implementations.

The facade holds an existing native model. Generation delegates to the
existing TextGenerator. Training, serving and persistent sessions will be
migrated in later 0103 patches; unavailable functions fail explicitly.

The legacy backend is selected globally at import time (MINI_LLM_BACKEND).
This facade never silently pretends to have switched that backend.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from .model_contracts import (
    ArchitectureSpec, Capabilities, ExecutionReport, GenerationResult,
    ModelInfo, UnsupportedExecutionBackend, UnsupportedModelOperation,
)


def _validate_backend(requested_backend: str, device: str) -> str:
    from .backend import BACKEND_NAME

    if requested_backend not in {"auto", "numpy", "cupy", "torch"}:
        raise ValueError("backend must be auto, numpy, cupy, or torch")
    if requested_backend == "torch":
        raise UnsupportedExecutionBackend(
            "PyTorch adapter is planned but is not implemented in 0103-0"
        )
    if requested_backend != "auto" and requested_backend != BACKEND_NAME:
        raise UnsupportedExecutionBackend(
            f"Process backend is {BACKEND_NAME!r}, requested {requested_backend!r}; "
            f"start a separate process with MINI_LLM_BACKEND={requested_backend}."
        )
    if device not in {"auto", "cpu", "cuda"}:
        raise ValueError("device must be auto, cpu, or cuda")
    actual_device = "cuda" if BACKEND_NAME == "cupy" else "cpu"
    if device != "auto" and device != actual_device:
        raise UnsupportedExecutionBackend(
            f"Process backend {BACKEND_NAME!r} runs on {actual_device!r}, "
            f"requested device {device!r}."
        )
    return BACKEND_NAME


class Model:
    """Single public entry point to supported native numerical model objects.

    This first patch supports native checkpoint loading, inspection, and
    legacy generation with a tokenizer. It does not claim verified fast GPU
    inference, automatic execution selection, or unified training yet.
    """

    def __init__(self, native_model: Any, *, tokenizer: Any = None,
                 model_id: str = "native", checkpoint: str | None = None,
                 context_length: int | None = None):
        if not callable(getattr(native_model, "parameters", None)) or not hasattr(native_model, "config"):
            raise TypeError("Native model must provide .config and .parameters()")
        self._native_model = native_model
        self.tokenizer = tokenizer
        self.model_id = str(model_id)
        self.checkpoint = checkpoint
        self._context_length = int(context_length or native_model.config.context_length)
        self.spec = ArchitectureSpec.from_config(native_model.config)

    @classmethod
    def wrap_native(cls, native_model, *, tokenizer=None, model_id="native",
                    context_length=None) -> "Model":
        """Wrap an existing instance without copying weights or grad buffers."""
        return cls(native_model, tokenizer=tokenizer, model_id=model_id,
                   context_length=context_length)

    @classmethod
    def load(cls, path, *, backend="auto", device="auto", model_id=None) -> "Model":
        """Load a native checkpoint using its canonical config/parameter names.

        Model formats other than native are handled by the legacy serving
        loader until the 0103C provider migration. No benchmark code imported.
        """
        _validate_backend(backend, device)
        path = Path(path).expanduser().resolve()
        config_path = path / "config.json"
        if not config_path.is_file():
            raise FileNotFoundError(
                f"Native checkpoint config not found: {config_path}; "
                "pretrained/remote adapters are scheduled for 0103C."
            )
        from .config import ModelConfig
        from .model.decoder_lm import DecoderLanguageModel
        from .checkpoint import load_checkpoint
        from .tokenizer.tokenizer import FastBPETokenizer, SimpleBPETokenizer

        config = ModelConfig(**json.loads(config_path.read_text(encoding="utf-8")))
        tokenizer_path = path / "tokenizer.json"
        if not tokenizer_path.is_file():
            tokenizer_path = path.parent / "tokenizer.json"
        if not tokenizer_path.is_file():
            raise FileNotFoundError(f"Tokenizer missing for native checkpoint: {path}")
        token_json = json.loads(tokenizer_path.read_text(encoding="utf-8"))
        tokenizer_cls = (SimpleBPETokenizer if "vocab_size" in token_json and
                         "token_to_id" in token_json else FastBPETokenizer)
        tokenizer = tokenizer_cls.load(str(tokenizer_path))
        native = DecoderLanguageModel(config, rng_seed=42, dtype=config.dtype)
        params = native.parameters()
        names = [param.name for param in params]
        if len(names) != len(set(names)):
            raise ValueError("Model contains duplicate canonical parameter names")
        saved, _, train_state = load_checkpoint(path, param_names=names,
                                                 skip_optimizer=True)
        missing = set(names) - set(saved)
        if missing:
            raise ValueError(
                f"Incomplete checkpoint: {len(missing)} missing parameter arrays; "
                f"examples: {sorted(missing)[:5]}"
            )
        for param in params:
            if tuple(saved[param.name].shape) != tuple(param.shape):
                raise ValueError(f"Checkpoint parameter shape mismatch: {param.name}")
        for param in params:
            param.data[...] = saved[param.name]
        if config.progressive_depth:
            active = int((train_state or {}).get("progressive_active_layers",
                                                  config.progressive_initial_layers))
            native.set_active_layers(active)
        return cls(native, tokenizer=tokenizer, model_id=model_id or path.name,
                   checkpoint=str(path))

    @property
    def capabilities(self) -> Capabilities:
        return Capabilities(generate=self.tokenizer is not None,
                            accessible_weights=True)

    def inspect(self) -> ModelInfo:
        params = self._native_model.parameters()
        shapes = tuple((p.name, tuple(int(v) for v in p.shape)) for p in params)
        return ModelInfo(self.model_id, self.spec,
                         sum(int(p.size) for p in params), shapes,
                         self.capabilities, self.checkpoint)

    def execution_info(self) -> ExecutionReport:
        """Report only observed backend/device, not inferred fast kernels."""
        from .backend import BACKEND_NAME
        return ExecutionReport(
            backend=BACKEND_NAME,
            device="cuda" if BACKEND_NAME == "cupy" else "cpu",
            dtype=str(self._native_model.dtype),
            notes=("0103-0 does not instrument active kernels or CUDA Graph capture; "
                   "acceleration is not verified",),
        )

    def generate(self, prompt: str, *, max_new_tokens: int = 128,
                 require_fast: bool = False, **kwargs) -> GenerationResult:
        if not self.capabilities.generate:
            raise UnsupportedModelOperation("Generation requires a tokenizer")
        report = self.execution_info()
        if require_fast:
            report.require_fast()
        # Existing optimized/native implementation; no new decoding maths.
        from inference import TextGenerator
        generator = TextGenerator(self._native_model, self.tokenizer,
                                  max_context=self._context_length)
        output, timing = generator.generate(
            prompt, max_new_tokens=max_new_tokens,
            use_kv_cache=True, return_token_ids=True, **kwargs,
        )
        ids = tuple(int(i) for i in timing.pop("generated_token_ids", ()))
        return GenerationResult(text=output, token_ids=ids,
                                timing=timing, execution=report)

    def score(self, *args, **kwargs):
        raise UnsupportedModelOperation("Unified score API is scheduled for 0103G")

    def train(self, *args, **kwargs):
        raise UnsupportedModelOperation("Unified training API is scheduled for 0103H")

    def session(self, *args, **kwargs):
        raise UnsupportedModelOperation("Persistent inference sessions are scheduled for 0103E")

    def save_weights(self, *args, **kwargs):
        raise UnsupportedModelOperation("Weights-only export via facade is not wired in 0103-0")
