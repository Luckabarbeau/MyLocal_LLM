"""0103-0: additive public model facade; NO new numerical implementations.

The facade holds an existing native model. Generation delegates to the
shared generation loop through TextGenerator. Request-scoped sessions are
available; multi-turn KV continuation and training remain separate work.

The legacy backend is selected globally at import time (MINI_LLM_BACKEND).
This facade never silently pretends to have switched that backend.
"""

from __future__ import annotations

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

    Provides native checkpoint loading, inspection, request-scoped generation
    and streaming via the common inference session. Training is not wired yet.
    """

    def __init__(self, native_model: Any, *, tokenizer: Any = None,
                 model_id: str = "native", checkpoint: str | None = None,
                 context_length: int | None = None,
                 execution_policy: str = "auto"):
        if not callable(getattr(native_model, "parameters", None)) or not hasattr(native_model, "config"):
            raise TypeError("Native model must provide .config and .parameters()")
        self._native_model = native_model
        self.tokenizer = tokenizer
        self.model_id = str(model_id)
        self.checkpoint = checkpoint
        self._context_length = int(context_length or native_model.config.context_length)
        self.spec = ArchitectureSpec.from_config(native_model.config)
        self.execution_policy = execution_policy
        self._last_execution_report = None

    @classmethod
    def wrap_native(cls, native_model, *, tokenizer=None, model_id="native",
                    context_length=None, execution_policy="auto") -> "Model":
        """Wrap an existing instance without copying weights or grad buffers."""
        return cls(native_model, tokenizer=tokenizer, model_id=model_id,
                   context_length=context_length,
                   execution_policy=execution_policy)

    @classmethod
    def load(cls, path, *, backend="auto", device="auto", model_id=None,
             execution_policy="auto") -> "Model":
        """Load a native checkpoint using its canonical config/parameter names.

        The canonical native loader is shared with the CLI, benchmarks and
        serving. GGUF/Safetensors sources use load_pretrained_weights from
        model_loading until a common generation facade is available in 0103E.
        """
        _validate_backend(backend, device)
        from .model_loading import load_native_checkpoint
        loaded = load_native_checkpoint(path, execution_policy=execution_policy)
        return cls(loaded.model, tokenizer=loaded.tokenizer,
                   model_id=model_id or loaded.checkpoint.name,
                   checkpoint=str(loaded.checkpoint),
                   context_length=loaded.context_length,
                   execution_policy=execution_policy)

    @property
    def capabilities(self) -> Capabilities:
        return Capabilities(generate=self.tokenizer is not None,
                            streaming=self.tokenizer is not None,
                            persistent_sessions=False,  # multi-turn KV continuation not implemented
                            accessible_weights=True)

    def inspect(self) -> ModelInfo:
        params = self._native_model.parameters()
        shapes = tuple((p.name, tuple(int(v) for v in p.shape)) for p in params)
        return ModelInfo(self.model_id, self.spec,
                         sum(int(p.size) for p in params), shapes,
                         self.capabilities, self.checkpoint)

    def execution_info(self) -> ExecutionReport:
        """Last observed session, otherwise an explicitly unverified baseline."""
        if self._last_execution_report is not None:
            return self._last_execution_report
        from .backend import BACKEND_NAME
        return ExecutionReport(
            backend=BACKEND_NAME,
            device="cuda" if BACKEND_NAME == "cupy" else "cpu",
            dtype=str(self._native_model.dtype),
            notes=("No completed inference session has been observed",),
        )

    def generate(self, prompt: str, *, max_new_tokens: int = 128,
                 require_fast: bool = False, **kwargs) -> GenerationResult:
        if not self.capabilities.generate:
            raise UnsupportedModelOperation("Generation requires a tokenizer")
        # The model facade always configures BEFORE constructing the inference engine.
        from .backend import BACKEND_NAME
        from .inference_policy import configure_inference
        selected_policy = "strict-fast" if require_fast else self.execution_policy
        configure_inference(self._native_model.config, backend=BACKEND_NAME,
                            dtype=self._native_model.dtype, policy=selected_policy)
        # Existing optimized/native implementation; no new decoding maths.
        from .legacy_generation import TextGenerator
        generator = TextGenerator(self._native_model, self.tokenizer,
                                  max_context=self._context_length)
        output, timing = generator.generate(
            prompt, max_new_tokens=max_new_tokens,
            use_kv_cache=True, return_token_ids=True, **kwargs,
        )
        ids = tuple(int(i) for i in timing.pop("generated_token_ids", ()))
        report = getattr(generator, "last_execution_report", None) or self.execution_info()
        self._last_execution_report = report
        if require_fast or selected_policy == "strict-fast":
            report.require_fast()
        return GenerationResult(text=output, token_ids=ids,
                                timing=timing, execution=report)

    def score(self, *args, **kwargs):
        raise UnsupportedModelOperation("Unified score API is scheduled for 0103G")

    def train(self, *args, **kwargs):
        raise UnsupportedModelOperation("Unified training API is scheduled for 0103H")

    def session(self):
        """Create an independent generation session over shared model weights.

        Successive calls start independent requests; append-only multi-turn KV
        continuation is not yet supported. Each session retains its own engine
        state, so callers must not drive the same session concurrently.
        """
        if not self.capabilities.generate:
            raise UnsupportedModelOperation("Generation requires a tokenizer")
        from .backend import BACKEND_NAME
        from .inference_policy import configure_inference
        configure_inference(self._native_model.config, backend=BACKEND_NAME,
                            dtype=self._native_model.dtype, policy=self.execution_policy)
        from .generation import GenerationSession, NativeGeneratorAdapter
        from .legacy_generation import TextGenerator
        return GenerationSession(NativeGeneratorAdapter(TextGenerator(
            self._native_model, self.tokenizer, max_context=self._context_length)))

    def save_weights(self, *args, **kwargs):
        raise UnsupportedModelOperation("Weights-only export via facade is not wired in 0103-0")
