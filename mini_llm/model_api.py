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
    if requested_backend not in {"auto", "numpy", "cupy", "torch", "pytorch"}:
        raise ValueError("backend must be auto, numpy, cupy, or pytorch")
    if requested_backend in {"torch", "pytorch"}:
        try:
            import torch
        except ImportError as exc:
            raise UnsupportedExecutionBackend(
                "PyTorch was requested but torch is not installed"
            ) from exc
        if device not in {"auto", "cpu", "cuda"}:
            raise ValueError("device must be auto, cpu, or cuda")
        if device == "cuda" and not torch.cuda.is_available():
            raise UnsupportedExecutionBackend("PyTorch CUDA was requested but is unavailable")
        return "pytorch"

    from .backend import BACKEND_NAME
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
                 execution_policy: str = "auto", backend: str | None = None):
        if not callable(getattr(native_model, "parameters", None)) or not hasattr(native_model, "config"):
            raise TypeError("Native model must provide .config and .parameters()")
        self._native_model = native_model
        self.tokenizer = tokenizer
        self.model_id = str(model_id)
        self.checkpoint = checkpoint
        self._context_length = int(context_length or native_model.config.context_length)
        self.spec = ArchitectureSpec.from_config(native_model.config)
        self.execution_policy = execution_policy
        self.backend = backend or "native"
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
        selected_backend = _validate_backend(backend, device)
        if selected_backend == "pytorch":
            import torch
            from .backends.pytorch.checkpoint import load_checkpoint
            from .backends.pytorch.model import TorchDecoderLanguageModel
            actual_device = (
                "cuda" if device == "auto" and torch.cuda.is_available() else
                "cpu" if device == "auto" else device
            )
            from .model_loading import read_native_config
            config = read_native_config(path)
            dtype = "bfloat16" if actual_device == "cuda" else config.dtype
            loaded = load_checkpoint(
                path,
                model_factory=lambda cfg: TorchDecoderLanguageModel(
                    cfg, dtype=dtype, device=actual_device
                ),
                device=actual_device, dtype=dtype, skip_optimizer=True,
            )
            return cls(
                loaded.model, tokenizer=loaded.tokenizer,
                model_id=model_id or loaded.checkpoint.name,
                checkpoint=str(loaded.checkpoint),
                context_length=loaded.context_length,
                execution_policy=execution_policy, backend="pytorch",
            )
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
                            score=True,  # exact native chunked teacher forcing
                            train=True,  # ExtendedTrainer via TrainingRun adapter
                            persistent_sessions=False,  # multi-turn KV continuation not implemented
                            accessible_weights=True)

    def inspect(self) -> ModelInfo:
        if self.backend == "pytorch":
            params = list(self._native_model.canonical_named_parameters())
            shapes = tuple((name, tuple(int(v) for v in p.shape)) for name, p in params)
            count = sum(int(p.numel()) for _, p in params)
        else:
            params = self._native_model.parameters()
            shapes = tuple((p.name, tuple(int(v) for v in p.shape)) for p in params)
            count = sum(int(p.size) for p in params)
        return ModelInfo(self.model_id, self.spec,
                         count, shapes,
                         self.capabilities, self.checkpoint)

    def execution_info(self) -> ExecutionReport:
        """Last observed session, otherwise an explicitly unverified baseline."""
        if self._last_execution_report is not None:
            return self._last_execution_report
        if self.backend == "pytorch":
            parameter = next(self._native_model.parameters())
            return ExecutionReport(
                backend="pytorch", device=str(parameter.device),
                dtype=str(parameter.dtype),
                active_optimizations=("native-pytorch-ops",),
                notes=("No completed inference session has been observed",),
            )
        from .backend import BACKEND_NAME
        return ExecutionReport(
            backend=BACKEND_NAME,
            device="cuda" if BACKEND_NAME == "cupy" else "cpu",
            dtype=str(self._native_model.dtype),
            notes=("No completed inference session has been observed",),
        )

    def generate(self, prompt: str, *, max_new_tokens: int = 128,
                 require_fast: bool = False, strategy: str = "greedy",
                 temperature: float = 1.0, top_k: int | None = None,
                 top_p: float | None = None, seed: int | None = None,
                 log_speed: bool = False, **kwargs) -> GenerationResult:
        """One-shot generation through the same request session as serving.

        Preserve the legacy facade's decoding strategy names and EOS inclusion
        while avoiding a second generation/sampling implementation. This does
        not provide append-only multi-turn KV continuation.
        """
        if not self.capabilities.generate:
            raise UnsupportedModelOperation("Generation requires a tokenizer")
        if strategy not in {"greedy", "temperature", "top-k", "top-p"}:
            raise ValueError(f"Unknown strategy: {strategy}")
        if strategy == "top-k" and top_k is None:
            raise ValueError("top-k sampling requires top_k")
        if strategy == "top-p" and top_p is None:
            raise ValueError("top-p sampling requires top_p")
        if require_fast or self.execution_policy == "strict-fast":
            # Fail *before* expensive generation if selected path is unverified.
            from .backend import BACKEND_NAME
            from .inference_policy import configure_inference
            configure_inference(self._native_model.config, backend=BACKEND_NAME,
                                dtype=self._native_model.dtype, policy="strict-fast")
        session = self.session()
        temp = 0.0 if strategy == "greedy" else temperature
        outcome = session.generate(
            prompt, max_tokens=max_new_tokens, temperature=temp,
            top_k=top_k if strategy == "top-k" else None,
            top_p=(top_p if strategy == "top-p" else 1.0),
            seed=seed, log_speed=log_speed,
            include_stop_token=True, **kwargs,
        )
        report = session.execution_info() or self.execution_info()
        self._last_execution_report = report
        if require_fast or self.execution_policy == "strict-fast":
            report.require_fast()
        timing = {
            "prefill_time_s": session.prefill_time_s,
            "forward_times": tuple(session.forward_times),
            "sample_times": tuple(session.sample_times),
            "total_tokens": outcome.completion_tokens,
        }
        return GenerationResult(outcome.text, tuple(outcome.token_ids), timing, report)

    def score(self, token_ids, target_ids, *, loss_mask=None,
              position_ids=None, memory_metadata=None,
              dual_resolution_metadata=None, chunk_tokens=512) -> float:
        """Teacher-forced token CE via the existing chunked native LM head.

        No optimizer, backward graph or persistent full-vocabulary logits are
        allocated. Callers own target shifting, EOS masks and document metadata.
        Other providers without a native decoder must reject this operation.
        """
        if self.backend == "pytorch":
            import torch
            parameter = next(self._native_model.parameters())
            ids = torch.as_tensor(token_ids, device=parameter.device, dtype=torch.long)
            targets = torch.as_tensor(target_ids, device=parameter.device, dtype=torch.long)
            if ids.ndim != 2 or targets.ndim != 2 or ids.shape != targets.shape:
                raise ValueError("token_ids and target_ids must have equal [batch, seq] shapes")
            with torch.no_grad():
                return float(self._native_model.forward_loss(
                    ids, targets, loss_mask=loss_mask,
                    position_ids=position_ids,
                    dual_resolution_metadata=dual_resolution_metadata,
                ).item())
        from .backend import xp
        ids = xp.asarray(token_ids, dtype=xp.int32)
        targets = xp.asarray(target_ids, dtype=xp.int32)
        if ids.ndim != 2 or targets.ndim != 2:
            raise ValueError("token_ids and target_ids must be [batch, seq]")
        if ids.shape != targets.shape:
            raise ValueError("token IDs and targets must have equal shapes")
        hidden, target_slice = self._native_model.forward_body(
            ids, return_cache=False, return_target_slice=True,
            position_ids=position_ids, memory_metadata=memory_metadata,
            dual_resolution_metadata=dual_resolution_metadata,
        )
        if target_slice is not None:
            start, stop = map(int, target_slice)
            targets = targets[:, start:stop]
            if loss_mask is not None:
                loss_mask = xp.asarray(loss_mask)[:, start:stop]
        loss, _ = self._native_model.chunked_lm_head_loss_forward(
            hidden, targets, target_slice=target_slice,
            loss_mask=loss_mask, chunk_tokens=chunk_tokens,
        )
        return float(loss)

    def train(self, *, train_shard_paths=(), val_shard_paths=(), **training_options):
        """Create a native TrainingRun using the validated ExtendedTrainer path.

        Composite specialist trainers are not yet supported by this adapter.
        """
        if self.backend == "pytorch":
            from .backends.pytorch.training import TorchTrainingRun
            return TorchTrainingRun.from_model(
                self, train_shard_paths=train_shard_paths,
                val_shard_paths=val_shard_paths, **training_options,
            )
        from .training_run import TrainingRun
        return TrainingRun.from_native(
            self, train_shard_paths=train_shard_paths,
            val_shard_paths=val_shard_paths, **training_options,
        )

    def session(self):
        """Create an independent generation session over shared model weights.

        Successive calls start independent requests; append-only multi-turn KV
        continuation is not yet supported. Each session retains its own engine
        state, so callers must not drive the same session concurrently.
        """
        if not self.capabilities.generate:
            raise UnsupportedModelOperation("Generation requires a tokenizer")
        if self.backend == "pytorch":
            from .backends.pytorch.inference import TorchGeneratorAdapter, sample_token
            from .generation import GenerationSession
            return GenerationSession(
                TorchGeneratorAdapter(
                    self._native_model, self.tokenizer, self._context_length
                ),
                sampler=sample_token,
            )
        from .backend import BACKEND_NAME
        from .inference_policy import configure_inference
        configure_inference(self._native_model.config, backend=BACKEND_NAME,
                            dtype=self._native_model.dtype, policy=self.execution_policy)
        from .generation import GenerationSession, NativeGeneratorAdapter
        from .legacy_generation import TextGenerator
        return GenerationSession(NativeGeneratorAdapter(TextGenerator(
            self._native_model, self.tokenizer, max_context=self._context_length)))

    def save_weights(self, path):
        """Write a canonical, weights-only native checkpoint snapshot.

        No optimizer state is serialized. The configuration and tokenizer must
        already exist at ``path`` or be supplied by the calling application;
        this method deliberately does not fabricate tokenizer metadata.
        """
        if self.backend == "pytorch":
            from .backends.pytorch.checkpoint import save_checkpoint
            save_checkpoint(path, self._native_model, tokenizer=self.tokenizer)
            return path
        from .checkpoint import save_checkpoint
        save_checkpoint(path, {p.name: p.data for p in self._native_model.parameters()})
        return path
