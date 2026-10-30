"""0103C: canonical, lazy model loading for native and supported Llama weights.

Only loads architecture/weights/tokenizer: no sampling, serving protocol,
benchmark code, or new mathematical kernels. Loading a native checkpoint is
strict about its canonical parameter schema (no random missing weights).

The CuPy/NumPy backend is still process-global; PyTorch is not silently used.
"""

from __future__ import annotations

from dataclasses import dataclass
import json
from pathlib import Path
from typing import Any


@dataclass(frozen=True)
class LoadedNative:
    model: Any
    tokenizer: Any
    context_length: int
    training_state: dict
    tokenizer_path: Path
    checkpoint: Path
    runtime_profile: Any = None
    execution_policy: Any = None

    @property
    def config(self):
        return self.model.config


@dataclass(frozen=True)
class LoadedPretrained:
    engine: Any
    tokenizer: Any
    context_length: int
    format: str


def read_native_config(checkpoint: str | Path):
    from .config import ModelConfig
    path = Path(checkpoint).expanduser().resolve()
    config_file = path / "config.json"
    if not config_file.is_file():
        raise FileNotFoundError(f"Native checkpoint config not found: {config_file}")
    return ModelConfig(**json.loads(config_file.read_text(encoding="utf-8")))


def resolve_native_context(config: Any, requested: int | None = None) -> int:
    """One context policy for CLI, server, model facade and benchmarks.

    Historical-memory models distinguish the working window and addressable
    horizon. For ordinary models, a caller may request a cache horizon beyond
    the training context, as permitted by the original inference engine.
    """
    memory = getattr(config, "memory_context", None)
    historical = bool(memory is not None and memory.enabled and
                      memory.integration_mode in
                      {"terminal_landmark", "routed_prefix", "historical_read"})
    horizon = int(memory.memory_length) if historical else int(config.context_length)
    resolved = horizon if requested is None else int(requested)
    if resolved < 1:
        raise ValueError("Inference context must be positive")
    if historical and resolved > horizon:
        raise ValueError(f"Requested context {resolved} exceeds checkpoint memory horizon {horizon}")
    return resolved


def _load_native_tokenizer(checkpoint: Path):
    from .tokenizer.tokenizer import SimpleBPETokenizer, FastBPETokenizer
    token_path = checkpoint / "tokenizer.json"
    if not token_path.is_file():
        token_path = checkpoint.parent / "tokenizer.json"
    if not token_path.is_file():
        raise FileNotFoundError(f"Tokenizer missing for native checkpoint: {checkpoint}")
    content = json.loads(token_path.read_text(encoding="utf-8"))
    is_simple = "vocab_size" in content and "token_to_id" in content
    cls = SimpleBPETokenizer if is_simple else FastBPETokenizer
    return cls.load(str(token_path)), token_path


def load_native_checkpoint(checkpoint: str | Path, *,
                           requested_context: int | None = None,
                           runtime_profile: str | None = None,
                           apply_profile: bool = False,
                           execution_policy: str | None = None) -> LoadedNative:
    """Load one native model with strict completeness and shape checks.

    Runtime profiles are applied *before* constructing model/kernels if
    requested. Default is no environment mutation for API compatibility;
    launcher wrappers opt in to their prior profile behavior.
    """
    from .checkpoint import load_checkpoint
    from .model.decoder_lm import DecoderLanguageModel

    path = Path(checkpoint).expanduser().resolve()
    config = read_native_config(path)
    profile = None
    if apply_profile:
        from .backend import BACKEND_NAME
        from .runtime_defaults import apply_runtime_defaults
        profile = apply_runtime_defaults(runtime_profile, backend_name=BACKEND_NAME,
                                         config=config, training=False,
                                         precision=getattr(config, "dtype", None))
    policy_result = None
    if execution_policy is not None:
        from .backend import BACKEND_NAME
        from .inference_policy import configure_inference
        policy_result = configure_inference(config, backend=BACKEND_NAME,
                                            dtype=config.dtype, policy=execution_policy)
    context = resolve_native_context(config, requested_context)
    tokenizer, tokenizer_path = _load_native_tokenizer(path)
    model = DecoderLanguageModel(config, rng_seed=42, dtype=config.dtype)
    parameters = model.parameters()
    names = [p.name for p in parameters]
    if len(names) != len(set(names)):
        raise ValueError("Model has duplicate canonical parameter names")
    weights, _, state = load_checkpoint(path, param_names=names, skip_optimizer=True)
    missing = set(names).difference(weights)
    if missing:
        raise ValueError(f"Incomplete checkpoint: {len(missing)} missing parameter arrays; "
                         f"examples: {sorted(missing)[:5]}")
    for param in parameters:
        if tuple(weights[param.name].shape) != tuple(param.shape):
            raise ValueError(f"Checkpoint parameter shape mismatch: {param.name}")
    for param in parameters:
        param.data[...] = weights[param.name]
    if config.progressive_depth:
        active = int((state or {}).get("progressive_active_layers",
                                       config.progressive_initial_layers))
        model.set_active_layers(active)
    return LoadedNative(model, tokenizer, context, state or {},
                        tokenizer_path, path, profile, policy_result)


def load_pretrained_weights(path: str | Path, *, format: str, dtype: str = "float16") -> LoadedPretrained:
    """Reuse existing in-process Llama-family GGUF/Safetensors engines.

    Unsupported GGUF architectures fail in the existing Llama loader. Other
    GGUF architectures may eventually use a remote llama.cpp adapter.
    """
    path = Path(path).expanduser().resolve()
    if format == "gguf":
        from gguf import GGUFReader
        from tokenizers import Tokenizer
        from .external import load_gguf_model
        from .external.gguf_loader import GGUFWeightMap
        from .external.gguf_tokenizer import load_gguf_tokenizer
        inspected = GGUFWeightMap(GGUFReader(str(path)))
        try:
            tokenizer = load_gguf_tokenizer(inspected.reader,
                                            vocab_size=inspected.config.vocab_size)
        except ValueError:
            sibling = path.parent / "tokenizer.json"
            if not sibling.is_file():
                raise ValueError("GGUF tokenizer metadata unsupported; supply adjacent tokenizer.json")
            tokenizer = Tokenizer.from_file(str(sibling))
        engine, _ = load_gguf_model(path, dtype=dtype)
    elif format == "safetensors":
        from tokenizers import Tokenizer
        from .external import load_safetensors_model
        tokenizer_path = path / "tokenizer.json"
        if not tokenizer_path.is_file():
            raise ValueError("Safetensors folder lacks tokenizer.json")
        tokenizer = Tokenizer.from_file(str(tokenizer_path))
        engine, _ = load_safetensors_model(path, dtype=dtype)
    else:
        raise ValueError(f"Unsupported in-process pretrained format: {format!r}")
    return LoadedPretrained(engine, tokenizer, int(engine.config.max_position_embeddings), format)
