"""Single-resident inference adapters for local OpenAI-compatible serving.

No inference mathematics are implemented here: GGUF, HF Safetensors, and
native checkpoints all reuse the repository's existing KV-cache engine.
"""

from dataclasses import dataclass
import gc
import json
from pathlib import Path


@dataclass(frozen=True)
class ModelSpec:
    id: str
    path: Path
    format: str = "auto"
    dtype: str = "float16"
    chat_template: str = "chatml"
    tool_format: str = "none"
    context_length: int | None = None

    @classmethod
    def from_dict(cls, obj, root):
        allowed = set(cls.__dataclass_fields__)
        unknown = set(obj) - allowed
        if unknown:
            raise ValueError(f"Unrecognized model configuration fields: {sorted(unknown)}")
        if not obj.get("id") or not obj.get("path"):
            raise ValueError("Each model must have a nonempty id and path")
        p = Path(obj["path"]).expanduser()
        if not p.is_absolute():
            p = Path(root) / p
        kw = dict(obj, path=p.resolve())
        spec = cls(**kw)
        if spec.format not in {"auto", "gguf", "safetensors", "native"}:
            raise ValueError(f"Unsupported model format: {spec.format}")
        if spec.dtype not in {"float16", "float32"}:
            raise ValueError("Serving currently supports float16 and float32 (native uses checkpoint dtype)")
        if spec.chat_template not in {"chatml", "plain"}:
            raise ValueError(f"Unsupported chat template: {spec.chat_template}")
        if spec.tool_format not in {"none", "json_envelope"}:
            raise ValueError(f"Unsupported tool format: {spec.tool_format}")
        if spec.context_length is not None and spec.context_length <= 0:
            raise ValueError("context_length must be positive")
        return spec

    def resolved_format(self):
        if self.format != "auto":
            return self.format
        if self.path.suffix.lower() == ".gguf":
            return "gguf"
        if self.path.is_dir() and (self.path / "model.safetensors").is_file():
            return "safetensors"
        if self.path.is_dir() and (self.path / "model.safetensors.index.json").is_file():
            return "safetensors"
        if self.path.is_dir() and (self.path / "config.json").is_file():
            return "native"
        raise ValueError(f"Could not identify model format at {self.path}")


def load_specs(manifest):
    path = Path(manifest).expanduser().resolve()
    obj = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(obj, dict) or not isinstance(obj.get("models"), list):
        raise ValueError("Registry must be a JSON object containing a 'models' array")
    specs = [ModelSpec.from_dict(row, path.parent) for row in obj["models"]]
    if not specs or len({x.id for x in specs}) != len(specs):
        raise ValueError("Registry must contain at least one model and unique IDs")
    return {spec.id: spec for spec in specs}


class ServingModel:
    def __init__(self, engine, tokenizer, context_length, *, native=False, stop_ids=()):
        self.engine = engine
        self.tokenizer = tokenizer
        self.context_length = int(context_length)
        self.native = native
        self.stop_ids = set(stop_ids)

    def encode(self, text):
        if self.native:
            return list(self.tokenizer.encode(text))
        return list(self.tokenizer.encode(text, add_special_tokens=False).ids)

    def decode(self, ids):
        if self.native:
            return self.tokenizer.decode(ids)
        return self.tokenizer.decode(ids, skip_special_tokens=False)

    def create_state(self, total_tokens):
        return self.engine.create_generation_state(1, total_tokens)

    def prefill(self, ids, state):
        from mini_llm.backend import xp
        return self.engine.prefill(xp.asarray([ids], dtype=xp.int32), state)

    def decode_one(self, token_id, state):
        from mini_llm.backend import xp
        return self.engine.decode_one(xp.asarray([token_id], dtype=xp.int32), state)


def _external_stops(tokenizer):
    return {int(i) for i in (
        tokenizer.token_to_id(t) for t in
        ("<|im_end|>", "<|endoftext|>", "</s>", "<|eos|>")) if i is not None}


def _native_stops(tokenizer):
    return {int(i) for t in ("<|eos|>", "</s>") for ids in [tokenizer.encode(t)]
            if len(ids) == 1 for i in ids}


def load_model(spec):
    fmt = spec.resolved_format()
    if fmt == "gguf":
        from gguf import GGUFReader
        from tokenizers import Tokenizer
        from mini_llm.external import load_gguf_model
        from mini_llm.external.gguf_loader import GGUFWeightMap
        from mini_llm.external.gguf_tokenizer import load_gguf_tokenizer
        inspected = GGUFWeightMap(GGUFReader(str(spec.path)))
        try:
            tokenizer = load_gguf_tokenizer(inspected.reader, vocab_size=inspected.config.vocab_size)
        except ValueError:
            sibling = spec.path.parent / "tokenizer.json"
            if not sibling.is_file():
                raise ValueError("GGUF tokenizer metadata unsupported; supply adjacent tokenizer.json")
            tokenizer = Tokenizer.from_file(str(sibling))
        engine, _ = load_gguf_model(spec.path, dtype=spec.dtype)
        cap = int(engine.config.max_position_embeddings)
        native = False
    elif fmt == "safetensors":
        from tokenizers import Tokenizer
        from mini_llm.external import load_safetensors_model
        tok_path = spec.path / "tokenizer.json"
        if not tok_path.is_file():
            raise ValueError("Safetensors folder lacks tokenizer.json")
        tokenizer = Tokenizer.from_file(str(tok_path))
        engine, _ = load_safetensors_model(spec.path, dtype=spec.dtype)
        cap = int(engine.config.max_position_embeddings)
        native = False
    elif fmt == "native":
        # Preserve checkpoint's full architectural config, weights, native
        # historical-memory router and tokenizer. The regular inference model
        # reuses weight references and is not an independent train model.
        from benchmarks.evaluate_generation import load_generator
        generator, _ = load_generator(spec.path, requested_context=None, runtime_profile=None)
        from mini_llm.inference_model import InferenceModel
        engine = InferenceModel(generator.model.config, dtype=generator.model.dtype)
        engine.set_weights(generator.model)
        tokenizer = generator.tokenizer
        cap = int(generator.max_context)
        memory = getattr(generator.model.config, "memory_context", None)
        if memory is not None and getattr(memory, "enabled", False) and (
            getattr(memory, "integration_mode", None) == "historical_read"
        ):
            # Existing 0083B inference explicitly forbids longer horizons.
            cap = min(cap, int(memory.target_length))
        native = True
        del generator
        gc.collect()
    else:
        raise ValueError(fmt)
    if spec.context_length is not None:
        if int(spec.context_length) > cap:
            raise ValueError(f"Configured context {spec.context_length} exceeds model capacity {cap}")
        cap = int(spec.context_length)
    stops = _native_stops(tokenizer) if native else _external_stops(tokenizer)
    return ServingModel(engine, tokenizer, cap, native=native, stop_ids=stops)
