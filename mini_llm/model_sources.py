"""0103C: framework-neutral model source catalogue and serving defaults.

No numerical imports, model allocation or HTTP server dependencies. The
remote model records specify external completion-service connections.
"""

from dataclasses import dataclass, field
import json
import math
from pathlib import Path

@dataclass(frozen=True)
class SamplingConfig:
    """Per-model decoding defaults; API requests override individual values."""

    temperature: float = 0.0
    top_p: float = 1.0
    top_k: int | None = None
    repetition_penalty: float = 1.0
    repetition_window: int = 256
    frequency_penalty: float = 0.0
    presence_penalty: float = 0.0

    @classmethod
    def from_dict(cls, values):
        if not isinstance(values, dict):
            raise ValueError("sampling must be a JSON object")
        unknown = set(values) - set(cls.__dataclass_fields__)
        if unknown:
            raise ValueError(f"Unrecognized sampling fields: {sorted(unknown)}")
        obj = cls(**values)
        for key in ("temperature", "top_p", "repetition_penalty",
                    "frequency_penalty", "presence_penalty"):
            value = getattr(obj, key)
            if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value):
                raise ValueError(f"sampling.{key} must be a finite number")
        if obj.temperature < 0:
            raise ValueError("sampling.temperature must be >= 0")
        if not 0 < obj.top_p <= 1:
            raise ValueError("sampling.top_p must be within (0, 1]")
        if obj.repetition_penalty <= 0:
            raise ValueError("sampling.repetition_penalty must be > 0")
        for key in ("top_k", "repetition_window"):
            value = getattr(obj, key)
            if key == "top_k" and value is None:
                continue
            if type(value) is not int or (value < 1 if key == "top_k" else not 0 <= value <= 65536):
                raise ValueError(f"sampling.{key} is not a valid integer")
        for key in ("frequency_penalty", "presence_penalty"):
            if not -2 <= getattr(obj, key) <= 2:
                raise ValueError(f"sampling.{key} must be between -2 and 2")
        return obj

    def with_overrides(self, overrides):
        return type(self).from_dict({**self.__dict__, **overrides})


@dataclass(frozen=True)
class ModelSpec:
    id: str
    path: Path | None = None
    format: str = "auto"
    dtype: str = "float16"
    chat_template: str = "chatml"
    tool_format: str = "none"
    context_length: int | None = None
    execution_policy: str = "auto"
    sampling: SamplingConfig = field(default_factory=SamplingConfig)
    # Optional external completion provider (0103F).
    api_url: str | None = None
    remote_model: str | None = None
    api_key_env: str | None = None

    @classmethod
    def from_dict(cls, obj, root):
        allowed = set(cls.__dataclass_fields__)
        unknown = set(obj) - allowed
        if unknown:
            raise ValueError(f"Unrecognized model configuration fields: {sorted(unknown)}")
        if not obj.get("id"):
            raise ValueError("Each model must have a nonempty id")
        fmt = obj.get("format", "auto")
        if fmt == "remote":
            if obj.get("path"):
                raise ValueError("Remote model must use api_url, not a local path")
            from urllib.parse import urlsplit
            url = obj.get("api_url", "")
            parsed = urlsplit(url)
            if (parsed.scheme not in {"http", "https"} or not parsed.hostname
                or parsed.username or parsed.password or parsed.query or parsed.fragment):
                raise ValueError("Remote model requires an http(s) api_url without embedded credentials")
            if not obj.get("remote_model"):
                raise ValueError("Remote model requires remote_model")
            env_name = obj.get("api_key_env")
            if env_name is not None and (not isinstance(env_name, str) or
                                          not env_name.isidentifier()):
                raise ValueError("api_key_env must be an environment-variable name")
            p = None
        else:
            if not obj.get("path"):
                raise ValueError("Local model requires a nonempty path")
            if obj.get("api_url") or obj.get("remote_model") or obj.get("api_key_env"):
                raise ValueError("Remote connection fields require format='remote'")
            p = Path(obj["path"]).expanduser()
            if not p.is_absolute():
                p = Path(root) / p
            p = p.resolve()
        kw = dict(obj, path=p, sampling=SamplingConfig.from_dict(obj.get("sampling", {})))
        spec = cls(**kw)
        if spec.format not in {"auto", "gguf", "safetensors", "native", "remote"}:
            raise ValueError(f"Unsupported model format: {spec.format}")
        if spec.dtype not in {"float16", "float32"}:
            raise ValueError("Serving currently supports float16 and float32 (native uses checkpoint dtype)")
        if spec.chat_template not in {"chatml", "plain"}:
            raise ValueError(f"Unsupported chat template: {spec.chat_template}")
        if spec.tool_format not in {"none", "json_envelope", "native"}:
            raise ValueError(f"Unsupported tool format: {spec.tool_format}")
        if spec.tool_format == "native" and spec.format != "remote":
            raise ValueError("Native upstream tool calling requires format='remote'")
        if spec.execution_policy not in {"auto", "reference", "balanced", "fast", "strict-fast"}:
            raise ValueError("Invalid execution_policy")
        if spec.context_length is not None and spec.context_length <= 0:
            raise ValueError("context_length must be positive")
        return spec

    def resolved_format(self):
        if self.format != "auto":
            return self.format
        if self.path is None:
            raise ValueError("Local model format detection requires a path")
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


class ModelCatalog:
    """Resolved manifest with distinct model IDs, independent of HTTP serving."""

    def __init__(self, specs):
        self._specs = dict(specs)
        if len(self._specs) != len(specs):
            raise ValueError("Duplicate model IDs")

    @classmethod
    def from_manifest(cls, manifest):
        return cls(load_specs(manifest))

    def names(self):
        return tuple(self._specs)

    def get(self, model_id):
        try:
            return self._specs[model_id]
        except KeyError as exc:
            raise KeyError(f"Unknown registered model: {model_id}") from exc

    def as_dict(self):
        return dict(self._specs)
