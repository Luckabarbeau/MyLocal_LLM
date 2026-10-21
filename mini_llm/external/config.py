"""Strict, read-only configuration for the initial HF Llama inference family.

Avoids passing foreign config JSON directly into native training ModelConfig.
Unsupported architectural features are errors, never implicit fallbacks.
"""

from dataclasses import dataclass
import json
from pathlib import Path


@dataclass(frozen=True)
class ExternalLlamaConfig:
    hidden_size: int
    intermediate_size: int
    num_hidden_layers: int
    num_attention_heads: int
    num_key_value_heads: int
    head_dim: int
    vocab_size: int
    max_position_embeddings: int
    rope_theta: float
    rms_norm_eps: float
    tie_word_embeddings: bool

    @classmethod
    def from_json(cls, path):
        with Path(path).open("r", encoding="utf-8") as stream:
            return cls.from_dict(json.load(stream))

    @classmethod
    def from_dict(cls, data):
        if not isinstance(data, dict):
            raise TypeError("HF model config must be a JSON object")
        if data.get("model_type") != "llama":
            raise ValueError(f"Unsupported model_type: {data.get('model_type')!r}")
        architectures = data.get("architectures", [])
        if architectures and architectures != ["LlamaForCausalLM"]:
            raise ValueError(f"Unsupported HF architectures: {architectures!r}")
        if data.get("hidden_act", "silu") != "silu":
            raise ValueError("Only silu/SwiGLU Llama FFN is supported")
        if data.get("attention_bias", False) or data.get("mlp_bias", False):
            raise ValueError("Llama projection biases are not supported")
        if data.get("rope_scaling") is not None:
            raise ValueError("Scaled/variant RoPE requires a separately validated implementation")
        if data.get("rope_interleaved", False):
            raise ValueError("Interleaved RoPE is not the supported HF Llama layout")
        if int(data.get("pretraining_tp", 1)) != 1:
            raise ValueError("Tensor-parallel pretraining layouts require separate validation")
        if float(data.get("partial_rotary_factor", 1.0)) != 1.0:
            raise ValueError("Partial rotary dimensions are unsupported")
        if data.get("rope_type", "default") != "default":
            raise ValueError("Unsupported RoPE variant")
        required = (
            "hidden_size", "intermediate_size", "num_hidden_layers",
            "num_attention_heads", "vocab_size", "max_position_embeddings",
        )
        missing = [name for name in required if name not in data]
        if missing:
            raise ValueError(f"Missing HF model config fields: {', '.join(missing)}")
        hidden = int(data["hidden_size"])
        qheads = int(data["num_attention_heads"])
        if qheads <= 0:
            raise ValueError("num_attention_heads must be positive")
        head = int(data.get("head_dim", hidden // qheads))
        kvheads = int(data.get("num_key_value_heads", qheads))
        obj = cls(
            hidden_size=hidden,
            intermediate_size=int(data["intermediate_size"]),
            num_hidden_layers=int(data["num_hidden_layers"]),
            num_attention_heads=qheads,
            num_key_value_heads=kvheads,
            head_dim=head,
            vocab_size=int(data["vocab_size"]),
            max_position_embeddings=int(data["max_position_embeddings"]),
            rope_theta=float(data.get("rope_theta", 10000.0)),
            rms_norm_eps=float(data.get("rms_norm_eps", 1e-6)),
            tie_word_embeddings=data.get("tie_word_embeddings", False),
        )
        obj.validate()
        return obj

    def validate(self):
        for name in (
            "hidden_size", "intermediate_size", "num_hidden_layers",
            "num_attention_heads", "num_key_value_heads", "head_dim",
            "vocab_size", "max_position_embeddings",
        ):
            value = getattr(self, name)
            if isinstance(value, bool) or value <= 0:
                raise ValueError(f"{name} must be a positive integer")
        if self.hidden_size != self.num_attention_heads * self.head_dim:
            raise ValueError("Expected hidden_size = num_attention_heads * head_dim")
        if self.num_attention_heads % self.num_key_value_heads:
            raise ValueError("num_attention_heads must be divisible by num_key_value_heads")
        if self.head_dim % 2:
            raise ValueError("RoPE head_dim must be even")
        if self.rope_theta <= 0 or self.rms_norm_eps <= 0:
            raise ValueError("RoPE theta and RMSNorm epsilon must be positive")
        if not isinstance(self.tie_word_embeddings, bool):
            raise TypeError("tie_word_embeddings must be a bool")
