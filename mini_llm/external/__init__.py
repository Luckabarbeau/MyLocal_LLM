"""Format-independent, inference-only external architecture adapters."""

from .config import ExternalLlamaConfig
from .llama_model import ExternalLlamaModel
from .safetensors_loader import load_safetensors_model, SafetensorsWeightMap
from .gguf_loader import load_gguf_model, GGUFWeightMap

__all__ = ["ExternalLlamaConfig", "ExternalLlamaModel",
           "load_safetensors_model", "SafetensorsWeightMap",
           "load_gguf_model", "GGUFWeightMap"]
