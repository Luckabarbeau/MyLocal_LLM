"""Sparse expert indexing only: no Qwen4Exp model graph or bank decoding."""

from numbers import Integral
from types import MappingProxyType

from .gguf_store import MultiShardTensorStore
from mini_llm.ops.expert_residency import ExpertKey
from mini_llm.ops.expert_source import ExpertBundle
from mini_llm.ops.quantized_weight import BLOCK_FORMATS, PackedGGUFWeight


class Qwen4ExpGGUFIndex:
    def __init__(self, store: MultiShardTensorStore, *, model_id="qwen4exp"):
        if store.closed:
            raise RuntimeError("GGUF store is closed")
        if not model_id:
            raise ValueError("model_id cannot be empty")
        metadata = store.metadata
        if metadata.get("general.architecture") != "qwen4exp":
            raise ValueError("Expected general.architecture=qwen4exp")
        for attr, name in (("n_layers", "block_count"), ("d_model", "embedding_length"),
                           ("n_experts", "expert_count"), ("top_k", "expert_used_count"),
                           ("d_ff", "expert_feed_forward_length")):
            value = metadata.get("qwen4exp." + name)
            if isinstance(value, bool) or not isinstance(value, Integral) or value <= 0:
                raise ValueError(f"qwen4exp.{name} must be a positive integer")
            setattr(self, attr, int(value))
        if self.top_k > self.n_experts:
            raise ValueError("expert_used_count exceeds expert_count")
        self.store, self.model_id = store, model_id
        descriptors = {}
        for layer in range(self.n_layers):
            expected = {"gate_inp": (self.n_experts, self.d_model),
                        "gate_exps": (self.n_experts, self.d_ff, self.d_model),
                        "up_exps": (self.n_experts, self.d_ff, self.d_model),
                        "down_exps": (self.n_experts, self.d_model, self.d_ff)}
            for suffix, shape in expected.items():
                name = f"blk.{layer}.ffn_{suffix}.weight"
                desc = store.tensors.get(name)
                if desc is None or desc.shape != shape:
                    raise ValueError(f"{name}: expected shape {shape}")
                if suffix != "gate_inp" and BLOCK_FORMATS.get(desc.quant_type) != (
                        desc.format.block_elements, desc.format.block_bytes):
                    raise NotImplementedError(f"{name}: unsupported packed {desc.quant_type}")
                descriptors[name] = desc
        self.descriptors = MappingProxyType(descriptors)

    @classmethod
    def open(cls, path, *, model_id="qwen4exp"):
        store = MultiShardTensorStore.open(path)
        try:
            return cls(store, model_id=model_id)
        except Exception:
            store.close()
            raise

    def source(self, key):
        if not isinstance(key, ExpertKey):
            raise TypeError("key must be ExpertKey")
        if key.model_id != self.model_id:
            raise ValueError("Expert model identity mismatch")
        for value, limit in ((key.layer_id, self.n_layers), (key.expert_id, self.n_experts)):
            if isinstance(value, bool) or not isinstance(value, Integral) or not 0 <= value < limit:
                raise IndexError(key)
        weights = []
        for projection in ("gate", "up", "down"):
            name = f"blk.{key.layer_id}.ffn_{projection}_exps.weight"
            desc = self.descriptors[name]
            raw = self.store.rows(name, 0, desc.shape[-2], outer=(key.expert_id,))
            weights.append(PackedGGUFWeight(raw, desc.shape[1:], desc.quant_type))
        return ExpertBundle(key, tuple(weights), self.store, "mmap")

    def provider(self, key):
        return self.source(key).weights

    def memory_plan(self, *, static, state, activation, cache, staging, headroom, device_total):
        """Validate explicit device reservations; mmap bytes are not RAM allocation."""
        parts = dict(static=static, state=state, activation=activation, cache=cache,
                     staging=staging, headroom=headroom, device_total=device_total)
        for name, value in parts.items():
            if isinstance(value, bool) or not isinstance(value, Integral) or value < 0:
                raise ValueError(f"{name} must be nonnegative integer bytes")
        reserved = sum(value for name, value in parts.items() if name != "device_total")
        if reserved > device_total:
            raise ValueError("Device memory reservations exceed device_total")
        return {**parts, "reserved": reserved, "remaining": device_total - reserved,
                "canonical_mmap_bytes": sum(d.nbytes for d in self.store.tensors.values())}
