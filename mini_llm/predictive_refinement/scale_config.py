"""Short-window *trunk-scale* predictive-refinement configurations.

This is deliberately not the 4K/64K routed-memory production architecture.
The current predictive-refinement kernels support only static attention heads;
the two learned-retrieval heads in the wide preset are replaced with bounded
local heads. The original 12-layer MoE dimensions and top-2 routing are kept.
"""

from dataclasses import dataclass, replace

from mini_llm.config import LocalAttentionConfig, ModelConfig, AttentionLayerConfig


WIDE_TRUNK_NAME = "wide-500m-static-trunk"


def wide_static_trunk_config(*, dtype="bfloat16", context_length=256):
    """Keep true 500M trunk geometry; isolate unsupported retrieval/64K routing."""
    original = ModelConfig.wide_500m_context_4k()
    layers = []
    for layer in original.attention_layers:
        heads = tuple(
            LocalAttentionConfig(window=128) if head.kind == "retrieval" else head
            for head in layer.heads
        )
        layers.append(AttentionLayerConfig(heads=heads))
    return replace(
        original, context_length=int(context_length), dtype=dtype,
        attention_layers=tuple(layers),
    )


def parameter_memory_gib(parameter_count, dtype):
    """Lower-bound resident model/gradient/optimizer storage; excludes activations."""
    n = int(parameter_count)
    bytes_weight = 2 if str(dtype) in ("bfloat16", "float16") else 4
    gib = 1024 ** 3
    return {
        "weights": n * bytes_weight / gib,
        "fp32_grads": n * 4 / gib,
        "fp32_master": n * 4 / gib,
        "fp32_adam_moments": n * 8 / gib,
        "no_offload_total": n * (bytes_weight + 16) / gib,
        "full_offload_gpu_lower_bound": n * (bytes_weight + 4) / gib,
    }
