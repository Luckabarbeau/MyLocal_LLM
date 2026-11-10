"""Shared lazy GGUF linear projection for external architecture adapters."""

from mini_llm.backend import BACKEND_NAME, xp
from mini_llm.ops.quantized_linear import packed_to_device, quantized_linear
from mini_llm.ops.quantized_weight import BLOCK_FORMATS, PackedGGUFWeight


def project_gguf_linear(hidden, index, tensor_name, host_cache, device_cache, cache_key, *,
                        cpu_packed_backend="reference", linear=None, upload=None):
    """Project [B,T,in] by one canonical GGUF [out,in] tensor.

    The adapter validates its architecture's headers first and owns the index
    and caches. Packed host weights stay packed; device promotion occurs only
    on explicit projection use. Dense tensors are materialized once as FP32.
    Optional operator injection preserves existing adapter instrumentation.
    """
    linear = quantized_linear if linear is None else linear
    upload = packed_to_device if upload is None else upload
    weight = host_cache.get(cache_key)
    if weight is None:
        tensor = index.tensors[tensor_name]
        kind = tensor.tensor_type.name
        if kind in BLOCK_FORMATS:
            shape = tuple(int(d) for d in reversed(tensor.shape))
            weight = PackedGGUFWeight(tensor.data, shape, kind)
        else:
            # Keep adapter-local flags independent even when asarray aliases F32 storage.
            weight = xp.asarray(index.non_expert_weight(tensor_name), dtype=xp.float32).view()
        host_cache[cache_key] = weight
    if isinstance(weight, PackedGGUFWeight):
        if BACKEND_NAME == "cupy":
            if cache_key not in device_cache:
                device_cache[cache_key] = upload(weight)
            weight = device_cache[cache_key]
        return linear(xp.ascontiguousarray(hidden), weight,
                      cpu_packed_backend=cpu_packed_backend)
    return hidden @ weight.T
