"""Shared lazy GGUF linear projection for external architecture adapters."""

from mini_llm.backend import BACKEND_NAME, xp
from mini_llm.ops.quantized_linear import (
    DevicePackedGGUFWeight, packed_to_device, quantized_linear, quantized_swiglu,
)
from mini_llm.ops.quantized_weight import BLOCK_FORMATS, PackedGGUFWeight
from mini_llm.ops.silu import silu
from mini_llm.ops.inference_matmul import matmul_inference


def resolve_gguf_weight(index, tensor_name, host_cache, device_cache, cache_key, *,
                        upload=None):
    """Lazily resolve one canonical [out,in] weight using caller-owned caches.

    Packed host views stay packed, with device promotion only on explicit use.
    Dense tensors are materialized once as FP32. Cache keys and optional upload
    injection belong to the adapter, independently of projection execution.
    """
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
    return weight


def _project_resolved_linear(hidden, weight, *, packed, cpu_packed_backend, linear):
    # Dispatch follows the host layout, not the representation returned by an
    # injected uploader (which may intentionally be an opaque test handle).
    if packed:
        linear = quantized_linear if linear is None else linear
        return linear(xp.ascontiguousarray(hidden), weight,
                      cpu_packed_backend=cpu_packed_backend)
    if BACKEND_NAME == 'cupy' and hasattr(hidden, '__cuda_array_interface__'):
        return matmul_inference(hidden, weight.T)
    return hidden @ weight.T


def project_gguf_linear(hidden, index, tensor_name, host_cache, device_cache, cache_key, *,
                        cpu_packed_backend="reference", linear=None, upload=None):
    """Project [B,T,in] by a lazily resolved GGUF [out,in] tensor.

    Optional linear/upload injection preserves adapter instrumentation.
    """
    weight = resolve_gguf_weight(index, tensor_name, host_cache, device_cache, cache_key,
                                 upload=upload)
    return _project_resolved_linear(hidden, weight,
                                    packed=isinstance(host_cache[cache_key], PackedGGUFWeight),
                                    cpu_packed_backend=cpu_packed_backend, linear=linear)


def project_gguf_swiglu(hidden, index, gate_tensor_name, up_tensor_name,
                        host_cache, device_cache, gate_cache_key, up_cache_key, *,
                        cpu_packed_backend="reference", linear=None, upload=None):
    """Fuse device-packed gate/up only for CuPy FP32; retain ordinary fallback.

    Quantization types may differ. Shape and kernel errors are deliberately
    left to the operators rather than retried through a different graph.
    """
    gate = resolve_gguf_weight(index, gate_tensor_name, host_cache, device_cache,
                               gate_cache_key, upload=upload)
    up = resolve_gguf_weight(index, up_tensor_name, host_cache, device_cache,
                             up_cache_key, upload=upload)
    if (BACKEND_NAME == "cupy" and isinstance(hidden, xp.ndarray)
            and hidden.dtype == xp.float32
            and isinstance(gate, DevicePackedGGUFWeight)
            and isinstance(up, DevicePackedGGUFWeight)):
        return quantized_swiglu(xp.ascontiguousarray(hidden), gate, up)
    gate = _project_resolved_linear(hidden, gate,
                                    packed=isinstance(host_cache[gate_cache_key], PackedGGUFWeight),
                                    cpu_packed_backend=cpu_packed_backend, linear=linear)
    up = _project_resolved_linear(hidden, up,
                                  packed=isinstance(host_cache[up_cache_key], PackedGGUFWeight),
                                  cpu_packed_backend=cpu_packed_backend, linear=linear)
    return silu(gate) * up
