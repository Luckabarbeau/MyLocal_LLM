"""0078J: non-invasive accounting of known persistent GPU-array caches.

This is deliberately *not* a generic object-graph walk or a leak detector.
Only known long-lived caches are inspected. Reading ``nbytes`` and cache keys
never transfers array contents from the GPU, and no cached state is modified.
The reported bytes are the nominal sizes of stored arrays, not their allocator
slab sizes; views and allocator fragmentation can make these differ.
"""


def _cuda_array_bytes(value, seen):
    """Sum nominal CUDA array bytes in nested cache payloads without copies."""
    if isinstance(value, dict):
        return sum(_cuda_array_bytes(item, seen) for item in value.values())
    if isinstance(value, (tuple, list)):
        return sum(_cuda_array_bytes(item, seen) for item in value)
    if not hasattr(value, "__cuda_array_interface__"):
        return 0
    identity = id(value)
    if identity in seen:
        return 0
    seen.add(identity)
    return int(value.nbytes)


def persistent_gpu_cache_inventory(model=None):
    """Return {cache name: {entries, cuda_bytes}} for known persistent caches.

    CPU-only tests may still call this function; NumPy arrays are excluded.
    This inventory intentionally does not include parameters, optimizer state,
    transient forward/backward caches, CUDA library workspaces, or the CuPy
    memory pool. It must never be interpreted as total live GPU memory.
    """
    from mini_llm.ops.rope import _ROPE_TABLE_CACHE
    from mini_llm.ops.attention import GQAAttention

    groups = {
        "rope_tables": _ROPE_TABLE_CACHE,
        "causal_attention_masks": GQAAttention._causal_mask_cache,
    }

    router = getattr(model, "memory_router", None) if model is not None else None
    if router is not None:
        groups["routing_metadata"] = getattr(router, "_routing_metadata_cache", {})

    inventory = {}
    for name, cache in groups.items():
        if not isinstance(cache, dict):
            continue
        inventory[name] = {
            "entries": len(cache),
            "cuda_bytes": _cuda_array_bytes(cache, set()),
        }

    # Packed QKV is a deliberately retained compute buffer on each layer.
    # Capture its size separately to distinguish one-time initialization from
    # unexpectedly increasing shape-keyed metadata caches.
    blocks = getattr(model, "blocks", ()) if model is not None else ()
    packed = []
    for block in blocks:
        attention = getattr(block, "attention", None)
        weight = getattr(attention, "_packed_qkv_weight", None)
        if weight is not None:
            packed.append(weight)
    inventory["packed_qkv_weights"] = {
        "entries": len(packed),
        "cuda_bytes": _cuda_array_bytes(packed, set()),
    }
    return inventory
