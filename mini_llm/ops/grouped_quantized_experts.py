"""Single-row FP32 grouped packed expert inference, without dense weight copies.

The caller supplies handles/routing weights in ascending expert-ID order (handles
have no ID field). Repeated handles are legal and remain separate ordered terms.
Only an E-by-3 (or E-by-4 with routing slots) uint64 table is uploaded;
packed weights stay in place. Optional route-slot indirection avoids a separate
GPU routing-weight permutation without changing expert accumulation order.
No cache, model, CUDA context, or backend selection is created by this module.
"""

from typing import Any

import numpy as np

from mini_llm.ops import quantized_linear as _ql
from mini_llm.ops.quantized_linear import (
    DevicePackedGGUFWeight,
    PackedExpertFFNInference,
    _triton_cupy_pointer as as_triton_tensor,
)
from mini_llm.ops.quantized_weight import BLOCK_FORMATS


if _ql.triton is not None:
    import triton
    import triton.language as tl
    # Reuse the exact decoder and fused activation graph, including GEMV tiling.
    _load_gguf_lanes = _ql._load_gguf_lanes
    _swiglu = _ql._packed_gate_up_swiglu_kernel

    @triton.jit
    def _grouped_gate_up(
        X, TABLE, H, D: tl.constexpr, F: tl.constexpr,
        GQ: tl.constexpr, GB: tl.constexpr, GF: tl.constexpr,
        UQ: tl.constexpr, UB: tl.constexpr, UF: tl.constexpr,
        BK: tl.constexpr, TABLE_WIDTH: tl.constexpr = 3,
    ):
        expert = tl.program_id(2)
        pointers = TABLE.to(tl.pointer_type(tl.uint64))
        gate = tl.load(pointers + expert * TABLE_WIDTH).to(tl.pointer_type(tl.uint8))
        up = tl.load(pointers + expert * TABLE_WIDTH + 1).to(tl.pointer_type(tl.uint8))
        _swiglu(X, gate, up, H + expert * F, D, F,
                GQ, GB, GF, UQ, UB, UF, BK)

    @triton.jit
    def _grouped_down_weighted(
        H, TABLE, ROUTES, TERMS, D: tl.constexpr, F: tl.constexpr,
        RSTRIDE: tl.constexpr, QK: tl.constexpr, QB: tl.constexpr,
        FORMAT: tl.constexpr, BK: tl.constexpr, TABLE_WIDTH: tl.constexpr = 3,
    ):
        row = tl.program_id(0)
        expert = tl.program_id(2)
        pointers = TABLE.to(tl.pointer_type(tl.uint64))
        down = tl.load(pointers + expert * TABLE_WIDTH + 2).to(tl.pointer_type(tl.uint8))
        lane = tl.arange(0, BK)
        acc = tl.full((BK,), 0.0, tl.float32)
        for tile in range(tl.cdiv(F, BK)):
            col = tile * BK + lane
            h = tl.load(H + expert * F + col, col < F, 0)
            w = _load_gguf_lanes(down, col, row, F, QK, QB, FORMAT)
            acc += h * w
        result = tl.sum(acc, axis=0)
        route_slot = expert
        if TABLE_WIDTH == 4:
            route_slot = tl.load(pointers + expert * TABLE_WIDTH + 3).to(tl.int32)
        route = tl.load(ROUTES + route_slot * RSTRIDE)
        # A real FP32 storage boundary prevents contraction with the later add.
        tl.store(TERMS + expert * D + row, result * route)

    @triton.jit
    def _ordered_reduce(TERMS, OUT, D: tl.constexpr, E: tl.constexpr,
                        BLOCK: tl.constexpr):
        col = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
        acc = tl.full((BLOCK,), 0.0, tl.float32)
        # Do not reduce the expert axis as a tree, even for duplicate handles.
        for expert in range(E):
            term = tl.load(TERMS + expert * D + col, col < D, 0)
            acc = acc + term
        tl.store(OUT + col, acc, col < D)


def _unsupported_reason(x, experts, routing_weights, block_k, routing_slots=None):
    if block_k not in (256, 512, 1024):
        return "block_k must be 256, 512 or 1024"
    if _ql.BACKEND_NAME != "cupy" or _ql.triton is None:
        return "requires the CuPy backend and Triton"
    cp: Any = _ql.xp
    if not isinstance(x, cp.ndarray) or x.dtype != cp.float32:
        return "x must be a CuPy FP32 array"
    if x.ndim != 2 or x.shape[0] != 1 or x.shape[1] <= 0 or not x.flags.c_contiguous:
        return "x must be contiguous [1, D] with positive D"
    if not isinstance(experts, (list, tuple)) or not 1 <= len(experts) <= 65535:
        return "experts must be a nonempty list/tuple of at most 65535 handles"
    if routing_slots is not None and (
            not isinstance(routing_slots, (tuple, list)) or len(routing_slots) != len(experts)
            or any(not isinstance(slot, (int, np.integer)) or isinstance(slot, bool)
                   or not 0 <= slot < len(experts) for slot in routing_slots)):
        return "routing_slots must be host integer indices into the [E] routing weights"
    if (not isinstance(routing_weights, cp.ndarray)
            or routing_weights.dtype != cp.float32
            or routing_weights.shape != (len(experts),)):
        return "routing_weights must be a CuPy FP32 [E] array (slices allowed)"
    device = x.device.id
    if routing_weights.device.id != device or cp.cuda.runtime.getDevice() != device:
        return "all operands must be on the current CUDA device"
    signature = None
    for expert in experts:
        if not isinstance(expert, PackedExpertFFNInference):
            return "experts must contain PackedExpertFFNInference handles"
        projections = (expert.gate, expert.up, expert.down)
        if not all(isinstance(w, DevicePackedGGUFWeight) for w in projections):
            return "all projections must be DevicePackedGGUFWeight (no host fallback)"
        gate, up, down = projections
        if (len(gate.shape) != 2 or min(gate.shape) <= 0
                or gate.shape != up.shape or gate.shape[1] != x.shape[1]
                or down.shape != (x.shape[1], gate.shape[0])):
            return "expert shapes must be gate/up [F, D], down [D, F] matching x"
        current = tuple((w.shape, w.quant_type) for w in projections)
        if signature is not None and current != signature:
            return "all experts must share per-projection shapes and quantization formats"
        signature = current
        for w in projections:
            if w.quant_type not in _ql._FORMAT_ID:
                return "unsupported packed quantization format"
            qk, qb = BLOCK_FORMATS[w.quant_type]
            if (not isinstance(w.data, cp.ndarray) or w.data.dtype != cp.uint8
                    or not w.data.flags.c_contiguous or w.shape[1] % qk
                    or w.data.size != w.shape[0] * w.shape[1] // qk * qb):
                return "invalid packed projection storage or block alignment"
            if w.data.device.id != device:
                return "all operands must be on the current CUDA device"
    return None


def supports_grouped_quantized_experts(x, experts, routing_weights, *, block_k=512,
                                       routing_slots=None) -> bool:
    """Metadata-only support check; no allocation, transfers, or GPU kernels.

    All handles share each projection's shape/format, but gate/up/down formats
    may differ from one another. FP32 routes may be strided, including reversed
    or broadcast views. Ordering is a caller contract, not inferred from handles.
    """
    return _unsupported_reason(x, experts, routing_weights, block_k, routing_slots) is None


def _launch(kernel, grid, *args, **options):
    compiled = kernel.warmup(*args, grid=grid, num_warps=4, **options)
    cp: Any = _ql.xp
    compiled[grid](*args, stream=int(cp.cuda.get_current_stream().ptr))


def grouped_quantized_experts_forward(x, experts, routing_weights, *, block_k=512,
                                      routing_slots=None) -> Any:
    """Return a fresh FP32 [1, D] weighted expert sum on CuPy's current stream.

    Graph: packed FP32 gate/up GEMVs -> stored FP32 SwiGLU -> FP32 down
    GEMV -> stored FP32 multiply by route -> ascending, sequential FP32 adds
    starting at zero. No FMA or tree reduction across experts. The handles'
    fuse_gate_up flags do not alter this explicitly fused graph. Inputs and
    handles are never mutated. Unsupported calls raise ValueError, not fallback.

    routing_slots optionally maps each ordered handle to its original route
    position. It is a host tuple/list; duplicate slots are allowed. The down
    kernel reads routes by slot, avoiding a separate gather/concatenation.

    Temporary storage: uint64 [E, 3 or 4] metadata, FP32 [E, F] activations and
    FP32 [E, D] weighted terms, plus the returned FP32 [1, D]. The caller must
    retain resident weight ownership until work on the current stream completes,
    as for existing packed inference (no cross-stream residency management).
    """
    reason = _unsupported_reason(x, experts, routing_weights, block_k, routing_slots)
    if reason is not None:
        raise ValueError(f"Unsupported grouped quantized experts: {reason}")
    cp: Any = _ql.xp
    e, d, f = len(experts), x.shape[1], experts[0].gate.shape[0]
    host_pointers = np.asarray([
        [int(w.data.data.ptr) for w in (h.gate, h.up, h.down)] for h in experts
    ], dtype=np.uint64)
    table_width = 3 if routing_slots is None else 4
    if routing_slots is not None:
        host_pointers = np.column_stack((host_pointers, np.asarray(routing_slots, dtype=np.uint64)))
    table = cp.asarray(host_pointers)
    # Existing pointer bridge supports uint8, not uint64. This zero-copy view
    # keeps the original allocation; kernels reinterpret it as a uint64 table.
    table_arg = as_triton_tensor(table.view(cp.uint8))
    hidden = cp.empty((e, f), dtype=cp.float32)
    terms = cp.empty((e, d), dtype=cp.float32)
    out = cp.empty((1, d), dtype=cp.float32)
    gate, up, down = experts[0].gate, experts[0].up, experts[0].down
    gq, gb = BLOCK_FORMATS[gate.quant_type]
    uq, ub = BLOCK_FORMATS[up.quant_type]
    dq, db = BLOCK_FORMATS[down.quant_type]
    _launch(_grouped_gate_up, (1, f, e), as_triton_tensor(x), table_arg,
            as_triton_tensor(hidden), d, f,
            gq, gb, _ql._FORMAT_ID[gate.quant_type],
            uq, ub, _ql._FORMAT_ID[up.quant_type], block_k, TABLE_WIDTH=table_width)
    # A strided route needs the same pointer protocol but not the bridge's
    # contiguous check. A one-element view retains its exact base pointer.
    routes_arg = as_triton_tensor(routing_weights[:1])
    _launch(_grouped_down_weighted, (d, 1, e), as_triton_tensor(hidden),
            table_arg, routes_arg, as_triton_tensor(terms), d, f,
            routing_weights.strides[0] // routing_weights.itemsize,
            dq, db, _ql._FORMAT_ID[down.quant_type], block_k, TABLE_WIDTH=table_width)
    _launch(_ordered_reduce, (triton.cdiv(d, 256), 1, 1),
            as_triton_tensor(terms), as_triton_tensor(out), d, e, 256,
            enable_fp_fusion=False)
    return out
