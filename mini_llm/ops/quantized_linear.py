"""Quantized GGUF linear inference operator for the shared NumPy/CuPy backend.

Execution contract:
    y[..., N] = x[..., K] @ W[N, K].T

The CuPy/Triton decode GEMV operates on packed Q8_0/Q4_0/Q3_K/Q4_K/Q5_K
bytes *directly*, reconstructing only lane-local dequantized values inside the
kernel. No floating-point weight matrix is allocated on the GPU.  The NumPy
route deliberately uses the small-matrix reference dequantizer; it is for
correctness and limited CPU fallback, not the production offload fast path.

This first kernel specializes for autoregressive decode / small token batches;
large-prefill GEMM and fast quantized CPU kernels are separate milestones.
"""

from dataclasses import dataclass
import math

import numpy as np

from mini_llm.backend import BACKEND_NAME, xp
from mini_llm.ops.quantized_weight import BLOCK_FORMATS, PackedGGUFWeight

try:
    import triton
    import triton.language as tl
except ImportError:
    triton = None
    tl = None


@dataclass(frozen=True)
class DevicePackedGGUFWeight:
    """GPU-resident packed tensor; shares the PackedGGUFWeight layout contract."""

    data: object
    shape: tuple[int, ...]
    quant_type: str

    def __post_init__(self):
        if BACKEND_NAME != "cupy":
            raise RuntimeError("DevicePackedGGUFWeight requires MINI_LLM_BACKEND=cupy")
        if not isinstance(self.data, xp.ndarray) or self.data.dtype != xp.uint8:
            raise TypeError("Device packed bytes must be a CuPy uint8 array")
        if not self.data.flags.c_contiguous:
            raise ValueError("Device packed bytes must be contiguous")
        if self.quant_type not in BLOCK_FORMATS:
            raise ValueError(f"Unsupported GGUF type {self.quant_type}")
        shape = tuple(int(i) for i in self.shape)
        if len(shape) != 2 or min(shape) <= 0:
            raise ValueError("Device packed linear weights must be [N, K]")
        elems, block_bytes = BLOCK_FORMATS[self.quant_type]
        if shape[-1] % elems or int(self.data.size) != math.prod(shape) // elems * block_bytes:
            raise ValueError("Invalid packed byte length or K block alignment")
        object.__setattr__(self, "shape", shape)

    @property
    def nbytes(self):
        return int(self.data.nbytes)


def packed_to_device(weight: PackedGGUFWeight):
    """Transfer *packed bytes* of one matrix/expert to CuPy VRAM once."""
    if BACKEND_NAME != "cupy":
        raise RuntimeError("Packed GPU weights require MINI_LLM_BACKEND=cupy")
    if not isinstance(weight, PackedGGUFWeight) or len(weight.shape) != 2:
        raise TypeError("Expected a two-dimensional PackedGGUFWeight")
    return DevicePackedGGUFWeight(xp.asarray(weight.data, dtype=xp.uint8),
                                  weight.shape, weight.quant_type)


if triton is not None:
    @triton.jit
    def _load_gguf_lanes(
        W, col, row, K: tl.constexpr,
        QK: tl.constexpr, QBYTES: tl.constexpr, FORMAT: tl.constexpr,
    ):
        """Common GGML block decoder used by GEMV and fused expert kernels.

        Decode only one vector tile in registers; never materialize dense W.
        Keep format offsets in one place to prevent divergent implementations.
        """
        valid = col < K
        within = col % QK
        block = row * (K // QK) * QBYTES + col // QK * QBYTES
        if FORMAT == 0:  # Q8_0: 2-byte FP16 scale + 32 signed int8
            lo = tl.load(W + block, valid, 0).to(tl.uint16)
            hi = tl.load(W + block + 1, valid, 0).to(tl.uint16)
            scale = (lo | (hi << 8)).to(tl.float16, bitcast=True).to(tl.float32)
            q = tl.load(W + block + 2 + within, valid, 0).to(tl.int32)
            signed = tl.where(q >= 128, q - 256, q)
            wv = scale * signed.to(tl.float32)
        elif FORMAT == 1:  # Q4_0: first 16 low nibbles, next 16 high
            lo = tl.load(W + block, valid, 0).to(tl.uint16)
            hi = tl.load(W + block + 1, valid, 0).to(tl.uint16)
            scale = (lo | (hi << 8)).to(tl.float16, bitcast=True).to(tl.float32)
            q = tl.load(W + block + 2 + within % 16, valid, 0).to(tl.int32)
            v = ((q >> tl.where(within < 16, 0, 4)) & 15) - 8
            wv = scale * v.to(tl.float32)
        elif FORMAT == 2 or FORMAT == 4:  # Q4_K/Q5_K: shared 8 scale/min groups
            lo = tl.load(W + block, valid, 0).to(tl.uint16)
            hi = tl.load(W + block + 1, valid, 0).to(tl.uint16)
            d = (lo | (hi << 8)).to(tl.float16, bitcast=True).to(tl.float32)
            lo_m = tl.load(W + block + 2, valid, 0).to(tl.uint16)
            hi_m = tl.load(W + block + 3, valid, 0).to(tl.uint16)
            dmin = (lo_m | (hi_m << 8)).to(tl.float16, bitcast=True).to(tl.float32)
            group = within // 32
            sb = tl.load(W + block + 4 + tl.where(group < 4, group, group + 4),
                         valid, 0).to(tl.int32)
            mb = tl.load(W + block + 4 + tl.where(group < 4, group + 4, group + 4),
                         valid, 0).to(tl.int32)
            hsb = tl.load(W + block + 4 + group - 4, valid & (group >= 4), 0).to(tl.int32)
            hmb = tl.load(W + block + 4 + group, valid & (group >= 4), 0).to(tl.int32)
            sc = tl.where(group < 4, sb & 63, (sb & 15) | ((hsb >> 2) & 48))
            mn = tl.where(group < 4, mb & 63, (mb >> 4) | ((hmb >> 2) & 48))
            q_idx = (within // 64) * 32 + within % 32
            if FORMAT == 4:  # Q5_K high-bit planes are between scales and qs.
                q = tl.load(W + block + 48 + q_idx, valid, 0).to(tl.int32)
                hb = tl.load(W + block + 16 + within % 32, valid, 0).to(tl.int32)
                high = ((hb >> (within // 32)) & 1) << 4
            else:
                q = tl.load(W + block + 16 + q_idx, valid, 0).to(tl.int32)
                high = 0
            v = ((q >> ((within % 64) // 32 * 4)) & 15) | high
            wv = d * sc.to(tl.float32) * v.to(tl.float32) - dmin * mn.to(tl.float32)
        else:  # FORMAT == 3, Q3_K: 2 low bits + high/sign mask
            lo = tl.load(W + block + 108, valid, 0).to(tl.uint16)
            hi = tl.load(W + block + 109, valid, 0).to(tl.uint16)
            d = (lo | (hi << 8)).to(tl.float16, bitcast=True).to(tl.float32)
            q = tl.load(W + block + 32 + within // 128 * 32 + within % 32,
                        valid, 0).to(tl.int32)
            v = (q >> (2 * (within % 128 // 32))) & 3
            mask = tl.load(W + block + within % 32, valid, 0).to(tl.int32)
            signed = v - 4 * (1 - ((mask >> (within // 32)) & 1))
            sg = within // 16
            sb = tl.load(W + block + 96 + sg % 8, valid, 0).to(tl.int32)
            hb = tl.load(W + block + 104 + sg % 4, valid, 0).to(tl.int32)
            sc = (((sb >> (4 * (sg // 8))) & 15)
                  | (((hb >> (2 * (sg // 4))) & 3) << 4)) - 32
            wv = d * sc.to(tl.float32) * signed.to(tl.float32)
        return tl.where(valid, wv, 0.0)

    @triton.jit
    def _packed_linear_decode_kernel(
        X, W, Y,
        K: tl.constexpr, N: tl.constexpr, M: tl.constexpr,
        QK: tl.constexpr, QBYTES: tl.constexpr,
        FORMAT: tl.constexpr, BK: tl.constexpr,
    ):
        token = tl.program_id(0)
        row = tl.program_id(1)
        lane = tl.arange(0, BK)
        acc = tl.full((BK,), 0.0, tl.float32)
        for tile in range(tl.cdiv(K, BK)):
            col = tile * BK + lane
            xv = tl.load(X + token * K + col, col < K, 0).to(tl.float32)
            wv = _load_gguf_lanes(W, col, row, K, QK, QBYTES, FORMAT)
            acc += xv * wv
        tl.store(Y + token * N + row, tl.sum(acc, axis=0))

    @triton.jit
    def _packed_gate_up_swiglu_kernel(
        X, WG, WU, H,
        K: tl.constexpr, F: tl.constexpr,
        G_QK: tl.constexpr, G_QBYTES: tl.constexpr, G_FORMAT: tl.constexpr,
        U_QK: tl.constexpr, U_QBYTES: tl.constexpr, U_FORMAT: tl.constexpr,
        BK: tl.constexpr,
    ):
        """One launch for both input projections and SiLU(gate)*up.

        One program produces one intermediate expert feature for one token.
        FP32 accumulation/activation; H has the original activation dtype.
        """
        token = tl.program_id(0)
        row = tl.program_id(1)
        lane = tl.arange(0, BK)
        acc_gate = tl.full((BK,), 0.0, tl.float32)
        acc_up = tl.full((BK,), 0.0, tl.float32)
        for tile in range(tl.cdiv(K, BK)):
            col = tile * BK + lane
            xv = tl.load(X + token * K + col, col < K, 0).to(tl.float32)
            gate_weight = _load_gguf_lanes(WG, col, row, K, G_QK, G_QBYTES, G_FORMAT)
            up_weight = _load_gguf_lanes(WU, col, row, K, U_QK, U_QBYTES, U_FORMAT)
            acc_gate += xv * gate_weight
            acc_up += xv * up_weight
        gate = tl.sum(acc_gate, axis=0)
        up = tl.sum(acc_up, axis=0)
        tl.store(H + token * F + row, (gate * tl.sigmoid(gate)) * up)


class _TritonCuPyPointer:
    """Triton's tensor protocol for existing CuPy arrays (no copies).

    Unlike PyTorch, CuPy ndarrays do not expose ``data_ptr()``. The native
    MyLocal_LLM Triton kernels already use this adapter pattern. Keep a strong
    reference to the backing allocation for the duration of the launch.
    """

    def __init__(self, array, dtype):
        self.array = array
        self.dtype = dtype

    def data_ptr(self):
        return int(self.array.data.ptr)


def _triton_cupy_pointer(array):
    """Wrap a CuPy float activation/output or a packed uint8 GGUF payload.

    All operands for quantized Triton kernels must be real CuPy allocations;
    do not accidentally accept NumPy arrays or force a host/device transfer.
    """
    if BACKEND_NAME != "cupy" or not isinstance(array, xp.ndarray):
        raise TypeError("Triton quantized kernel operands must be CuPy arrays")
    if tl is None:
        raise RuntimeError("Triton is required for packed GPU inference")
    if not array.flags.c_contiguous:
        raise ValueError("Triton quantized kernel operands must be contiguous")
    types = {
        "float16": tl.float16,
        "float32": tl.float32,
        "bfloat16": tl.bfloat16,
        "uint8": tl.uint8,
    }
    dtype = types.get(str(array.dtype))
    if dtype is None:
        raise TypeError(f"Unsupported Triton quantized operand dtype: {array.dtype}")
    return _TritonCuPyPointer(array, dtype)


_FORMAT_ID = {"Q8_0": 0, "Q4_0": 1, "Q4_K": 2, "Q3_K": 3, "Q5_K": 4}


def quantized_linear(x, weight, *, block_k=512, cpu_packed_backend="reference"):
    """Compute x @ weight.T for packed GGUF [out_features, in_features].

    Host PackedGGUFWeight uses reference CPU dequantization. CuPy requires a
    DevicePackedGGUFWeight and Triton, and never dequantizes the matrix. This
    explicit distinction prevents an accidental GPU full-matrix expansion.

    For now x must be contiguous, 2D or 3D, and have a floating-point dtype.
    """
    if not isinstance(x, (np.ndarray,)) and (
        BACKEND_NAME != "cupy" or not isinstance(x, xp.ndarray)
    ):
        raise TypeError("Packed linear input must be a NumPy or CuPy array")
    if x.ndim not in (2, 3):
        raise ValueError("Expected x as [M, K] or [B, T, K]")
    if x.shape[-1] != weight.shape[-1]:
        raise ValueError("Input K does not match packed weight K")
    if len(weight.shape) != 2:
        raise ValueError("Packed linear weights must be [N, K]")
    shape = tuple(x.shape[:-1]) + (weight.shape[0],)
    if isinstance(weight, PackedGGUFWeight):
        if not isinstance(x, np.ndarray):
            raise TypeError("Packed host weight requires a NumPy input; no GPU dequantization fallback")
        x_cpu = np.asarray(x)
        if x_cpu.dtype not in (np.float16, np.float32, np.float64):
            raise TypeError("Reference CPU inference requires FP16/FP32/FP64 input")
        if cpu_packed_backend != "reference":
            from mini_llm.ops.native_packed_cpu import packed_cpu_linear
            native = packed_cpu_linear(x_cpu, weight, backend=cpu_packed_backend)
            if native is not None:
                return native
        w = weight.dequantize_reference()
        y = np.asarray(x_cpu, dtype=np.float32).reshape(-1, x.shape[-1]) @ w.T
        return y.astype(x_cpu.dtype).reshape(shape)
    if not isinstance(weight, DevicePackedGGUFWeight):
        raise TypeError("Expected PackedGGUFWeight or DevicePackedGGUFWeight")
    if BACKEND_NAME != "cupy" or triton is None:
        raise RuntimeError("Native packed GPU inference requires CuPy backend and Triton")
    if not isinstance(x, xp.ndarray) or not x.flags.c_contiguous:
        raise ValueError("CuPy input must be a contiguous GPU array")
    if str(x.dtype) not in ("float16", "float32", "bfloat16"):
        raise TypeError("GPU input must be FP16, BF16, or FP32")
    if block_k not in (256, 512, 1024):
        raise ValueError("block_k must be 256, 512 or 1024")
    n, k = weight.shape
    m = int(x.size) // k
    out = xp.empty((m, n), dtype=x.dtype)
    if m:
        qk, qb = BLOCK_FORMATS[weight.quant_type]
        _packed_linear_decode_kernel[(m, n)](
            _triton_cupy_pointer(x), _triton_cupy_pointer(weight.data),
            _triton_cupy_pointer(out), k, n, m, qk, qb,
            _FORMAT_ID[weight.quant_type], block_k,
            num_warps=4,
        )
    return out.reshape(shape)


def quantized_swiglu(x, gate, up, *, block_k=512):
    """Compute SiLU(x @ gate.T) * (x @ up.T) with packed GGUF weights.

    On GPU, one Triton launch replaces two packed GEMVs plus the separate
    activation/multiplication; neither projection is stored as an intermediate.
    The CPU route is an intentionally simple correctness/reference fallback.
    All operations remain inference-only (no backward or training changes).
    Different GGUF quantization types for gate and up are supported.
    """
    if not isinstance(gate, (PackedGGUFWeight, DevicePackedGGUFWeight)) or not isinstance(
        up, (PackedGGUFWeight, DevicePackedGGUFWeight)
    ):
        raise TypeError("gate and up must be packed GGUF weights")
    if type(gate) is not type(up):
        raise TypeError("gate and up must both be host-packed or both device-packed")
    if len(gate.shape) != 2 or up.shape != gate.shape:
        raise ValueError("gate/up must share two-dimensional [F,K] shape")
    if not isinstance(x, np.ndarray) and (BACKEND_NAME != "cupy" or not isinstance(x, xp.ndarray)):
        raise TypeError("Quantized SwiGLU input must be a NumPy or CuPy array")
    if x.ndim not in (2, 3):
        raise ValueError("Expected x as [M,K] or [B,T,K]")
    f, k = gate.shape
    if x.shape[-1] != k:
        raise ValueError("Input K does not match gate/up K")
    if isinstance(gate, PackedGGUFWeight):
        if not isinstance(x, np.ndarray):
            raise TypeError("Host packed gate/up require NumPy activations")
        # Avoid implementing a separate CPU decoder; reuse the shared linear
        # correctness oracle and ensure FP32 activation accumulation.
        g = quantized_linear(x, gate)
        u = quantized_linear(x, up)
        gf = np.asarray(g, dtype=np.float32)
        return ((gf / (1.0 + np.exp(-gf))) * np.asarray(u, dtype=np.float32)).astype(x.dtype)
    if BACKEND_NAME != "cupy" or triton is None:
        raise RuntimeError("Fused packed GPU SwiGLU requires CuPy backend and Triton")
    if not isinstance(x, xp.ndarray) or not x.flags.c_contiguous:
        raise ValueError("GPU activations must be a contiguous CuPy array")
    if str(x.dtype) not in ("float16", "float32", "bfloat16"):
        raise TypeError("GPU activations must be FP16, BF16, or FP32")
    if block_k not in (256, 512, 1024):
        raise ValueError("block_k must be 256, 512 or 1024")
    m = int(x.size) // k
    out = xp.empty((m, f), dtype=x.dtype)
    if m:
        gq, gb = BLOCK_FORMATS[gate.quant_type]
        uq, ub = BLOCK_FORMATS[up.quant_type]
        _packed_gate_up_swiglu_kernel[(m, f)](
            _triton_cupy_pointer(x), _triton_cupy_pointer(gate.data),
            _triton_cupy_pointer(up.data), _triton_cupy_pointer(out), k, f,
            gq, gb, _FORMAT_ID[gate.quant_type],
            uq, ub, _FORMAT_ID[up.quant_type], block_k, num_warps=4,
        )
    return out.reshape(x.shape[:-1] + (f,))


class PackedExpertFFNInference:
    """Three quantized GGUF projections; existing MoE executor adapter."""

    def __init__(self, gate, up, down, *, fuse_gate_up=True):
        self.gate, self.up, self.down = gate, up, down
        self.fuse_gate_up = bool(fuse_gate_up)
        if gate.shape != up.shape:
            raise ValueError("Gate and up projections must have identical dimensions")
        if down.shape != (gate.shape[1], gate.shape[0]):
            raise ValueError("Down projection dimensions must be transposed")

    def forward(self, x):
        if self.fuse_gate_up:
            h = quantized_swiglu(x, self.gate, self.up)
        else:
            # Explicit opt-out for GPU parity/performance troubleshooting.
            from mini_llm.ops.silu import silu
            g = quantized_linear(x, self.gate)
            u = quantized_linear(x, self.up)
            h = silu(g) * u
        # Always use the shared quantized down projection (no dense expansion).
        return quantized_linear(xp.ascontiguousarray(h), self.down)
