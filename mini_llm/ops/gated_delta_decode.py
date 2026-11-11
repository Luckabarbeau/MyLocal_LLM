"""Three-launch CuPy FP32 single-token gated-delta mixer.

Only global [all Q | all K | all V] packing is assumed. The public mixer
validates before calling here; unsupported metadata returns None before any
mutation. Compilation/allocation also precede mutation. A launch failure is
never retried through the portable path (history may already have advanced).
"""

from collections import deque
import math
import os

import numpy as np

from mini_llm.backend import BACKEND_NAME, xp


_MODULES = {}
_PENDING = {}
_NAMES = ("conv_silu_history", "normalized_delta", "gated_rms")
_SOURCE = r"""
extern "C" __global__ void conv_silu_history(
    const float* x, const float* weight, const float* bias,
    float* history, float* convolved, int channels, int taps, int count,
    int has_bias) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= count) return;
    int b = i / channels, c = i % channels;
    int base = b * (taps - 1) * channels + c;
    float current = x[i], z = 0.0f;
    // One thread owns all history slots of a channel: no shift races.
    for (int j = 0; j < taps - 1; ++j) {
        float raw = history[base + j * channels];
        z += raw * weight[c * taps + j];
    }
    z += current * weight[c * taps + taps - 1];
    if (has_bias) z += bias[c];
    convolved[i] = z * (1.0f / (1.0f + expf(-z)));
    for (int j = 0; j < taps - 2; ++j)
        history[base + j * channels] = history[base + (j + 1) * channels];
    if (taps > 1) history[base + (taps - 2) * channels] = current;
}

extern "C" __global__ void normalized_delta(
    const float* convolved, const float* log_decay, const float* beta,
    const int* head_map, float* memory, float* output,
    int channels, int q_width, int hv, int dk, int dv,
    float eps, float query_scale) {
    // Four key partitions, 32 contiguous value columns per block. Rectangular
    // memory is coalesced along Dv; each tile exclusively owns its columns.
    int row = blockIdx.x, b = row / hv, h = row % hv;
    int lane = threadIdx.x & 31, part = threadIdx.x >> 5;
    int col = blockIdx.y * 32 + lane, tid = threadIdx.x;
    int kh = head_map[h], packed = b * channels + kh * dk;
    __shared__ float q[256], k[256], sq[128], sk[128], partial[128];
    float qs = 0.0f, ks = 0.0f;
    for (int d = tid; d < dk; d += 128) {
        float a = convolved[packed + d];
        float z = convolved[packed + q_width + d];
        q[d] = a; k[d] = z;
        qs += a * a; ks += z * z;
    }
    sq[tid] = qs; sk[tid] = ks;
    __syncthreads();
    for (int s = 64; s; s >>= 1) {
        if (tid < s) { sq[tid] += sq[tid + s]; sk[tid] += sk[tid + s]; }
        __syncthreads();
    }
    float qden = sqrtf(sq[0] + eps), kden = sqrtf(sk[0] + eps);
    for (int d = tid; d < dk; d += 128) {
        // Preserve the FP32 normalized storage and scaled-query boundaries.
        q[d] = (q[d] / qden) * query_scale;
        k[d] = k[d] / kden;
    }
    __syncthreads();
    int base = row * dk * dv;
    float decay = expf(log_decay[row]), prediction = 0.0f;
    if (col < dv) {
        for (int d = part; d < dk; d += 4) {
            int index = base + d * dv + col;
            float decayed = memory[index] * decay;
            memory[index] = decayed;
            prediction += k[d] * decayed;
        }
    }
    partial[tid] = prediction;
    __syncthreads();
    prediction = ((partial[lane] + partial[32 + lane]) +
                  partial[64 + lane]) + partial[96 + lane];
    float correction = 0.0f;
    if (col < dv)
        correction = (convolved[b * channels + 2 * q_width + h * dv + col]
                      - prediction) * beta[row];
    // All partitions must finish reading prediction before shared reuse.
    __syncthreads();
    float readout = 0.0f;
    if (col < dv) {
        for (int d = part; d < dk; d += 4) {
            int index = base + d * dv + col;
            float updated = memory[index] + k[d] * correction;
            memory[index] = updated;
            readout += q[d] * updated;
        }
    }
    partial[tid] = readout;
    __syncthreads();
    if (part == 0 && col < dv)
        output[row * dv + col] = ((partial[lane] + partial[32 + lane]) +
                                  partial[64 + lane]) + partial[96 + lane];
}

extern "C" __global__ void gated_rms(
    const float* x, const float* gate, const float* gamma, float* output,
    int dv, float eps) {
    int row = blockIdx.x, tid = threadIdx.x;
    __shared__ float sums[128];
    float sum = 0.0f;
    for (int d = tid; d < dv; d += 128) {
        float z = x[row * dv + d];
        sum += z * z;
    }
    sums[tid] = sum;
    __syncthreads();
    for (int s = 64; s; s >>= 1) {
        if (tid < s) sums[tid] += sums[tid + s];
        __syncthreads();
    }
    float inv = 1.0f / sqrtf(sums[0] / (float)dv + eps);
    for (int d = tid; d < dv; d += 128) {
        int i = row * dv + d;
        float g = gate[i];
        float activated = g * (1.0f / (1.0f + expf(-g)));
        output[i] = ((x[i] * inv) * gamma[d]) * activated;
    }
}
"""


def decode_enabled():
    """CPU-safe escape hatch, captured by each mixer at construction."""
    return BACKEND_NAME == "cupy" and os.environ.get(
        "MINI_LLM_GATED_DELTA_DECODE", "1"
    ).lower() not in {"0", "false", "off", "no"}


def _get_kernels(device):
    if device not in _MODULES:
        try:
            module = xp.RawModule(code=_SOURCE, options=("--std=c++14", "--fmad=false"))
            kernels = tuple(module.get_function(name) for name in _NAMES)
            _MODULES[device] = (module, kernels)
        except Exception:
            # No launch has occurred. Preserve the portable path on systems
            # without a usable NVRTC/toolchain, independently for each device.
            _MODULES[device] = None
    entry = _MODULES[device]
    return None if entry is None else entry[1]


def _eligible(mixer, qkv, gate, log_decay, beta, state):
    if BACKEND_NAME != "cupy" or qkv.shape[1] != 1:
        return False
    rule, conv, norm = mixer.rule, mixer.conv, mixer.norm
    if (rule.state_dtype != xp.float32 or conv.compute_dtype != xp.float32 or
            norm.compute_dtype != xp.float32 or conv.activation != "silu" or
            norm.gate_activation != "silu" or not 1 <= rule.key_dim <= 256):
        return False
    gamma, weight, bias = norm.norm.gamma, conv.weight, conv.bias
    if (gamma.shape != (rule.value_dim,) or
            weight.shape != (mixer.channels, conv.kernel_size)):
        return False
    if bias is not None and bias.shape != (mixer.channels,):
        return False
    arrays = (qkv, gate, log_decay, beta, state.conv.history,
              state.delta.memory, weight, gamma)
    if bias is not None:
        arrays += (bias,)
    indices = rule._head_indices
    if (not isinstance(indices, xp.ndarray) or indices.dtype != xp.int32 or
            indices.shape != (rule.n_value_heads,) or not indices.flags.c_contiguous):
        return False
    device = xp.cuda.runtime.getDevice()
    if not all(isinstance(a, xp.ndarray) and a.dtype == xp.float32 and
               a.flags.c_contiguous and a.device.id == device and
               a.size < 2**31 for a in arrays) or indices.device.id != device:
        return False
    if (qkv.shape[0] * rule.n_value_heads >= 2**31 or
            (rule.value_dim + 31) // 32 > 65535):
        return False
    # Constants must survive the same FP32 scalar conversion as the graph.
    for value in (mixer.l2_eps, norm.norm.eps, rule.query_scale):
        if not math.isfinite(value) or abs(value) > np.finfo(np.float32).max:
            return False
    if min(mixer.l2_eps, norm.norm.eps) < np.finfo(np.float32).tiny:
        return False
    # Aliased state/inputs need portable operation ordering, not concurrent
    # tiles. Pointer intervals are metadata, never device-value checks.
    for target in (state.conv.history, state.delta.memory):
        if not target.size:
            continue
        lo, hi = target.data.ptr, target.data.ptr + target.nbytes
        for source in (*arrays, indices):
            if source is target or not source.size:
                continue
            if lo < source.data.ptr + source.nbytes and source.data.ptr < hi:
                return False
    return True


def try_decode(mixer, qkv, gate, log_decay, beta, state):
    """Return a fused output, or None without mutation for unsupported metadata.

    Retain actual arrays (including intermediates/weights) until the recorded
    current-stream event completes. Event queries reclaim prior calls without
    synchronizing, including when callers immediately drop temporary inputs.
    """
    if not _eligible(mixer, qkv, gate, log_decay, beta, state):
        return None
    device = qkv.device.id
    kernels = _get_kernels(device)
    if kernels is None:
        return None
    stream = xp.cuda.get_current_stream()
    pending = _PENDING.setdefault((device, stream.ptr), deque())
    while pending and pending[0][0].done:
        pending.popleft()
    rule, conv = mixer.rule, mixer.conv
    batch = qkv.shape[0]
    convolved = xp.empty_like(qkv)
    readout = xp.empty_like(gate)
    output = xp.empty_like(gate)
    event = xp.cuda.Event(disable_timing=True)
    bias = conv.weight if conv.bias is None else conv.bias
    refs = (qkv, gate, log_decay, beta, state.conv.history, state.delta.memory,
            conv.weight, bias, mixer.norm.norm.gamma, rule._head_indices,
            convolved, readout, output, kernels)
    i32, f32 = np.int32, np.float32
    try:
        kernels[0](((qkv.size + 127) // 128,), (128,), (
            qkv, conv.weight, bias, state.conv.history, convolved,
            i32(mixer.channels), i32(conv.kernel_size), i32(qkv.size),
            i32(conv.bias is not None)), stream=stream)
        kernels[1]((batch * rule.n_value_heads, (rule.value_dim + 31) // 32), (128,), (
            convolved, log_decay, beta, rule._head_indices, state.delta.memory, readout,
            i32(mixer.channels), i32(mixer.q_width), i32(rule.n_value_heads),
            i32(rule.key_dim), i32(rule.value_dim), f32(mixer.l2_eps),
            f32(rule.query_scale)), stream=stream)
        kernels[2]((batch * rule.n_value_heads,), (128,), (
            readout, gate, mixer.norm.norm.gamma, output, i32(rule.value_dim),
            f32(mixer.norm.norm.eps)), stream=stream)
    finally:
        event.record(stream)
        pending.append((event, refs))
    state.conv.length = int(state.conv.length) + 1
    state.delta.length = int(state.delta.length) + 1
    return output
