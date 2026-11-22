"""Four-launch causal chunk mixer with the singleton FP32 reduction graph.

Convolution and output normalization parallelize over prompt tokens. Each delta
state tile advances all tokens inside one CUDA launch; temporal dependence stays
ordered on device instead of returning to Python for every token.
"""
from collections import deque
from functools import lru_cache

import numpy as np

from mini_llm.backend import BACKEND_NAME, xp
from . import gated_delta_decode as decode


_CONV = r'''
extern "C" __global__ void chunk_conv(const float* x, const float* weight,
        const float* bias, const float* history, float* out,
        int channels, int taps, int time, int has_bias) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= time * channels) return;
    int t = i / channels, c = i % channels;
    float current = x[i], z = 0.0f;
    for (int j = 0; j < taps - 1; ++j) {
        int source = t - (taps - 1) + j;
        float raw = source < 0 ? history[(source + taps - 1)*channels+c] : x[source*channels+c];
        z += raw * weight[c*taps+j];
    }
    z += current * weight[c*taps+taps-1];
    if (has_bias) z += bias[c];
    out[i] = z * (1.0f / (1.0f + expf(-z)));
}
extern "C" __global__ void chunk_history(const float* x, float* history,
        int channels, int taps, int time) {
    int c = blockIdx.x * blockDim.x + threadIdx.x;
    if (c >= channels) return;
    // Ascending destination order only reads old slots to its right.
    for (int j = 0; j < taps - 1; ++j) {
        int source = time - (taps - 1) + j;
        history[j*channels+c] = source < 0 ? history[(source+taps-1)*channels+c] : x[source*channels+c];
    }
}
'''


def _source():
    # Share the established singleton's arithmetic rather than maintain a
    # separate recurrence equation or change its floating-point partitions.
    source = decode._SOURCE
    begin = source.index('extern "C" __global__ void normalized_delta(')
    end = source.index('extern "C" __global__ void gated_rms(', begin)
    core = source[begin:end]
    anchors = ('void normalized_delta(', 'float eps, float query_scale) {',
               'int kh = head_map[h], packed = b * channels + kh * dk;',
               'log_decay[row]', 'beta[row]', 'output[row * dv + col]')
    if any(core.count(anchor) != 1 for anchor in anchors):
        raise RuntimeError('Native delta source contract changed; review chunk adapter')
    core = core.replace('void normalized_delta(', 'void chunk_delta(')
    core = core.replace('float eps, float query_scale) {', 'float eps, float query_scale, int time) {')
    core = core.replace('int kh = head_map[h], packed = b * channels + kh * dk;',
        'int kh = head_map[h];\n    for (int t = 0; t < time; ++t) {\n    int packed = t * channels + kh * dk;')
    core = core.replace('log_decay[row]', 'log_decay[t*hv+h]').replace('beta[row]', 'beta[t*hv+h]')
    core = core.replace('convolved[b * channels + 2 * q_width + h * dv + col]',
                        'convolved[t * channels + 2 * q_width + h * dv + col]')
    core = core.replace('output[row * dv + col]', 'output[(t*hv+h) * dv + col]')
    closing = core.rfind('}')
    core = core[:closing] + '    __syncthreads();\n    }\n' + core[closing:]
    return _CONV + core + source[end:]


@lru_cache(maxsize=8)
def _kernels(device):
    with xp.cuda.Device(device):
        module = xp.RawModule(code=_source(), options=('--std=c++14', '--fmad=false'))
        return tuple(module.get_function(name) for name in
                     ('chunk_conv', 'chunk_history', 'chunk_delta', 'gated_rms'))


def try_chunk(mixer, qkv, gate, log_decay, beta, state, *, capture_owned=False):
    """Single sequence only; decline before mutation for unsupported metadata."""
    if BACKEND_NAME != 'cupy' or qkv.shape[0] != 1 or qkv.shape[1] < 1:
        return None
    if not all(a.flags.c_contiguous for a in (qkv, gate, log_decay, beta)):
        return None
    if not decode._eligible(mixer, qkv[:, :1], gate[:, :1], log_decay[:, :1], beta[:, :1], state):
        return None
    stream = xp.cuda.get_current_stream()
    if stream.is_capturing():
        return None
    kernels = _kernels(qkv.device.id)  # Compilation precedes any state mutation.
    time, channels = qkv.shape[1:]
    rule, conv = mixer.rule, mixer.conv
    convolved = xp.empty_like(qkv)
    readout, output = xp.empty_like(gate), xp.empty_like(gate)
    bias = conv.weight if conv.bias is None else conv.bias
    key = (qkv.device.id, stream.ptr)
    pending = decode._PENDING.setdefault(key, deque())
    while pending and pending[0][0].done:
        pending.popleft()
    event = xp.cuda.Event(disable_timing=True)
    refs = (qkv, gate, log_decay, beta, state.conv.history, state.delta.memory,
            conv.weight, bias, rule._head_indices, mixer.norm.norm.gamma,
            convolved, readout, output, kernels)
    i, f = np.int32, np.float32
    try:
        kernels[0](((time*channels+127)//128,), (128,),
            (qkv, conv.weight, bias, state.conv.history, convolved,
             i(channels), i(conv.kernel_size), i(time), i(conv.bias is not None)))
        kernels[1](((channels+127)//128,), (128,),
            (qkv, state.conv.history, i(channels), i(conv.kernel_size), i(time)))
        kernels[2]((rule.n_value_heads, (rule.value_dim+31)//32), (128,),
            (convolved, log_decay, beta, rule._head_indices, state.delta.memory, readout,
             i(channels), i(mixer.q_width), i(rule.n_value_heads), i(rule.key_dim),
             i(rule.value_dim), f(mixer.l2_eps), f(rule.query_scale), i(time)))
        kernels[3]((time*rule.n_value_heads,), (128,),
            (readout, gate, mixer.norm.norm.gamma, output, i(rule.value_dim),
             f(mixer.norm.norm.eps), i(mixer.norm.gate_activation == 'sigmoid')))
    finally:
        event.record(stream)
        pending.append((event, refs))
    state.conv.length = int(state.conv.length) + time
    state.delta.length = int(state.delta.length) + time
    return output
