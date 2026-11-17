"""Bounded Qwen4Exp storage/placement. Inspect headers before touching payloads."""
from dataclasses import dataclass
import math
from numbers import Integral
import os

import numpy as np

from mini_llm.backend import BACKEND_NAME, asnumpy, xp
from mini_llm.ops.native_packed_cpu import packed_cpu_linear
from mini_llm.ops.quantized_linear import (
    packed_to_device, quantized_linear, supports_packed_gpu_type,
)
from mini_llm.ops.quantized_weight import BLOCK_FORMATS, PackedGGUFWeight
from .gguf_store import MultiShardTensorStore
from .qwen4exp_gguf import Qwen4ExpGGUFIndex
from .qwen4exp_mixer import Qwen4ExpMixerSpec
from .qwen4exp_attention import QSAConfig
from .qwen4exp_ple import PLEConfig


DENSE = frozenset(('F32', 'F16', 'BF16'))
GiB = 1024**3


def integer(value, name, minimum=0):
    if isinstance(value, (bool, np.bool_)) or not isinstance(value, Integral) or value < minimum:
        raise ValueError(f'{name} must be an integer >= {minimum}')
    return int(value)


@dataclass(frozen=True)
class Qwen4ExpHeaders:
    width: int
    vocab: int
    context: int
    rank: int
    recurrent: tuple
    ple_layers: tuple
    mixer: Qwen4ExpMixerSpec
    attention: QSAConfig
    shapes: dict
    dense_bytes: int
    constructor_bytes: int
    expert_scale: float


def inspect_headers(index):
    """Validate the complete supported text graph using descriptors alone."""
    s, m = index.store, index.store.metadata
    if m.get('general.architecture') != 'qwen4exp':
        raise ValueError('Expected qwen4exp architecture')
    d, n = index.d_model, index.n_layers
    get = lambda key: m['qwen4exp.' + key]
    if get('hyper_connection.count') != 4:
        raise NotImplementedError('Only four HC streams are supported')
    rank = integer(get('hyper_connection.low_rank'), 'HC rank', 1)
    context = integer(get('context_length'), 'context_length', 1)
    r = Qwen4ExpMixerSpec.from_metadata(m)
    if get('attention.key_length') != get('attention.value_length'):
        raise NotImplementedError('Unequal QSA key/value widths')
    a = QSAConfig(d, n_q_heads=get('attention.head_count'),
        n_kv_heads=get('attention.head_count_kv'), head_dim=get('attention.key_length'),
        rotary_dim=get('rope.dimension_count'), index_heads=get('attention.indexer.head_count'),
        index_dim=get('attention.indexer.key_length'), index_topk=get('attention.indexer.top_k'),
        rope_base=get('rope.freq_base'), sections=tuple(get('rope.dimension_sections')),
        epsilon=r.rms_eps, tie_policy='ggml_cpu')
    ratios = get('attention.compress_ratios')
    if isinstance(ratios, Integral):
        ratios = [ratios] * n
    if len(ratios) != n:
        raise ValueError('compress_ratios length mismatch')
    ratios = tuple(integer(v, 'compress ratio') for v in ratios)
    explicit = m.get('qwen4exp.attention.recurrent_layers')
    if explicit is None:
        interval = integer(m.get('qwen4exp.full_attention_interval', 4), 'full interval', 1)
        recurrent = tuple((i + 1) % interval != 0 for i in range(n))
    else:
        values = [explicit] * n if isinstance(explicit, (Integral, bool)) else explicit
        if len(values) != n or any(v not in (0, 1, False, True) for v in values):
            raise ValueError('Invalid recurrent_layers metadata')
        recurrent = tuple(bool(v) for v in values)
    if any(ratio != (0 if rec else a.block_size) for rec, ratio in zip(recurrent, ratios)):
        raise NotImplementedError('Supported layout requires recurrent ratio 0 and QSA ratio 4')
    ple = tuple(integer(i, 'PLE layer') for i in m.get('qwen4exp.ple.layers', ()))
    if len(ple) > 1 or len(set(ple)) != len(ple) or any(i >= n or not recurrent[i] for i in ple):
        raise NotImplementedError('At most one PLE on a recurrent layer is supported')
    emb = s.tensors.get('token_embd.weight')
    if emb is None or len(emb.shape) != 2 or emb.shape[1] != d:
        raise ValueError('token_embd.weight must be [vocab,width]')
    if emb.quant_type != 'IQ4_XS':
        raise NotImplementedError('Token embedding requires selected IQ4_XS rows')
    vocab = emb.shape[0]
    shapes = {'token_embd.weight': (vocab, d), 'output.weight': (vocab, d)}
    controls = set()
    def hc(stem, inject=True):
        shapes.update({stem + 'norm.weight': (4*d,), stem + 'down.weight': (rank, 4*d),
                       stem + 'up.weight': (4*d, rank)})
        if inject:
            shapes[stem + 'inject.weight'] = (4, 4*d)
        controls.update(name for name in shapes if name.startswith(stem))
    hc('output_hc_', False)
    shared = integer(get('expert_shared_feed_forward_length'), 'shared width', 1)
    scale = float(m.get('qwen4exp.expert_weights_scale', 1.0))
    if not math.isfinite(scale) or scale <= 0:
        raise ValueError('Invalid expert_weights_scale')
    for key, expected in (('expert_gating_func', 1), ('expert_weights_norm', True)):
        if 'qwen4exp.' + key in m and get(key) != expected:
            raise NotImplementedError(f'Unsupported routing metadata {key}')
    for key in ('expert_group_count', 'expert_group_used_count'):
        if 'qwen4exp.' + key in m:
            raise NotImplementedError(f'Unsupported routing metadata {key}')
    for i, rec in enumerate(recurrent):
        p = f'blk.{i}.'
        hc(p + 'hc_attn_'); hc(p + 'hc_ffn_')
        shapes.update({p+'ffn_gate_inp.weight': (index.n_experts, d),
            p+'ffn_gate_inp_shexp.weight': (d,),
            p+'ffn_gate_shexp.weight': (shared, d), p+'ffn_up_shexp.weight': (shared, d),
            p+'ffn_down_shexp.weight': (d, shared),
            p+'ffn_gate_exps.weight': (index.n_experts, index.d_ff, d),
            p+'ffn_up_exps.weight': (index.n_experts, index.d_ff, d),
            p+'ffn_down_exps.weight': (index.n_experts, d, index.d_ff)})
        controls.update((p+'ffn_gate_inp.weight', p+'ffn_gate_inp_shexp.weight'))
        if rec:
            shapes.update({p+'attn_qkv.weight': (r.qkv_channels, d),
                p+'attn_gate.weight': (r.inner_size, d),
                p+'ssm_alpha.weight': (r.n_value_heads, d), p+'ssm_beta.weight': (r.n_value_heads, d),
                p+'ssm_out.weight': (d, r.inner_size),
                p+'ssm_conv1d.weight': (r.qkv_channels, r.conv_kernel_size),
                p+'ssm_norm.weight': (r.head_dim,), p+'ssm_a': (r.n_value_heads,),
                p+'ssm_dt.bias': (r.n_value_heads,)})
            controls.update(p+k for k in ('ssm_conv1d.weight','ssm_norm.weight','ssm_a','ssm_dt.bias'))
        else:
            shapes.update({p+'attn_q.weight': (2*a.n_q_heads*a.head_dim, d),
                p+'attn_k.weight': (a.n_kv_heads*a.head_dim, d),
                p+'attn_v.weight': (a.n_kv_heads*a.head_dim, d),
                p+'attn_output.weight': (d, a.n_q_heads*a.head_dim),
                p+'attn_q_norm.weight': (a.head_dim,), p+'attn_k_norm.weight': (a.head_dim,),
                p+'indexer.q_proj.weight': (a.index_heads*a.index_dim, d),
                p+'indexer.k_proj.weight': (a.index_dim, d),
                p+'indexer.q_norm.weight': (a.index_dim,), p+'indexer.k_norm.weight': (a.index_dim,)})
            controls.update(p+k for k in ('attn_q_norm.weight','attn_k_norm.weight',
                                          'indexer.q_norm.weight','indexer.k_norm.weight'))
        if i in ple:
            table = s.tensors.get('per_layer_token_embd.weight')
            if table is None or table.quant_type != 'IQ4_NL':
                raise ValueError('Missing IQ4_NL PLE table')
            PLEConfig.from_metadata(m, table.shape[0])
            shapes['per_layer_token_embd.weight'] = (table.shape[0], 160)
            shapes.update({p+'ple_key.weight': (4*d, d), p+'ple_value.weight': (d, d),
                p+'ple_conv1d.weight': (4*d, 4)})
            for part in ('key','query','conv'):
                shapes[p+f'ple_norm_{part}.weight'] = (4*d,)
            controls.update(p+k for k in ('ple_conv1d.weight','ple_norm_key.weight',
                                          'ple_norm_query.weight','ple_norm_conv.weight'))
    extra = set(s.tensors) - set(shapes)
    if extra:
        raise NotImplementedError(f'Unsupported tensors: {sorted(extra)[:5]}')
    dense_sizes = []
    for name, shape in shapes.items():
        desc = s.tensors.get(name)
        # Converted gammas can be flat or already grouped, with identical order.
        grouped = name.endswith('norm.weight') and ('hc_' in name or 'ple_norm_' in name)
        valid_shape = desc is not None and (desc.shape == shape or
            (grouped and desc.shape == (4, d)))
        if not valid_shape:
            raise ValueError(f'{name}: expected shape {shape}')
        kind = desc.quant_type
        if name in controls and kind not in DENSE:
            raise NotImplementedError(f'{name}: control weight must be dense')
        if kind not in DENSE and kind not in BLOCK_FORMATS:
            raise NotImplementedError(f'{name}: unsupported {kind}')
        if kind in BLOCK_FORMATS and BLOCK_FORMATS[kind] != (
                desc.format.block_elements, desc.format.block_bytes):
            raise ValueError(f'{name}: packed block layout mismatch')
        if '_exps.weight' in name and kind not in BLOCK_FORMATS:
            raise NotImplementedError('Routed experts must remain packed')
        if kind in DENSE:
            dense_sizes.append(math.prod(shape)*4)
    # Constructor copies and BF16 conversion: conservatively reserve a whole HC
    # group (both matrices) plus three copies of the largest dense tensor.
    temporary = 4 * (2*rank*4*d + 4*4*d + 4*d) + 3*max(dense_sizes, default=0)
    return Qwen4ExpHeaders(d, vocab, context, rank, recurrent, ple, r, a, shapes,
                          sum(dense_sizes), temporary, scale)


class Qwen4ExpWeights:
    """Own only a factory-opened store. No dense or packed host weight cache."""
    def __init__(self, index, *, owns_store=False):
        self.index, self.store = index, index.store
        self.owns_store = owns_store
        self.headers = inspect_headers(index)
        self.plan = None
        self.device_names = frozenset()
        self.device = {}
        self.closed = False

    @classmethod
    def open(cls, path):
        store = MultiShardTensorStore.open(path)
        try:
            return cls(Qwen4ExpGGUFIndex(store, model_id=str(store.paths[0])), owns_store=True)
        except Exception:
            store.close()
            raise

    def memory_plan(self, max_length, *, host_budget_bytes=8*GiB,
                    static_device_budget_bytes=4*GiB, device_budget_bytes=8*GiB,
                    cache_bytes=0, headroom_bytes=512*1024**2, staging_bytes=64*1024**2,
                    activation_bytes=64*1024**2, cpu_workers=2, resident_device_bytes=0):
        h = self.headers
        resident_device_bytes = integer(resident_device_bytes, 'resident_device_bytes')
        length = integer(max_length, 'max_length', 1)
        if length > h.context:
            raise ValueError('max_length exceeds checkpoint context')
        workers = integer(cpu_workers, 'cpu_workers', 1)
        cap = min(32, len(os.sched_getaffinity(0)) if hasattr(os, 'sched_getaffinity') else (os.cpu_count() or 1))
        if workers > cap:
            raise ValueError(f'cpu_workers exceeds fixed positive cap {cap}')
        budgets = dict(host_budget_bytes=host_budget_bytes, static_device_budget_bytes=static_device_budget_bytes,
            device_budget_bytes=device_budget_bytes, cache_bytes=cache_bytes, headroom_bytes=headroom_bytes,
            staging_bytes=staging_bytes, activation_bytes=activation_bytes)
        for k,v in budgets.items():
            integer(v,k)
        r, a = h.mixer, h.attention
        nr = sum(h.recurrent); na = len(h.recurrent)-nr
        recurrent = 4*nr*((r.conv_kernel_size-1)*r.qkv_channels + r.n_value_heads*r.head_dim**2)
        qsa = 4*na*(length*(2*a.n_kv_heads*a.head_dim+a.index_dim) +
                      (length//a.block_size+1)*a.index_dim)
        ple = 4*len(h.ple_layers)*9*4*h.width
        state = recurrent+qsa+ple
        # Sequential QSA has O(length) selection scratch and bounded K/V scratch.
        scratch = 4*(32*max(r.qkv_channels,2*a.n_q_heads*a.head_dim,4*h.width)+
            h.vocab + 8*length + 4*min(length,a.index_topk+a.block_size-1)*a.head_dim)
        activation = max(activation_bytes, scratch)
        # Native kernels use per-call row/output scratch, never a full matrix.
        staging = max(staging_bytes, 4*workers*(8*h.width+8*self.index.d_ff))
        device_static = 0
        names = []
        if BACKEND_NAME == 'cupy':
            device_static = h.dense_bytes
            if device_static > static_device_budget_bytes:
                raise MemoryError('Dense FP32 backbone exceeds static_device_budget_bytes; use NumPy CPU')
            for name, desc in self.store.tensors.items():
                if self.plan is not None and name not in self.device_names:
                    continue
                if name in ('token_embd.weight','per_layer_token_embd.weight') or '_exps.weight' in name:
                    continue
                if supports_packed_gpu_type(desc.quant_type) and device_static+desc.nbytes <= static_device_budget_bytes:
                    names.append(name); device_static += desc.nbytes
        host_static = h.dense_bytes if BACKEND_NAME == 'numpy' else 0
        host_total = host_static+h.constructor_bytes+staging+headroom_bytes
        device_total = device_static+cache_bytes+headroom_bytes
        if BACKEND_NAME == 'numpy':
            host_total += state+activation
            if cache_bytes:
                raise ValueError('NumPy runtime has no device expert cache')
        else:
            device_total += state+activation+staging+h.constructor_bytes
        if resident_device_bytes > device_total:
            raise ValueError('resident_device_bytes exceeds planned device total')
        if host_total > host_budget_bytes:
            raise MemoryError(f'Insufficient host budget: need {host_total}, budget {host_budget_bytes}')
        if BACKEND_NAME == 'cupy':
            if device_total > device_budget_bytes:
                raise MemoryError(f'Insufficient device budget: need {device_total}, budget {device_budget_bytes}')
            free, _ = xp.cuda.runtime.memGetInfo()
            # Credit only this model's live allocations, never global pool usage.
            # Unused pool-cached blocks may still make this check conservative.
            remaining = max(device_total-resident_device_bytes, 0)
            if remaining > free:
                raise MemoryError(f'Insufficient free device memory: need {remaining}, free {free}')
        budgets.update(activation_bytes=activation, staging_bytes=staging)
        return dict(max_length=length, cpu_workers=workers, dense_fp32_bytes=h.dense_bytes,
            constructor_temporary_bytes=h.constructor_bytes, host_static_bytes=host_static,
            device_static_bytes=device_static, recurrent_bytes=recurrent, qsa_bytes=qsa, ple_bytes=ple,
            state_bytes=state,
            host_reserved_bytes=host_total, device_reserved_bytes=device_total,
            canonical_mmap_bytes=sum(t.nbytes for t in self.store.tensors.values()),
            device_names=tuple(names), **budgets)

    def approve(self, plan):
        if self.plan is not None:
            raise RuntimeError('Weight placement already approved')
        self.plan = dict(plan)
        self.device_names = frozenset(plan['device_names'])

    def _ready(self):
        if self.closed or self.store.closed:
            raise RuntimeError('Weight source is closed')
        if self.plan is None:
            raise RuntimeError('Memory plan must be approved before payload access')

    def dense(self, name):
        self._ready()
        t = self.store.tensors[name]
        if t.quant_type not in DENSE:
            raise TypeError('Full decode of packed weights is forbidden')
        raw = self.store.tensor_bytes(name)
        if t.quant_type == 'BF16':
            host = (raw.view('<u2').astype(np.uint32) << 16).view(np.float32)
        else:
            host = raw.view('<f4' if t.quant_type == 'F32' else '<f2').astype(np.float32, copy=True)
        return xp.asarray(host.reshape(t.shape), dtype=xp.float32)

    def projection(self, name):
        self._ready()
        desc = self.store.tensors[name]
        if desc.quant_type in DENSE:
            weight = self.dense(name)
            return lambda x: x @ weight.T
        weight = self.store.packed_weight(name)
        if name in self.device_names:
            device = packed_to_device(weight)
            self.device[name] = device
            def project(x):
                shape = x.shape[:-1]+(weight.shape[0],)
                return quantized_linear(xp.ascontiguousarray(x.reshape(-1,x.shape[-1])), device).reshape(shape)
            return project
        def project(x):
            shape = x.shape[:-1]+(weight.shape[0],)
            host = np.ascontiguousarray(asnumpy(x).reshape(-1,x.shape[-1]), dtype=np.float32)
            result = packed_cpu_linear(host, weight, backend='native')
            if result is None:
                raise RuntimeError('Native packed projection returned no result')
            return xp.asarray(result, dtype=xp.float32).reshape(shape)
        return project

    def embedding(self, token):
        self._ready()
        d = self.headers.width
        raw = self.store.rows('token_embd.weight', token, token+1)
        row = PackedGGUFWeight(raw, (1,d), 'IQ4_XS').dequantize_rows_reference(
            np.array([0],dtype=np.int64),tile_rows=1)[0]
        return xp.asarray(row,dtype=xp.float32)

    def close(self):
        self.device.clear()
        if self.owns_store:
            self.store.close()
        self.closed = True
