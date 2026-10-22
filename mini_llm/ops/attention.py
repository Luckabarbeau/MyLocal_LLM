import os
import numpy as np

from ..backend import (
    xp, BACKEND_NAME, is_low_precision_dtype, is_bfloat16_dtype,
)
from ..init import matrix_parameter
from .rope import rope_forward, rope_backward, clear_rope_cache
from .attention_selection import (
    build_local_causal_plan,
)
from .indexed_attention import (
    indexed_attention_forward,
    indexed_attention_backward,
    local_window_attention_forward,
    local_window_attention_backward,
    dilated_attention_forward,
    dilated_attention_backward,
    global_sparse_attention_forward,
    global_sparse_attention_backward,
)
from .retrieval_attention import ContextRetrievalAttention
from .hierarchical_memory import TerminalLandmarkAttention
from ..performance_profiler import performance_scope
from ..config import (
    AttentionLayerConfig,
    DenseAttentionConfig,
    LocalAttentionConfig,
    DilatedAttentionConfig,
    GlobalSparseAttentionConfig,
    RetrievalAttentionConfig,
)


def _bf16_mixed_context_enabled(dtype):
    """Keep mixed-attention context in BF16 storage through the output projection."""
    if not is_bfloat16_dtype(dtype):
        return False
    raw = os.environ.get("MINI_LLM_BF16_MIXED_CONTEXT", "0").strip().lower()
    return raw not in {"0", "false", "off", "no"}


def _skip_empty_retrieval_enabled():
    """Avoid the learned router if no causal historical block is selectable."""
    raw = os.environ.get("MINI_LLM_SKIP_EMPTY_RETRIEVAL", "0").strip().lower()
    return raw not in {"0", "false", "off", "no", ""}


def _has_eligible_retrieval_route(seq_len, config):
    """Host-only equivalent of the first eligible `eligible_route_starts` row.

    A routing boundary is a positive multiple of `routing_stride`, strictly
    before seq_len. At least top_k_blocks complete historical blocks must end
    before the exclude-recent margin. This uses static config/shape metadata,
    without any GPU scalar reductions or allocations.
    """
    stride = int(config.routing_stride)
    earliest = max(
        stride,
        int(config.exclude_recent_tokens)
        + int(config.top_k_blocks) * int(config.history_block_size),
    )
    first_route = ((earliest + stride - 1) // stride) * stride
    return int(seq_len) > first_route


def _packed_qkv_enabled():
    """Use one packed QKV projection GEMM instead of three separate GEMMs."""
    raw = os.environ.get("MINI_LLM_PACKED_QKV", "0").strip().lower()
    return raw not in {"0", "false", "off", "no"}


def _packed_qkv_strict():
    raw = os.environ.get("MINI_LLM_PACKED_QKV_STRICT", "0").strip().lower()
    return raw not in {"0", "false", "off", "no", ""}


def _compact_packed_v_enabled():
    """Copy packed-QKV V into compact storage so the large QKV backing can die.

    Packed Q/K are immediately transformed by RoPE into independent tensors,
    while V would otherwise remain a strided view that keeps the full packed
    [Q,K,V] projection alive until backward.  The compact copy is tiny compared
    with the retained backing allocation and is therefore enabled by default.
    """
    raw = os.environ.get("MINI_LLM_COMPACT_PACKED_V", "1").strip().lower()
    return raw not in {"0", "false", "off", "no"}


_PACKED_QKV_BF16_PACK_KERNEL = None
_PACKED_QKV_BF16_PACK_DISABLED = False


def _get_packed_qkv_bf16_pack_kernel():
    """Compile/cache 0052B inverse-RoPE + BF16 QKV packing kernel."""
    global _PACKED_QKV_BF16_PACK_KERNEL, _PACKED_QKV_BF16_PACK_DISABLED
    if BACKEND_NAME != "cupy" or _PACKED_QKV_BF16_PACK_DISABLED:
        return None
    if _PACKED_QKV_BF16_PACK_KERNEL is not None:
        return _PACKED_QKV_BF16_PACK_KERNEL

    code = r"""
    #include <cuda_bf16.h>

    extern "C" __global__
    void inverse_rope_pack_qkv_bf16(
        const float* __restrict__ dq,
        const float* __restrict__ dk,
        const float* __restrict__ dv,
        const __nv_bfloat16* __restrict__ cos_table,
        const __nv_bfloat16* __restrict__ sin_table,
        __nv_bfloat16* __restrict__ out,
        int rows, int seq_len, int q_width, int kv_width, int d_head)
    {
        const long long total_width = (long long)q_width + 2LL * kv_width;
        const long long total = (long long)rows * total_width;
        const int half = d_head >> 1;

        for (long long idx = (long long)blockIdx.x * blockDim.x + threadIdx.x;
             idx < total;
             idx += (long long)blockDim.x * gridDim.x) {
            const int row = (int)(idx / total_width);
            const int col = (int)(idx - (long long)row * total_width);
            const int pos = row % seq_len;
            float value;

            if (col < q_width) {
                const int local = col;
                const int lane = local % d_head;
                const int pair = lane >> 1;
                const int pair_base = local - (lane & 1);
                const long long base = (long long)row * q_width + pair_base;
                const float a = dq[base];
                const float b = dq[base + 1];
                const long long trig = (long long)pos * half + pair;
                const float c = __bfloat162float(cos_table[trig]);
                const float s = __bfloat162float(sin_table[trig]);
                value = (lane & 1) ? (-a * s + b * c) : (a * c + b * s);
            } else if (col < q_width + kv_width) {
                const int local = col - q_width;
                const int lane = local % d_head;
                const int pair = lane >> 1;
                const int pair_base = local - (lane & 1);
                const long long base = (long long)row * kv_width + pair_base;
                const float a = dk[base];
                const float b = dk[base + 1];
                const long long trig = (long long)pos * half + pair;
                const float c = __bfloat162float(cos_table[trig]);
                const float s = __bfloat162float(sin_table[trig]);
                value = (lane & 1) ? (-a * s + b * c) : (a * c + b * s);
            } else {
                const int local = col - q_width - kv_width;
                value = dv[(long long)row * kv_width + local];
            }
            out[idx] = __float2bfloat16_rn(value);
        }
    }
    """
    try:
        _PACKED_QKV_BF16_PACK_KERNEL = xp.RawKernel(
            code,
            "inverse_rope_pack_qkv_bf16",
            options=("--std=c++11",),
        )
    except Exception:
        _PACKED_QKV_BF16_PACK_DISABLED = True
        if _packed_qkv_strict():
            raise
        return None
    return _PACKED_QKV_BF16_PACK_KERNEL


def _fused_inverse_rope_pack_bf16(dq, dk, dv, rope_cache, *, seq_len, d_head):
    """Return packed BF16 [dQ_pre|dK_pre|dV], or None for fallback."""
    if BACKEND_NAME != "cupy":
        return None
    if dq.dtype != xp.float32 or dk.dtype != xp.float32 or dv.dtype != xp.float32:
        return None
    if any(not a.flags.c_contiguous for a in (dq, dk, dv)):
        return None
    cos = rope_cache["cos"]
    sin = rope_cache["sin"]
    # The 0052B fused kernel assumes one contiguous position table shared by
    # every batch item (row % seq_len).  Hierarchical memory can have different
    # selected source positions per sample, so use the exact vectorized
    # rope_backward fallback for batch-specific explicit position tables.
    if int(cos.shape[0]) != 1 or int(sin.shape[0]) != 1:
        return None
    if not is_bfloat16_dtype(cos.dtype) or not is_bfloat16_dtype(sin.dtype):
        return None
    if not cos.flags.c_contiguous or not sin.flags.c_contiguous:
        return None

    rows = int(dq.shape[0] * dq.shape[1])
    q_width = int(dq.shape[2] * dq.shape[3])
    kv_width = int(dk.shape[2] * dk.shape[3])
    if tuple(dv.shape) != tuple(dk.shape) or int(dq.shape[3]) != int(d_head):
        return None
    out = xp.empty((rows, q_width + 2 * kv_width), dtype=cos.dtype)
    kernel = _get_packed_qkv_bf16_pack_kernel()
    if kernel is None:
        return None
    total = int(out.size)
    threads = 256
    blocks = min((total + threads - 1) // threads, 65535)
    try:
        kernel(
            (blocks,),
            (threads,),
            (
                dq, dk, dv, cos, sin, out,
                np.int32(rows), np.int32(seq_len), np.int32(q_width),
                np.int32(kv_width), np.int32(d_head),
            ),
        )
    except Exception:
        global _PACKED_QKV_BF16_PACK_DISABLED
        _PACKED_QKV_BF16_PACK_DISABLED = True
        if _packed_qkv_strict():
            raise
        return None
    return out


def softmax_forward(x, axis=-1, logit_multiplier=1.0):
    """Stable softmax with FP32 reductions for low-precision logits.

    Attention score GEMMs stay FP16/BF16, while max/subtract/exp/sum/divide
    execute in FP32.  ``logit_multiplier`` restores the FP16 score pre-scale
    only after max subtraction, so positive overflow cannot occur.
    """
    work = x.astype("float32", copy=False) if is_low_precision_dtype(x.dtype) else x
    x_max = xp.max(work, axis=axis, keepdims=True)
    shifted = work - x_max
    if logit_multiplier != 1.0:
        shifted = shifted * logit_multiplier
    e = xp.exp(shifted)
    return e / xp.sum(e, axis=axis, keepdims=True)


def softmax_backward(dp, p, axis=-1):
    dot = xp.sum(dp * p, axis=axis, keepdims=True)
    return p * (dp - dot)


class GQAAttention:
    """Explicit grouped-query causal self-attention.

    K/V stay in their native ``n_kv_heads`` representation.  Query heads are
    viewed as ``[n_kv_heads, group_size]`` and batched matmul broadcasts the
    shared K/V heads across each query group.  This avoids physically repeating
    K/V tensors in forward and their gradients in backward.
    """

    _causal_mask_cache = {}

    def __init__(
        self, d_model, n_q_heads, n_kv_heads, d_head,
        input_std, output_std, rng, rope_base=10_000.0,
        name="attention", dtype="float32", attention_config=None,
        terminal_memory_config=None,
    ):
        if n_q_heads * d_head != d_model:
            raise ValueError("n_q_heads * d_head must equal d_model.")
        if n_q_heads % n_kv_heads != 0:
            raise ValueError("n_q_heads must be divisible by n_kv_heads.")

        self.d_model = d_model
        self.n_q_heads = n_q_heads
        self.n_kv_heads = n_kv_heads
        self.d_head = d_head
        self.group_size = n_q_heads // n_kv_heads
        self.scale = 1.0 / (d_head ** 0.5)
        self.rope_base = float(rope_base)
        self.dtype = dtype
        if attention_config is not None and not isinstance(
            attention_config, AttentionLayerConfig
        ):
            raise TypeError("attention_config must be an AttentionLayerConfig or None")
        self.attention_config = attention_config
        self._static_head_groups = []
        self._retrieval_groups = []
        self.terminal_memory = None
        self.historical_reader = None
        self.historical_layer = None
        self._terminal_memory_head_indices = ()
        self._terminal_memory_group_name = None

        self.Wq = matrix_parameter(
            (d_model, n_q_heads * d_head), input_std, rng, f"{name}.Wq", dtype=dtype
        )
        self.Wk = matrix_parameter(
            (d_model, n_kv_heads * d_head), input_std, rng, f"{name}.Wk", dtype=dtype
        )
        self.Wv = matrix_parameter(
            (d_model, n_kv_heads * d_head), input_std, rng, f"{name}.Wv", dtype=dtype
        )
        self.Wo = matrix_parameter(
            (n_q_heads * d_head, d_model), output_std, rng, f"{name}.Wo", dtype=dtype
        )

        # 0052: packed QKV is a non-parameter compute buffer.  Wq/Wk/Wv remain
        # the canonical trainable/checkpoint parameters so optimizer/checkpoint
        # compatibility is unchanged.  The buffer is refreshed once after each
        # optimizer update instead of being rebuilt for every microbatch.
        self._packed_qkv_weight = None
        self._packed_qkv_valid = False
        self._packed_qkv_grad = None
        self._q_width = self.n_q_heads * self.d_head
        self._kv_width = self.n_kv_heads * self.d_head
        self._qkv_width = self._q_width + 2 * self._kv_width

        if self.attention_config is not None:
            if len(self.attention_config.heads) != self.n_q_heads:
                raise ValueError(
                    "attention_config must define exactly one topology per query head"
                )
            self._configure_attention_patterns(
                self.attention_config, rng, input_std, name, dtype
            )

        self._terminal_init_name = name
        self._terminal_init_dtype = dtype
        self._terminal_init_std = input_std
        if terminal_memory_config is not None:
            self.configure_terminal_memory(terminal_memory_config, rng)

    def configure_terminal_memory(self, terminal_memory_config, rng):
        """Attach 0058C memory parameters after base-model initialization.

        DecoderLanguageModel calls this only after every pre-existing base
        parameter has been initialized, preserving checkpoint/RNG compatibility.
        """
        if self.terminal_memory is not None:
            return
        if not self._retrieval_groups:
            raise ValueError(
                "terminal Landmark memory requires at least one retrieval-head group"
            )
        if len(self._retrieval_groups) != 1:
            raise ValueError(
                "0058C currently supports one retrieval-head group in the memory layer"
            )
        terminal_group = self._retrieval_groups[0]
        head_indices = tuple(terminal_group["head_indices"])
        if len(head_indices) != int(terminal_memory_config.read_heads):
            raise ValueError(
                "memory read_heads must equal retrieval heads in memory-aware layer"
            )
        self._terminal_memory_head_indices = head_indices
        self._terminal_memory_group_name = terminal_group["name"]
        self.terminal_memory = TerminalLandmarkAttention(
            d_model=self.d_model, d_head=self.d_head,
            n_query_heads=len(head_indices), config=terminal_memory_config,
            rng=rng, input_std=self._terminal_init_std, rope_base=self.rope_base,
            name=f"{self._terminal_init_name}.terminal_memory",
            dtype=self._terminal_init_dtype,
        )

    def parameters(self):
        params = [self.Wq, self.Wk, self.Wv, self.Wo]
        for group in self._retrieval_groups:
            params.extend(group["module"].parameters())
        if self.terminal_memory is not None:
            params.extend(self.terminal_memory.parameters())
        return params

    def _configure_attention_patterns(self, attention_config, rng, input_std, name, dtype):
        static_groups = {}
        retrieval_groups = {}

        for head_index, head_config in enumerate(attention_config.heads):
            if isinstance(head_config, DenseAttentionConfig):
                static_groups.setdefault(
                    ("dense", None, None, None, None, None), []
                ).append(head_index)
            elif isinstance(head_config, LocalAttentionConfig):
                static_groups.setdefault(
                    ("local", int(head_config.window), None, None, None, None), []
                ).append(head_index)
            elif isinstance(head_config, DilatedAttentionConfig):
                static_groups.setdefault(
                    (
                        "dilated",
                        int(head_config.window),
                        int(head_config.dilation),
                        int(head_config.offset),
                        None,
                        None,
                    ),
                    [],
                ).append(head_index)
            elif isinstance(head_config, GlobalSparseAttentionConfig):
                static_groups.setdefault(
                    (
                        "global_sparse",
                        None,
                        None,
                        int(head_config.offset),
                        int(head_config.stride),
                        bool(head_config.include_current),
                    ),
                    [],
                ).append(head_index)
            elif isinstance(head_config, RetrievalAttentionConfig):
                group = retrieval_groups.setdefault(
                    head_config.group,
                    {"head_indices": [], "config": head_config.context_router},
                )
                if group["config"] != head_config.context_router:
                    raise ValueError(
                        f"retrieval group {head_config.group!r} must share one "
                        "ContextRouterConfig"
                    )
                group["head_indices"].append(head_index)
            else:
                raise TypeError(
                    f"unsupported attention head config: {type(head_config).__name__}"
                )

        self._static_head_groups = [
            {
                "kind": kind,
                "window": window,
                "dilation": dilation,
                "offset": offset,
                "stride": stride,
                "include_current": include_current,
                "head_indices": tuple(indices),
            }
            for (
                kind,
                window,
                dilation,
                offset,
                stride,
                include_current,
            ), indices in static_groups.items()
        ]

        self._retrieval_groups = []
        for group_name, group in retrieval_groups.items():
            head_indices = tuple(group["head_indices"])
            router_config = group["config"]
            if router_config.num_queries not in {1, len(head_indices)}:
                raise ValueError(
                    f"retrieval group {group_name!r} requires num_queries=1 or "
                    f"num_queries={len(head_indices)}"
                )
            module = ContextRetrievalAttention(
                d_model=self.d_model,
                config=router_config,
                rng=rng,
                input_std=input_std,
                name=f"{name}.retrieval.{group_name}",
                dtype=dtype,
            )
            self._retrieval_groups.append(
                {
                    "name": group_name,
                    "head_indices": head_indices,
                    "module": module,
                }
            )

    def zero_grad(self):
        for p in self.parameters():
            p.zero_grad()

    def _ensure_packed_qkv_grad_views(self):
        """Make Wq/Wk/Wv.grad non-overlapping views of one packed buffer.

        This is purely a storage/layout change.  The three Parameter objects
        remain canonical for checkpointing and the optimizer, but packed QKV
        backward can now accumulate one contiguous dW matrix instead of
        splitting/copying it back into three independent arrays.
        """
        grad_dtype = self.Wq.grad.dtype
        shape = (self.d_model, self._qkv_width)
        if (
            self._packed_qkv_grad is None
            or self._packed_qkv_grad.shape != shape
            or self._packed_qkv_grad.dtype != grad_dtype
        ):
            packed = xp.zeros(shape, dtype=grad_dtype)
            q_end = self._q_width
            k_end = q_end + self._kv_width
            packed[:, :q_end] = self.Wq.grad
            packed[:, q_end:k_end] = self.Wk.grad
            packed[:, k_end:] = self.Wv.grad
            self._packed_qkv_grad = packed
        q_end = self._q_width
        k_end = q_end + self._kv_width
        self.Wq.grad = self._packed_qkv_grad[:, :q_end]
        self.Wk.grad = self._packed_qkv_grad[:, q_end:k_end]
        self.Wv.grad = self._packed_qkv_grad[:, k_end:]

    def refresh_compute_buffers(self):
        """Refresh non-parameter compute buffers after a parameter update.

        The packed QKV matrix is intentionally not a Parameter.  That keeps
        checkpoint names, optimizer moments, and master-weight state exactly
        the same as the reference Wq/Wk/Wv implementation.
        """
        if not _packed_qkv_enabled():
            self._packed_qkv_valid = False
            return
        self._ensure_packed_qkv_grad_views()
        if (
            self._packed_qkv_weight is None
            or self._packed_qkv_weight.shape != (self.d_model, self._qkv_width)
            or self._packed_qkv_weight.dtype != self.Wq.data.dtype
        ):
            self._packed_qkv_weight = xp.empty(
                (self.d_model, self._qkv_width), dtype=self.Wq.data.dtype
            )
        q_end = self._q_width
        k_end = q_end + self._kv_width
        self._packed_qkv_weight[:, :q_end] = self.Wq.data
        self._packed_qkv_weight[:, q_end:k_end] = self.Wk.data
        self._packed_qkv_weight[:, k_end:] = self.Wv.data
        self._packed_qkv_valid = True

    def _get_packed_qkv_weight(self):
        if not self._packed_qkv_valid or self._packed_qkv_weight is None:
            self.refresh_compute_buffers()
        return self._packed_qkv_weight

    @classmethod
    def clear_caches(cls):
        cls._causal_mask_cache.clear()
        clear_rope_cache()

    def _group_queries(self, q):
        """[B,T,Hq,D] -> [B,Hkv,G,T,D] as a view+transpose."""
        b, t, _, d = q.shape
        return q.reshape(
            b, t, self.n_kv_heads, self.group_size, d
        ).transpose(0, 2, 3, 1, 4)

    def _ungroup_queries(self, q_grouped):
        """[B,Hkv,G,T,D] -> [B,T,Hq,D]."""
        b, _, _, t, d = q_grouped.shape
        return q_grouped.transpose(0, 3, 1, 2, 4).reshape(
            b, t, self.n_q_heads, d
        )

    def _get_causal_mask(self, t):
        if t not in self._causal_mask_cache:
            self._causal_mask_cache[t] = xp.triu(
                xp.ones((t, t), dtype=bool), k=1
            )
        return self._causal_mask_cache[t]

    @staticmethod
    def _head_selection(indices):
        """Use a view for contiguous attention-head groups; index others normally."""
        if not indices:
            return indices
        start = indices[0]
        if all(index == start + offset for offset, index in enumerate(indices)):
            return slice(start, start + len(indices))
        return indices

    def _mixed_context_forward(self, x, q, k, v, return_cache, terminal_memory=None,
                               historical_skip_heads=()):
        b, t, _, _ = q.shape
        keep_bf16_context = _bf16_mixed_context_enabled(q.dtype)
        context_dtype = (
            q.dtype
            if keep_bf16_context
            else (xp.float32 if is_bfloat16_dtype(q.dtype) else q.dtype)
        )
        context = xp.zeros(q.shape, dtype=context_dtype)
        group_caches = []
        routing = {}

        skipped = set(historical_skip_heads)
        for group in self._static_head_groups:
            head_indices = tuple(h for h in group["head_indices"] if h not in skipped)
            if not head_indices:
                continue
            q_subset = q[:, :, self._head_selection(head_indices), :]
            topology_kind = group["kind"]
            with performance_scope(f"attention.{topology_kind}.forward"):
                with performance_scope(f"attention.{topology_kind}.plan.forward"):
                    if topology_kind == "dense":
                        plan = build_local_causal_plan(b, t, t)
                    elif topology_kind == "local":
                        # The specialized contiguous local kernel derives its
                        # band mask directly from query/key ranges and does not
                        # need a per-token KeySelectionPlan.
                        plan = None
                    elif topology_kind == "dilated":
                        # Fixed dilation is executed by a specialized phase
                        # kernel; no per-token indexed plan is needed.
                        plan = None
                    elif topology_kind == "global_sparse":
                        # Fixed global anchors are executed by a specialized
                        # kernel that gathers each anchor K/V only once.
                        plan = None
                    else:
                        raise RuntimeError(
                            f"unsupported static attention kind: {topology_kind!r}"
                        )
                    kv_head_indices = xp.asarray(
                        head_indices, dtype=xp.int64
                    ) // self.group_size
                with performance_scope(f"attention.{topology_kind}.kernel.forward"):
                    if topology_kind == "local":
                        kv_mapping = tuple(
                            index // self.group_size for index in head_indices
                        )
                        if return_cache:
                            subset, subset_cache = local_window_attention_forward(
                                q_subset,
                                k,
                                v,
                                group["window"],
                                kv_head_indices=kv_mapping,
                                scale=self.scale,
                                return_cache=True,
                            )
                            group_caches.append(
                                {
                                    "kind": "local_window",
                                    "topology_kind": topology_kind,
                                    "head_indices": head_indices,
                                    "cache": subset_cache,
                                }
                            )
                        else:
                            subset = local_window_attention_forward(
                                q_subset,
                                k,
                                v,
                                group["window"],
                                kv_head_indices=kv_mapping,
                                scale=self.scale,
                                return_cache=False,
                            )
                    elif topology_kind == "dilated":
                        kv_mapping = tuple(
                            index // self.group_size for index in head_indices
                        )
                        if return_cache:
                            subset, subset_cache = dilated_attention_forward(
                                q_subset,
                                k,
                                v,
                                group["window"],
                                group["dilation"],
                                group["offset"],
                                kv_head_indices=kv_mapping,
                                scale=self.scale,
                                return_cache=True,
                            )
                            group_caches.append(
                                {
                                    "kind": "dilated_window",
                                    "topology_kind": topology_kind,
                                    "head_indices": head_indices,
                                    "cache": subset_cache,
                                }
                            )
                        else:
                            subset = dilated_attention_forward(
                                q_subset,
                                k,
                                v,
                                group["window"],
                                group["dilation"],
                                group["offset"],
                                kv_head_indices=kv_mapping,
                                scale=self.scale,
                                return_cache=False,
                            )
                    elif topology_kind == "global_sparse":
                        kv_mapping = tuple(
                            index // self.group_size for index in head_indices
                        )
                        if return_cache:
                            subset, subset_cache = global_sparse_attention_forward(
                                q_subset,
                                k,
                                v,
                                group["stride"],
                                group["offset"],
                                group["include_current"],
                                kv_head_indices=kv_mapping,
                                scale=self.scale,
                                return_cache=True,
                            )
                            group_caches.append(
                                {
                                    "kind": "global_sparse",
                                    "topology_kind": topology_kind,
                                    "head_indices": head_indices,
                                    "cache": subset_cache,
                                }
                            )
                        else:
                            subset = global_sparse_attention_forward(
                                q_subset,
                                k,
                                v,
                                group["stride"],
                                group["offset"],
                                group["include_current"],
                                kv_head_indices=kv_mapping,
                                scale=self.scale,
                                return_cache=False,
                            )
                    elif return_cache:
                        subset, subset_cache = indexed_attention_forward(
                            q_subset,
                            k,
                            v,
                            plan,
                            kv_head_indices=kv_head_indices,
                            scale=self.scale,
                            return_cache=True,
                        )
                        group_caches.append(
                            {
                                "kind": "indexed",
                                "topology_kind": topology_kind,
                                "head_indices": head_indices,
                                "cache": subset_cache,
                            }
                        )
                    else:
                        subset = indexed_attention_forward(
                            q_subset,
                            k,
                            v,
                            plan,
                            kv_head_indices=kv_head_indices,
                            scale=self.scale,
                            return_cache=False,
                        )
                    context[:, :, self._head_selection(head_indices), :] = subset

        for group in self._retrieval_groups:
            head_indices = group["head_indices"]
            if all(h in skipped for h in head_indices):
                continue
            if any(h in skipped for h in head_indices):
                raise ValueError("historical_read must replace whole current retrieval groups")
            module = group["module"]
            if (
                terminal_memory is None
                and _skip_empty_retrieval_enabled()
                and not _has_eligible_retrieval_route(t, module.config)
            ):
                # Exactly match the no-route branch of ContextRouter.forward
                # and block_retrieval_attention_forward: retrieval output and
                # all retrieval gradients are zero. Crucially, skip the
                # history pool, learned query pool, Q/K/V gathering and empty
                # backward entirely. Static attention heads still run.
                empty_shape = (
                    b, 0, module.router.num_queries,
                    int(module.config.top_k_blocks),
                )
                routing[group["name"]] = {
                    "weights": xp.zeros(empty_shape, dtype=x.dtype),
                    "selected_blocks": xp.zeros(empty_shape, dtype=xp.int64),
                    "route_starts": xp.zeros((0,), dtype=xp.int64),
                }
                continue
            q_subset = q[:, :, self._head_selection(head_indices), :]
            kv_head_indices = tuple(
                index // self.group_size for index in head_indices
            )
            with performance_scope("attention.retrieval.forward"):
                if return_cache:
                    subset, group_routing, subset_cache = group["module"].forward(
                        x,
                        q_subset,
                        k,
                        v,
                        kv_head_indices=kv_head_indices,
                        return_cache=True,
                    )
                    group_caches.append(
                        {
                            "kind": "retrieval",
                            "name": group["name"],
                            "head_indices": head_indices,
                            "module": group["module"],
                            "cache": subset_cache,
                        }
                    )
                else:
                    subset, group_routing = group["module"].forward(
                        x,
                        q_subset,
                        k,
                        v,
                        kv_head_indices=kv_head_indices,
                        return_cache=False,
                    )
                context[:, :, self._head_selection(head_indices), :] = subset
                routing[group["name"]] = group_routing

        if terminal_memory is not None and self.terminal_memory is not None:
            head_indices = self._terminal_memory_head_indices
            q_subset = q[:, :, self._head_selection(head_indices), :]
            kv_head_indices = tuple(index // self.group_size for index in head_indices)
            with performance_scope("attention.terminal_memory.forward"):
                if return_cache:
                    terminal_context, terminal_cache = self.terminal_memory.forward(
                        q_subset, k, v, kv_head_indices, terminal_memory,
                        return_cache=True,
                    )
                else:
                    terminal_context = self.terminal_memory.forward(
                        q_subset, k, v, kv_head_indices, terminal_memory,
                        return_cache=False,
                    )
                    terminal_cache = None
            if terminal_context is not None:
                rows = xp.asarray(terminal_memory.terminal_rows, dtype=xp.int64)
                valid_batches = xp.nonzero(terminal_memory.route_valid)[0]
                if valid_batches.size:
                    heads = xp.asarray(head_indices, dtype=xp.int64)
                    context[
                        valid_batches[:, None], rows[valid_batches, None], heads[None, :], :
                    ] = terminal_context[valid_batches]
                if return_cache:
                    group_caches.append({
                        "kind": "terminal_landmark",
                        "head_indices": head_indices,
                        "memory": terminal_memory,
                        "cache": terminal_cache,
                        "overridden_group": self._terminal_memory_group_name,
                    })

        return context, group_caches, routing

    def _mixed_context_backward(self, dcontext, cache):
        q, k, v = cache["q"], cache["k"], cache["v"]
        grad_dtype = xp.float32 if is_bfloat16_dtype(q.dtype) else q.dtype
        dq = xp.zeros(q.shape, dtype=grad_dtype)
        dk = xp.zeros(k.shape, dtype=grad_dtype)
        dv = xp.zeros(v.shape, dtype=grad_dtype)
        # Most current/historical-read layers have no current-window router.
        # Allocate a router-input gradient only when a retrieval group needs it.
        drouter_input = None

        for group_cache in cache["pattern_caches"]:
            head_indices = group_cache["head_indices"]
            dsubset = dcontext[:, :, self._head_selection(head_indices), :]
            if group_cache["kind"] == "retrieval":
                terminal_entry = next(
                    (entry for entry in cache["pattern_caches"]
                     if entry.get("kind") == "terminal_landmark"
                     and entry.get("overridden_group") == group_cache.get("name")),
                    None,
                )
                if terminal_entry is not None:
                    dsubset = xp.array(dsubset, copy=True)
                    mem = terminal_entry["memory"]
                    rows = xp.asarray(mem.terminal_rows, dtype=xp.int64)
                    valid_batches = xp.nonzero(mem.route_valid)[0]
                    if valid_batches.size:
                        dsubset[valid_batches, rows[valid_batches], :, :] = 0
            if group_cache["kind"] == "terminal_landmark":
                mem = group_cache["memory"]
                rows = xp.asarray(mem.terminal_rows, dtype=xp.int64)
                dterm = xp.zeros((dcontext.shape[0], len(head_indices), self.d_head), dtype=dcontext.dtype)
                valid_batches = xp.nonzero(mem.route_valid)[0]
                if valid_batches.size:
                    heads = xp.asarray(head_indices, dtype=xp.int64)
                    dterm[valid_batches] = dcontext[
                        valid_batches[:, None], rows[valid_batches, None], heads[None, :], :
                    ]
                with performance_scope("attention.terminal_memory.backward"):
                    dq_sub, dk_sub, dv_sub = self.terminal_memory.backward(
                        dterm, group_cache["cache"], mem
                    )
            elif group_cache["kind"] in {"indexed", "local_window", "dilated_window", "global_sparse"}:
                topology_kind = group_cache.get("topology_kind", "indexed")
                with performance_scope(f"attention.{topology_kind}.backward"):
                    with performance_scope(f"attention.{topology_kind}.kernel.backward"):
                        if group_cache["kind"] == "local_window":
                            dq_sub, dk_sub, dv_sub = local_window_attention_backward(
                                dsubset, group_cache["cache"]
                            )
                        elif group_cache["kind"] == "dilated_window":
                            dq_sub, dk_sub, dv_sub = dilated_attention_backward(
                                dsubset, group_cache["cache"]
                            )
                        elif group_cache["kind"] == "global_sparse":
                            dq_sub, dk_sub, dv_sub = global_sparse_attention_backward(
                                dsubset, group_cache["cache"]
                            )
                        else:
                            dq_sub, dk_sub, dv_sub, _ = indexed_attention_backward(
                                dsubset, group_cache["cache"]
                            )
            else:
                with performance_scope("attention.retrieval.backward"):
                    drouter, dq_sub, dk_sub, dv_sub = group_cache["module"].backward(
                        dsubset, group_cache["cache"]
                    )
                    if drouter_input is None:
                        drouter_input = xp.zeros(cache["x"].shape, dtype=grad_dtype)
                    drouter_input += drouter.astype(grad_dtype, copy=False)

            dq[:, :, self._head_selection(head_indices), :] += dq_sub.astype(grad_dtype, copy=False)
            dk += dk_sub.astype(grad_dtype, copy=False)
            dv += dv_sub.astype(grad_dtype, copy=False)

        return dq, dk, dv, drouter_input

    def _projection_backward(self, x, dq, dk, dv, cache):
        b, t, _ = x.shape
        x2 = x.reshape(-1, self.d_model)

        if _packed_qkv_enabled():
            self._ensure_packed_qkv_grad_views()
            dqkv2 = None
            if is_bfloat16_dtype(x.dtype):
                with performance_scope("attention.qkv_projection.backward.fused_rope_pack"):
                    dqkv2 = _fused_inverse_rope_pack_bf16(
                        dq,
                        dk,
                        dv,
                        cache["q_rope_cache"],
                        seq_len=t,
                        d_head=self.d_head,
                    )

            if dqkv2 is None:
                # Portable/reference fallback: preserve the 0052 semantics.
                dq_pre = rope_backward(dq, cache["q_rope_cache"])
                dk_pre = rope_backward(dk, cache["k_rope_cache"])
                dq2 = dq_pre.reshape(-1, self._q_width)
                dk2 = dk_pre.reshape(-1, self._kv_width)
                dv2 = dv.reshape(-1, self._kv_width)
                if is_bfloat16_dtype(x.dtype):
                    dq2 = dq2.astype(x.dtype, copy=False)
                    dk2 = dk2.astype(x.dtype, copy=False)
                    dv2 = dv2.astype(x.dtype, copy=False)
                with performance_scope("attention.qkv_projection.backward.pack"):
                    dqkv2 = xp.concatenate((dq2, dk2, dv2), axis=1)

            with performance_scope("attention.qkv_projection.backward.wgrad"):
                dw_packed = x2.T @ dqkv2
            with performance_scope("attention.qkv_projection.backward.grad_accumulate"):
                self._packed_qkv_grad += dw_packed

            with performance_scope("attention.qkv_projection.backward.dx"):
                dx2 = dqkv2 @ self._get_packed_qkv_weight().T
        else:
            dq_pre = rope_backward(dq, cache["q_rope_cache"])
            dk_pre = rope_backward(dk, cache["k_rope_cache"])
            dq2 = dq_pre.reshape(-1, self._q_width)
            dk2 = dk_pre.reshape(-1, self._kv_width)
            dv2 = dv.reshape(-1, self._kv_width)
            if is_bfloat16_dtype(x.dtype):
                dq2 = dq2.astype(x.dtype, copy=False)
                dk2 = dk2.astype(x.dtype, copy=False)
                dv2 = dv2.astype(x.dtype, copy=False)

            self.Wq.grad += x2.T @ dq2
            self.Wk.grad += x2.T @ dk2
            self.Wv.grad += x2.T @ dv2

            dx2 = (
                dq2 @ self.Wq.data.T
                + dk2 @ self.Wk.data.T
                + dv2 @ self.Wv.data.T
            )
        return dx2.reshape(b, t, self.d_model)

    def forward_shared_context_block(
        self, x, context_kv, *, context_length, valid_block_mask=None,
        return_cache=True, block_position_starts=None, group_plans=None,
        retrieval_tasks=None,
    ):
        """0079G: run *only* K active queries against read-only causal K/V.

        The context cache is constructed by an ordinary causal Transformer
        execution, using this same attention module and parameters. Only K
        query/key/value projections and RoPE evaluations occur per refinement.
        A bounded K+context K/V view is assembled for the indexed reference
        kernel; no causal-context attention or FFN is rerun.
        """
        if self.terminal_memory is not None:
            raise NotImplementedError("provisional terminal memory is not supported")
        if self._retrieval_groups and (
            retrieval_tasks is None or len(retrieval_tasks) != len(self._retrieval_groups)
        ):
            raise ValueError("learned retrieval needs a task for every retrieval group")
        b, ksize, d = x.shape
        c = int(context_length)
        k_context, v_context = context_kv
        expected = (b, c, self.n_kv_heads, self.d_head)
        if k_context.shape != expected or v_context.shape != expected:
            raise ValueError("context KV must match the active batch and causal prefix")
        if ksize < 1 or d != self.d_model:
            raise ValueError("expected [B,K,d_model] with K >= 1")
        x2 = x.reshape(-1, d)
        # The active stream uses the exact original weights and RoPE positions.
        q_pre = (x2 @ self.Wq.data).reshape(b, ksize, self.n_q_heads, self.d_head)
        k_pre = (x2 @ self.Wk.data).reshape(b, ksize, self.n_kv_heads, self.d_head)
        v_active = (x2 @ self.Wv.data).reshape(b, ksize, self.n_kv_heads, self.d_head)
        if block_position_starts is None:
            positions = xp.broadcast_to(
                xp.arange(c, c + ksize, dtype=xp.int64)[None, :], (b, ksize)
            )
        else:
            starts = xp.asarray(block_position_starts, dtype=xp.int64)
            if starts.shape != (b,):
                raise ValueError("block_position_starts must have shape [B_active]")
            positions = starts[:, None] + xp.arange(ksize, dtype=xp.int64)[None, :]
        q, q_rope_cache = rope_forward(q_pre, self.rope_base, position_ids=positions)
        k_active, k_rope_cache = rope_forward(k_pre, self.rope_base, position_ids=positions)
        shared_resident = getattr(k_context, '_is_shared_indexed_kv', False)
        if shared_resident:
            k_full = k_context.with_active(k_active)
            v_full = v_context.with_active(v_active)
        else:
            k_full = xp.concatenate((k_context, k_active), axis=1)
            v_full = xp.concatenate((v_context, v_active), axis=1)
        from mini_llm.predictive_refinement.compact_sparse import compact_provisional_plan
        groups = self._static_head_groups if self.attention_config is not None else ({
            "kind": "dense", "head_indices": tuple(range(self.n_q_heads)),
            "window": None, "dilation": None, "offset": 0, "stride": None,
        },)
        pieces, group_caches = [], []
        if group_plans is not None and len(group_plans) != len(groups):
            raise ValueError("precomputed plan count must match attention head groups")
        for group_idx, group in enumerate(groups):
            heads = group["head_indices"]
            if group_plans is not None:
                plan = group_plans[group_idx]
            else:
                if block_position_starts is not None:
                    raise ValueError("grouped starts require precomputed attention plans")
                plan = compact_provisional_plan(
                    c, ksize, b, kind=group["kind"], window=group["window"],
                    dilation=group["dilation"], offset=group["offset"] or 0,
                    stride=group["stride"], valid_block_mask=valid_block_mask,
                )
            indices = tuple(h // self.group_size for h in heads)
            if return_cache:
                out, attn_cache = indexed_attention_forward(
                    q[:, :, heads, :], k_full, v_full, plan,
                    kv_head_indices=indices, scale=self.scale, return_cache=True,
                )
                group_caches.append((heads, attn_cache))
            else:
                out = indexed_attention_forward(
                    q[:, :, heads, :], k_full, v_full, plan,
                    kv_head_indices=indices, scale=self.scale, return_cache=False,
                )
            pieces.append((heads, out))
        # Q3 retrieval heads use EXACTLY the learned causal-trunk router's
        # selected blocks and logit-bias prior. The selected keys live in the
        # original resident context cache; no 6K K/V copy is created.
        for task in (retrieval_tasks or ()):
            heads = task["heads"]
            rk, rv = task["kv"]
            retrieval_k = rk.with_active(k_active)
            retrieval_v = rv.with_active(v_active)
            if return_cache:
                out, attn_cache = indexed_attention_forward(
                    q[:, :, heads, :], retrieval_k, retrieval_v,
                    task["plan"], kv_head_indices=tuple(h // self.group_size for h in heads),
                    scale=self.scale, return_cache=True,
                )
                group_caches.append((heads, attn_cache, task))
            else:
                out = indexed_attention_forward(
                    q[:, :, heads, :], retrieval_k, retrieval_v,
                    task["plan"], kv_head_indices=tuple(h // self.group_size for h in heads),
                    scale=self.scale, return_cache=False,
                )
            pieces.append((heads, out))
        dtype = xp.float32 if is_bfloat16_dtype(q.dtype) else q.dtype
        merged_heads = xp.zeros(q.shape, dtype=dtype)
        for heads, out in pieces:
            merged_heads[:, :, heads, :] = out
        merged = merged_heads.reshape(b, ksize, self.n_q_heads * self.d_head)
        merged_compute = merged.astype(x.dtype, copy=False) if is_bfloat16_dtype(q.dtype) else merged
        y = (merged_compute.reshape(-1, self.n_q_heads * self.d_head) @ self.Wo.data).reshape(b, ksize, d)
        if not return_cache:
            return y
        return y, {
            "shared_context_active": True, "context_length": c,
            "x": x, "q": q, "k": k_active, "v": v_active,
            "merged": merged, "q_rope_cache": q_rope_cache,
            "k_rope_cache": k_rope_cache, "block_groups": group_caches,
            "shared_resident_kv": shared_resident,
        }

    def forward(
        self, x, return_cache=True, position_ids=None, terminal_memory=None,
        attention_allow_mask=None, block_context_length=None,
        block_valid_mask=None, historical_state=None,
    ):
        if x.ndim != 3:
            raise ValueError("attention input must have shape [B,T,D].")
        b, t, d_model = x.shape
        if d_model != self.d_model:
            raise ValueError("input final dimension != d_model.")

        # Keep projection GEMMs strictly 2-D. CuPy 14 supports BF16 2-D GEMM
        # efficiently, but its generic N-D @ 2-D path currently fails on BF16.
        x2 = x.reshape(-1, self.d_model)
        with performance_scope("attention.qkv_projection.forward"):
            if _packed_qkv_enabled():
                with performance_scope("attention.qkv_projection.forward.packed_gemm"):
                    qkv = x2 @ self._get_packed_qkv_weight()
                q_end = self._q_width
                k_end = q_end + self._kv_width
                q_pre = qkv[:, :q_end].reshape(
                    b, t, self.n_q_heads, self.d_head
                )
                k_pre = qkv[:, q_end:k_end].reshape(
                    b, t, self.n_kv_heads, self.d_head
                )
                v = qkv[:, k_end:].reshape(
                    b, t, self.n_kv_heads, self.d_head
                )
            else:
                q_pre = (x2 @ self.Wq.data).reshape(
                    b, t, self.n_q_heads, self.d_head
                )
                k_pre = (x2 @ self.Wk.data).reshape(
                    b, t, self.n_kv_heads, self.d_head
                )
                v = (x2 @ self.Wv.data).reshape(
                    b, t, self.n_kv_heads, self.d_head
                )

        with performance_scope("attention.rope.forward"):
            q, q_rope_cache = rope_forward(
                q_pre, self.rope_base, position_ids=position_ids
            )
            k, k_rope_cache = rope_forward(
                k_pre, self.rope_base, position_ids=position_ids
            )

        # 0055A: with packed QKV, V is a narrow strided view into the much
        # larger packed projection.  Q and K already have independent RoPE
        # outputs, so keeping that V view alive unnecessarily pins the entire
        # QKV backing allocation in every layer until backward.  Copy only V
        # (~1/6 of the packed tensor for the production GQA geometry) and let
        # the original packed storage be reclaimed.
        if _packed_qkv_enabled() and _compact_packed_v_enabled():
            with performance_scope("attention.qkv_projection.forward.compact_v"):
                v = xp.array(v, copy=True, order="C")
            del qkv, q_pre, k_pre

        if block_context_length is not None:
            if attention_allow_mask is not None or self.attention_config is None:
                raise ValueError("compact block mode requires static mixed heads and no dense mask")
            if self._retrieval_groups or self.terminal_memory is not None:
                raise NotImplementedError("routed/terminal memory requires boundary-causal router support")
            c = int(block_context_length)
            if c < 0 or c >= t:
                raise ValueError("compact block requires 0 <= context length < total length")
            if c:
                context, context_groups, _ = self._mixed_context_forward(
                    x[:, :c], q[:, :c], k[:, :c], v[:, :c], return_cache
                )
            else:
                context, context_groups = None, []
            from mini_llm.predictive_refinement.compact_sparse import compact_provisional_plan
            active_parts = []
            block_groups = []
            for group in self._static_head_groups:
                heads = group["head_indices"]
                plan = compact_provisional_plan(
                    c, t - c, b, kind=group["kind"], window=group["window"],
                    dilation=group["dilation"], offset=group["offset"] or 0,
                    stride=group["stride"], valid_block_mask=block_valid_mask,
                )
                kv_map = tuple(h // self.group_size for h in heads)
                if return_cache:
                    part, part_cache = indexed_attention_forward(
                        q[:, c:, heads, :], k, v, plan,
                        kv_head_indices=kv_map, scale=self.scale, return_cache=True,
                    )
                    block_groups.append((heads, part_cache))
                else:
                    part = indexed_attention_forward(
                        q[:, c:, heads, :], k, v, plan,
                        kv_head_indices=kv_map, scale=self.scale, return_cache=False,
                    )
                active_parts.append((heads, part))
            context_dtype = (
                xp.float32 if is_bfloat16_dtype(q.dtype) else q.dtype
            )
            merged_heads = xp.zeros(q.shape, dtype=context_dtype)
            if c:
                merged_heads[:, :c] = context
            for heads, part in active_parts:
                merged_heads[:, c:, heads, :] = part
            merged = merged_heads.reshape(b, t, self.n_q_heads * self.d_head)
            merged_compute = merged.astype(x.dtype, copy=False) if is_bfloat16_dtype(q.dtype) else merged
            with performance_scope("attention.output_projection.forward"):
                y = (merged_compute.reshape(-1, merged_compute.shape[-1]) @ self.Wo.data).reshape(b, t, self.d_model)
            if not return_cache:
                return y
            return y, {
                "x": x, "q": q, "k": k, "v": v, "merged": merged,
                "q_rope_cache": q_rope_cache, "k_rope_cache": k_rope_cache,
                "compact_context_length": c,
                "compact_context_groups": context_groups,
                "compact_block_groups": block_groups,
            }

        if attention_allow_mask is not None and self.attention_config is not None:
            raise NotImplementedError(
                "block-causal masks require dense reference attention; "
                "sparse/routed kernels must be audited separately"
            )
        if self.attention_config is not None:
            historical_skip_heads = ()
            if historical_state is not None and self.historical_reader is not None:
                # A batch containing any memoryless example retains its
                # corresponding current-window head fallback.
                if historical_state.all_rows_have_valid:
                    historical_skip_heads = tuple(
                        slot.head for slot in
                        self.historical_reader.slots[self.historical_layer]
                    )
            context, pattern_caches, routing = self._mixed_context_forward(
                x, q, k, v, return_cache, terminal_memory=terminal_memory,
                historical_skip_heads=historical_skip_heads,
            )
            historical_cache = None
            if historical_state is not None and self.historical_reader is not None:
                with performance_scope("attention.historical.forward"):
                    context, historical_cache = self.historical_reader.apply(
                        context, q, self.historical_layer, historical_state,
                        return_cache=return_cache,
                    )
            merged = context.reshape(b, t, self.n_q_heads * self.d_head)
            merged_compute = (
                merged.astype(x.dtype, copy=False)
                if is_bfloat16_dtype(q.dtype) else merged
            )
            with performance_scope("attention.output_projection.forward"):
                y = (
                    merged_compute.reshape(-1, merged_compute.shape[-1]) @ self.Wo.data
                ).reshape(b, t, self.d_model)
            if not return_cache:
                return y
            return y, {
                "x": x,
                "q": q,
                "k": k,
                "v": v,
                "merged": merged,
                "q_rope_cache": q_rope_cache,
                "k_rope_cache": k_rope_cache,
                "pattern_caches": pattern_caches,
                "routing": routing,
                "mixed": True,
                "historical_state": historical_state,
                "historical_read_cache": historical_cache,
            }

        # Native GQA representation -- no xp.repeat(K/V).
        q_grouped = self._group_queries(q)             # [B,Hkv,G,T,Dh]
        k_heads = k.transpose(0, 2, 1, 3)             # [B,Hkv,T,Dh]
        v_heads = v.transpose(0, 2, 1, 3)             # [B,Hkv,T,Dh]

        # Scale Q before the dot product.  For FP16, additionally pre-scale
        # the attention logits before the GEMM.  This keeps the expensive QK^T
        # matmul on the fast FP16 tensor-core path while increasing its overflow
        # headroom by 32x.  The exact attention temperature is restored inside
        # the stable softmax *after* subtracting the row maximum, where all
        # values are non-positive and therefore cannot overflow to +Inf.
        #
        # Algebraically, for s = scale * QK^T and r = 32:
        #   softmax(s) = softmax(r * (s/r))
        # and subtracting max(s/r) before multiplying by r is exact up to FP16
        # roundoff.  Backward continues to use ds/dQ for the original s, so no
        # gradient rescaling is required below.
        score_prescale = 1.0 / 32.0 if q_grouped.dtype == xp.float16 else 1.0

        # CuPy 14's generic batched matmul does not currently accept BF16
        # (dtype code 'E'). For BF16 training, run only the comparatively small
        # attention batched products in FP32. The large projection/MoE GEMMs
        # remain BF16 tensor-core GEMMs, while this path is both robust and
        # numerically stronger.
        bf16_attention = is_bfloat16_dtype(q_grouped.dtype)
        if bf16_attention:
            q_scaled = q_grouped.astype("float32") * self.scale
            k_mat = k_heads.astype("float32")
        else:
            q_scaled = q_grouped * (self.scale * score_prescale)
            k_mat = k_heads

        scores_grouped = xp.matmul(
            q_scaled,
            k_mat[:, :, xp.newaxis, :, :].swapaxes(-2, -1),
        )                                               # [B,Hkv,G,T,T]
        scores = scores_grouped.reshape(b, self.n_q_heads, t, t)

        if attention_allow_mask is None:
            causal = self._get_causal_mask(t)
            scores_masked = xp.where(causal[None, None, :, :], -xp.inf, scores)
        else:
            allow = xp.asarray(attention_allow_mask, dtype=xp.bool_)
            if allow.shape == (t, t):
                allow = allow[None, None, :, :]
            elif allow.shape == (b, t, t):
                allow = allow[:, None, :, :]
            else:
                raise ValueError("attention_allow_mask must be [T,T] or [B,T,T]")
            scores_masked = xp.where(allow, scores, -xp.inf)
        probs = softmax_forward(
            scores_masked,
            axis=-1,
            logit_multiplier=(1.0 / score_prescale),
        )

        # Softmax is FP32 for low-precision inputs. Cast probabilities back to
        # the branch compute dtype only for the tensor-core P@V GEMM.
        if bf16_attention:
            probs_compute = probs
            v_mat = v_heads.astype("float32")
        else:
            probs_compute = (
                probs.astype(q_grouped.dtype, copy=False)
                if is_low_precision_dtype(q_grouped.dtype) else probs
            )
            v_mat = v_heads
        probs_grouped_compute = probs_compute.reshape(
            b, self.n_kv_heads, self.group_size, t, t
        )
        context_grouped = xp.matmul(
            probs_grouped_compute,
            v_mat[:, :, xp.newaxis, :, :],
        )                                               # [B,Hkv,G,T,Dh]
        context = self._ungroup_queries(context_grouped)
        merged = context.reshape(b, t, self.n_q_heads * self.d_head)
        merged_compute = (
            merged.astype(x.dtype, copy=False) if bf16_attention else merged
        )
        with performance_scope("attention.output_projection.forward"):
            y = (
                merged_compute.reshape(-1, merged_compute.shape[-1]) @ self.Wo.data
            ).reshape(b, t, self.d_model)

        if not return_cache:
            return y

        return y, {
            "x": x,
            "q": q,
            "k": k,
            "v": v,
            "probs": probs,
            "merged": merged,
            "q_rope_cache": q_rope_cache,
            "k_rope_cache": k_rope_cache,
        }

    def backward(
        self, dy, cache, *, output_scale=None, return_output_scale_grad=False,
        shared_context_kv_grad=None, return_shared_context_kv_grad=False,
    ):
        x = cache["x"]
        q, k, v = cache["q"], cache["k"], cache["v"]
        merged = cache["merged"]
        b, t, _ = x.shape

        with performance_scope("attention.output_projection.backward"):
            merged2 = merged.reshape(-1, merged.shape[-1])
            dy2 = dy.reshape(-1, dy.shape[-1])
            output_scale_grad = None
            if output_scale is None:
                self.Wo.grad += merged2.T @ dy2
                dmerged = (dy2 @ self.Wo.data.T).reshape(merged.shape)
            else:
                # For y = alpha * (merged @ Wo), form the ungated output
                # projection gradient once.  Its Frobenius product with Wo is
                # exactly dL/d(alpha), while all ordinary branch gradients are
                # scaled by alpha.  This avoids retaining the full attention
                # output solely for the scalar gate gradient.
                raw_wgrad = merged2.T @ dy2
                if return_output_scale_grad:
                    acc_dtype = (
                        xp.float32
                        if is_low_precision_dtype(self.Wo.data.dtype)
                        else self.Wo.data.dtype
                    )
                    output_scale_grad = xp.sum(
                        raw_wgrad.astype(acc_dtype, copy=False)
                        * self.Wo.data.astype(acc_dtype, copy=False),
                        dtype=acc_dtype,
                    )
                self.Wo.grad += output_scale * raw_wgrad
                dmerged = (
                    (dy2 @ self.Wo.data.T) * output_scale
                ).reshape(merged.shape)
        dcontext = dmerged.reshape(
            b, t, self.n_q_heads, self.d_head
        )

        if cache.get("shared_context_active", False):
            c = int(cache["context_length"])
            q, k_active, v_active = cache["q"], cache["k"], cache["v"]
            b, ksize = q.shape[:2]
            grad_dtype = xp.float32 if is_bfloat16_dtype(q.dtype) else q.dtype
            dq = xp.zeros(q.shape, dtype=grad_dtype)
            shared_resident = bool(cache.get('shared_resident_kv', False))
            dk_context = (None if shared_resident else
                          xp.zeros((b, c, self.n_kv_heads, self.d_head), dtype=grad_dtype))
            dv_context = (None if shared_resident else xp.zeros_like(dk_context))
            dk_active = xp.zeros(k_active.shape, dtype=grad_dtype)
            dv_active = xp.zeros(v_active.shape, dtype=grad_dtype)
            for entry in cache["block_groups"]:
                heads, group_cache = entry[:2]
                dqa, dka, dva, dbias = indexed_attention_backward(
                    dcontext[:, :, heads, :], group_cache
                )
                if len(entry) == 3 and dbias is not None:
                    from mini_llm.predictive_refinement.routed_retrieval import accumulate_retrieval_bias_gradient
                    accumulate_retrieval_bias_gradient(entry[2], dbias)
                dq[:, :, heads, :] += dqa
                if shared_resident:
                    dk_active += dka
                    dv_active += dva
                else:
                    dk_context += dka[:, :c]
                    dv_context += dva[:, :c]
                    dk_active += dka[:, c:]
                    dv_active += dva[:, c:]
            dx = self._projection_backward(x, dq, dk_active, dv_active, cache)
            if return_shared_context_kv_grad:
                if return_output_scale_grad:
                    return dx, output_scale_grad, (dk_context, dv_context)
                return dx, (dk_context, dv_context)
            raise RuntimeError("shared-context active backward must collect context KV gradients")

        if "compact_context_length" in cache:
            c = cache["compact_context_length"]
            work_dtype = xp.float32 if is_bfloat16_dtype(q.dtype) else q.dtype
            dq = xp.zeros(q.shape, dtype=work_dtype)
            dk = xp.zeros(k.shape, dtype=work_dtype)
            dv = xp.zeros(v.shape, dtype=work_dtype)
            prefix_cache = {
                "x": x[:, :c], "q": q[:, :c], "k": k[:, :c], "v": v[:, :c],
                "pattern_caches": cache["compact_context_groups"],
            }
            if c:
                dq_c, dk_c, dv_c, _ = self._mixed_context_backward(dcontext[:, :c], prefix_cache)
                dq[:, :c] += dq_c
                dk[:, :c] += dk_c
                dv[:, :c] += dv_c
            for heads, group_cache in cache["compact_block_groups"]:
                dq_a, dk_a, dv_a, _ = indexed_attention_backward(
                    dcontext[:, c:, heads, :], group_cache
                )
                dq[:, c:, heads, :] += dq_a
                dk += dk_a
                dv += dv_a
            with performance_scope("attention.qkv_projection.backward"):
                dx = self._projection_backward(x, dq, dk, dv, cache)
            if return_output_scale_grad:
                return dx, output_scale_grad
            return dx

        if self.attention_config is not None:
            historical_dq = None
            if cache.get("historical_read_cache") is not None:
                with performance_scope("attention.historical.backward"):
                    dcontext, historical_dq = self.historical_reader.backward_reads(
                        dcontext, q, self.historical_layer,
                        cache["historical_state"], cache["historical_read_cache"],
                    )
            dq, dk, dv, drouter_input = self._mixed_context_backward(
                dcontext, cache
            )
            if historical_dq is not None:
                dq += historical_dq.astype(dq.dtype, copy=False)
            if shared_context_kv_grad is not None:
                dk += shared_context_kv_grad[0]
                dv += shared_context_kv_grad[1]
            with performance_scope("attention.qkv_projection.backward"):
                dx = self._projection_backward(x, dq, dk, dv, cache)
            if drouter_input is not None:
                dx = dx + drouter_input.astype(dx.dtype, copy=False)
            if return_output_scale_grad:
                return dx, output_scale_grad
            return dx

        probs = cache["probs"]
        q_grouped = self._group_queries(q)              # [B,Hkv,G,T,D]
        dcontext_grouped = self._group_queries(dcontext)
        k_heads = k.transpose(0, 2, 1, 3)              # [B,Hkv,T,D]
        v_heads = v.transpose(0, 2, 1, 3)
        bf16_attention = is_bfloat16_dtype(q_grouped.dtype)
        if bf16_attention:
            probs_compute = probs
            dcontext_mat = dcontext_grouped.astype("float32")
            v_mat = v_heads.astype("float32")
        else:
            probs_compute = (
                probs.astype(q_grouped.dtype, copy=False)
                if is_low_precision_dtype(q_grouped.dtype) else probs
            )
            dcontext_mat = dcontext_grouped
            v_mat = v_heads
        probs_grouped_compute = probs_compute.reshape(
            b, self.n_kv_heads, self.group_size, t, t
        )

        dprobs_grouped = xp.matmul(
            dcontext_mat,
            v_mat[:, :, xp.newaxis, :, :].swapaxes(-2, -1),
        )                                               # [B,Hkv,G,T,T]

        # V is shared by all query heads in a group; sum group contributions.
        dv_heads = xp.matmul(
            probs_grouped_compute.swapaxes(-2, -1),
            dcontext_mat,
        ).sum(axis=2)                                   # [B,Hkv,T,D]

        dprobs = dprobs_grouped.reshape(
            b, self.n_q_heads, t, t
        )
        dscores = softmax_backward(dprobs, probs, axis=-1)
        dscores_grouped = dscores.reshape(
            b, self.n_kv_heads, self.group_size, t, t
        )
        if bf16_attention:
            dscores_compute = dscores_grouped.astype("float32", copy=False)
            k_grad_mat = k_heads.astype("float32")
            q_grad_mat = q_grouped.astype("float32")
        else:
            dscores_compute = (
                dscores_grouped.astype(q_grouped.dtype, copy=False)
                if is_low_precision_dtype(q_grouped.dtype) else dscores_grouped
            )
            k_grad_mat = k_heads
            q_grad_mat = q_grouped

        dq_grouped = xp.matmul(
            dscores_compute,
            k_grad_mat[:, :, xp.newaxis, :, :],
        ) * self.scale

        # K is also shared by every query head in a group.
        dk_heads = xp.matmul(
            dscores_compute.swapaxes(-2, -1),
            q_grad_mat * self.scale,
        ).sum(axis=2)

        dq = self._ungroup_queries(dq_grouped)
        dk = dk_heads.transpose(0, 2, 1, 3)
        dv = dv_heads.transpose(0, 2, 1, 3)

        if shared_context_kv_grad is not None:
            dk += shared_context_kv_grad[0]
            dv += shared_context_kv_grad[1]
        with performance_scope("attention.qkv_projection.backward"):
            dx = self._projection_backward(x, dq, dk, dv, cache)
        if return_output_scale_grad:
            return dx, output_scale_grad
        return dx
