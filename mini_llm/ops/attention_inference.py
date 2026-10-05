"""Inference-only grouped-query attention with a preallocated KV cache."""

import math
import numpy as _np

from mini_llm.backend import xp, BACKEND_NAME, is_bfloat16_dtype, is_low_precision_dtype
from mini_llm.ops.rope import get_rope_cos_sin
from mini_llm.ops.indexed_attention import (
    local_window_attention_forward,
    dilated_attention_forward,
    global_sparse_attention_forward,
)
from mini_llm.ops.topk import selected_topk_softmax_forward
from mini_llm.ops.sparse_decode_attention import (
    HEAD_DENSE,
    HEAD_LOCAL,
    HEAD_DILATED,
    HEAD_GLOBAL,
    HEAD_RETRIEVAL,
    fused_sparse_decode_bf16_d64,
)
from mini_llm.ops.decode_qkv_fusion import (
    fused_decode_qkv_enabled,
    fused_unpack_rope_store_bf16_d64,
)
from mini_llm.ops.terminal_landmark_decode import terminal_landmark_decode


_CAUSAL_ALLOW_MASK_CACHE = {}


def _get_causal_allow_mask(length):
    """Return cached lower-triangular allow mask [length,length]."""
    length = int(length)
    mask = _CAUSAL_ALLOW_MASK_CACHE.get(length)
    if mask is None:
        pos = xp.arange(length, dtype=xp.int32)
        mask = pos[None, :] <= pos[:, None]
        _CAUSAL_ALLOW_MASK_CACHE[length] = mask
    return mask


class GQAAttentionInference:
    """Grouped-query attention for autoregressive inference with KV cache.

    K/V remain stored only for ``n_kv_heads``.  Query heads are viewed as
    ``[n_kv_heads, group_size]`` and batched matmul broadcasts shared K/V
    heads across each group, avoiding ``xp.repeat`` of the active KV cache.
    """

    def __init__(
        self, d_model, n_q_heads, n_kv_heads, d_head,
        rope_base=10_000.0, dtype="float32", max_context=None,
        attention_config=None,
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
        self.attention_config = attention_config
        self.sparse_attention = attention_config is not None
        self._static_head_groups = []
        self._retrieval_groups = []
        self._fused_decode_ready = False
        self._fused_head_kind = None
        self._fused_kv_head = None
        self._fused_p0 = None
        self._fused_p1 = None
        self._fused_p2 = None
        self._fused_retrieval_qmap = None
        self._fused_dummy_selected = None
        self._fused_dummy_bias = None
        # 0059H decode-only post-QKV fusion. This becomes True after the first
        # successful device launch, allowing the benchmarker to distinguish an
        # enabled flag from an actually active kernel.
        self._fused_qkv_decode_active = False
        self._terminal_memory_head_indices = ()
        self._terminal_memory_config = None

        self.Wq = None
        self.Wk = None
        self.Wv = None
        self.Wo = None
        self.Wqkv = None
        self._q_width = n_q_heads * d_head
        self._kv_width = n_kv_heads * d_head

        self._rope_capacity = 0
        self._rope_cos = None
        self._rope_sin = None
        if max_context is not None:
            self.ensure_rope_capacity(max_context)

    def set_weights(self, Wq, Wk, Wv, Wo, training_attention=None):
        """Set weights and build one persistent inference QKV matrix.

        For configurable sparse attention, ``training_attention`` supplies the
        already-constructed topology and learned ContextRouter modules.  The
        inference model shares those read-only router parameters instead of
        creating a second parameter set.
        """
        self.Wq = xp.asarray(Wq) if BACKEND_NAME == "cupy" else Wq
        self.Wk = xp.asarray(Wk) if BACKEND_NAME == "cupy" else Wk
        self.Wv = xp.asarray(Wv) if BACKEND_NAME == "cupy" else Wv
        self.Wo = xp.asarray(Wo) if BACKEND_NAME == "cupy" else Wo

        # Inference repeatedly multiplies the same hidden state by Q, K and V.
        # Pack the weights once at model construction/load time so prefill and
        # especially decode issue one GEMM instead of three tiny GEMMs.
        self.Wqkv = xp.concatenate(
            (self.Wq, self.Wk, self.Wv), axis=1
        )
        if self.sparse_attention:
            if training_attention is None:
                raise ValueError(
                    "configurable sparse inference requires training_attention "
                    "when weights are attached"
                )
            # These dictionaries are immutable topology metadata plus read-only
            # router modules during inference.  Copy the outer containers so
            # inference-local route state can never mutate the training module.
            self._static_head_groups = [dict(g) for g in training_attention._static_head_groups]
            self._retrieval_groups = [dict(g) for g in training_attention._retrieval_groups]
            if getattr(training_attention, "terminal_memory", None) is not None:
                self._terminal_memory_head_indices = tuple(
                    int(h) for h in training_attention._terminal_memory_head_indices
                )
                self._terminal_memory_config = training_attention.terminal_memory.config
            self._build_fused_decode_metadata()

    def _build_fused_decode_metadata(self):
        """Build persistent per-head metadata for the one-launch decode kernel.

        0059E intentionally supports one learned retrieval group per layer,
        which covers the current medium/wide presets.  More exotic topologies
        transparently keep using the exact legacy sparse decode implementation.
        """
        if BACKEND_NAME != "cupy" or self.d_head != 64:
            return
        if len(self._retrieval_groups) > 1:
            return

        kinds = _np.full(self.n_q_heads, -1, dtype=_np.int32)
        kv_heads = _np.empty(self.n_q_heads, dtype=_np.int32)
        p0 = _np.zeros(self.n_q_heads, dtype=_np.int32)
        p1 = _np.zeros(self.n_q_heads, dtype=_np.int32)
        p2 = _np.zeros(self.n_q_heads, dtype=_np.int32)
        rqmap = _np.full(self.n_q_heads, -1, dtype=_np.int32)

        kind_codes = {
            "dense": HEAD_DENSE,
            "local": HEAD_LOCAL,
            "dilated": HEAD_DILATED,
            "global_sparse": HEAD_GLOBAL,
        }
        for group in self._static_head_groups:
            kind = str(group["kind"])
            if kind not in kind_codes:
                return
            for h in group["head_indices"]:
                h = int(h)
                if kinds[h] != -1:
                    return
                kinds[h] = kind_codes[kind]
                kv_heads[h] = h // self.group_size
                if kind == "local":
                    p0[h] = int(group["window"])
                elif kind == "dilated":
                    p0[h] = int(group["window"])
                    p1[h] = int(group["dilation"])
                    p2[h] = int(group["offset"])
                elif kind == "global_sparse":
                    p0[h] = int(group["stride"])
                    p1[h] = int(group["offset"])
                    p2[h] = int(bool(group["include_current"]))

        if self._retrieval_groups:
            group = self._retrieval_groups[0]
            heads = tuple(int(h) for h in group["head_indices"])
            num_queries = int(group["module"].config.num_queries)
            if num_queries not in {1, len(heads)}:
                return
            for local_idx, h in enumerate(heads):
                if kinds[h] != -1:
                    return
                kinds[h] = HEAD_RETRIEVAL
                kv_heads[h] = h // self.group_size
                rqmap[h] = 0 if num_queries == 1 else local_idx

        if _np.any(kinds < 0):
            return

        self._fused_head_kind = xp.asarray(kinds)
        self._fused_kv_head = xp.asarray(kv_heads)
        self._fused_p0 = xp.asarray(p0)
        self._fused_p1 = xp.asarray(p1)
        self._fused_p2 = xp.asarray(p2)
        self._fused_retrieval_qmap = xp.asarray(rqmap)
        self._fused_dummy_selected = xp.zeros((1, 1, 1), dtype=xp.int64)
        self._fused_dummy_bias = xp.zeros((1, 1, 1), dtype=xp.float32)
        self._fused_decode_ready = True

    def ensure_rope_capacity(self, length):
        """Ensure absolute-position RoPE tables exist through ``length``."""
        length = int(length)
        if length <= self._rope_capacity:
            return
        self._rope_cos, self._rope_sin = get_rope_cos_sin(
            length, self.d_head, self.rope_base, self.dtype
        )
        self._rope_capacity = length

    def _weight_data(self):
        if BACKEND_NAME == "cupy":
            return self.Wq, self.Wk, self.Wv, self.Wo
        return tuple(
            w.data if hasattr(w, "data") else w
            for w in (self.Wq, self.Wk, self.Wv, self.Wo)
        )

    def _group_queries(self, q):
        """[B,T,Hq,D] -> [B,Hkv,G,T,D]."""
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

    def _apply_rope_absolute(self, x, start_pos):
        """Apply cached RoPE for contiguous positions beginning at start_pos."""
        _, t_new, _, d = x.shape
        end_pos = int(start_pos) + t_new
        self.ensure_rope_capacity(end_pos)

        cos = self._rope_cos[:, int(start_pos):end_pos, :, :]
        sin = self._rope_sin[:, int(start_pos):end_pos, :, :]

        # q_pre/k_pre are inference temporaries and are never reused after
        # RoPE. Rotate them in place to avoid allocating a second full tensor.
        # Preserve the even lanes because they are needed when writing odds.
        x0 = x[..., 0::2].copy()
        x1 = x[..., 1::2]
        x[..., 0::2] = x0 * cos - x1 * sin
        x[..., 1::2] = x0 * sin + x1 * cos
        return x

    def _apply_rope_positions(self, x, position_ids):
        """Apply cached RoPE for explicit [T] or [B,T] source positions."""
        b, t_new, _, _ = x.shape
        positions = xp.asarray(position_ids)
        if positions.ndim == 1:
            if int(positions.shape[0]) != int(t_new):
                raise ValueError("1D position_ids must have shape [T]")
        elif positions.ndim == 2:
            if tuple(positions.shape) != (int(b), int(t_new)):
                raise ValueError("2D position_ids must have shape [B,T]")
        else:
            raise ValueError("position_ids must have shape [T] or [B,T]")
        if positions.dtype.kind not in {"i", "u"}:
            raise TypeError("position_ids must use an integer dtype")
        if positions.size and bool(xp.any(positions < 0)):
            raise ValueError("position_ids must be non-negative")

        max_position = int(xp.max(positions).item()) if positions.size else -1
        self.ensure_rope_capacity(max_position + 1)
        base_cos = self._rope_cos[0, :, 0, :]
        base_sin = self._rope_sin[0, :, 0, :]
        if positions.ndim == 1:
            cos = base_cos[positions][None, :, None, :]
            sin = base_sin[positions][None, :, None, :]
        else:
            cos = base_cos[positions][:, :, None, :]
            sin = base_sin[positions][:, :, None, :]

        x0 = x[..., 0::2].copy()
        x1 = x[..., 1::2]
        x[..., 0::2] = x0 * cos - x1 * sin
        x[..., 1::2] = x0 * sin + x1 * cos
        return x

    def _attention(self, q, k_native, v_native, causal_mask=None):
        """Attend with native Hkv K/V tensors; never physically repeat them."""
        b, t_query, _, _ = q.shape
        t_key = k_native.shape[2]

        bf16_attention = is_bfloat16_dtype(q.dtype)

        if t_query == 1:
            # Decode fast path. CuPy's N-D BF16 matmul currently fails, so
            # promote only the batched attention products to FP32.
            q_decode = q[:, 0, :, :].reshape(
                b, self.n_kv_heads, self.group_size, self.d_head
            )
            q_scores = q_decode.astype(xp.float32, copy=False) if bf16_attention else q_decode
            k_scores = k_native.astype(xp.float32, copy=False) if bf16_attention else k_native
            scores_decode = xp.matmul(
                q_scores * self.scale,
                k_scores.swapaxes(-2, -1),
            )  # [B,Hkv,G,Tk]
            scores = scores_decode.reshape(
                b, self.n_q_heads, 1, t_key
            )
        else:
            q_grouped = self._group_queries(q)  # [B,Hkv,G,Tq,D]
            q_scores = q_grouped.astype(xp.float32, copy=False) if bf16_attention else q_grouped
            k_scores = k_native.astype(xp.float32, copy=False) if bf16_attention else k_native
            q_scaled = q_scores * self.scale
            scores_grouped = xp.matmul(
                q_scaled,
                k_scores[:, :, xp.newaxis, :, :].swapaxes(-2, -1),
            )                                  # [B,Hkv,G,Tq,Tk]
            scores = scores_grouped.reshape(
                b, self.n_q_heads, t_query, t_key
            )

        if causal_mask is not None:
            scores = xp.where(
                causal_mask[None, None, :, :], scores, -xp.inf
            )

        scores_f32 = scores.astype(xp.float32)
        scores_f32 -= xp.max(scores_f32, axis=-1, keepdims=True)
        probs = xp.exp(scores_f32)
        probs /= xp.sum(probs, axis=-1, keepdims=True)

        # Keep CuPy's unsupported N-D BF16 matmul out of P@V as well.
        # The result is cast back to the branch compute dtype before the large
        # output projection GEMM.
        if t_query == 1:
            probs_decode = probs.reshape(
                b, self.n_kv_heads, self.group_size, t_key
            )
            v_compute = v_native.astype(xp.float32, copy=False) if bf16_attention else v_native
            probs_compute = probs_decode if bf16_attention else probs_decode.astype(self.dtype, copy=False)
            context_decode = xp.matmul(
                probs_compute,
                v_compute,
            )  # [B,Hkv,G,D]
            context = context_decode.reshape(
                b, 1, self.n_q_heads, self.d_head
            )
            return context.astype(q.dtype, copy=False) if bf16_attention else context

        probs_grouped = probs.reshape(
            b, self.n_kv_heads, self.group_size, t_query, t_key
        )
        v_compute = v_native.astype(xp.float32, copy=False) if bf16_attention else v_native
        probs_compute = probs_grouped if bf16_attention else probs_grouped.astype(self.dtype, copy=False)
        context_grouped = xp.matmul(
            probs_compute,
            v_compute[:, :, xp.newaxis, :, :],
        )
        context = self._ungroup_queries(context_grouped)
        return context.astype(q.dtype, copy=False) if bf16_attention else context

    def _project_and_store(
        self, x, k_cache, v_cache, start_pos, *, single_token_decode=False,
        rope_start_pos=None, cache_position=None, position_ids=None,
    ):
        """Project one span and write native-Hkv K/V into the working cache.

        ``start_pos`` is the logical position inside the current working window.
        Long-memory inference may rotate Q/K at a much larger absolute RoPE
        position and write a one-token decode into a ring-buffer cache slot.
        """
        b, t_new, _ = x.shape
        logical_start = int(start_pos)
        logical_end = logical_start + int(t_new)
        rope_start = logical_start if rope_start_pos is None else int(rope_start_pos)
        if cache_position is None:
            cache_start = logical_start
            cache_end = cache_start + int(t_new)
            if cache_end > int(k_cache.shape[2]):
                raise ValueError(
                    f"input length {t_new} at cache position {cache_start} exceeds "
                    f"capacity {k_cache.shape[2]}"
                )
        else:
            if int(t_new) != 1:
                raise ValueError("ring-cache insertion is only supported for one-token decode")
            cache_start = int(cache_position)
            cache_end = cache_start + 1
            if not (0 <= cache_start < int(k_cache.shape[2])):
                raise ValueError("ring-cache insertion slot lies outside cache")

        qkv = (x.reshape(-1, self.d_model) @ self.Wqkv).reshape(b, t_new, -1)

        if (
            single_token_decode
            and int(t_new) == 1
            and self.d_head == 64
            and fused_decode_qkv_enabled()
            and is_bfloat16_dtype(qkv.dtype)
        ):
            self.ensure_rope_capacity(rope_start + 1)
            q = xp.empty((b, 1, self.n_q_heads, self.d_head), dtype=qkv.dtype)
            ran = fused_unpack_rope_store_bf16_d64(
                qkv, q, k_cache, v_cache, self._rope_cos, self._rope_sin,
                position=rope_start,
                cache_position=cache_start,
                n_q_heads=self.n_q_heads,
                n_kv_heads=self.n_kv_heads,
            )
            if ran:
                self._fused_qkv_decode_active = True
                return q, None, None, logical_end
            self._fused_qkv_decode_active = False

        q_end = self._q_width
        k_end = q_end + self._kv_width
        q_pre = qkv[..., :q_end].reshape(b, t_new, self.n_q_heads, self.d_head)
        k_pre = qkv[..., q_end:k_end].reshape(b, t_new, self.n_kv_heads, self.d_head)
        v = qkv[..., k_end:].reshape(b, t_new, self.n_kv_heads, self.d_head)
        if position_ids is None:
            q = self._apply_rope_absolute(q_pre, rope_start)
            k = self._apply_rope_absolute(k_pre, rope_start)
        else:
            q = self._apply_rope_positions(q_pre, position_ids)
            k = self._apply_rope_positions(k_pre, position_ids)
        if cache_position is None:
            k_cache[:, :, cache_start:cache_end, :] = k.transpose(0, 2, 1, 3)
            v_cache[:, :, cache_start:cache_end, :] = v.transpose(0, 2, 1, 3)
        else:
            k_cache[:, :, cache_start:cache_end, :] = k.transpose(0, 2, 1, 3)
            v_cache[:, :, cache_start:cache_end, :] = v.transpose(0, 2, 1, 3)
        return q, k, v, logical_end

    def _apply_terminal_memory_override(
        self, q, context, k_cache, v_cache, *, memory_store, memory_route,
        local_count, cache_start=0, terminal_row=-1,
    ):
        """Replace terminal retrieval-head context with 0058C Landmark memory."""
        if (
            memory_store is None
            or memory_route is None
            or not self._terminal_memory_head_indices
        ):
            return context
        heads = tuple(self._terminal_memory_head_indices)
        kv_heads = tuple(int(h) // self.group_size for h in heads)
        row = int(terminal_row)
        if row < 0:
            row += int(context.shape[1])
        q_term = xp.ascontiguousarray(q[:, row:row + 1, :, :])
        replacement = terminal_landmark_decode(
            q_term, k_cache, v_cache, memory_store, memory_route,
            query_heads=heads,
            local_kv_heads=kv_heads,
            local_count=int(local_count),
            cache_start=int(cache_start),
            gate_scale=float(self._terminal_memory_config.router_weight_scale),
            attn_scale=float(self.scale),
        )
        # TerminalMemoryInferenceStore only materializes a route once every
        # batch element has enough complete external blocks.  Assign the whole
        # batch directly; this avoids a tiny nonzero/advanced-index kernel and
        # an implicit host-side size check on every generated token.
        head_ids = xp.asarray(heads, dtype=xp.int64)
        context[:, row, head_ids, :] = replacement[:, 0].astype(
            context.dtype, copy=False
        )
        return context

    def _prefill_sparse(
        self, x, k_cache, v_cache, start_pos, return_all=False,
        router_input_cache=None, rope_start_pos=None, position_ids=None,
        terminal_memory_store=None, terminal_memory_route=None,
    ):
        """Exact sparse prefill using the same kernels as training forward."""
        if int(start_pos) != 0:
            raise NotImplementedError(
                "sparse prefill currently expects a fresh cache (start_pos=0)"
            )
        b, t_prompt, _ = x.shape
        q, k, v, end_pos = self._project_and_store(
            x, k_cache, v_cache, start_pos, rope_start_pos=rope_start_pos,
            position_ids=position_ids,
        )
        if router_input_cache is not None:
            router_input_cache[:, :end_pos, :] = x

        # Match training GQAAttention._mixed_context_forward exactly, but keep
        # no backward state.  Prompt prefill therefore remains vectorized while
        # autoregressive decode below touches only selected cache entries.
        context_dtype = xp.float32 if is_bfloat16_dtype(q.dtype) else q.dtype
        context = xp.zeros(q.shape, dtype=context_dtype)
        for group in self._static_head_groups:
            head_indices = tuple(group["head_indices"])
            q_subset = q[:, :, head_indices, :]
            kv_mapping = tuple(i // self.group_size for i in head_indices)
            kind = group["kind"]
            if kind == "dense":
                subset = local_window_attention_forward(
                    q_subset, k, v, t_prompt,
                    kv_head_indices=kv_mapping, scale=self.scale,
                    return_cache=False,
                )
            elif kind == "local":
                subset = local_window_attention_forward(
                    q_subset, k, v, group["window"],
                    kv_head_indices=kv_mapping, scale=self.scale,
                    return_cache=False,
                )
            elif kind == "dilated":
                subset = dilated_attention_forward(
                    q_subset, k, v, group["window"], group["dilation"],
                    group["offset"], kv_head_indices=kv_mapping,
                    scale=self.scale, return_cache=False,
                )
            elif kind == "global_sparse":
                subset = global_sparse_attention_forward(
                    q_subset, k, v, group["stride"], group["offset"],
                    group["include_current"], kv_head_indices=kv_mapping,
                    scale=self.scale, return_cache=False,
                )
            else:
                raise RuntimeError(f"unsupported sparse inference head kind {kind!r}")
            context[:, :, head_indices, :] = subset

        for group in self._retrieval_groups:
            head_indices = tuple(group["head_indices"])
            q_subset = q[:, :, head_indices, :]
            kv_mapping = tuple(i // self.group_size for i in head_indices)
            subset, _ = group["module"].forward(
                x, q_subset, k, v,
                kv_head_indices=kv_mapping,
                return_cache=False,
            )
            context[:, :, head_indices, :] = subset

        context = self._apply_terminal_memory_override(
            q, context, k_cache, v_cache,
            memory_store=terminal_memory_store,
            memory_route=terminal_memory_route,
            local_count=t_prompt,
            cache_start=0,
            terminal_row=t_prompt - 1,
        )

        merged = context.astype(q.dtype, copy=False).reshape(
            b, t_prompt, self.n_q_heads * self.d_head
        )
        y = (merged.reshape(-1, self.d_model) @ self.Wo).reshape(
            b, t_prompt, self.d_model
        )
        return y if return_all else y[:, -1, :]

    def _decode_indices(self, group, position):
        """Return exact causal cache indices for one static sparse head group."""
        p = int(position)
        kind = group["kind"]
        if kind == "dense":
            return xp.arange(p + 1, dtype=xp.int64)
        if kind == "local":
            first = max(0, p - int(group["window"]) + 1)
            return xp.arange(first, p + 1, dtype=xp.int64)
        if kind == "dilated":
            window = int(group["window"])
            dilation = int(group["dilation"])
            offset = int(group["offset"])
            last = p - offset
            if last < 0:
                return xp.zeros((0,), dtype=xp.int64)
            first = max(0, p - window + 1)
            # Smallest key >= first with the same phase as ``last``.
            n = (last - first) // dilation
            start = last - n * dilation
            return xp.arange(start, last + 1, dilation, dtype=xp.int64)
        if kind == "global_sparse":
            stride = int(group["stride"])
            offset = int(group["offset"])
            anchors = xp.arange(offset, p + 1, stride, dtype=xp.int64)
            if bool(group["include_current"]):
                current_is_anchor = (p >= offset) and ((p - offset) % stride == 0)
                if not current_is_anchor:
                    anchors = xp.concatenate((anchors, xp.asarray([p], dtype=xp.int64)))
            return anchors
        raise RuntimeError(f"unsupported sparse inference head kind {kind!r}")

    def _single_query_attention(
        self, q_subset, k_cache, v_cache, indices, kv_mapping, logit_bias=None,
        cache_start=0, fixed_prefix_length=0, working_capacity=None,
    ):
        """One-query sparse attention without expanding native GQA KV heads."""
        b, tq, n_heads, d = q_subset.shape
        if tq != 1:
            raise AssertionError("single-query sparse attention expects Tq=1")
        n_keys = int(indices.shape[-1]) if indices.ndim > 1 else int(indices.size)
        if n_keys == 0:
            return xp.zeros(q_subset.shape, dtype=q_subset.dtype)

        kv_ids = xp.asarray(kv_mapping, dtype=xp.int64)
        indices = self._map_logical_indices(
            indices,
            cache_start=cache_start,
            fixed_prefix_length=fixed_prefix_length,
            working_capacity=working_capacity,
            cache_capacity=int(k_cache.shape[2]),
        )
        # Common static patterns use [K]; retrieval may use [B,H,K].
        if indices.ndim == 1:
            k_native = k_cache[:, kv_ids, :, :][:, :, indices, :]
            v_native = v_cache[:, kv_ids, :, :][:, :, indices, :]
        elif indices.ndim == 3:
            batch_ids = xp.arange(b, dtype=xp.int64)[:, None, None]
            kv_grid = kv_ids[None, :, None]
            k_native = k_cache[batch_ids, kv_grid, indices, :]
            v_native = v_cache[batch_ids, kv_grid, indices, :]
        else:
            raise ValueError("decode indices must have shape [K] or [B,H,K]")

        qh = q_subset[:, 0, :, :]
        qf = qh.astype(xp.float32, copy=False)
        kf = k_native.astype(xp.float32, copy=False)
        scores = xp.matmul(
            (qf * self.scale)[:, :, None, :], kf.swapaxes(-1, -2)
        )[:, :, 0, :]
        if logit_bias is not None:
            scores = scores + logit_bias.astype(xp.float32, copy=False)
        scores -= xp.max(scores, axis=-1, keepdims=True)
        probs = xp.exp(scores)
        probs /= xp.sum(probs, axis=-1, keepdims=True)
        vf = v_native.astype(xp.float32, copy=False)
        ctx = xp.matmul(probs[:, :, None, :], vf)[:, :, 0, :]
        return ctx[:, None, :, :].astype(q_subset.dtype, copy=False)

    @staticmethod
    def _logical_ring_view(cache, cache_start, count):
        """Return logical [oldest..newest] rows from a physical ring cache."""
        count = int(count)
        cache_start = int(cache_start)
        capacity = int(cache.shape[1])
        if count <= 0:
            return cache[:, :0, :]
        if cache_start == 0 and count <= capacity:
            return cache[:, :count, :]
        end = cache_start + count
        if end <= capacity:
            return cache[:, cache_start:end, :]
        return xp.concatenate(
            (cache[:, cache_start:, :], cache[:, : end - capacity, :]), axis=1
        )

    @staticmethod
    def _map_logical_indices(
        indices, *, cache_start=0, fixed_prefix_length=0,
        working_capacity=None, cache_capacity=None,
    ):
        """Map compact active-sequence indices onto a fixed-prefix + ring cache."""
        prefix = int(fixed_prefix_length)
        if working_capacity is None:
            if cache_capacity is None:
                raise ValueError("cache_capacity is required when working_capacity is omitted")
            working_capacity = int(cache_capacity) - prefix
        working_capacity = int(working_capacity)
        if working_capacity <= 0:
            return indices
        start = int(cache_start)
        if prefix == 0:
            return (indices + start) % working_capacity
        return xp.where(
            indices < prefix,
            indices,
            prefix + ((indices - prefix + start) % working_capacity),
        )

    @classmethod
    def _logical_active_view(
        cls, cache, *, cache_start=0, count=None, fixed_prefix_length=0,
        working_capacity=None,
    ):
        """Materialize [fixed prefix | logical working ring] in compact order."""
        prefix = int(fixed_prefix_length)
        if working_capacity is None:
            working_capacity = int(cache.shape[1]) - prefix
        working_capacity = int(working_capacity)
        if count is None:
            count = prefix + working_capacity
        count = int(count)
        work_count = max(0, count - prefix)
        if prefix == 0:
            physical = cls._map_logical_indices(
                xp.arange(work_count, dtype=xp.int64),
                cache_start=cache_start,
                fixed_prefix_length=0,
                working_capacity=working_capacity,
            )
            return cache[:, physical, :]
        prefix_view = cache[:, :prefix, :]
        if work_count <= 0:
            return prefix_view
        work_logical = xp.arange(prefix, prefix + work_count, dtype=xp.int64)
        physical = cls._map_logical_indices(
            work_logical,
            cache_start=cache_start,
            fixed_prefix_length=prefix,
            working_capacity=working_capacity,
        )
        return xp.concatenate((prefix_view, cache[:, physical, :]), axis=1)

    def _retrieval_route_for_position(
        self, group, router_input_cache, end_pos, position,
        retrieval_route_cache, route_cache_key, *, cache_start=0,
        working_count=None, route_clock=None, working_start_abs=0,
        fixed_prefix_length=0, working_capacity=None,
    ):
        """Return the learned within-working-window retrieval route.

        0059I keeps the base 4k sparse router alive while the working cache is a
        ring. Routes are refreshed at the original routing stride. At a refresh
        boundary the small 4k router-input ring is materialized contiguously;
        between boundaries selected logical blocks are shifted so they continue
        to refer to the same absolute tokens while the window advances.
        """
        module = group["module"]
        router = module.router
        cfg = module.config
        p = int(position)
        count = int(end_pos if working_count is None else working_count)
        stride = int(cfg.routing_stride)
        block_size = int(cfg.history_block_size)
        route_start = (p // stride) * stride
        candidate_blocks = max(
            0, (route_start - int(cfg.exclude_recent_tokens)) // block_size
        )
        if route_start < stride or candidate_blocks < int(cfg.top_k_blocks):
            return None

        clock = p if route_clock is None else int(route_clock)
        refresh_id = clock // stride
        state = retrieval_route_cache.get(route_cache_key)
        if state is not None and int(state.get("refresh_id", -1)) == refresh_id:
            return state

        logical_cache = self._logical_active_view(
            router_input_cache,
            cache_start=cache_start,
            count=count,
            fixed_prefix_length=fixed_prefix_length,
            working_capacity=working_capacity,
        )
        batch = int(logical_cache.shape[0])
        token_end = candidate_blocks * block_size
        pooled, _ = router.history_pooler.forward(logical_cache[:, :token_end, :])
        history_proj = (
            pooled.reshape(-1, router.d_model) @ router.W_history.data
        ).reshape(batch, candidate_blocks, router.router_dim)

        route_starts = xp.asarray([route_start], dtype=xp.int64)
        query_pooled, _ = router.query_pooler.forward(
            logical_cache[:, :route_start, :], route_starts
        )
        query_proj = (
            query_pooled.reshape(-1, router.d_model) @ router.W_query.data
        ).reshape(batch, 1, router.num_queries, router.router_dim)

        q3 = query_proj.reshape(batch, router.num_queries, router.router_dim)
        if is_low_precision_dtype(q3.dtype):
            qscore = q3.astype(xp.float32, copy=False)
            hscore = history_proj.astype(xp.float32, copy=False)
        else:
            qscore = q3
            hscore = history_proj
        scores = xp.matmul(qscore, hscore.swapaxes(1, 2)) / math.sqrt(
            router.router_dim
        )
        weights, selected, _ = selected_topk_softmax_forward(
            scores[:, None, :, :], int(cfg.top_k_blocks),
            output_dtype=router_input_cache.dtype,
        )
        state = {
            "refresh_id": refresh_id,
            "route_start": route_start,
            "weights": xp.ascontiguousarray(weights[:, 0, :, :]),
            "selected": xp.ascontiguousarray(selected[:, 0, :, :]),
            "window_start_abs": int(working_start_abs),
        }
        if cfg.router_weight_mode == "logit_bias":
            wf = state["weights"].astype(xp.float32, copy=False)
            state["block_bias"] = xp.ascontiguousarray(
                float(cfg.router_weight_scale)
                * xp.log(wf + float(cfg.router_weight_eps))
            )
        else:
            state["block_bias"] = xp.zeros(
                state["selected"].shape, dtype=xp.float32
            )
        retrieval_route_cache[route_cache_key] = state
        return state

    def _fused_sparse_decode_context(
        self, q, k_cache, v_cache, router_input_cache, end_pos, position,
        retrieval_route_cache, layer_idx, *, cache_start=0, working_count=None,
        route_clock=None, working_start_abs=0, fixed_prefix_length=0,
        working_capacity=None,
    ):
        """Run the one-launch BF16 sparse read, including ring-cache mapping."""
        # The current fused kernel assumes one uniformly rotating cache.  0060B
        # keeps the routed prefix fixed while only the current-window segment
        # rotates, so use the exact fallback until a dedicated split-ring fused
        # kernel is added.
        if int(fixed_prefix_length) != 0 or working_capacity is not None:
            return None
        if not self._fused_decode_ready:
            return None

        selected = self._fused_dummy_selected
        block_bias = self._fused_dummy_bias
        retrieval_active = False
        retrieval_queries = 1
        retrieval_blocks = 1
        retrieval_block_size = 1
        retrieval_key_shift = 0

        if self._retrieval_groups:
            group = self._retrieval_groups[0]
            route = self._retrieval_route_for_position(
                group, router_input_cache, end_pos, position,
                retrieval_route_cache, (int(layer_idx), str(group["name"])),
                cache_start=cache_start, working_count=working_count,
                route_clock=route_clock, working_start_abs=working_start_abs,
                fixed_prefix_length=fixed_prefix_length,
                working_capacity=working_capacity,
            )
            cfg = group["module"].config
            retrieval_block_size = int(cfg.history_block_size)
            retrieval_blocks = int(cfg.top_k_blocks)
            retrieval_queries = int(cfg.num_queries)
            if route is not None:
                selected = route["selected"]
                block_bias = route["block_bias"]
                retrieval_active = True
                retrieval_key_shift = int(route.get("window_start_abs", working_start_abs)) - int(working_start_abs)

        context = xp.empty_like(q)
        ran = fused_sparse_decode_bf16_d64(
            q, k_cache, v_cache, context,
            head_kind=self._fused_head_kind,
            kv_head=self._fused_kv_head,
            p0=self._fused_p0,
            p1=self._fused_p1,
            p2=self._fused_p2,
            retrieval_qmap=self._fused_retrieval_qmap,
            selected_blocks=selected,
            retrieval_block_bias=block_bias,
            position=position,
            retrieval_active=retrieval_active,
            retrieval_queries=retrieval_queries,
            retrieval_blocks=retrieval_blocks,
            retrieval_block_size=retrieval_block_size,
            scale=self.scale,
            cache_start=cache_start,
            retrieval_key_shift=retrieval_key_shift,
        )
        if not ran:
            self._fused_decode_ready = False
            return None
        return context

    def _decode_retrieval_group(
        self, group, q, k_cache, v_cache, router_input_cache,
        end_pos, position, retrieval_route_cache, route_cache_key, *,
        cache_start=0, working_count=None, route_clock=None, working_start_abs=0,
        fixed_prefix_length=0, working_capacity=None,
    ):
        head_indices = tuple(group["head_indices"])
        q_subset = q[:, :, head_indices, :]
        kv_mapping = tuple(i // self.group_size for i in head_indices)
        route = self._retrieval_route_for_position(
            group, router_input_cache, end_pos, position,
            retrieval_route_cache, route_cache_key,
            cache_start=cache_start, working_count=working_count,
            route_clock=route_clock, working_start_abs=working_start_abs,
            fixed_prefix_length=fixed_prefix_length,
            working_capacity=working_capacity,
        )
        if route is None:
            return xp.zeros(q_subset.shape, dtype=q_subset.dtype)

        selected = route["selected"]
        weights = route["weights"]
        n_heads = len(head_indices)
        if int(selected.shape[1]) == 1 and n_heads != 1:
            selected = xp.broadcast_to(
                selected, (selected.shape[0], n_heads, selected.shape[-1])
            )
            weights = xp.broadcast_to(weights, selected.shape)
        block_size = int(group["module"].config.history_block_size)
        offsets = xp.arange(block_size, dtype=xp.int64)
        indices = (
            selected[..., None] * block_size + offsets[None, None, None, :]
        ).reshape(selected.shape[0], n_heads, -1)
        key_shift = int(route.get("window_start_abs", working_start_abs)) - int(working_start_abs)
        if key_shift:
            indices = indices + key_shift
        valid_idx = (indices >= 0) & (indices < int(working_count if working_count is not None else end_pos))

        logit_bias = None
        cfg = group["module"].config
        if cfg.router_weight_mode == "logit_bias":
            wf = weights.astype(xp.float32, copy=False)
            block_bias = float(cfg.router_weight_scale) * xp.log(
                wf + float(cfg.router_weight_eps)
            )
            logit_bias = xp.repeat(block_bias, block_size, axis=-1)
        if logit_bias is None:
            logit_bias = xp.zeros(indices.shape, dtype=xp.float32)
        logit_bias = xp.where(valid_idx, logit_bias, -1.0e30)
        indices = xp.where(valid_idx, indices, 0)
        return self._single_query_attention(
            q_subset, k_cache, v_cache, indices, kv_mapping,
            logit_bias=logit_bias, cache_start=cache_start,
            fixed_prefix_length=fixed_prefix_length,
            working_capacity=working_capacity,
        )

    def _decode_sparse(
        self, x, k_cache, v_cache, start_pos,
        router_input_cache=None, retrieval_route_cache=None, layer_idx=0, *,
        cache_position=None, rope_position=None, cache_start=0,
        working_count=None, route_clock=None, working_start_abs=0,
        fixed_prefix_length=0, working_capacity=None,
        terminal_memory_store=None, terminal_memory_route=None,
    ):
        if router_input_cache is None or retrieval_route_cache is None:
            raise ValueError("sparse decode requires router and route inference state")
        b, t_new, _ = x.shape
        if t_new != 1:
            raise AssertionError("decode_one processes exactly one token")
        logical_position = int(start_pos)
        if working_count is None:
            working_count = logical_position + 1
        cache_slot = logical_position if cache_position is None else int(cache_position)
        rope_pos = logical_position if rope_position is None else int(rope_position)
        q, _, _, end_pos = self._project_and_store(
            x, k_cache, v_cache, logical_position, single_token_decode=True,
            rope_start_pos=rope_pos, cache_position=cache_slot,
        )
        router_input_cache[:, cache_slot:cache_slot + 1, :] = x

        context = self._fused_sparse_decode_context(
            q, k_cache, v_cache, router_input_cache, end_pos, logical_position,
            retrieval_route_cache, layer_idx,
            cache_start=cache_start, working_count=working_count,
            route_clock=route_clock, working_start_abs=working_start_abs,
            fixed_prefix_length=fixed_prefix_length,
            working_capacity=working_capacity,
        )
        if context is None:
            context = xp.zeros(q.shape, dtype=q.dtype)
            for group in self._static_head_groups:
                head_indices = tuple(group["head_indices"])
                indices = self._decode_indices(group, logical_position)
                subset = self._single_query_attention(
                    q[:, :, head_indices, :], k_cache, v_cache, indices,
                    tuple(i // self.group_size for i in head_indices),
                    cache_start=cache_start,
                    fixed_prefix_length=fixed_prefix_length,
                    working_capacity=working_capacity,
                )
                context[:, :, head_indices, :] = subset

            for group in self._retrieval_groups:
                subset = self._decode_retrieval_group(
                    group, q, k_cache, v_cache, router_input_cache,
                    end_pos, logical_position, retrieval_route_cache,
                    (int(layer_idx), str(group["name"])),
                    cache_start=cache_start, working_count=working_count,
                    route_clock=route_clock, working_start_abs=working_start_abs,
                    fixed_prefix_length=fixed_prefix_length,
                    working_capacity=working_capacity,
                )
                context[:, :, tuple(group["head_indices"]), :] = subset

        context = self._apply_terminal_memory_override(
            q, context, k_cache, v_cache,
            memory_store=terminal_memory_store,
            memory_route=terminal_memory_route,
            local_count=working_count,
            cache_start=cache_start,
            terminal_row=0,
        )
        merged = context.reshape(b, 1, self.n_q_heads * self.d_head)
        y = (merged.reshape(-1, self.d_model) @ self.Wo).reshape(
            b, 1, self.d_model
        )
        return y[:, -1, :]

    def prefill(
        self, x, k_cache, v_cache, start_pos, return_all=False,
        router_input_cache=None, retrieval_route_cache=None, layer_idx=0, *,
        rope_start_pos=None, position_ids=None,
        terminal_memory_store=None, terminal_memory_route=None,
    ):
        """Prefill the cache for a prompt and run causal attention."""
        b, t_prompt, _ = x.shape
        cache_capacity = k_cache.shape[2]
        end_pos = int(start_pos) + t_prompt
        if end_pos > cache_capacity:
            raise ValueError(
                f"Prompt length {t_prompt} at position {start_pos} "
                f"exceeds cache capacity {cache_capacity}."
            )

        if BACKEND_NAME == "cupy" and isinstance(x, _np.ndarray):
            x = xp.asarray(x)

        if self.sparse_attention:
            return self._prefill_sparse(
                x, k_cache, v_cache, start_pos, return_all=return_all,
                router_input_cache=router_input_cache,
                rope_start_pos=rope_start_pos,
                position_ids=position_ids,
                terminal_memory_store=terminal_memory_store,
                terminal_memory_route=terminal_memory_route,
            )

        _, _, _, Wo = self._weight_data()
        qkv = (x.reshape(-1, self.d_model) @ self.Wqkv).reshape(
            b, t_prompt, -1
        )
        q_end = self._q_width
        k_end = q_end + self._kv_width
        q_pre = qkv[..., :q_end].reshape(
            b, t_prompt, self.n_q_heads, self.d_head
        )
        k_pre = qkv[..., q_end:k_end].reshape(
            b, t_prompt, self.n_kv_heads, self.d_head
        )
        v = qkv[..., k_end:].reshape(
            b, t_prompt, self.n_kv_heads, self.d_head
        )

        if position_ids is None:
            q = self._apply_rope_absolute(q_pre, start_pos)
            k = self._apply_rope_absolute(k_pre, start_pos)
        else:
            q = self._apply_rope_positions(q_pre, position_ids)
            k = self._apply_rope_positions(k_pre, position_ids)

        k_cache[:, :, int(start_pos):end_pos, :] = k.transpose(0, 2, 1, 3)
        v_cache[:, :, int(start_pos):end_pos, :] = v.transpose(0, 2, 1, 3)

        # Native cache views: [B,Hkv,Tk,D].
        k_all = k_cache[:, :, :end_pos, :]
        v_all = v_cache[:, :, :end_pos, :]

        # Reuse a shared absolute-position causal mask rather than rebuilding
        # arange/comparison tensors for every layer during prompt prefill.
        # Rows are the absolute query positions [start_pos:end_pos], columns
        # are keys [0:end_pos]. decode_one still needs no causal mask.
        causal_full = _get_causal_allow_mask(cache_capacity)
        causal_mask = causal_full[int(start_pos):end_pos, :end_pos]

        context = self._attention(q, k_all, v_all, causal_mask=causal_mask)
        merged = context.reshape(
            b, t_prompt, self.n_q_heads * self.d_head
        )
        y = (merged.reshape(-1, self.d_model) @ Wo).reshape(
            b, t_prompt, self.d_model
        )
        return y if return_all else y[:, -1, :]

    def decode_one(
        self, x, k_cache, v_cache, start_pos, router_input_cache=None,
        retrieval_route_cache=None, layer_idx=0, *, cache_position=None,
        rope_position=None, cache_start=0, working_count=None, route_clock=None,
        working_start_abs=0, fixed_prefix_length=0, working_capacity=None,
        terminal_memory_store=None, terminal_memory_route=None,
    ):
        """Decode exactly one token using cached native-Hkv K/V tensors."""
        b, t_new, _ = x.shape
        if t_new != 1:
            raise AssertionError("decode_one processes exactly one token")

        if BACKEND_NAME == "cupy" and isinstance(x, _np.ndarray):
            x = xp.asarray(x)

        if self.sparse_attention:
            return self._decode_sparse(
                x, k_cache, v_cache, start_pos,
                router_input_cache=router_input_cache,
                retrieval_route_cache=retrieval_route_cache,
                layer_idx=layer_idx, cache_position=cache_position,
                rope_position=rope_position, cache_start=cache_start,
                working_count=working_count, route_clock=route_clock,
                working_start_abs=working_start_abs,
                fixed_prefix_length=fixed_prefix_length,
                working_capacity=working_capacity,
                terminal_memory_store=terminal_memory_store,
                terminal_memory_route=terminal_memory_route,
            )

        _, _, _, Wo = self._weight_data()
        qkv = (x.reshape(-1, self.d_model) @ self.Wqkv).reshape(
            b, 1, -1
        )
        q_end = self._q_width
        k_end = q_end + self._kv_width
        q_pre = qkv[..., :q_end].reshape(
            b, 1, self.n_q_heads, self.d_head
        )
        k_pre = qkv[..., q_end:k_end].reshape(
            b, 1, self.n_kv_heads, self.d_head
        )
        v = qkv[..., k_end:].reshape(
            b, 1, self.n_kv_heads, self.d_head
        )

        rope_pos = int(start_pos) if rope_position is None else int(rope_position)
        q = self._apply_rope_absolute(q_pre, rope_pos)
        k = self._apply_rope_absolute(k_pre, rope_pos)

        end_pos = int(start_pos) + 1
        cache_slot = int(start_pos) if cache_position is None else int(cache_position)
        k_cache[:, :, cache_slot:cache_slot + 1, :] = k.transpose(0, 2, 1, 3)
        v_cache[:, :, cache_slot:cache_slot + 1, :] = v.transpose(0, 2, 1, 3)

        if working_capacity is None and int(fixed_prefix_length) == 0:
            k_all = k_cache[:, :, :end_pos, :]
            v_all = v_cache[:, :, :end_pos, :]
        else:
            active_count = int(working_count if working_count is not None else end_pos)
            logical = xp.arange(active_count, dtype=xp.int64)
            physical = self._map_logical_indices(
                logical,
                cache_start=cache_start,
                fixed_prefix_length=fixed_prefix_length,
                working_capacity=working_capacity,
                cache_capacity=int(k_cache.shape[2]),
            )
            k_all = k_cache[:, :, physical, :]
            v_all = v_cache[:, :, physical, :]
        context = self._attention(q, k_all, v_all, causal_mask=None)
        merged = context.reshape(b, 1, self.n_q_heads * self.d_head)
        y = (merged.reshape(-1, self.d_model) @ Wo).reshape(
            b, 1, self.d_model
        )
        return y[:, -1, :]
