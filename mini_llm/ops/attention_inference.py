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
        self, x, k_cache, v_cache, start_pos, *, single_token_decode=False
    ):
        """Project one contiguous input span and append its native-Hkv K/V.

        0059H specializes the post-GEMM work for BF16 autoregressive decode.
        Prefill deliberately keeps the existing vectorized path because it has
        many query rows and is already efficient.
        """
        b, t_new, _ = x.shape
        end_pos = int(start_pos) + int(t_new)
        if end_pos > int(k_cache.shape[2]):
            raise ValueError(
                f"input length {t_new} at position {start_pos} exceeds cache "
                f"capacity {k_cache.shape[2]}"
            )
        qkv = (x.reshape(-1, self.d_model) @ self.Wqkv).reshape(b, t_new, -1)

        if (
            single_token_decode
            and int(t_new) == 1
            and self.d_head == 64
            and fused_decode_qkv_enabled()
            and is_bfloat16_dtype(qkv.dtype)
        ):
            self.ensure_rope_capacity(end_pos)
            q = xp.empty(
                (b, 1, self.n_q_heads, self.d_head), dtype=qkv.dtype
            )
            ran = fused_unpack_rope_store_bf16_d64(
                qkv, q, k_cache, v_cache, self._rope_cos, self._rope_sin,
                position=int(start_pos),
                n_q_heads=self.n_q_heads,
                n_kv_heads=self.n_kv_heads,
            )
            if ran:
                self._fused_qkv_decode_active = True
                # Decode callers only consume Q; K/V have already been written
                # directly into their native cache slots.
                return q, None, None, end_pos
            self._fused_qkv_decode_active = False

        q_end = self._q_width
        k_end = q_end + self._kv_width
        q_pre = qkv[..., :q_end].reshape(b, t_new, self.n_q_heads, self.d_head)
        k_pre = qkv[..., q_end:k_end].reshape(b, t_new, self.n_kv_heads, self.d_head)
        v = qkv[..., k_end:].reshape(b, t_new, self.n_kv_heads, self.d_head)
        q = self._apply_rope_absolute(q_pre, start_pos)
        k = self._apply_rope_absolute(k_pre, start_pos)
        k_cache[:, :, int(start_pos):end_pos, :] = k.transpose(0, 2, 1, 3)
        v_cache[:, :, int(start_pos):end_pos, :] = v.transpose(0, 2, 1, 3)
        return q, k, v, end_pos

    def _prefill_sparse(
        self, x, k_cache, v_cache, start_pos, return_all=False,
        router_input_cache=None,
    ):
        """Exact sparse prefill using the same kernels as training forward."""
        if int(start_pos) != 0:
            raise NotImplementedError(
                "sparse prefill currently expects a fresh cache (start_pos=0)"
            )
        b, t_prompt, _ = x.shape
        q, k, v, end_pos = self._project_and_store(
            x, k_cache, v_cache, start_pos
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

    def _single_query_attention(self, q_subset, k_cache, v_cache, indices, kv_mapping, logit_bias=None):
        """One-query sparse attention without expanding native GQA KV heads."""
        b, tq, n_heads, d = q_subset.shape
        if tq != 1:
            raise AssertionError("single-query sparse attention expects Tq=1")
        n_keys = int(indices.shape[-1]) if indices.ndim > 1 else int(indices.size)
        if n_keys == 0:
            return xp.zeros(q_subset.shape, dtype=q_subset.dtype)

        kv_ids = xp.asarray(kv_mapping, dtype=xp.int64)
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

    def _retrieval_route_for_position(
        self, group, router_input_cache, end_pos, position,
        retrieval_route_cache, route_cache_key,
    ):
        """Return the learned block route controlling one decode position.

        Unlike the training/full-prefix router, incremental inference evaluates
        only the *current* route.  Historical block projections are cached once
        when blocks first become eligible, so a 64k decode does not repeatedly
        repool/reproject the entire prefix at every routing boundary.
        """
        module = group["module"]
        router = module.router
        cfg = module.config
        p = int(position)
        stride = int(cfg.routing_stride)
        block_size = int(cfg.history_block_size)
        route_start = (p // stride) * stride
        candidate_blocks = max(
            0, (route_start - int(cfg.exclude_recent_tokens)) // block_size
        )
        if route_start < stride or candidate_blocks < int(cfg.top_k_blocks):
            return None

        state = retrieval_route_cache.get(route_cache_key)
        if state is not None and int(state.get("route_start", -1)) == route_start:
            return state

        batch = int(router_input_cache.shape[0])
        if state is None:
            max_blocks = int(router_input_cache.shape[1]) // block_size
            state = {
                "route_start": -1,
                "weights": None,
                "selected": None,
                "block_bias": None,
                "projected_blocks": 0,
                "history_proj": xp.empty(
                    (batch, max_blocks, int(router.router_dim)),
                    dtype=router_input_cache.dtype,
                ),
            }
            retrieval_route_cache[route_cache_key] = state

        projected = int(state["projected_blocks"])
        if candidate_blocks > projected:
            token_start = projected * block_size
            token_end = candidate_blocks * block_size
            pooled, _ = router.history_pooler.forward(
                router_input_cache[:, token_start:token_end, :]
            )
            hproj = (
                pooled.reshape(-1, router.d_model) @ router.W_history.data
            ).reshape(batch, candidate_blocks - projected, router.router_dim)
            state["history_proj"][:, projected:candidate_blocks, :] = hproj
            state["projected_blocks"] = candidate_blocks

        route_starts = xp.asarray([route_start], dtype=xp.int64)
        query_pooled, _ = router.query_pooler.forward(
            router_input_cache[:, :route_start, :], route_starts
        )
        query_proj = (
            query_pooled.reshape(-1, router.d_model) @ router.W_query.data
        ).reshape(batch, 1, router.num_queries, router.router_dim)

        history_proj = state["history_proj"][:, :candidate_blocks, :]
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
        # selected_topk_softmax_forward also supplies the exact deterministic
        # tie handling used by training.  Its backward cache is intentionally
        # discarded during inference.
        weights, selected, _ = selected_topk_softmax_forward(
            scores[:, None, :, :], int(cfg.top_k_blocks),
            output_dtype=router_input_cache.dtype,
        )
        state["route_start"] = route_start
        state["weights"] = xp.ascontiguousarray(weights[:, 0, :, :])
        state["selected"] = xp.ascontiguousarray(selected[:, 0, :, :])
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
        return state

    def _fused_sparse_decode_context(
        self, q, k_cache, v_cache, router_input_cache, end_pos, position,
        retrieval_route_cache, layer_idx,
    ):
        """Run the 0059E one-launch BF16 sparse attention path when possible.

        Routing remains identical to the reference implementation.  The only
        difference is execution of the already-defined sparse K/V read.  A
        ``None`` return means the topology/dtype/backend is unsupported and the
        caller must use the legacy per-group implementation.
        """
        if not self._fused_decode_ready:
            return None

        selected = self._fused_dummy_selected
        block_bias = self._fused_dummy_bias
        retrieval_active = False
        retrieval_queries = 1
        retrieval_blocks = 1
        retrieval_block_size = 1

        if self._retrieval_groups:
            group = self._retrieval_groups[0]
            route = self._retrieval_route_for_position(
                group,
                router_input_cache,
                end_pos,
                position,
                retrieval_route_cache,
                (int(layer_idx), str(group["name"])),
            )
            cfg = group["module"].config
            retrieval_block_size = int(cfg.history_block_size)
            retrieval_blocks = int(cfg.top_k_blocks)
            retrieval_queries = int(cfg.num_queries)
            if route is not None:
                selected = route["selected"]
                block_bias = route["block_bias"]
                retrieval_active = True

        context = xp.empty_like(q)
        ran = fused_sparse_decode_bf16_d64(
            q,
            k_cache,
            v_cache,
            context,
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
        )
        if not ran:
            # Unsupported layout/toolchain: permanently use the proven legacy
            # path for this attention instance instead of retrying/allocating a
            # fused output buffer on every generated token.
            self._fused_decode_ready = False
            return None
        return context

    def _decode_retrieval_group(
        self, group, q, k_cache, v_cache, router_input_cache,
        end_pos, position, retrieval_route_cache, route_cache_key,
    ):
        head_indices = tuple(group["head_indices"])
        q_subset = q[:, :, head_indices, :]
        kv_mapping = tuple(i // self.group_size for i in head_indices)
        route = self._retrieval_route_for_position(
            group, router_input_cache, end_pos, position,
            retrieval_route_cache, route_cache_key,
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

        logit_bias = None
        cfg = group["module"].config
        if cfg.router_weight_mode == "logit_bias":
            wf = weights.astype(xp.float32, copy=False)
            block_bias = float(cfg.router_weight_scale) * xp.log(
                wf + float(cfg.router_weight_eps)
            )
            logit_bias = xp.repeat(block_bias, block_size, axis=-1)
        return self._single_query_attention(
            q_subset, k_cache, v_cache, indices, kv_mapping,
            logit_bias=logit_bias,
        )

    def _decode_sparse(
        self, x, k_cache, v_cache, start_pos,
        router_input_cache=None, retrieval_route_cache=None,
        layer_idx=0,
    ):
        if router_input_cache is None or retrieval_route_cache is None:
            raise ValueError("sparse decode requires router and route inference state")
        b, t_new, _ = x.shape
        if t_new != 1:
            raise AssertionError("decode_one processes exactly one token")
        q, _, _, end_pos = self._project_and_store(
            x, k_cache, v_cache, start_pos, single_token_decode=True
        )
        router_input_cache[:, int(start_pos):end_pos, :] = x

        fused_context = self._fused_sparse_decode_context(
            q,
            k_cache,
            v_cache,
            router_input_cache,
            end_pos,
            start_pos,
            retrieval_route_cache,
            layer_idx,
        )
        if fused_context is not None:
            merged = fused_context.reshape(
                b, 1, self.n_q_heads * self.d_head
            )
            y = (merged.reshape(-1, self.d_model) @ self.Wo).reshape(
                b, 1, self.d_model
            )
            return y[:, -1, :]

        context = xp.zeros(q.shape, dtype=q.dtype)
        for group_index, group in enumerate(self._static_head_groups):
            head_indices = tuple(group["head_indices"])
            indices = self._decode_indices(group, start_pos)
            subset = self._single_query_attention(
                q[:, :, head_indices, :], k_cache, v_cache, indices,
                tuple(i // self.group_size for i in head_indices),
            )
            context[:, :, head_indices, :] = subset

        for group_index, group in enumerate(self._retrieval_groups):
            subset = self._decode_retrieval_group(
                group, q, k_cache, v_cache, router_input_cache,
                end_pos, start_pos, retrieval_route_cache,
                (int(layer_idx), str(group["name"])),
            )
            context[:, :, tuple(group["head_indices"]), :] = subset

        merged = context.reshape(b, 1, self.n_q_heads * self.d_head)
        y = (merged.reshape(-1, self.d_model) @ self.Wo).reshape(
            b, 1, self.d_model
        )
        return y[:, -1, :]

    def prefill(
        self, x, k_cache, v_cache, start_pos, return_all=False,
        router_input_cache=None, retrieval_route_cache=None, layer_idx=0,
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

        q = self._apply_rope_absolute(q_pre, start_pos)
        k = self._apply_rope_absolute(k_pre, start_pos)

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
        retrieval_route_cache=None, layer_idx=0,
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
                layer_idx=layer_idx,
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

        q = self._apply_rope_absolute(q_pre, start_pos)
        k = self._apply_rope_absolute(k_pre, start_pos)

        end_pos = int(start_pos) + 1
        k_cache[:, :, int(start_pos):end_pos, :] = k.transpose(0, 2, 1, 3)
        v_cache[:, :, int(start_pos):end_pos, :] = v.transpose(0, 2, 1, 3)

        context = self._attention(
            q,
            k_cache[:, :, :end_pos, :],
            v_cache[:, :, :end_pos, :],
            causal_mask=None,
        )
        merged = context.reshape(b, 1, self.n_q_heads * self.d_head)
        y = (merged.reshape(-1, self.d_model) @ Wo).reshape(
            b, 1, self.d_model
        )
        return y[:, -1, :]
