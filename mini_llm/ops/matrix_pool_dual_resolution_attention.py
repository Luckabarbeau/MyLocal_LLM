"""Model B: mixed-window local attention plus lightweight matrix-pooled history."""

from __future__ import annotations

import math
import os

import numpy as np

from ..backend import BACKEND_NAME, is_low_precision_dtype, resolve_dtype, xp
from ..init import matrix_parameter
from ..parameter import Parameter
from ..performance_profiler import performance_scope
from .attention import softmax_backward, softmax_forward
from .rope import rope_backward, rope_forward
from .dual_resolution_attention import (
    DualResolutionAttention,
    _PackedProjection,
    _add_parameter_grad,
    _fast_projection_enabled,
    _linear_backward,
    _linear_forward,
    _work_array,
)


class MatrixPoolDualResolutionAttention:
    """Full-context Model B attention with learned matrix-pooled summaries.

    The operator returns ``local_delta + global_delta``.  ``residual_input`` is
    used only to construct the post-local state consumed by pooling/global
    attention; the Transformer block owns the actual residual addition.
    Backward therefore returns gradients for both the normalized attention input
    and this direct residual dependency.
    """

    def __init__(
        self, d_model, n_q_heads, n_kv_heads, d_head, input_std, output_std,
        rng, rope_base=10_000.0, name="matrix_pool_attention", dtype="float32",
        config=None, dual_resolution_config=None,
    ):
        if config is None:
            config = dual_resolution_config
        elif dual_resolution_config is not None and config is not dual_resolution_config:
            raise ValueError("provide only one dual-resolution config object")
        if config is None or getattr(config, "compression_mode", None) != "matrix_pool":
            raise ValueError("matrix-pool dual-resolution config required")
        self.d_model = int(d_model)
        self.n_q_heads = int(n_q_heads)
        self.n_kv_heads = int(n_kv_heads)
        self.d_head = int(d_head)
        if self.n_q_heads * self.d_head != self.d_model:
            raise ValueError("n_q_heads * d_head must equal d_model")
        if self.n_q_heads % self.n_kv_heads:
            raise ValueError("n_q_heads must be divisible by n_kv_heads")
        self.group_size = self.n_q_heads // self.n_kv_heads
        self.scale = 1.0 / math.sqrt(self.d_head)
        self.rope_base = float(rope_base)
        self.dtype = resolve_dtype(dtype)
        self.config = config
        self.block_size = int(config.block_size)
        self.local_window = int(config.local_window)
        self.long_local_window = int(config.long_local_window)
        self.long_local_heads = int(config.long_local_heads)
        self.summaries_per_block = int(config.summaries_per_block)
        self.global_heads = int(config.global_heads)
        self.global_head_dim = int(config.global_head_dim)
        self.global_width = self.global_heads * self.global_head_dim
        self.global_enabled = bool(config.global_enabled)
        self.pool_position_bias = bool(config.pool_position_bias)
        self.global_query_chunk = int(config.global_query_chunk)
        self.recompute_attention_probs = bool(config.recompute_attention_probs)
        if self.global_head_dim != self.d_head:
            raise ValueError("global_head_dim must equal the model d_head")
        self.local_kernel = os.environ.get(
            "MINI_LLM_DUAL_RESOLUTION_LOCAL_KERNEL",
            os.environ.get("MINI_LLM_DUAL_RESOLUTION_KERNEL", "cupy"),
        ).strip().lower()
        self.global_kernel = os.environ.get(
            "MINI_LLM_DUAL_RESOLUTION_GLOBAL_KERNEL", "cupy"
        ).strip().lower()
        if self.local_kernel not in {"cupy", "triton"}:
            raise ValueError("matrix-pool local kernel must be cupy or triton")
        if self.global_kernel not in {"cupy", "triton"}:
            raise ValueError("matrix-pool global kernel must be cupy or triton")

        q_width = self.n_q_heads * self.d_head
        kv_width = self.n_kv_heads * self.d_head
        self.Wq = matrix_parameter((self.d_model, q_width), input_std, rng,
                                   f"{name}.Wq", dtype=dtype)
        self.Wk = matrix_parameter((self.d_model, kv_width), input_std, rng,
                                   f"{name}.Wk", dtype=dtype)
        self.Wv = matrix_parameter((self.d_model, kv_width), input_std, rng,
                                   f"{name}.Wv", dtype=dtype)
        self.Wo = matrix_parameter((q_width, self.d_model), output_std, rng,
                                   f"{name}.Wo", dtype=dtype)
        self.W_pool_score = matrix_parameter(
            (self.d_model, self.summaries_per_block), input_std, rng,
            f"{name}.matrix_pool.score", dtype=dtype,
        )
        self.pool_position = Parameter(
            xp.zeros((self.block_size, self.summaries_per_block), dtype=self.dtype),
            name=f"{name}.matrix_pool.position_bias", decay=False,
        ) if self.pool_position_bias else None
        if self.global_enabled:
            self.W_global_q = matrix_parameter(
                (self.d_model, self.global_width), input_std, rng,
                f"{name}.global.Wq", dtype=dtype,
            )
            self.W_global_k = matrix_parameter(
                (self.d_model, self.global_width), input_std, rng,
                f"{name}.global.Wk", dtype=dtype,
            )
            self.W_global_v = matrix_parameter(
                (self.d_model, self.global_width), input_std, rng,
                f"{name}.global.Wv", dtype=dtype,
            )
            self.W_global_o = matrix_parameter(
                (self.global_width, self.d_model), output_std, rng,
                f"{name}.global.Wo", dtype=dtype,
            )
        else:
            self.W_global_q = self.W_global_k = None
            self.W_global_v = self.W_global_o = None

    _regular_metadata = DualResolutionAttention._regular_metadata
    _prepare_metadata = DualResolutionAttention._prepare_metadata

    def parameters(self):
        params = [self.Wq, self.Wk, self.Wv, self.Wo, self.W_pool_score]
        if self.pool_position is not None:
            params.append(self.pool_position)
        if self.global_enabled:
            params.extend([
                self.W_global_q, self.W_global_k,
                self.W_global_v, self.W_global_o,
            ])
        return params

    def zero_grad(self):
        for parameter in self.parameters():
            parameter.zero_grad()

    def _ensure_projection_groups(self):
        if not hasattr(self, "_packed_projection_groups"):
            self._packed_main = _PackedProjection((self.Wq, self.Wk, self.Wv))
            groups = [self._packed_main]
            if self.global_enabled:
                self._packed_global_kv = _PackedProjection(
                    (self.W_global_k, self.W_global_v)
                )
                groups.append(self._packed_global_kv)
            self._packed_projection_groups = tuple(groups)

    def refresh_compute_buffers(self):
        if _fast_projection_enabled():
            self._ensure_projection_groups()
            for group in self._packed_projection_groups:
                group.refresh()

    def _local_forward(self, q, k, v, documents, return_cache):
        if self.local_kernel == "triton":
            if BACKEND_NAME != "cupy":
                raise RuntimeError("Triton local attention requires CuPy")
            from .triton_dual_resolution_local import local_attention_forward
            result, cache = local_attention_forward(
                q, k, v, documents, window=self.local_window,
                long_window=self.long_local_window,
                long_heads=self.long_local_heads, scale=self.scale,
            )
            return result, ({"triton": cache} if return_cache else None)
        batch, length = q.shape[:2]
        output = xp.zeros(q.shape, dtype=_work_array(q).dtype)
        caches = []
        chunk = min(self.global_query_chunk, max(1, length))
        for row in range(batch):
            for head in range(self.n_q_heads):
                window = (self.long_local_window if head < self.long_local_heads
                          else self.local_window)
                kv_head = head // self.group_size
                for qs in range(0, length, chunk):
                    qe = min(length, qs + chunk)
                    ks = max(0, qs - window + 1)
                    ke = qe
                    qi = xp.arange(qs, qe)[:, None]
                    ki = xp.arange(ks, ke)[None, :]
                    allowed = ((ki <= qi) & (ki >= qi - window + 1)
                               & (documents[row, qs:qe, None]
                                  == documents[row, None, ks:ke]))
                    qh = _work_array(q[row, qs:qe, head])
                    kh = _work_array(k[row, ks:ke, kv_head])
                    vh = _work_array(v[row, ks:ke, kv_head])
                    scores = (qh * self.scale) @ kh.T
                    probabilities = softmax_forward(
                        xp.where(allowed, scores, -xp.inf), axis=-1
                    )
                    output[row, qs:qe, head] = probabilities @ vh
                    if return_cache:
                        caches.append((row, head, kv_head, qs, qe, ks, ke,
                                       None if self.recompute_attention_probs
                                       else probabilities))
        return output, caches

    def _local_backward(self, dout, q, k, v, cache, documents):
        if isinstance(cache, dict) and "triton" in cache:
            from .triton_dual_resolution_local import local_attention_backward
            return local_attention_backward(
                _work_array(dout).astype(xp.float32, copy=False), cache["triton"]
            )
        dq = xp.zeros(q.shape, dtype=_work_array(dout).dtype)
        dk = xp.zeros(k.shape, dtype=dq.dtype)
        dv = xp.zeros(v.shape, dtype=dq.dtype)
        for row, head, kv_head, qs, qe, ks, ke, probabilities in cache:
            window = (self.long_local_window if head < self.long_local_heads
                      else self.local_window)
            qh = _work_array(q[row, qs:qe, head])
            kh = _work_array(k[row, ks:ke, kv_head])
            vh = _work_array(v[row, ks:ke, kv_head])
            if probabilities is None:
                qi = xp.arange(qs, qe)[:, None]
                ki = xp.arange(ks, ke)[None, :]
                allowed = ((ki <= qi) & (ki >= qi - window + 1)
                           & (documents[row, qs:qe, None]
                              == documents[row, None, ks:ke]))
                probabilities = softmax_forward(
                    xp.where(allowed, (qh * self.scale) @ kh.T, -xp.inf), axis=-1
                )
            dc = _work_array(dout[row, qs:qe, head])
            dp = dc @ vh.T
            ds = softmax_backward(dp, probabilities, axis=-1)
            dq[row, qs:qe, head] += (ds @ kh) * self.scale
            dk[row, ks:ke, kv_head] += (ds.T @ qh) * self.scale
            dv[row, ks:ke, kv_head] += probabilities.T @ dc
        return dq, dk, dv

    def _valid_blocks(self, hidden, metadata):
        rows = metadata.get("valid_block_rows")
        slots = metadata.get("valid_block_slots")
        if rows is None or slots is None:
            rows, slots = xp.nonzero(metadata["block_valid"])
        indices = metadata["block_indices"][rows, slots]
        return rows, slots, hidden[rows[:, None], indices]

    def _pool_forward(self, hidden, metadata, return_cache):
        rows, slots, blocks = self._valid_blocks(hidden, metadata)
        block_count = metadata["block_valid"].shape[1]
        summaries = xp.zeros(
            (hidden.shape[0], block_count, self.summaries_per_block, self.d_model),
            dtype=xp.float32 if is_low_precision_dtype(hidden.dtype) else hidden.dtype,
        )
        if blocks.shape[0] == 0:
            return summaries, ({"rows": rows, "slots": slots, "blocks": blocks,
                                "probabilities": None} if return_cache else None)
        block_work = _work_array(blocks)
        score_input = blocks.reshape(-1, self.d_model).astype(
            self.W_pool_score.data.dtype, copy=False
        )
        scores = _work_array(score_input @ self.W_pool_score.data).reshape(
            blocks.shape[0], self.block_size, self.summaries_per_block
        )
        if self.pool_position is not None:
            scores = scores + _work_array(self.pool_position.data)[None, :, :]
        probabilities = softmax_forward(scores, axis=1)
        valid_summaries = probabilities.swapaxes(1, 2) @ block_work
        summaries[rows, slots] = valid_summaries
        cache = None
        if return_cache:
            cache = {"rows": rows, "slots": slots, "blocks": blocks,
                     "probabilities": probabilities}
        return summaries, cache

    def _pool_backward(self, d_summaries, cache, hidden_shape,
                       accumulate_parameter_grads):
        dx = xp.zeros(hidden_shape, dtype=xp.float32)
        probabilities = cache["probabilities"]
        if probabilities is None:
            return dx
        rows, slots = cache["rows"], cache["slots"]
        blocks = _work_array(cache["blocks"])
        ds = _work_array(d_summaries[rows, slots])
        dprob = blocks @ ds.swapaxes(1, 2)
        dz = softmax_backward(dprob, probabilities, axis=1)
        if accumulate_parameter_grads:
            _add_parameter_grad(
                self.W_pool_score,
                blocks.reshape(-1, self.d_model).T
                @ dz.reshape(-1, self.summaries_per_block),
            )
            if self.pool_position is not None:
                _add_parameter_grad(self.pool_position, xp.sum(dz, axis=0))
        dblocks = probabilities @ ds + dz @ _work_array(self.W_pool_score.data).T
        indices = cache["rows"][:, None]
        token_indices = cache.get("token_indices")
        if token_indices is None:
            # Recover the non-overlapping token indices from the prepared metadata
            # supplied by the caller through an attached cache entry.
            token_indices = cache["block_indices"]
        dx[indices, token_indices] = dblocks
        return dx

    def _summary_metadata(self, metadata):
        count = self.summaries_per_block
        return {
            "valid": xp.repeat(metadata["block_valid"], count, axis=1),
            "documents": xp.repeat(metadata["block_document_ids"], count, axis=1),
            "ends": xp.repeat(metadata["block_end_positions"], count, axis=1),
            "positions": xp.repeat(metadata["block_rope_positions"], count, axis=1),
        }

    def _global_forward(self, q, k, v, metadata, summary_metadata, return_cache):
        if self.global_kernel == "triton":
            if BACKEND_NAME != "cupy":
                raise RuntimeError("Triton global attention requires CuPy")
            from .triton_dual_resolution_dense import forward
            output, cache = forward(
                q, k, v, scale=self.scale,
                documents=metadata["document_ids"],
                positions=metadata["position_ids"],
                block_documents=summary_metadata["documents"],
                block_ends=summary_metadata["ends"],
                block_valid=summary_metadata["valid"],
            )
            return output, ({"triton": cache} if return_cache else None)
        batch, length = q.shape[:2]
        output = xp.zeros(q.shape, dtype=_work_array(q).dtype)
        caches = []
        if k.shape[1] == 0:
            return output, caches
        chunk = min(self.global_query_chunk, max(1, length))
        for row in range(batch):
            kh = _work_array(k[row]).transpose(1, 0, 2)
            vh = _work_array(v[row]).transpose(1, 0, 2)
            for qs in range(0, length, chunk):
                qe = min(length, qs + chunk)
                allowed = (summary_metadata["valid"][row, None, :]
                           & (metadata["document_ids"][row, qs:qe, None]
                              == summary_metadata["documents"][row, None, :])
                           & (summary_metadata["ends"][row, None, :]
                              < metadata["position_ids"][row, qs:qe, None]))
                active = xp.any(allowed, axis=-1)
                qh = _work_array(q[row, qs:qe]).transpose(1, 0, 2)
                scores = (qh * self.scale) @ kh.swapaxes(-2, -1)
                scores = xp.where(allowed[None, :, :], scores, -xp.inf)
                scores = xp.where(active[None, :, None], scores, 0.0)
                probabilities = softmax_forward(scores, axis=-1)
                probabilities *= allowed[None, :, :]
                output[row, qs:qe] = (probabilities @ vh).transpose(1, 0, 2)
                if return_cache:
                    caches.append((row, qs, qe, None if self.recompute_attention_probs
                                   else probabilities))
        return output, caches

    def _global_backward(self, dout, q, k, v, cache, metadata, summary_metadata):
        if isinstance(cache, dict) and "triton" in cache:
            from .triton_dual_resolution_dense import backward
            return backward(_work_array(dout).astype(xp.float32, copy=False),
                            cache["triton"])
        dq = xp.zeros(q.shape, dtype=_work_array(dout).dtype)
        dk = xp.zeros(k.shape, dtype=dq.dtype)
        dv = xp.zeros(v.shape, dtype=dq.dtype)
        for row, qs, qe, probabilities in cache:
            kh = _work_array(k[row]).transpose(1, 0, 2)
            vh = _work_array(v[row]).transpose(1, 0, 2)
            qh = _work_array(q[row, qs:qe]).transpose(1, 0, 2)
            if probabilities is None:
                allowed = (summary_metadata["valid"][row, None, :]
                           & (metadata["document_ids"][row, qs:qe, None]
                              == summary_metadata["documents"][row, None, :])
                           & (summary_metadata["ends"][row, None, :]
                              < metadata["position_ids"][row, qs:qe, None]))
                active = xp.any(allowed, axis=-1)
                scores = (qh * self.scale) @ kh.swapaxes(-2, -1)
                scores = xp.where(allowed[None, :, :], scores, -xp.inf)
                scores = xp.where(active[None, :, None], scores, 0.0)
                probabilities = softmax_forward(scores, axis=-1)
                probabilities *= allowed[None, :, :]
            dc = _work_array(dout[row, qs:qe]).transpose(1, 0, 2)
            dp = dc @ vh.swapaxes(-2, -1)
            ds = softmax_backward(dp, probabilities, axis=-1)
            dq[row, qs:qe] += ((ds @ kh) * self.scale).transpose(1, 0, 2)
            dk[row] += ((ds.swapaxes(-2, -1) @ qh).transpose(1, 0, 2)
                        * self.scale)
            dv[row] += (probabilities.swapaxes(-2, -1) @ dc).transpose(1, 0, 2)
        return dq, dk, dv

    def forward(self, x, return_cache=True, position_ids=None,
                dual_resolution_metadata=None, residual_input=None, **ignored):
        del ignored
        if residual_input is None:
            raise ValueError("Model B attention requires residual_input")
        batch, length, width = x.shape
        if width != self.d_model or residual_input.shape != x.shape:
            raise ValueError("attention and residual inputs must be [B,T,d_model]")
        metadata = self._prepare_metadata(
            dual_resolution_metadata, batch, length, position_ids
        )
        positions = metadata["position_ids"]
        if _fast_projection_enabled():
            self._ensure_projection_groups()
            qp, kp, vp = self._packed_main.forward(x)
        else:
            qp = _linear_forward(x, self.Wq)
            kp = _linear_forward(x, self.Wk)
            vp = _linear_forward(x, self.Wv)
        q_pre = qp.reshape(batch, length, self.n_q_heads, self.d_head)
        k_pre = kp.reshape(batch, length, self.n_kv_heads, self.d_head)
        v = vp.reshape(batch, length, self.n_kv_heads, self.d_head)
        q, q_rope_cache = rope_forward(q_pre, self.rope_base, positions)
        k, k_rope_cache = rope_forward(k_pre, self.rope_base, positions)
        with performance_scope("matrix_pool.local.forward"):
            local_context, local_cache = self._local_forward(
                q, k, v, metadata["document_ids"], return_cache
            )
        local_merged = local_context.reshape(batch, length, self.d_model)
        local_compute = local_merged.astype(x.dtype, copy=False)
        local_delta = _linear_forward(local_compute, self.Wo)
        local_state = (_work_array(residual_input)
                       + _work_array(local_delta))

        global_delta = xp.zeros_like(local_delta)
        global_cache = None
        if self.global_enabled and length > self.block_size:
            with performance_scope("matrix_pool.pool.forward"):
                summaries, pool_cache = self._pool_forward(
                    local_state, metadata, return_cache
                )
            summary_meta = self._summary_metadata(metadata)
            summary_flat = summaries.reshape(
                batch, -1, self.d_model
            ).astype(x.dtype, copy=False)
            state_compute = local_state.astype(x.dtype, copy=False)
            qg_projection = _linear_forward(state_compute, self.W_global_q)
            if _fast_projection_enabled():
                kg_projection, vg_projection = self._packed_global_kv.forward(summary_flat)
            else:
                kg_projection = _linear_forward(summary_flat, self.W_global_k)
                vg_projection = _linear_forward(summary_flat, self.W_global_v)
            qg_pre = qg_projection.reshape(batch, length, self.global_heads,
                                           self.global_head_dim)
            kg_pre = kg_projection.reshape(batch, -1, self.global_heads,
                                           self.global_head_dim)
            vg = vg_projection.reshape(batch, -1, self.global_heads,
                                       self.global_head_dim)
            qg, qg_rope_cache = rope_forward(qg_pre, self.rope_base, positions)
            kg, kg_rope_cache = rope_forward(
                kg_pre, self.rope_base, summary_meta["positions"]
            )
            with performance_scope("matrix_pool.global.forward"):
                global_context, global_attention_cache = self._global_forward(
                    qg, kg, vg, metadata, summary_meta, return_cache
                )
            global_merged = global_context.reshape(batch, length, self.global_width)
            global_compute = global_merged.astype(x.dtype, copy=False)
            global_delta = _linear_forward(global_compute, self.W_global_o)
            if return_cache:
                rows = pool_cache["rows"]
                slots = pool_cache["slots"]
                pool_cache["block_indices"] = metadata["block_indices"][rows, slots]
                global_cache = {
                    "pool": pool_cache, "summaries": summaries,
                    "summary_flat": summary_flat, "summary_meta": summary_meta,
                    "state_compute": state_compute, "qg": qg, "kg": kg, "vg": vg,
                    "qg_rope_cache": qg_rope_cache,
                    "kg_rope_cache": kg_rope_cache,
                    "attention": global_attention_cache,
                    "global_compute": global_compute,
                }
        output = local_delta + global_delta.astype(local_delta.dtype, copy=False)
        if not return_cache:
            return output
        return output, {
            "x": x, "residual_input": residual_input, "metadata": metadata,
            "q": q, "k": k, "v": v,
            "q_rope_cache": q_rope_cache, "k_rope_cache": k_rope_cache,
            "local_context": local_context, "local_compute": local_compute,
            "local_cache": local_cache, "global": global_cache,
        }

    def backward(self, dy, cache, output_scale=None,
                 return_output_scale_grad=False, shared_context_kv_grad=None,
                 return_shared_context_kv_grad=False,
                 accumulate_parameter_grads=True):
        if output_scale is not None or return_output_scale_grad:
            raise NotImplementedError("Model B does not support progressive residual gates")
        if shared_context_kv_grad is not None or return_shared_context_kv_grad:
            raise NotImplementedError("Model B does not use shared context KV")
        dy_work = _work_array(dy)
        batch, length, _ = dy.shape
        d_local_delta = dy_work
        d_residual_input = xp.zeros(cache["residual_input"].shape, dtype=xp.float32)
        global_cache = cache["global"]
        if global_cache is not None:
            global_compute = global_cache["global_compute"]
            dy2 = dy_work.reshape(-1, self.d_model)
            if accumulate_parameter_grads:
                _add_parameter_grad(
                    self.W_global_o,
                    _work_array(global_compute.reshape(-1, self.global_width)).T @ dy2,
                )
            dgc = (dy2 @ _work_array(self.W_global_o.data).T).reshape(
                batch, length, self.global_heads, self.global_head_dim
            )
            with performance_scope("matrix_pool.global.backward"):
                dqg, dkg, dvg = self._global_backward(
                    dgc, global_cache["qg"], global_cache["kg"], global_cache["vg"],
                    global_cache["attention"], cache["metadata"],
                    global_cache["summary_meta"],
                )
            dqg_pre = rope_backward(dqg, global_cache["qg_rope_cache"])
            dkg_pre = rope_backward(dkg, global_cache["kg_rope_cache"])
            dqg_flat = dqg_pre.reshape(batch, length, self.global_width)
            dkg_flat = dkg_pre.reshape(global_cache["summary_flat"].shape[:-1]
                                       + (self.global_width,))
            dvg_flat = dvg.reshape(dkg_flat.shape)
            dx_q = _linear_backward(
                dqg_flat, global_cache["state_compute"], self.W_global_q,
                accumulate_parameter_grads,
            )
            if _fast_projection_enabled():
                dsummary_flat = self._packed_global_kv.backward(
                    (dkg_flat, dvg_flat), global_cache["summary_flat"],
                    accumulate_parameter_grads,
                )
            else:
                dsummary_flat = _linear_backward(
                    dkg_flat, global_cache["summary_flat"], self.W_global_k,
                    accumulate_parameter_grads,
                )
                dsummary_flat += _linear_backward(
                    dvg_flat, global_cache["summary_flat"], self.W_global_v,
                    accumulate_parameter_grads,
                )
            dsummary = dsummary_flat.reshape(global_cache["summaries"].shape)
            with performance_scope("matrix_pool.pool.backward"):
                dx_pool = self._pool_backward(
                    dsummary, global_cache["pool"], cache["residual_input"].shape,
                    accumulate_parameter_grads,
                )
            dh = _work_array(dx_q) + dx_pool
            d_residual_input += dh
            d_local_delta = d_local_delta + dh

        dl2 = d_local_delta.reshape(-1, self.d_model)
        local_compute = cache["local_compute"]
        if accumulate_parameter_grads:
            _add_parameter_grad(
                self.Wo,
                _work_array(local_compute.reshape(-1, self.d_model)).T @ dl2,
            )
        dlocal_context = (dl2 @ _work_array(self.Wo.data).T).reshape(
            batch, length, self.n_q_heads, self.d_head
        )
        with performance_scope("matrix_pool.local.backward"):
            dq, dk, dv = self._local_backward(
                dlocal_context, cache["q"], cache["k"], cache["v"],
                cache["local_cache"], cache["metadata"]["document_ids"],
            )
        dq_pre = rope_backward(dq, cache["q_rope_cache"])
        dk_pre = rope_backward(dk, cache["k_rope_cache"])
        q_width = self.n_q_heads * self.d_head
        kv_width = self.n_kv_heads * self.d_head
        if _fast_projection_enabled():
            dx_norm = self._packed_main.backward(
                (dq_pre.reshape(-1, q_width), dk_pre.reshape(-1, kv_width),
                 dv.reshape(-1, kv_width)),
                cache["x"], accumulate_parameter_grads,
            )
        else:
            dx_norm = _linear_backward(
                dq_pre.reshape(batch, length, q_width), cache["x"], self.Wq,
                accumulate_parameter_grads,
            )
            dx_norm += _linear_backward(
                dk_pre.reshape(batch, length, kv_width), cache["x"], self.Wk,
                accumulate_parameter_grads,
            )
            dx_norm += _linear_backward(
                dv.reshape(batch, length, kv_width), cache["x"], self.Wv,
                accumulate_parameter_grads,
            )
        return dx_norm, d_residual_input
