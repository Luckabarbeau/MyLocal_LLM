"""Optional independent 64-wide exact attention over raw history + current states.

Each enabled layer has independent Q/K/V/O.  The original multi-head attention
is unaffected. The native kernel uses lower-right causality, which gives query
row i access to all valid historical rows and current rows 0..i. No N_query by
N_history score tensor is explicitly allocated.

The CUDA path uses a native Triton streaming forward/backward kernel over CuPy
allocations. It retains only linear output/LSE state and never materializes a
Q-by-K score, probability, or boolean causal-mask tensor.
"""

import math

from mini_llm.backend import xp, BACKEND_NAME
from mini_llm.init import matrix_parameter
from mini_llm.parameter import Parameter



def _linear(a, w):
    # Use 2D BF16 tensor-core GEMM rather than CuPy's unsupported ND BF16 dot.
    b = a.reshape(-1, a.shape[-1]).astype(w.dtype, copy=False)
    return (b @ w).reshape(*a.shape[:-1], w.shape[-1])


def _reference_attention(q, k, v, allow, scale):
    """Small NumPy reference with defined zero output for fully masked rows."""
    scores = (q.astype(xp.float32) @ k.astype(xp.float32).T) * float(scale)
    masked = xp.where(allow, scores, -xp.inf)
    row_valid = xp.any(allow, axis=-1)
    safe_max = xp.max(xp.where(allow, scores, -xp.inf), axis=-1, keepdims=True)
    safe_max = xp.where(row_valid[:, None], safe_max, 0.0)
    probabilities = xp.where(allow, xp.exp(masked - safe_max), 0.0)
    denominator = xp.sum(probabilities, axis=-1, keepdims=True)
    probabilities = xp.where(
        row_valid[:, None], probabilities / xp.maximum(denominator, 1e-30), 0.0
    )
    output = probabilities @ v.astype(xp.float32)
    return output.astype(q.dtype, copy=False), (q, k, v, probabilities, allow, float(scale))


def _reference_attention_backward(grad, cache):
    q, k, v, probabilities, allow, scale = cache
    go = grad.astype(xp.float32, copy=False)
    p = probabilities.astype(xp.float32, copy=False)
    qf, kf, vf = (a.astype(xp.float32, copy=False) for a in (q, k, v))
    dv = p.T @ go
    dp = go @ vf.T
    ds = p * (dp - xp.sum(dp * p, axis=-1, keepdims=True))
    ds = xp.where(allow, ds * scale, 0.0)
    return ds @ kf, ds.T @ qf, dv


def _rope(a, positions, base, *, reverse=False):
    """Interleaved RoPE on [T,64] with exact source positions, FP32 math."""
    d = a.shape[-1]
    freq = 1.0 / (float(base) ** (xp.arange(0, d, 2, dtype=xp.float32) / d))
    angle = positions.astype(xp.float32)[:, None] * freq[None, :]
    c, s = xp.cos(angle), xp.sin(angle)
    if reverse:
        s = -s
    original = a.astype(xp.float32, copy=False)
    result = xp.empty_like(original)
    result[:, ::2] = original[:, ::2] * c - original[:, 1::2] * s
    result[:, 1::2] = original[:, ::2] * s + original[:, 1::2] * c
    return result.astype(a.dtype, copy=False)


class FullHistoryHead:
    """An additive head; W_o starts at zero for exact checkpoint parity."""

    def __init__(self, *, d_model, d_head, embedding, rng, init_std, rope_base,
                 name, dtype, zero_output_init=True, output_init_std=None):
        self.d_model = int(d_model)
        self.width = int(d_head)
        self.embedding = embedding
        self.rope_base = float(rope_base)
        self.name = name
        self.q = matrix_parameter((d_model, d_head), init_std, rng,
                                  name + ".Wq", dtype=dtype)
        self.k = matrix_parameter((d_model, d_head), init_std, rng,
                                  name + ".Wk", dtype=dtype)
        self.v = matrix_parameter((d_model, d_head), init_std, rng,
                                  name + ".Wv", dtype=dtype)
        if zero_output_init:
            self.o = Parameter(xp.zeros((d_head, d_model), dtype=dtype),
                               name=name + ".Wo")
        else:
            self.o = matrix_parameter(
                (d_head, d_model),
                init_std if output_init_std is None else output_init_std,
                rng,
                name + ".Wo",
                dtype=dtype,
            )
        self.project_chunk = 4096

    def parameters(self):
        return [self.q, self.k, self.v, self.o]

    def _history_kv(self, ids):
        """Project raw token embeddings in bounded chunks (no 65K x 1024 copy)."""
        n = int(ids.size)
        k = xp.empty((n, self.width), dtype=self.k.data.dtype)
        v = xp.empty_like(k)
        # Group independent K/V projections into one GEMM per chunk. The
        # weights remain independently registered, optimized and checkpointed.
        packed = xp.concatenate((self.k.data, self.v.data), axis=-1)
        for start in range(0, n, self.project_chunk):
            stop = min(start + self.project_chunk, n)
            embeddings = self.embedding.data[ids[start:stop]]
            kv = _linear(embeddings, packed)
            k[start:stop] = kv[:, :self.width]
            v[start:stop] = kv[:, self.width:]
        return k, v

    def forward(self, current, historical_state, positions, *, return_cache=True,
                attention_allow_mask=None, current_valid_mask=None,
                historical_allow_mask=None):
        """Unified causal softmax over compacted valid history and current rows.

        ``attention_allow_mask`` uses current-window coordinates [T,T] or [B,T,T].
        ``historical_allow_mask`` optionally uses source-history coordinates
        [B,T,H_capacity]. Invalid query rows return an exact zero correction.
        """
        if positions is None:
            raise ValueError("full_history_head requires absolute current positions")
        if current.ndim != 3:
            raise ValueError("current must have shape [B,T,D]")
        batch, length, _ = current.shape
        if positions.shape != (batch, length):
            raise ValueError("absolute positions must be [B,T]")
        if current_valid_mask is None:
            current_valid = xp.ones((batch, length), dtype=xp.bool_)
        else:
            current_valid = xp.asarray(current_valid_mask, dtype=xp.bool_)
            if current_valid.shape != (batch, length):
                raise ValueError("current_valid_mask must be [B,T]")
        current_allow = None
        if attention_allow_mask is not None:
            current_allow = xp.asarray(attention_allow_mask, dtype=xp.bool_)
            if current_allow.shape == (length, length):
                current_allow = xp.broadcast_to(current_allow[None], (batch, length, length))
            elif current_allow.shape != (batch, length, length):
                raise ValueError("attention_allow_mask must be [T,T] or [B,T,T]")
        history_allow = None
        if historical_allow_mask is not None:
            history_allow = xp.asarray(historical_allow_mask, dtype=xp.bool_)
            if history_allow.ndim != 3 or history_allow.shape[:2] != (batch, length):
                raise ValueError("historical_allow_mask must be [B,T,H_capacity]")
        results, rows = [], []
        scale = 1.0 / math.sqrt(self.width)
        for row in range(int(batch)):
            if historical_state is None:
                indices = xp.empty((0,), dtype=xp.int64)
                ids = xp.empty((0,), dtype=xp.int32)
                h_pos = xp.empty((0,), dtype=xp.int64)
            else:
                indices = xp.nonzero(historical_state.valid[row])[0]
                ids = historical_state.ids[row, indices]
                positions_source = historical_state.positions
                h_pos = (positions_source[row, indices] if positions_source.ndim == 2
                         else positions_source[indices])
            valid_current_indices = xp.nonzero(current_valid[row])[0]
            if indices.size and valid_current_indices.size and bool(
                xp.any(h_pos >= positions[row, valid_current_indices[0]])
            ):
                raise ValueError("historical tokens must strictly precede current window")
            h_k, h_v = self._history_kv(ids)
            c = current[row]
            q_all = _linear(c, self.q.data)
            c_k_all = _linear(c, self.k.data)
            c_v_all = _linear(c, self.v.data)
            q = q_all[valid_current_indices]
            c_k = c_k_all[valid_current_indices]
            c_v = c_v_all[valid_current_indices]
            c_pos = positions[row, valid_current_indices]
            q_rope = _rope(q, c_pos, self.rope_base)
            k_rope = xp.concatenate((
                _rope(h_k, h_pos, self.rope_base), _rope(c_k, c_pos, self.rope_base),
            ), axis=0)
            values = xp.concatenate((h_v, c_v), axis=0)
            compact_length = int(valid_current_indices.size)
            h_len = int(ids.size)
            allow = None
            if BACKEND_NAME != "cupy":
                causal = (xp.arange(compact_length)[:, None] >=
                          xp.arange(compact_length)[None, :])
                if current_allow is not None:
                    selected = current_allow[row][
                        valid_current_indices[:, None], valid_current_indices[None, :]
                    ]
                    causal &= selected
                history_visible = xp.ones((compact_length, h_len), dtype=xp.bool_)
                if history_allow is not None and h_len:
                    if history_allow.shape[2] != historical_state.valid.shape[1]:
                        raise ValueError("historical_allow_mask capacity mismatch")
                    history_visible &= history_allow[row][
                        valid_current_indices[:, None], indices[None, :]
                    ]
                allow = xp.concatenate((history_visible, causal), axis=1)
            if BACKEND_NAME != "cupy" and compact_length * int(k_rope.shape[0]) > 2_000_000:
                raise RuntimeError(
                    "CPU full-history reference limited to 2M attention scores; "
                    "use MINI_LLM_BACKEND=cupy with native Triton attention for long context"
                )
            if compact_length == 0:
                out_backend = xp.empty((0, self.width), dtype=current.dtype)
                attention_cache = None
            elif BACKEND_NAME == "cupy":
                if self.width != 64:
                    raise RuntimeError(
                        "native full-history Triton attention currently requires d_head=64"
                    )
                if current_allow is not None or history_allow is not None:
                    raise NotImplementedError(
                        "native full-history Triton attention requires compact causal "
                        "validity; arbitrary query-dependent masks need a structured "
                        "document-boundary representation"
                    )
                from mini_llm.ops.triton_full_history_attention import (
                    full_history_attention_forward,
                )
                result = full_history_attention_forward(
                    q_rope, k_rope, values, return_cache=return_cache
                )
                if return_cache:
                    out_backend, attention_cache = result
                else:
                    out_backend, attention_cache = result, None
            else:
                out_backend, attention_cache = _reference_attention(
                    q_rope, k_rope, values, allow, scale
                )
            full_out = xp.zeros((length, self.width), dtype=out_backend.dtype)
            full_out[valid_current_indices] = out_backend
            results.append(full_out)
            if return_cache:
                rows.append((c, ids, indices, h_pos, positions[row], valid_current_indices,
                             attention_cache, h_len))
        retrieved = xp.stack(results, axis=0)
        output = _linear(retrieved, self.o.data)
        return output, ((retrieved, rows) if return_cache else None)

    def backward(self, grad, cache, *, output_scale=None,
                 return_output_scale_grad=False,
                 accumulate_embedding_grad=True):
        if cache is None:
            raise ValueError("full history backward requires forward cache")
        retrieved, rows = cache
        grad32 = grad.astype(xp.float32, copy=False)
        scale = 1.0 if output_scale is None else output_scale
        output_scale_grad = None
        if return_output_scale_grad:
            branch = _linear(retrieved, self.o.data).astype(xp.float32, copy=False)
            output_scale_grad = xp.sum(grad32 * branch)
        branch_grad = grad32 * scale
        self.o.grad += (retrieved.reshape(-1, self.width).astype(xp.float32).T @
                        branch_grad.reshape(-1, self.d_model))
        d_retrieved = _linear(branch_grad.astype(self.o.data.dtype), self.o.data.T)
        dxs = []
        for row, (c, ids, indices, h_pos, c_pos_all, valid_current_indices,
                  attention_cache, h_len) in enumerate(rows):
            compact_grad = d_retrieved[row, valid_current_indices]
            compact_positions = c_pos_all[valid_current_indices]
            if int(valid_current_indices.size) == 0:
                dq = xp.empty((0, self.width), dtype=xp.float32)
                dk = xp.empty((h_len, self.width), dtype=xp.float32)
                dv = xp.empty((h_len, self.width), dtype=xp.float32)
            elif BACKEND_NAME == "cupy":
                from mini_llm.ops.triton_full_history_attention import (
                    full_history_attention_backward,
                )
                dq, dk, dv = full_history_attention_backward(
                    compact_grad.astype(xp.float32, copy=False), attention_cache
                )
            else:
                dq, dk, dv = _reference_attention_backward(compact_grad, attention_cache)
            dq = _rope(dq, compact_positions, self.rope_base, reverse=True).astype(xp.float32)
            dhk = _rope(dk[:h_len], h_pos, self.rope_base, reverse=True).astype(xp.float32)
            dck = _rope(dk[h_len:], compact_positions, self.rope_base, reverse=True).astype(xp.float32)
            dhv = dv[:h_len].astype(xp.float32)
            dcv = dv[h_len:].astype(xp.float32)
            cf = c[valid_current_indices].astype(xp.float32)
            self.q.grad += cf.T @ dq
            self.k.grad += cf.T @ dck
            self.v.grad += cf.T @ dcv
            dx_compact = dq @ self.q.data.astype(xp.float32).T
            dx_compact += dck @ self.k.data.astype(xp.float32).T
            dx_compact += dcv @ self.v.data.astype(xp.float32).T
            dx = xp.zeros(c.shape, dtype=xp.float32)
            dx[valid_current_indices] = dx_compact
            # Recompute raw embeddings chunk by chunk instead of retaining
            # an additional 61K x 1024 activation for each selected layer.
            for start in range(0, h_len, self.project_chunk):
                stop = min(start + self.project_chunk, h_len)
                raw = self.embedding.data[ids[start:stop]].astype(xp.float32)
                self.k.grad += raw.T @ dhk[start:stop]
                self.v.grad += raw.T @ dhv[start:stop]
                if accumulate_embedding_grad and self.embedding.grad is not None:
                    d_emb = (
                        dhk[start:stop] @ self.k.data.astype(xp.float32).T
                        + dhv[start:stop] @ self.v.data.astype(xp.float32).T
                    )
                    xp.add.at(self.embedding.grad, ids[start:stop], d_emb)
            dxs.append(dx)
        result = xp.stack(dxs, axis=0)
        if return_output_scale_grad:
            return result, output_scale_grad
        return result

    def _inference_history(self, history_state, state, layer):
        cache = state.full_history_historical_kv
        if layer in cache:
            return cache[layer]
        rows = []
        for batch_row in range(int(state.batch_size)):
            if history_state is None:
                ids = xp.empty((0,), dtype=xp.int32)
                pos = xp.empty((0,), dtype=xp.int64)
            else:
                valid = xp.nonzero(history_state.valid[batch_row])[0]
                ids = history_state.ids[batch_row, valid]
                pos = history_state.positions[valid]
            k, v = self._history_kv(ids)
            rows.append((_rope(k, pos, self.rope_base), v))
        cache[layer] = rows
        return rows

    def inference_prefill(self, current, history_state, state, layer, rope_start):
        """Full-query exact causal prefill and current K/V ring-cache construction."""
        b, t, _ = current.shape
        cap = int(state.working_capacity)
        kv_k = xp.empty((b, cap, self.width), dtype=self.k.data.dtype)
        kv_v = xp.empty_like(kv_k)
        kv_k[:, :t] = xp.stack([
            _rope(_linear(current[row], self.k.data),
                  xp.arange(rope_start, rope_start+t, dtype=xp.int64), self.rope_base)
            for row in range(b)
        ])
        kv_v[:, :t] = _linear(current, self.v.data)
        state.full_history_current_kv[layer] = (kv_k, kv_v)
        self._inference_history(history_state, state, layer)
        positions = xp.broadcast_to(
            xp.arange(rope_start, rope_start+t, dtype=xp.int64)[None, :], (b, t)
        )
        # Uses the same native training kernel without retaining backward state.
        return self.forward(current, history_state, positions, return_cache=False)[0]

    def inference_decode(self, current, history_state, state, layer,
                         *, rope_position, cache_position, cache_start, working_count):
        """Single-token exact full-history read; independent cached K/V."""
        if layer not in state.full_history_current_kv:
            raise RuntimeError("historical-head K/V cache missing; prefill required")
        k_cache, v_cache = state.full_history_current_kv[layer]
        cap = int(state.working_capacity)
        index = int(cache_position)
        rope_ids = xp.asarray([int(rope_position)], dtype=xp.int64)
        indices = (int(cache_start) + xp.arange(working_count)) % cap
        rows = self._inference_history(history_state, state, layer)
        outputs = []
        for row, (old_k, old_v) in enumerate(rows):
            c = current[row]
            k_cache[row, index] = _rope(_linear(c, self.k.data), rope_ids, self.rope_base)[0]
            v_cache[row, index] = _linear(c, self.v.data)[0]
            q = _rope(_linear(c, self.q.data), rope_ids, self.rope_base).astype(xp.float32)
            keys = xp.concatenate((old_k, k_cache[row, indices]), axis=0).astype(xp.float32)
            values = xp.concatenate((old_v, v_cache[row, indices]), axis=0).astype(xp.float32)
            scores = (q @ keys.T) * (1. / math.sqrt(self.width))
            scores -= xp.max(scores, axis=-1, keepdims=True)
            probabilities = xp.exp(scores)
            probabilities /= xp.sum(probabilities, axis=-1, keepdims=True)
            retrieved = probabilities @ values
            outputs.append(retrieved.astype(self.o.data.dtype))
        return _linear(xp.stack(outputs, axis=0), self.o.data)
