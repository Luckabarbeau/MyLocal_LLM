"""0083B reference, block-encoded, read-only historical attention.

The memory is encoded once per forward; no historical rows enter the decoder.
A historical role replaces a query head's current-token attention, and uses
that head's existing Q and output projection. Queries are causal decoder states;
all historical K/V are derived exclusively from pre-window source tokens.

This implementation favors inspectable semantics over a production fused CUDA
kernel. Attention probabilities are reconstructed by query tile in backward.
"""
import os
from dataclasses import dataclass
from mini_llm.backend import xp, BACKEND_NAME, is_bfloat16_dtype
from mini_llm.init import matrix_parameter
from mini_llm.parameter import Parameter
from mini_llm.ops.historical_schedule import historical_slots, source_indices
from mini_llm.performance_profiler import performance_scope


def _f32(x):
    return x.astype('float32', copy=False)


def _env_enabled(name, default='0'):
    raw = os.environ.get(name, default).strip().lower()
    return raw not in ('', '0', 'false', 'no', 'off')


def _mat(x, w):
    """Reference batched GEMM. BF16 operation uses f32 for batched products."""
    if BACKEND_NAME == 'cupy' and is_bfloat16_dtype(w.dtype):
        # Large history encoder projections stay on the GPU BF16 GEMM path;
        # only the narrow residual/normalization stream is promoted to FP32.
        a = x.reshape(-1, x.shape[-1]).astype(w.dtype, copy=False)
        result = a @ w
        return result.reshape(*x.shape[:-1], w.shape[-1]).astype('float32')
    a = _f32(x)
    return (a.reshape(-1, a.shape[-1]) @ _f32(w)).reshape(*a.shape[:-1], w.shape[-1])


def _dot_backward(x, dy, w, p):
    p.grad += (_f32(x).reshape(-1, x.shape[-1]).T @
               _f32(dy).reshape(-1, dy.shape[-1]))
    return (_f32(dy).reshape(-1, dy.shape[-1]) @ _f32(w).T).reshape(x.shape)


def _rms(x, gamma, eps=1e-6):
    inv = 1. / xp.sqrt(xp.mean(x*x, axis=-1, keepdims=True) + eps)
    return x * inv * _f32(gamma.data), inv


def _rms_backward(x, dy, gamma, inv):
    f = _f32(dy) * _f32(gamma.data)
    normalized = x * inv
    gamma.grad += xp.sum(_f32(dy) * normalized, axis=tuple(range(dy.ndim - 1)))
    return inv * (f - normalized*xp.mean(f*normalized, axis=-1, keepdims=True))


def _silu(x):
    sigmoid = 1. / (1. + xp.exp(-x))
    return x * sigmoid, sigmoid


def _softmax(scores, mask=None):
    if mask is not None:
        scores = xp.where(mask, scores, -1e30)
        nonempty = xp.any(mask, axis=-1, keepdims=True)
    else:
        nonempty = True
    scores = scores - xp.max(scores, axis=-1, keepdims=True)
    p = xp.exp(scores)
    if mask is not None:
        p = xp.where(mask, p, 0.)
    return p / xp.maximum(xp.sum(p, axis=-1, keepdims=True), 1e-30)


def _rope(x, positions, base=10000., inverse=False):
    """Independent rotary transform using true source positions."""
    freqs = 1. / (float(base) ** (xp.arange(0, x.shape[-1], 2, dtype=xp.float32) / x.shape[-1]))
    angles = _f32(positions)[..., None] * freqs
    c, s = xp.cos(angles), xp.sin(angles)
    if inverse:
        s = -s
    y = xp.empty(x.shape, dtype=xp.float32)
    y[..., ::2] = x[..., ::2] * c - x[..., 1::2] * s
    y[..., 1::2] = x[..., ::2] * s + x[..., 1::2] * c
    return y


@dataclass
class HistoricalState:
    ids: object
    valid: object
    positions: object
    z: object
    k: object
    v: object
    block_keys: object
    block_valid: object
    encoder_cache: object
    dk: object = None
    dv: object = None
    dblock_keys: object = None
    has_valid: bool = False
    all_rows_have_valid: bool = False


class HistoricalReadMemory:
    def __init__(self, config, embedding_weight, rng, dtype, name='historical_read'):
        cfg = config.memory_context
        self.config = config
        self.cfg = cfg
        self.embedding_weight = embedding_weight
        self.block_size = int(cfg.block_size)
        self.slots = historical_slots(config.n_layers, config.n_q_heads)
        self.chunk = int(os.environ.get('MINI_LLM_HISTORY_QUERY_CHUNK', '128'))
        if self.chunk <= 0:
            raise ValueError('MINI_LLM_HISTORY_QUERY_CHUNK must be positive')
        self.recompute = os.environ.get('MINI_LLM_HISTORY_RECOMPUTE', '0').lower() not in ('0','false','no','off','')
        self.batch_routes = os.environ.get(
            'MINI_LLM_HISTORY_BATCH_ROUTES', '1'
        ).lower() not in ('0', 'false', 'no', 'off', '')
        self.triton_block_forward = (
            BACKEND_NAME == 'cupy'
            and _env_enabled('MINI_LLM_TRITON_HISTORICAL_BLOCK_FORWARD')
        )
        self.triton_block_backward = (
            self.triton_block_forward
            and _env_enabled('MINI_LLM_TRITON_HISTORICAL_BLOCK_BACKWARD')
        )
        self.memory_audit = (BACKEND_NAME == 'cupy' and
                             os.environ.get('MINI_LLM_HISTORY_MEMORY_AUDIT', '0') == '1')
        self.name = name
        self.dtype = dtype
        self.in_proj = matrix_parameter((config.d_model, 128), config.init_std, rng, f'{name}.in_proj', dtype=dtype)
        self.n1 = Parameter(xp.ones((128,), dtype=dtype), f'{name}.norm1', decay=False)
        self.qkv = matrix_parameter((128, 384), config.init_std, rng, f'{name}.qkv', dtype=dtype)
        self.o = matrix_parameter((128, 128), config.residual_init_std, rng, f'{name}.o', dtype=dtype)
        self.n2 = Parameter(xp.ones((128,), dtype=dtype), f'{name}.norm2', decay=False)
        self.gate = matrix_parameter((128, 512), config.init_std, rng, f'{name}.ff_gate', dtype=dtype)
        self.up = matrix_parameter((128, 512), config.init_std, rng, f'{name}.ff_up', dtype=dtype)
        self.down = matrix_parameter((512, 128), config.residual_init_std, rng, f'{name}.ff_down', dtype=dtype)
        self.key = matrix_parameter((128, config.d_head), config.init_std, rng, f'{name}.key', dtype=dtype)
        self.value = matrix_parameter((128, config.d_head), config.init_std, rng, f'{name}.value', dtype=dtype)
        self.router_key = matrix_parameter((128, config.d_head), config.init_std, rng, f'{name}.router_key', dtype=dtype)
        # The two learned reread heads share block IDs, not token-level K/V
        # projections; preserve independent Q (existing GQA) and K/V (here).
        self.learned_heads = tuple(sorted({s.head for layer in self.slots for s in layer
                                           if s.kind == 'learned'}))
        self.learned_keys = {
            head: matrix_parameter((128, config.d_head), config.init_std, rng,
                                   f'{name}.learned{head}.key', dtype=dtype)
            for head in self.learned_heads
        }
        self.learned_values = {
            head: matrix_parameter((128, config.d_head), config.init_std, rng,
                                   f'{name}.learned{head}.value', dtype=dtype)
            for head in self.learned_heads
        }
        self.head_gates = [Parameter(xp.full((len(layer),), 0.1, dtype=xp.float32),
                                     f'{name}.layer{layer_index}.gates', decay=False)
                           for layer_index, layer in enumerate(self.slots)]
        if self.triton_block_forward and _env_enabled(
                'MINI_LLM_TRITON_HISTORICAL_BLOCK_PREWARM', '1'):
            from mini_llm.ops.triton_historical_block import (
                prewarm_deterministic_block_read_forward,
            )
            block_slot = next(
                slot for layer in self.slots for slot in layer
                if slot.kind == 'block'
            )
            prewarm_deterministic_block_read_forward(
                query_dtype=self.dtype,
                query_length=int(cfg.target_length),
                history_length=int(cfg.distant_memory_length),
                phase=block_slot.phase,
                period=block_slot.period,
                backward=self.triton_block_backward,
            )

    def parameters(self):
        return [self.in_proj, self.n1, self.qkv, self.o, self.n2,
                self.gate, self.up, self.down, self.key, self.value,
                self.router_key,
                *(p for head in self.learned_heads for p in
                  (self.learned_keys[head], self.learned_values[head])),
                *self.head_gates]

    def _audit(self, stage, state=None):
        if not self.memory_audit:
            return
        xp.cuda.Stream.null.synchronize()
        free, total = xp.cuda.runtime.memGetInfo()
        pool = xp.get_default_memory_pool()
        available = None if state is None else int(xp.sum(state.valid).item())
        print(f"HISTORY_MEMORY stage={stage} device_used={total-free} "
              f"device_free={free} pool_live={pool.used_bytes()} "
              f"pool_reserved={pool.total_bytes()} valid_history={available}",
              flush=True)

    def encode(self, ids, valid):
        """Batches independent 32-token blocks without cross-block leakage."""
        with performance_scope('historical.encoder.forward'):
            b, h = ids.shape
            if h % self.block_size:
                raise ValueError('history must contain complete 32-token blocks')
            with performance_scope('historical.encoder.embedding_gather'):
                input_emb = self.embedding_weight.data[ids]
            with performance_scope('historical.encoder.input_projection'):
                x0 = _mat(input_emb, self.in_proj.data)
                x = x0.reshape(b, h // 32, 32, 128)
            with performance_scope('historical.encoder.rmsnorm1'):
                xnorm, inv1 = _rms(x, self.n1)
            with performance_scope('historical.encoder.qkv_projection'):
                qkv = _mat(xnorm, self.qkv.data).reshape(b, h // 32, 32, 3, 4, 32)
                q = qkv[..., 0, :, :].transpose(0, 1, 3, 2, 4)
                k = qkv[..., 1, :, :].transpose(0, 1, 3, 2, 4)
                v = qkv[..., 2, :, :].transpose(0, 1, 3, 2, 4)
            with performance_scope('historical.encoder.block_attention'):
                allowed = valid.reshape(b, h // 32, 32)
                p = _softmax(xp.matmul(q, k.swapaxes(-2, -1)) / 32.**.5,
                             allowed[:, :, None, None, :])
                context = xp.matmul(p, v).transpose(0, 1, 3, 2, 4).reshape(
                    b, h // 32, 32, 128)
            with performance_scope('historical.encoder.output_projection_residual'):
                post_attn = x + _mat(context, self.o.data)
            with performance_scope('historical.encoder.rmsnorm2'):
                ynorm, inv2 = _rms(post_attn, self.n2)
            with performance_scope('historical.encoder.ffn_projections'):
                g = _mat(ynorm, self.gate.data)
                u = _mat(ynorm, self.up.data)
            with performance_scope('historical.encoder.swiglu'):
                a, sigmoid = _silu(g)
                hidden = a * u
            with performance_scope('historical.encoder.down_projection_residual'):
                z = post_attn + _mat(hidden, self.down.data)
            with performance_scope('historical.encoder.validity_mask'):
                z *= allowed[..., None]
            if self.recompute:
                return z.reshape(b, h, 128), None
            return z.reshape(b, h, 128), (input_emb, x, xnorm, inv1, q, k, v, p,
                                          context, post_attn, ynorm, inv2, g, u,
                                          a, sigmoid, hidden, allowed)

    def open(self, history_ids, valid_starts, valid_mask=None, *, position_offset=0, inference=False):
        with performance_scope('historical.open.validity_crop'):
            ids = xp.asarray(history_ids)
            self._audit('history_start')
            b, h = ids.shape
            if h % self.block_size:
                raise ValueError('history length must be a multiple of 32')
            starts = xp.asarray(valid_starts, dtype=xp.int64)
            if starts.shape != (b,) or bool(xp.any((starts < 0) | (starts > h))):
                raise ValueError('invalid document history starts')
            # Crop the common, fully invalid prefix; source coordinates remain
            # absolute, with ragged earlier boundaries masked per sample.
            offset = min(max(0, h - 32), (int(xp.min(starts).item()) // 32) * 32)
            ids = ids[:, offset:]
            positions = xp.arange(offset, h, dtype=xp.int64) + int(position_offset)
            valid = positions[None, :] >= starts[:, None]
            if valid_mask is not None:
                supplied = xp.asarray(valid_mask, dtype=bool)
                if supplied.shape != (b, h):
                    raise ValueError('history_valid_mask must be [B,H]')
                valid = valid & supplied[:, offset:]
        previous_recompute = self.recompute
        if inference:
            self.recompute = True  # Avoid retaining backward intermediates during serving.
        try:
            z, cache = self.encode(ids, valid)
        finally:
            self.recompute = previous_recompute
        self._audit('after_history_encoder')
        with performance_scope('historical.shared_kv_projection'):
            k_unrotated = _mat(z, self.key.data)
            v = _mat(z, self.value.data)
        with performance_scope('historical.shared_rope'):
            k = _rope(k_unrotated, positions[None, :], self.config.rope_base)
        with performance_scope('historical.router_summary'):
            nblocks = z.shape[1] // 32
            block_valid_count = valid.reshape(b, nblocks, 32).sum(axis=-1)
            summary = (z.reshape(b, nblocks, 32, 128).sum(axis=2) /
                       xp.maximum(block_valid_count[..., None], 1))
            norm = xp.sqrt(xp.mean(summary**2, axis=-1, keepdims=True) + 1e-6)
        with performance_scope('historical.router_key_projection'):
            block_keys = _mat(summary / norm, self.router_key.data)
        state = HistoricalState(
            ids, valid, positions, z, k, v, block_keys,
            block_valid_count > 0, cache,
            has_valid=bool(xp.any(block_valid_count > 0).item()),
            all_rows_have_valid=bool(
                xp.all(xp.any(block_valid_count > 0, axis=1)).item()
            ),
        )
        state.learned_k = {}
        for head, p in self.learned_keys.items():
            with performance_scope('historical.learned_kv_projection'):
                learned_k = _mat(z, p.data)
            with performance_scope('historical.learned_rope'):
                state.learned_k[head] = _rope(
                    learned_k, positions[None, :], self.config.rope_base)
        state.learned_v = {}
        for head, p in self.learned_values.items():
            with performance_scope('historical.learned_kv_projection'):
                state.learned_v[head] = _mat(z, p.data)
        state.learned_dk = {}
        state.learned_dv = {}
        state.summary = summary
        state.summary_norm = norm
        state.source_offset = offset
        state.source_length = h
        self._audit('after_history_kv', state)
        return state

    def _encode_backward(self, dz, state):
        if state.encoder_cache is None:
            prev = self.recompute
            self.recompute = False
            try:
                with performance_scope('historical.encoder.recompute'):
                    _, cache = self.encode(state.ids, state.valid)
            finally:
                self.recompute = prev
        else:
            cache = state.encoder_cache
        (emb, x, xnorm, inv1, q, k, v, p, context, y, ynorm, inv2,
         g, u, a, sigmoid, hidden, allowed) = cache
        b, n, l, width = x.shape
        with performance_scope('historical.encoder.backward.validity_mask'):
            dz = _f32(dz).reshape(b, n, l, width) * allowed[..., None]
        with performance_scope('historical.encoder.backward.ffn'):
            dh = _dot_backward(hidden, dz, self.down.data, self.down)
            dg = dh * u * (sigmoid + g * sigmoid * (1. - sigmoid))
            du = dh * a
            dynorm = (_dot_backward(ynorm, dg, self.gate.data, self.gate) +
                       _dot_backward(ynorm, du, self.up.data, self.up))
        with performance_scope('historical.encoder.backward.rmsnorm2'):
            dy = dz + _rms_backward(y, dynorm, self.n2, inv2)
        with performance_scope('historical.encoder.backward.output_projection'):
            dctx = _dot_backward(context, dy, self.o.data, self.o)
            dctx = dctx.reshape(b, n, l, 4, 32).transpose(0, 1, 3, 2, 4)
        with performance_scope('historical.encoder.backward.block_attention'):
            dp = xp.matmul(dctx, v.swapaxes(-2, -1))
            dv = xp.matmul(p.swapaxes(-2, -1), dctx)
            ds = p * (dp - xp.sum(dp*p, axis=-1, keepdims=True)) * (32.**-.5)
            dq = xp.matmul(ds, k)
            dk = xp.matmul(ds.swapaxes(-2, -1), q)
            dqkv = xp.stack([dq, dk, dv], axis=4).transpose(0, 1, 3, 4, 2, 5)
            dqkv = dqkv.reshape(b, n, l, 384)
        with performance_scope('historical.encoder.backward.qkv_projection'):
            dxnorm = _dot_backward(xnorm, dqkv, self.qkv.data, self.qkv)
        with performance_scope('historical.encoder.backward.rmsnorm1'):
            dx = dy + _rms_backward(x, dxnorm, self.n1, inv1)
        with performance_scope('historical.encoder.backward.input_projection'):
            demb = _dot_backward(
                emb, dx.reshape(b, n*l, width), self.in_proj.data, self.in_proj)
        # Embeddings are tied to the output projection. Add to its shared grad.
        with performance_scope('historical.close.embedding_scatter'):
            grad = self.embedding_weight.grad
            xp.add.at(grad, state.ids.reshape(-1), demb.reshape(-1, demb.shape[-1]))

    def close(self, state):
        """Backpropagate all shared historical K/V contributions exactly once."""
        self._audit('before_history_backward', state)
        if state.dk is None:
            return
        h = state.z.shape[1]
        with performance_scope('historical.close.shared_rope_backward'):
            shared_dk = _rope(
                state.dk, state.positions[None, :],
                self.config.rope_base, inverse=True)
        with performance_scope('historical.close.shared_projection_backward'):
            dz = _dot_backward(state.z, shared_dk, self.key.data, self.key)
            dz += _dot_backward(state.z, state.dv, self.value.data, self.value)
        for head in state.learned_dk:
            with performance_scope('historical.close.learned_rope_backward'):
                learned_dk = _rope(
                    state.learned_dk[head], state.positions[None, :],
                    self.config.rope_base, inverse=True)
            with performance_scope('historical.close.learned_projection_backward'):
                dz += _dot_backward(
                    state.z, learned_dk,
                    self.learned_keys[head].data, self.learned_keys[head],
                )
                dz += _dot_backward(
                    state.z, state.learned_dv[head],
                    self.learned_values[head].data, self.learned_values[head],
                )
        if state.dblock_keys is not None:
            with performance_scope('historical.close.router_summary_backward'):
                b, n, width = state.summary.shape
                ns = state.summary / state.summary_norm
                dsummary = _dot_backward(
                    ns, state.dblock_keys, self.router_key.data, self.router_key)
                dsummary = ((dsummary - ns * xp.mean(
                    dsummary * ns, axis=-1, keepdims=True)) /
                    state.summary_norm)
                valid_blocks = state.valid.reshape(b, n, 32)
                counts = xp.maximum(valid_blocks.sum(axis=-1, keepdims=True), 1)
                dz += (dsummary[:, :, None, :] * valid_blocks[..., None] /
                       counts[..., None]).reshape(b, h, 128)
        with performance_scope('historical.encoder.backward'):
            self._encode_backward(dz, state)
        self._audit('after_history_backward', state)

    def _indices(self, slot, state):
        # Slot source-coordinate indices are independent of the cropped store.
        h = state.source_length
        offset = state.source_offset
        idx = source_indices(slot, h, valid_start=offset)
        return xp.asarray(idx, dtype=xp.int64) - offset

    def _route(self, q, layer_slots, state):
        """Hard top-8 decisions use only x[r] at each 128-token boundary."""
        with performance_scope('historical.layer.route'):
            head = next(slot.head for slot in layer_slots if slot.kind == 'learned')
            query = _f32(q[:, ::128, head, :])
            scores = xp.matmul(query, state.block_keys.swapaxes(-2, -1)) * (64. ** -.5)
            scores = xp.where(state.block_valid[:, None, :], scores, -1e30)
            ranks = xp.argsort(-scores, axis=-1)[..., :int(self.cfg.top_k_blocks)]
            valid = xp.take_along_axis(
                xp.broadcast_to(state.block_valid[:, None, :], scores.shape), ranks, axis=-1)
            return ranks, valid, scores

    def _read(self, query, k, v, valid, *, bias=None, return_cache=False):
        """Tile query rows; no Q-dependent K/V replication and no B*W*H mask."""
        q = _f32(query)
        k = _f32(k)
        v = _f32(v)
        outputs = []
        probabilities = [] if return_cache else None
        for j in range(0, q.shape[1], self.chunk):
            chunk = q[:, j:j+self.chunk, :]
            score = xp.matmul(chunk, k.swapaxes(-2, -1)) * (64.**-.5)
            if bias is not None:
                score += bias[:, None, :]
            p = _softmax(score, valid[:, None, :])
            outputs.append(xp.matmul(p, v))
            if return_cache:
                probabilities.append(p)
        return xp.concatenate(outputs, axis=1), probabilities

    def apply(self, context, q, layer_idx, state, *, return_cache=True):
        """Override designated current attention heads, preserving local heads."""
        slots = self.slots[layer_idx]
        # The document-first sampler frequently selects windows with no
        # preceding in-document history. Do not launch historical QK kernels
        # for an entirely invalid memory (or replace ordinary current heads).
        if not slots or not state.has_valid:
            return context, None
        with performance_scope('historical.layer.context_copy'):
            context = xp.array(context, copy=True)
        routes = self._route(q, slots, state) if any(s.kind == 'learned' for s in slots) else None
        entries = []
        block_valid = xp.any(state.valid, axis=1)
        b, w = q.shape[:2]
        for sidx, slot in enumerate(slots):
            query = q[:, :, slot.head, :]
            if slot.kind == 'learned':
                rank, valid_blocks, route_scores = routes
                head_k = state.learned_k[slot.head]
                head_v = state.learned_v[slot.head]
                route_count = int(rank.shape[1])
                if self.batch_routes and w == route_count * 128:
                    with performance_scope('historical.layer.learned_gather'):
                        idx = (
                            rank[..., None] * 32
                            + xp.arange(32, dtype=xp.int64)[None, None, None, :]
                        ).reshape(b, route_count, -1)
                        batch_rows = xp.arange(b, dtype=xp.int64)[:, None, None]
                        bk = head_k[batch_rows, idx]
                        bv = head_v[batch_rows, idx]
                        mask = (
                            state.valid[batch_rows, idx]
                            & xp.repeat(valid_blocks, 32, axis=2)
                        )
                        selected_scores = xp.take_along_axis(
                            route_scores, rank, axis=2
                        )
                        route_bias = 0.25 * xp.repeat(
                            selected_scores, 32, axis=2
                        )
                    with performance_scope('historical.layer.learned_read'):
                        query_routes = _f32(query).reshape(
                            b, route_count, 128, query.shape[-1]
                        )
                        score = (
                            xp.matmul(query_routes, _f32(bk).swapaxes(-2, -1))
                            * (64. ** -.5)
                            + route_bias[:, :, None, :]
                        )
                        p = _softmax(score, mask[:, :, None, :])
                        read = xp.matmul(p, _f32(bv)).reshape(b, w, -1)
                else:
                    all_outputs = []
                    for r in range(route_count):
                        with performance_scope('historical.layer.learned_gather'):
                            idx = rank[:, r, :, None] * 32 + xp.arange(32)[None, None, :]
                            idx = idx.reshape(b, -1)
                            bk = xp.take_along_axis(head_k, idx[:, :, None], axis=1)
                            bv = xp.take_along_axis(head_v, idx[:, :, None], axis=1)
                            mask = (xp.take_along_axis(state.valid, idx, axis=1) &
                                    xp.repeat(valid_blocks[:, r, :], 32, axis=1))
                            selected_scores = xp.take_along_axis(
                                route_scores[:, r], rank[:, r], axis=1)
                            route_bias = 0.25 * xp.repeat(selected_scores, 32, axis=1)
                        with performance_scope('historical.layer.learned_read'):
                            o, _ = self._read(
                                query[:, r*128:(r+1)*128], bk, bv,
                                mask, bias=route_bias)
                        all_outputs.append(o)
                    read = xp.concatenate(all_outputs, axis=1)
                selection = rank
            else:
                idx = self._indices(slot, state)
                selection = idx
                if idx.size:
                    with performance_scope(f'historical.layer.{slot.kind}_read'):
                        if slot.kind == 'block' and self.triton_block_forward:
                            from mini_llm.ops.triton_historical_block import (
                                deterministic_block_read_forward,
                            )
                            result = deterministic_block_read_forward(
                                query,
                                state.k,
                                state.v,
                                state.valid,
                                phase=slot.phase,
                                period=slot.period,
                                source_offset=state.source_offset,
                                source_length=state.source_length,
                                return_cache=(
                                    return_cache and self.triton_block_backward
                                ),
                            )
                            if return_cache and self.triton_block_backward:
                                read, selection = result
                            else:
                                read = result
                        else:
                            read, _ = self._read(
                                query, state.k[:, idx, :], state.v[:, idx, :],
                                state.valid[:, idx])
                else:
                    read = xp.zeros(query.shape, dtype=xp.float32)
            with performance_scope('historical.layer.context_replace'):
                old = context[:, :, slot.head, :]
                active = block_valid[:, None, None]
                gate = self.head_gates[layer_idx].data[sidx]
                context[:, :, slot.head, :] = xp.where(active, gate * read, old)
            if return_cache:
                entries.append((slot, selection, active))
        self._audit(f'layer{layer_idx}_history_forward', state)
        return context, (entries, routes) if return_cache else None

    def apply_decode(self, context, q, layer_idx, state, route_cache, *,
                     clock, refresh_id):
        """Historical replacement for one query without rerouting each token.

        The query at each 128-row local boundary selects the learned blocks;
        that selection persists for the remaining 127 rows. In sliding mode
        history rebuilds invalidate the selection explicitly.
        """
        slots = self.slots[layer_idx]
        if not slots or not state.has_valid:
            return context
        context = xp.array(context, copy=True)
        routes = None
        if any(slot.kind == 'learned' for slot in slots):
            key = int(layer_idx)
            cached = route_cache.get(key)
            if cached is None or cached[0] != refresh_id:
                routes = self._route(q, slots, state)
                route_cache[key] = (refresh_id, routes)
            else:
                routes = cached[1]
        active = xp.any(state.valid, axis=1)[:, None, None]
        for slot_idx, slot in enumerate(slots):
            query = q[:, :, slot.head, :]
            if slot.kind == 'learned':
                ranks, valid_blocks, route_scores = routes
                # apply_decode has one route boundary, hence shape [B,1,K].
                rank = ranks[:, 0]
                b = query.shape[0]
                idx = (rank[..., None] * 32 +
                       xp.arange(32, dtype=xp.int64)[None, None, :]).reshape(b, -1)
                k = xp.take_along_axis(state.learned_k[slot.head], idx[..., None], axis=1)
                v = xp.take_along_axis(state.learned_v[slot.head], idx[..., None], axis=1)
                valid = (xp.take_along_axis(state.valid, idx, axis=1) &
                         xp.repeat(valid_blocks[:, 0], 32, axis=1))
                bias = 0.25 * xp.repeat(
                    xp.take_along_axis(route_scores[:, 0], rank, axis=1),
                    32, axis=1,
                )
            else:
                idx = self._indices(slot, state)
                if idx.size == 0:
                    read = xp.zeros(query.shape, dtype=xp.float32)
                    context[:, :, slot.head, :] = xp.where(
                        active, self.head_gates[layer_idx].data[slot_idx] * read,
                        context[:, :, slot.head, :])
                    continue
                k, v = state.k[:, idx], state.v[:, idx]
                valid, bias = state.valid[:, idx], None
            read, _ = self._read(query, k, v, valid, bias=bias)
            previous = context[:, :, slot.head, :]
            context[:, :, slot.head, :] = xp.where(
                active, self.head_gates[layer_idx].data[slot_idx] * read, previous
            )
        return context

    def backward_reads(self, dcontext, q, layer_idx, state, read_cache):
        """VJP of historical token attention; probabilities replay by tile."""
        entries, routes = read_cache
        with performance_scope('historical.backward.gradient_init'):
            dcontext = xp.array(dcontext, copy=True)
            dq = xp.zeros(q.shape, dtype=xp.float32)
            if state.dk is None:
                state.dk = xp.zeros(state.k.shape, dtype=xp.float32)
                state.dv = xp.zeros(state.v.shape, dtype=xp.float32)
        b, w = q.shape[:2]
        route_grad = (xp.zeros_like(routes[2]) if routes is not None else None)
        for sidx, (slot, selection, active) in enumerate(entries):
            if slot.kind == 'block' and self.triton_block_backward:
                from mini_llm.ops.triton_historical_block import (
                    DeterministicBlockReadCache,
                    deterministic_block_read_backward,
                )
                if isinstance(selection, DeterministicBlockReadCache):
                    with performance_scope('historical.backward.block_read'):
                        deterministic_block_read_backward(
                            dcontext[:, :, slot.head, :],
                            q[:, :, slot.head, :],
                            state.k,
                            state.v,
                            state.valid,
                            active.reshape(-1),
                            self.head_gates[layer_idx].data[sidx:sidx + 1],
                            self.head_gates[layer_idx].grad[sidx:sidx + 1],
                            selection,
                            dquery=dq[:, :, slot.head, :],
                            dkey=state.dk,
                            dvalue=state.dv,
                        )
                    continue
            upstream = _f32(dcontext[:, :, slot.head, :]) * active
            dcontext[:, :, slot.head, :] = xp.where(active, 0., dcontext[:, :, slot.head, :])
            query = _f32(q[:, :, slot.head, :])
            key_store = (state.learned_k[slot.head] if slot.kind == 'learned'
                         else state.k)
            value_store = (state.learned_v[slot.head] if slot.kind == 'learned'
                           else state.v)
            if slot.kind == 'learned':
                if slot.head not in state.learned_dk:
                    state.learned_dk[slot.head] = xp.zeros_like(key_store)
                    state.learned_dv[slot.head] = xp.zeros_like(value_store)
                dk_store = state.learned_dk[slot.head]
                dv_store = state.learned_dv[slot.head]
                rank, valid_blocks, route_scores = routes
                route_count = int(rank.shape[1])
                if self.batch_routes and w == route_count * 128:
                    with performance_scope('historical.backward.learned_gather'):
                        idx = (
                            rank[..., None] * 32
                            + xp.arange(32, dtype=xp.int64)[None, None, None, :]
                        ).reshape(b, route_count, -1)
                        batch_rows = xp.arange(b, dtype=xp.int64)[:, None, None]
                        ksel = _f32(key_store[batch_rows, idx])
                        vsel = _f32(value_store[batch_rows, idx])
                        valid = (
                            state.valid[batch_rows, idx]
                            & xp.repeat(valid_blocks, 32, axis=2)
                        )
                        selected_scores = xp.take_along_axis(
                            route_scores, rank, axis=2
                        )
                        bias = 0.25 * xp.repeat(selected_scores, 32, axis=2)
                    with performance_scope('historical.backward.learned_read'):
                        qsel = query.reshape(b, route_count, 128, -1)
                        grad = upstream.reshape(b, route_count, 128, -1)
                        score = (
                            xp.matmul(qsel, ksel.swapaxes(-2, -1)) * (64. ** -.5)
                            + bias[:, :, None, :]
                        )
                        p = _softmax(score, valid[:, :, None, :])
                        out = xp.matmul(p, vsel)
                        grad_gate = xp.sum(grad * out)
                        dy = grad * self.head_gates[layer_idx].data[sidx]
                        dp = xp.matmul(dy, vsel.swapaxes(-2, -1))
                        dvsel = xp.matmul(p.swapaxes(-2, -1), dy)
                        dlogit = p * (dp - xp.sum(dp * p, axis=-1, keepdims=True))
                        db = 0.25 * dlogit.sum(axis=2).reshape(
                            b, route_count, -1, 32
                        ).sum(axis=-1)
                        route_rows = xp.arange(
                            route_count, dtype=xp.int64)[None, :, None]
                        xp.add.at(route_grad, (batch_rows, route_rows, rank), db)
                        ds = dlogit * (64. ** -.5)
                        dq[:, :, slot.head, :] += xp.matmul(ds, ksel).reshape(
                            b, w, -1
                        )
                        dksel = xp.matmul(ds.swapaxes(-2, -1), qsel)
                        flat_idx = idx.reshape(b, -1)
                        flat_batch = xp.arange(b, dtype=xp.int64)[:, None]
                        xp.add.at(
                            dk_store, (flat_batch, flat_idx),
                            dksel.reshape(b, -1, dksel.shape[-1]),
                        )
                        xp.add.at(
                            dv_store, (flat_batch, flat_idx),
                            dvsel.reshape(b, -1, dvsel.shape[-1]),
                        )
                        self.head_gates[layer_idx].grad[sidx] += grad_gate
                    continue
                sections = []
                for r in range(route_count):
                    idx = (rank[:, r, :, None]*32 + xp.arange(32)[None, None, :]).reshape(b, -1)
                    valid = xp.take_along_axis(state.valid, idx, axis=1) & xp.repeat(valid_blocks[:, r], 32, axis=1)
                    selected_scores = xp.take_along_axis(route_scores[:, r], rank[:, r], axis=1)
                    bias = 0.25 * xp.repeat(selected_scores, 32, axis=1)
                    sections.append((slice(r*128, min((r+1)*128, w)), idx, valid, bias, r))
            else:
                dk_store, dv_store = state.dk, state.dv
                idx = selection
                if not idx.size:
                    continue
                sections = [(slice(0, w), idx, state.valid[:, idx], None, None)]
            grad_gate = xp.asarray(0., dtype=xp.float32)
            for sl, idx, mask, bias, route_index in sections:
                qsel = query[:, sl]
                grad = upstream[:, sl]
                with performance_scope(f'historical.backward.{slot.kind}_gather'):
                    if idx.ndim == 1:
                        ksel, vsel = key_store[:, idx], value_store[:, idx]
                    else:
                        ksel = xp.take_along_axis(key_store, idx[..., None], axis=1)
                        vsel = xp.take_along_axis(value_store, idx[..., None], axis=1)
                for t in range(0, qsel.shape[1], self.chunk):
                    with performance_scope(f'historical.backward.{slot.kind}_read'):
                        qtile = qsel[:, t:t+self.chunk, :]
                        dytile = grad[:, t:t+self.chunk, :]
                        score = xp.matmul(qtile, ksel.swapaxes(-2, -1)) * (64.**-.5)
                        if bias is not None:
                            score += bias[:, None, :]
                        p = _softmax(score, mask[:, None])
                        out = xp.matmul(p, vsel)
                        grad_gate += xp.sum(dytile * out)
                        dytile *= self.head_gates[layer_idx].data[sidx]
                        dp = xp.matmul(dytile, vsel.swapaxes(-2, -1))
                        dvsel = xp.matmul(p.swapaxes(-2, -1), dytile)
                        dlogit = p * (dp - xp.sum(dp*p, axis=-1, keepdims=True))
                        if route_index is not None:
                            db = 0.25 * dlogit.sum(axis=1).reshape(
                                b, -1, 32).sum(axis=-1)
                            batch_rows = xp.arange(b, dtype=xp.int64)[:, None]
                            xp.add.at(
                                route_grad,
                                (batch_rows, route_index, routes[0][:, route_index]),
                                db,
                            )
                        ds = dlogit * (64.**-.5)
                        dq[:, sl, slot.head, :][:, t:t+self.chunk, :] += xp.matmul(
                            ds, ksel)
                        dksel = xp.matmul(ds.swapaxes(-2, -1), qtile)
                        if idx.ndim == 1:
                            dk_store[:, idx, :] += dksel
                            dv_store[:, idx, :] += dvsel
                        else:
                            batch_rows = xp.arange(b, dtype=xp.int64)[:, None]
                            xp.add.at(dk_store, (batch_rows, idx), dksel)
                            xp.add.at(dv_store, (batch_rows, idx), dvsel)
            self.head_gates[layer_idx].grad[sidx] += grad_gate
        if routes is not None:
            with performance_scope('historical.backward.router'):
                # Hard forward top-k with a differentiable full-candidate softmax
                # surrogate. The direct selected-score derivative also travels
                # through the 0.25 token-attention score bias above.
                rank, valid_blocks, scores = routes
                pfull = _softmax(scores, state.block_valid[:, None, :])
                surrogate = pfull * (
                    route_grad - xp.sum(pfull * route_grad, axis=-1, keepdims=True))
                dscore = route_grad + float(self.cfg.router_surrogate_scale) * surrogate
                route_head = next(
                    slot.head for slot, _, _ in entries if slot.kind == 'learned')
                route_q = _f32(q[:, ::128, route_head, :])
                dq[:, ::128, route_head, :] += (
                    xp.matmul(dscore, state.block_keys) * (64.**-.5))
                dk = xp.matmul(dscore.swapaxes(-2, -1), route_q) * (64.**-.5)
                if state.dblock_keys is None:
                    state.dblock_keys = dk
                else:
                    state.dblock_keys += dk
        self._audit(f'layer{layer_idx}_history_backward', state)
        return dcontext, dq
