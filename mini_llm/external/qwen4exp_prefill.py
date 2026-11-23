"""Layer-major, bounded chunk prefill; no decode-graph or model modifications.

Integration: return prefill_chunked(self, sequence, state, chunk_size) from the
model's prefill entry point. This function owns _call_lock (do not nest it).
Only the final token uses the output head, even with diagnostics enabled.
Diagnostics are chunk arrays, in layer-major order, not decode trace order.

Packed projections, native-GEMV coefficient/routing batches and native HC
kernels process whole chunks without changing their per-token reductions.
Gated-delta recurrence advances the chunk inside a CUDA kernel. All-cell QSA
uses batched causal attention and bulk host history commits; sparse QSA and PLE
retain reference operators. Packed handles never expand into dense experts.
CPU-only packed-HC fixtures use bounded bit expansion, never numeric uint16.

Ordinary heterogeneous execution groups token rows per expert using the existing
async submit/resolve dispatcher, protected residency, and ordered reductions.
Its CPU GGML executor supports multirow input even under ggml_pool; the separate
singleton native row pool is not a multirow API. Device expert plans can select
different GPU Q8 arithmetic: when enabled, routed experts retain singleton plan
execution with the already-computed routes in the explicit decode fallback.
The grouped preset retains that arithmetic with multirow expert execution;
stream uses GPU Q8 for misses; triton executes every prompt expert through
wave-bounded grouped kernels that dequantize on device into FP16 tiles and
compute FP16 tl.dot products accumulated in FP32 (opt-in policy, no external
kernel library). The latter two are distinct arithmetic policies, not claims of
mixed-CPU bitwise or decode equivalence. No expert skipping is performed.
Short checkpoint parity is not long-context validation or evidence of HTTP speedup.
"""
from copy import copy
import os

from mini_llm.backend import BACKEND_NAME, xp
from mini_llm.ops.expert_residency import HeterogeneousExpertExecutor
from .qwen4exp_hyperconnection import _sigmoid
from .qwen4exp_weights import integer


def _hc_read(hc, streams):
    if not getattr(hc, 'packed_bf16', False):
        return hc.read(streams)

    if BACKEND_NAME == 'cupy':
        if streams.ndim == 2:
            return hc.read(streams)
        from mini_llm.ops.hyperconnection_prefill_gpu import read_chunk
        chunk = read_chunk(hc, streams)
        if chunk is not None:
            return chunk
        rows = [hc.read(row) for row in streams.reshape(-1, 4, hc.width)]
        leading = streams.shape[:-2]
        hidden = xp.stack([value for value, _ in rows]).reshape(leading + (hc.width,))
        gains = (None if rows[0][1] is None else
                 xp.stack([gain for _, gain in rows]).reshape(leading + (4,)))
        return hidden, gains

    def project(value, bits):
        # BF16 stores the HIGH bits of FP32, not integer-valued weights.
        weight = (bits.astype(xp.uint32) << xp.uint32(16)).view(xp.float32)
        return value @ weight.T

    rms = xp.sqrt(xp.mean(streams * streams, axis=-1, keepdims=True,
                          dtype=xp.float32) + hc.epsilon)
    normalized = streams / rms * hc.gamma
    flat = normalized.reshape(streams.shape[:-2] + (4 * hc.width,))
    low = project(flat, hc.down) * xp.float32(.25)
    low = low * _sigmoid(low)
    gates = _sigmoid(project(low, hc.up)).reshape(streams.shape)
    mixed = xp.mean(normalized * gates, axis=-2, dtype=xp.float32)
    gains = (None if hc.inject is None else
             xp.float32(2) * _sigmoid(project(flat, hc.inject) * xp.float32(.25)))
    return mixed, gains


def _project(model, projection, name, hidden):
    """Batch projections using canonical packed handles, bypassing vector wrappers."""
    if not callable(projection):
        result = hidden @ projection.T
    else:
        projection = getattr(projection, 'batch_projection', projection)
        weights = getattr(model, 'weights', None)
        device = getattr(weights, 'device', {}).get(name)
        if device is None:
            device = getattr(projection, 'packed_device_weight', None)
        if device is not None:
            from mini_llm.ops.quantized_linear import quantized_linear
            rows = xp.ascontiguousarray(hidden.reshape(-1, hidden.shape[-1]))
            result = quantized_linear(rows, device)
            result = result.reshape(hidden.shape[:-1] + (device.shape[0],))
        elif getattr(projection, 'native_gemv', False):
            rows = hidden.reshape(-1, 1, hidden.shape[-1])
            value = projection(rows)
            result = value.reshape(hidden.shape[:-1] + (value.shape[-1],))
        else:
            result = projection(hidden)
    if (not isinstance(result, xp.ndarray) or result.dtype != xp.float32 or
            result.shape[:-1] != hidden.shape[:-1]):
        raise ValueError('Chunk projection must return FP32 with matching leading dimensions')
    return result


def _project_singletons(model, projection, name, hidden):
    rows = hidden.reshape(-1, hidden.shape[-1])
    if getattr(projection, 'native_gemv', False):
        # Broadcast a batch of one-row products: the same Triton GEMV reduction,
        # not a GEMM with a different accumulation order.
        result = projection(rows[:, None, :])[:, 0]
    else:
        result = xp.concatenate([_project(model, projection, name, row[None]) for row in rows], axis=0)
    return result.reshape(hidden.shape[:-1] + (result.shape[-1],))


def _recurrent(model, layer, mixer, hidden, state):
    # Private adapter view: do not replace projections or graph flags on the
    # shared decode operator. Disable paired vector-only decode callbacks.
    adapter = copy(mixer)
    names = dict(qkv='attn_qkv', z='attn_gate', beta='ssm_beta',
                 alpha='ssm_alpha', out='ssm_out')
    adapter.graph_decode = False
    adapter.qkv_z_projection = None
    if BACKEND_NAME == 'cupy' and mixer.mixer._decode_fastpath_enabled:
        core = mixer.mixer
        adapter.mixer = copy(core)

        def causal_forward(qkv, gate, log_decay, beta, mutable, *, capture_owned=False):
            from mini_llm.ops.gated_delta_prefill import try_chunk
            chunk = try_chunk(core, qkv, gate, log_decay, beta, mutable,
                              capture_owned=capture_owned)
            if chunk is not None:
                return chunk
            # Sequence reference arithmetic differs from the fused decode core.
            # Preserve its exact recurrence before the quantized output projection.
            return xp.concatenate([
                core.forward(xp.ascontiguousarray(qkv[:, t:t + 1]), gate[:, t:t + 1],
                             log_decay[:, t:t + 1], beta[:, t:t + 1], mutable,
                             capture_owned=capture_owned)
                for t in range(qkv.shape[1])], axis=1)

        adapter.mixer.forward = causal_forward
    adapter.projections = {
        key: (lambda x, p=p, name=f'blk.{layer}.{names[key]}.weight', key=key:
              (_project_singletons if BACKEND_NAME == 'cupy' and key in ('beta', 'alpha')
               else _project)(model, p, name, x))
        for key, p in mixer.projections.items()}
    return adapter.forward(hidden[None], state)[0]


def _qsa(model, layer, attention, hidden, state):
    from mini_llm.ops.qsa_prefill_gpu import try_qsa_chunk
    from mini_llm.ops.qsa_sparse_prefill_gpu import try_sparse_qsa_chunk
    width = attention.config.index_topk + attention.config.block_size - 1
    prefix = max(0, min(hidden.shape[0], width-state.length))
    if prefix:
        result = try_qsa_chunk(model, layer, attention, hidden[:prefix], state)
        if result is not None:
            if prefix == hidden.shape[0]:
                return result
            tail = try_sparse_qsa_chunk(model, layer, attention, hidden[prefix:], state)
            if tail is None:
                tail = attention.forward(hidden[prefix:], state, start_position=state.length)
            return xp.concatenate((result, tail), axis=0)
    result = try_sparse_qsa_chunk(model, layer, attention, hidden, state)
    if result is not None:
        return result
    return attention.forward(hidden, state, start_position=state.length)


def _routed(experts, x, weights, ids):
    """Keep singleton device-plan arithmetic, otherwise group rows per expert.

    No executor/cache/operator is patched. The existing dispatcher owns job
    draining on error, cache admission/protection and expert-ID reduction order.
    In particular, do not replace Q8 device plans with PackedExpertFFNInference:
    fused gate/up and down can use different activation quantizers there.
    """
    executor = getattr(experts, 'expert_executor', None)
    mode = os.environ.get('MINI_LLM_PREFILL_EXPERTS', 'decode')
    if mode not in ('decode', 'grouped', 'stream', 'triton'):
        raise ValueError(f'Unsupported prefill expert mode: {mode!r}')
    if mode == 'triton' and BACKEND_NAME == 'cupy':
        from mini_llm.ops.triton_prefill_experts import try_triton_prefill_experts
        fused = try_triton_prefill_experts(experts, x, weights, ids)
        if fused is not None:
            return fused
        if os.environ.get('MINI_LLM_TRITON_ARITHMETIC', 'fp16') == 'native_int8':
            raise RuntimeError('Native INT8 prefill is ineligible; refusing an arithmetic-policy fallback')
    if mode in ('grouped', 'stream') and BACKEND_NAME == 'cupy':
        from mini_llm.ops.prefill_experts import try_prefill_experts
        grouped = try_prefill_experts(experts, x, weights, ids, mode=mode)
        if grouped is not None:
            return grouped
    if (BACKEND_NAME == 'cupy' and isinstance(executor, HeterogeneousExpertExecutor)
            and executor.device_expert_plan):
        # Reuse routes, rather than rerunning the router for each token. Always
        # use [1,1,D], including the final partial chunk, to preserve plan policy.
        return xp.concatenate([
            experts.forward(x[:, t:t + 1], weights[:, t:t + 1], ids[:, t:t + 1]).copy()
            for t in range(x.shape[1])], axis=1)
    return experts.forward(x, weights, ids)


def _moe(model, layer, moe, hidden, pipeline=None):
    x = hidden[None]
    if BACKEND_NAME == 'cupy' and not getattr(moe.router, '_capturable_logits', False):
        # The remaining CuPy GEMV path must not be substituted by GEMM.
        routes = [moe.router.forward(row[None, None]) for row in hidden]
        weights = xp.concatenate([w for w, _ in routes], axis=1)
        ids = xp.concatenate([i for _, i in routes], axis=1)
    else:
        weights, ids = moe.router.forward(x)
    # In the streamed source path, shared work is queued while the independent
    # issuer continues H2D, before routed expert kernels on the compute stream.
    routed = None if pipeline is not None else _routed(moe.experts, x, weights * moe.scale, ids)[0]
    stem = f'blk.{layer}.ffn_'
    gate = _project(model, moe.gate, stem + 'gate_shexp.weight', hidden)
    up = _project(model, moe.up, stem + 'up_shexp.weight', hidden)
    shared = _project(model, moe.down, stem + 'down_shexp.weight',
                      (gate * _sigmoid(gate)) * up)
    if BACKEND_NAME == 'cupy':
        if moe.shared_graph_decode:
            from mini_llm.ops.inference_matmul import matmul_inference
            score = matmul_inference(hidden[:, None, :],
                                     moe.shared_gate.reshape(-1, 1)).reshape(-1)
        else:
            score = xp.stack([row @ moe.shared_gate for row in hidden])
    else:
        score = hidden @ moe.shared_gate
    if pipeline is not None:
        routed = pipeline.run(moe.experts, xp.ascontiguousarray(hidden),
                              xp.ascontiguousarray(weights[0] * moe.scale),
                              xp.ascontiguousarray(ids[0], dtype=xp.int32))
    return routed + _sigmoid(score)[:, None] * shared


def prefill_chunked(model, tokens, state, chunk_size):
    """Return final FP32 [1,vocab] logits and advance the supplied state.

    chunk_size is a positive integer; input accepts the model's [T]/[1,T]
    integer contract. Capacity/input errors do not poison
    state. Any execution failure poisons it, as layer state may have advanced.
    Completed chunks advance state.length; failed chunks are not resumable.
    Extra activation storage scales with chunk_size, not the prompt length.
    """
    size = integer(chunk_size, 'chunk_size', 1)
    model._validate(state)
    tokens = model._tokens(tokens)
    if not state.has_capacity(tokens.size):
        raise ValueError('Generation state capacity exceeded')

    if not model._call_lock.acquire(blocking=False):
        raise RuntimeError('Calls sharing this model/executor must execute serially')
    pipeline = None
    try:
        model._validate(state)
        if not state.has_capacity(tokens.size):
            raise ValueError('Generation state capacity exceeded')
        try:
            for first in range(0, len(tokens), size):
                chunk = tokens[first:first + size]
                if os.environ.get('MINI_LLM_NATIVE_PREFILL_PIPELINE', '0') == '1':
                    if BACKEND_NAME != 'cupy':
                        raise RuntimeError('Native prefill pipeline requires CuPy CUDA')
                    executor = model.expert_executor
                    if len(chunk) >= 3072 and executor.cache._admission_limit == 0:
                        from mini_llm.ops.native_prefill_pipeline import NativePrefillPipeline
                        pipeline = getattr(model, '_native_prefill_pipeline', None)
                        if pipeline is None:
                            pipeline = NativePrefillPipeline()
                            model._native_prefill_pipeline = pipeline
                        pipeline.begin_chunk(model, len(chunk), topk=model.blocks[0].moe.router.k)
                    else:
                        # Explicitly reported compatibility path, not a claim
                        # that the source's short CPU/MMQ branch is reproduced.
                        executor.counters['native_prefill_short_reference_chunks'] += 1
                streams = model.final_hc.initialize(xp.stack([
                    model.embedding(int(token)) for token in chunk]))
                for i, b in enumerate(model.blocks):
                    if b.ple is not None:
                        rows = []
                        for token, row in zip(chunk, streams):
                            row, state.ple[i] = b.ple.step(int(token), row, state.ple[i])
                            rows.append(row)
                        streams = xp.stack(rows)
                    hidden, gains = _hc_read(b.attention_hc, streams)
                    model._diagnostic(i, 'hc_attention_read', hidden)
                    if b.recurrent:
                        mixed = _recurrent(model, i, b.mixer, hidden, state.recurrent[i])
                    else:
                        mixed = _qsa(model, i, b.mixer, hidden, state.qsa[i])
                    streams = b.attention_hc.write(streams, mixed, gains)
                    model._diagnostic(i, 'post_mixer', streams)
                    hidden, gains = _hc_read(b.ffn_hc, streams)
                    model._diagnostic(i, 'hc_ffn_read', hidden)
                    streams = b.ffn_hc.write(streams, _moe(model, i, b.moe, hidden, pipeline), gains)
                    model._diagnostic(i, 'post_moe', streams)
                if pipeline is not None:

                    pipeline.end_chunk()
                    for name, value in pipeline.counters.items():
                        model.expert_executor.counters['native_prefill_' + name] = value
                    pipeline = None
                if first + len(chunk) == len(tokens):
                    hidden = _hc_read(model.final_hc, streams[-1])[0]
                    model._diagnostic(-1, 'hc_output_read', hidden)
                    logits = model.output(hidden).reshape(1, -1)
                    if logits.shape != (1, model.config.vocab_size) or logits.dtype != xp.float32:
                        raise ValueError('Output must be FP32 [1,vocab]')
                    model._diagnostic(-1, 'logits', logits)
                executor = getattr(model, 'expert_executor', None)
                if isinstance(executor, HeterogeneousExpertExecutor):
                    executor.check_native_service()
                state.length += len(chunk)
            return logits
        except Exception:
            state.failed = True
            raise
    finally:
        try:
            if pipeline is not None:
                pipeline.end_chunk()
        finally:
            model._call_lock.release()
