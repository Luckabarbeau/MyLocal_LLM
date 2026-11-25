"""Separate native decoder, not a replacement for the Python model.

Setup may iterate layers. A step is exactly one native call, including embedding,
all branches, CPU GGML jobs, and logits. Mutable owners are transferred, not shared.
"""
from contextlib import ExitStack
import ctypes as C
import math

from . import (_Owned, _DecoderLayer, _integer, NativeError, PayloadLease,
               UploadedTensor, HCWeights, HCState, RecurrentLayer, QSALayer, MoE, PLEState)


class NativeDecoder(_Owned):
    _destroy = 'ml_decoder_destroy'

    @classmethod
    def bind(cls, engine, embedding, output, final_hc, hc, layers, *, capacity,
             scale=1.0, trace=False):
        _integer(capacity, 'capacity', 262144)
        if not isinstance(trace, bool):
            raise TypeError('trace must be bool')
        layers = tuple(layers)  # attention HC, branch, FFN HC, MoE, optional PLE
        runtime = engine._runtime
        objects = [(embedding, PayloadLease), (output, UploadedTensor),
                   (final_hc, HCWeights), (hc, HCState)]
        mutable = [hc]
        bindings = []
        with runtime._lock:
            for a, branch, f, moe, ple in layers:
                objects += [(a, HCWeights), (f, HCWeights), (moe, MoE)]
                if not isinstance(branch, (RecurrentLayer, QSALayer)):
                    raise TypeError('native recurrent or QSA branch required')
                objects.append((branch, type(branch)))
                if ple is not None:
                    objects.append((ple, PLEState))
                mutable.extend([branch, moe] + ([ple] if ple is not None else []))
                bindings.append(_DecoderLayer(a._require_open(), branch._require_open(),
                    f._require_open(), moe._require_open(), ple._require_open() if ple else 0,
                    int(isinstance(branch, RecurrentLayer))))
            for obj, kind in objects:
                if not isinstance(obj, kind) or obj._runtime is not runtime:
                    raise TypeError('same-runtime native bindings required')
            out = C.c_uint64()
            runtime._call('ml_decoder_create', engine._require_open(), embedding._require_open(),
                output._require_open(), final_hc._require_open(), hc._require_open(),
                (_DecoderLayer*len(bindings))(*bindings), len(bindings), capacity,
                scale, int(trace), C.byref(out))
            for obj in mutable:
                obj._handle = 0  # Native registry transfer succeeded; close is now a no-op.
            result = cls(runtime, out.value)
            result.width = embedding.descriptor.shape[1]
            result.vocab = embedding.descriptor.shape[0]
            result.layers = len(bindings)
            result.capacity = capacity
            return result

    @property
    def state(self):
        position, failed = C.c_uint64(), C.c_int32()
        self._runtime._call('ml_decoder_state', self._require_open(), C.byref(position), C.byref(failed))
        return position.value, bool(failed.value)

    def reset(self):
        self._runtime._call('ml_decoder_reset', self._require_open())

    def step_device(self, token):
        """Commit a step leaving logits resident; returns only committed position."""
        _integer(token, 'token', 2**31-1)
        committed = C.c_uint64()
        self._runtime._call('ml_decoder_step', self._require_open(), token,
                           None, 0, C.byref(committed))
        return committed.value

    def step(self, token):
        import numpy as np
        _integer(token, 'token', 2**31-1)
        logits = np.empty((1, self.vocab), np.float32)
        committed = C.c_uint64()
        self._runtime._call('ml_decoder_step', self._require_open(), token,
            logits.ctypes.data_as(C.POINTER(C.c_float)), logits.size, C.byref(committed))
        return logits

    def trace_for_test(self, layer, stage):
        import numpy as np
        _integer(layer, 'layer', self.layers-1)
        stage = ('hc_attention_read', 'post_mixer', 'hc_ffn_read', 'post_moe').index(stage)
        result = np.empty((4, self.width) if stage % 2 else (self.width,), np.float32)
        self._runtime._call('ml_decoder_trace', self._require_open(), layer, stage,
            result.ctypes.data_as(C.POINTER(C.c_float)), result.size)
        return result

    @classmethod
    def open(cls, runtime, path, *, bridge_library, ggml_library, capacity=256,
             budget_bytes=10*1024**3, dense_budget_bytes=7*1024**3,
             mapped_budget_bytes=96*1024**3, host_budget_bytes=1024**3,
             page_size=16, physical_pages=2, threads=4, trace=False):
        """Actual 48-layer schema; all routed experts stay mmap-backed on CPU.

        No promotion, expert pinning, decoded expert bank, or numerical fallback.
        Each open owns a separate session and Engine. Weight sharing across opens
        is not yet implemented. The active capacity is independent of pagecap262144.
        Budgets are requested allocations, not CUDA context/driver or mmap RSS.
        """

        from mini_llm.external.qwen4exp_ple import PLEConfig
        for name, value in locals().copy().items():
            if name in ('capacity', 'budget_bytes', 'dense_budget_bytes', 'mapped_budget_bytes',
                        'host_budget_bytes', 'page_size', 'physical_pages', 'threads'):
                _integer(value, name, 2**63-1)
        if not capacity or capacity > 262144 or capacity % 4:
            raise ValueError('active capacity must be a positive multiple of four <=262144')
        if not page_size or page_size > 4096 or page_size % 4 or not physical_pages:
            raise ValueError('positive page slots and page size divisible by four <=4096 required')
        if physical_pages > math.ceil(capacity/page_size):
            raise ValueError('physical pages exceed active logical page count')
        with ExitStack() as stack:
            store = stack.enter_context(runtime.gguf(path))
            h = store.validate_qwen4exp()  # Native authoritative snapshots/schema, before payloads.
            if (h.width, h.vocab, h.rank, len(h.recurrent)) != (2560, 248320, 320, 48):
                raise ValueError('whole-checkpoint factory requires actual 48-layer geometry')
            if h.recurrent != tuple((i+1)%4 != 0 for i in range(48)):
                raise ValueError('unsupported actual recurrent/QSA schedule')
            import os
            mapped = sum(os.stat(p).st_size for p in store.paths)
            if mapped > mapped_budget_bytes:
                raise ValueError('mapped shard budget exceeded before payload allocation')
            names = [n for n in h.shapes if '_exps.weight' not in n and
                     n not in ('token_embd.weight', 'per_layer_token_embd.weight')]
            dense = sum(store.tensors[n].nbytes for n in names)
            # Native HC expands BF16 exactly; PLE expands dense controls, not its table.
            dense += sum(store.tensors[n].nbytes*2 for n in names if store.tensors[n].type_id == 30
                         and ('hc_' in n or 'ple_norm_' in n or 'ple_conv1d' in n))
            if dense > dense_budget_bytes:
                raise ValueError('dense GPU budget exceeded before allocation')
            # Conservative workspace upper bound for the validated actual geometry.
            # Recurrent includes both committed and staged matrices; QSA includes
            # pooled/index buffers, selected scratch, physical pages and one stage.
            workspace = 36*10*1024**2 + 12*(12*1024**2 + capacity*272 +
                         (physical_pages+1)*page_size*4096) + 16*1024**2
            if dense + workspace > budget_bytes:
                raise ValueError('GPU allocation budget exceeded before allocation')
            per_host = host_budget_bytes//12
            # KVPageRequests: three uint64 scalars and two arrays of 2051 IDs.
            metadata = (3+2*2051)*8 + 8*(math.ceil(capacity/page_size)+2*physical_pages)
            if per_host < metadata + page_size*4096:
                raise ValueError('host page budget cannot hold metadata and one page')
            engine = stack.enter_context(runtime.engine(budget_bytes=budget_bytes, max_sessions=1))
            def lease(name):
                return stack.enter_context(store.lease(name))
            def upload(name):
                return stack.enter_context(engine.upload(lease(name)))
            eps = h.mixer.rms_eps
            def hc(stem, inject=True):
                return stack.enter_context(engine.hc_weights(*[upload(stem+s) for s in
                    ('norm.weight','down.weight','up.weight')],
                    upload(stem+'inject.weight') if inject else None,
                    epsilon=eps, arithmetic='reference_fp32'))
            layers = []
            d_ff = h.shapes['blk.0.ffn_gate_exps.weight'][1]
            for i, recurrent in enumerate(h.recurrent):
                p = f'blk.{i}.'
                a, f = hc(p+'hc_attn_'), hc(p+'hc_ffn_')
                ple = None
                if i in h.ple_layers:
                    table = lease('per_layer_token_embd.weight')
                    config = PLEConfig.from_metadata(store.metadata, table.descriptor.shape[0])
                    w = stack.enter_context(engine.ple_weights(table, config,
                        *[upload(p+'ple_'+s+'.weight') for s in
                          ('key','value','norm_key','norm_query','norm_conv','conv1d')],
                        epsilon=eps, arithmetic='reference_fp32'))
                    ple = stack.enter_context(w.state(capacity))
                if recurrent:
                    branch = stack.enter_context(engine.recurrent_layer(h.mixer, capacity,
                        *[upload(p+s) for s in ('attn_qkv.weight','attn_gate.weight','ssm_beta.weight',
                          'ssm_alpha.weight','ssm_out.weight','ssm_conv1d.weight','ssm_norm.weight',
                          'ssm_a','ssm_dt.bias')], arithmetic='reference_fp32'))
                else:
                    branch = stack.enter_context(engine.qsa_layer(capacity,
                        *[upload(p+s) for s in ('attn_q.weight','attn_k.weight','attn_v.weight',
                          'indexer.q_proj.weight','indexer.k_proj.weight','attn_output.weight',
                          'attn_q_norm.weight','attn_k_norm.weight','indexer.q_norm.weight','indexer.k_norm.weight')],
                        policy='ggml_cpu', epsilon=eps, kv_policy='reference_fp32',
                        page_size=page_size, physical_pages=physical_pages, host_budget_bytes=per_host))
                moe = stack.enter_context(engine.moe(*[upload(p+s) for s in
                    ('ffn_gate_inp.weight','ffn_gate_shexp.weight','ffn_up_shexp.weight',
                     'ffn_down_shexp.weight','ffn_gate_inp_shexp.weight')],
                    *[lease(p+'ffn_'+s+'_exps.weight') for s in ('gate','up','down')],
                    d_model=h.width, d_ff=d_ff, gpu_experts=(), threads=threads,
                    bridge_library=bridge_library, ggml_library=ggml_library))
                layers.append((a, branch, f, moe, ple))
            decoder = engine.decoder(lease('token_embd.weight'), upload('output.weight'),
                hc('output_hc_', False), stack.enter_context(engine.hc_state(h.width, h.rank)),
                layers, capacity=capacity, scale=h.expert_scale, trace=trace)
            decoder.memory_plan = dict(dense_bytes=dense, workspace_bound=workspace,
                mapped_bytes=mapped, gpu_budget=budget_bytes, host_page_budget=host_budget_bytes,
                kv_policy='reference_fp32', expert_placement='all_cpu_mmap')
            return decoder
