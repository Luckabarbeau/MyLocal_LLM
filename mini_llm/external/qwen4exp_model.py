"""First Qwen4Exp text runtime: FP32 operators, sparse experts, explicit sessions.

Synthetic integration validation is not independent full-checkpoint parity.
"""
from dataclasses import dataclass, field
from types import SimpleNamespace
import threading
import weakref
import os

import numpy as np

from mini_llm.backend import BACKEND_NAME, asnumpy, xp
from mini_llm.ops.expert_residency import HeterogeneousExpertExecutor
from mini_llm.ops.experts_inference import ExpertsInference
from mini_llm.ops.router_inference import RouterInference
from .qwen4exp_attention import Qwen4ExpAttention
from .qwen4exp_hyperconnection import Qwen4ExpHyperConnection, _sigmoid
from .qwen4exp_mixer import Qwen4ExpRecurrentMixerInference
from .qwen4exp_ple import Qwen4ExpPLE
from .qwen4exp_weights import Qwen4ExpWeights, integer


@dataclass
class Qwen4ExpLayer:
    attention_hc: object
    mixer: object
    ffn_hc: object
    moe: object
    recurrent: bool
    ple: object = None


class Qwen4ExpMoE:
    def __init__(self, router, experts, gate, up, down, shared_gate, *, scale=1.0,
                 fast_elementwise=False, overlap_shared=False, shared_graph_decode=False,
                                  defer_shared=False):
        if not isinstance(fast_elementwise, bool):
            raise TypeError('fast_elementwise must be bool')
        if not isinstance(overlap_shared, bool):
            raise TypeError('overlap_shared must be bool')
        if overlap_shared and BACKEND_NAME != 'cupy':
            raise ValueError('overlap_shared requires CuPy')
        if not isinstance(shared_graph_decode,bool):
            raise TypeError('shared_graph_decode must be bool')
        if shared_graph_decode and BACKEND_NAME!='cupy':
            raise ValueError('Shared graph decode requires CuPy')
        if not isinstance(defer_shared, bool):
            raise TypeError('defer_shared must be bool')
        if defer_shared and not overlap_shared:
            raise ValueError('defer_shared requires overlap_shared')
        self.defer_shared = defer_shared
        self.shared_graph_decode = shared_graph_decode
        self._shared_graph_enabled = shared_graph_decode
        self._shared_graph = None
        self._fused_moe_graph = os.environ.get('MINI_LLM_NATIVE_MOE_FUSED_SHARED','0')=='1'
        self.fast_elementwise = fast_elementwise
        self._shared_stream = xp.cuda.Stream(non_blocking=True) if overlap_shared else None
        self.router, self.experts = router, experts
        self.gate, self.up, self.down = gate, up, down
        self.shared_gate, self.scale = shared_gate, np.float32(scale)

    def _compute_shared(self,hidden,unused=()):
        gate=self.gate(hidden)
        if self.fast_elementwise and BACKEND_NAME=='cupy':
            from mini_llm.ops.qwen4exp_elementwise_gpu import shared_hidden
            shared=self.down(shared_hidden(gate,self.up(hidden)))
        else:
            shared=self.down((gate*_sigmoid(gate))*self.up(hidden))
        if self.shared_graph_decode:
            from mini_llm.ops.inference_matmul import matmul_inference
            score=matmul_inference(hidden.reshape(1,-1),self.shared_gate.reshape(-1,1)).reshape(())
        else:
            score=hidden@self.shared_gate
        return shared,score

    def _run_shared(self,hidden):
        if not self._shared_graph_enabled:
            return self._compute_shared(hidden)
        if self._shared_graph is None:
            from mini_llm.ops.inference_graph import CapturedArrayInference
            self._shared_graph=CapturedArrayInference(self._compute_shared,hidden,(),pool_limit_bytes=2*1024**2)
        return self._shared_graph.forward(hidden,(),borrow_output=True)

    def graph_statistics(self):
        return (self._shared_graph.statistics() if self._shared_graph is not None else
                dict(calls=0,pool_reserved_bytes=0,pool_used_bytes=0))

    def enqueue_gpu_branch(self, hidden):
        """Pure GPU shared branch for the frozen native service graph."""
        return self._compute_shared(hidden.reshape(-1))

    def combine_gpu_branch(self, routed, shared_values):
        shared,score=shared_values
        routed=routed.reshape(-1)
        if self.fast_elementwise:
            from mini_llm.ops.qwen4exp_elementwise_gpu import combine_shared
            output=combine_shared(routed,shared,score)
        else:
            output=routed+_sigmoid(score)*shared
        return output.reshape(1,-1)

    def __call__(self, hidden, *, routing=None):
        x = hidden.reshape(1,1,-1)
        if self._fused_moe_graph and BACKEND_NAME=='cupy':
            device_plan=getattr(self.experts.expert_executor,'try_forward_device_plan',None)
            if device_plan is not None:
                routing=self.router.forward(x) if routing is None else routing
                weights,ids=routing
                output=device_plan(self.experts.layer_id,x.reshape(1,-1),
                    (weights*self.scale).reshape(-1),ids.reshape(-1),self.experts.n_experts,
                    gpu_branch=self)
                if output is not None:
                    return output.reshape(-1)
        if self._shared_stream is not None:
            caller = xp.cuda.get_current_stream()
            if caller.is_capturing():
                raise RuntimeError('Shared expert overlap requires an uncaptured dispatch seam')
            ready = xp.cuda.Event(disable_timing=True)
            ready.record(caller)  # Only hidden is needed, not routed/CPU completion.
            complete = xp.cuda.Event(disable_timing=True)
            launched = False
            shared = shared_score = None
            def launch_shared():
                nonlocal launched, shared, shared_score
                # The CPU input-ready fence is ahead of shared work in deferred
                # mode; stream ownership still protects every shared consumer.
                self._shared_stream.wait_event(ready)
                launched = True
                try:
                    with self._shared_stream:
                        shared,shared_score = self._run_shared(hidden)
                finally:
                    complete.record(self._shared_stream)
            try:
                if not self.defer_shared:
                    launch_shared()
                weights, ids = self.router.forward(x) if routing is None else routing
                if self.defer_shared:
                    routed = self.experts.forward(x, weights*self.scale, ids,
                        on_cpu_submitted=launch_shared).reshape(-1)
                else:
                    routed = self.experts.forward(x, weights*self.scale, ids).reshape(-1)
            finally:
                if launched:
                    # Fence also on failure before input/weight owners can retire.
                    caller.wait_event(complete)
            if self.fast_elementwise:
                from mini_llm.ops.qwen4exp_elementwise_gpu import combine_shared
                return combine_shared(routed, shared, shared_score)
            return routed + _sigmoid(shared_score)*shared
        weights, ids = self.router.forward(x) if routing is None else routing
        routed = self.experts.forward(x, weights*self.scale, ids).reshape(-1)
        shared,shared_score = self._run_shared(hidden)
        if self.fast_elementwise and BACKEND_NAME == 'cupy':
            from mini_llm.ops.qwen4exp_elementwise_gpu import combine_shared
            return combine_shared(routed,shared,shared_score)
        return routed+_sigmoid(shared_score)*shared


@dataclass(eq=False)
class Qwen4ExpGenerationState:
    owner: object = field(repr=False)
    max_length: int
    recurrent: dict
    qsa: dict
    ple: dict
    reserved_bytes: int = 0
    length: int = 0
    failed: bool = False
    batch_size: int = 1
    _busy: object = field(default_factory=threading.Lock, repr=False)

    def remaining_capacity(self):
        return self.max_length-self.length

    def has_capacity(self, n_tokens=1):
        return not self.failed and self.length+integer(n_tokens,'n_tokens') <= self.max_length


class Qwen4ExpGGUFModel:
    """Single-batch text only. Calls sharing an executor must execute serially.

    Use open(path,max_length=...) for an owned mmap source, or from_index for
    borrowed sources. Fresh states reset requests; failed states cannot resume.
    Diagnostic callback signature: (layer_id, stage, copied_FP32_array).
    """
    @classmethod
    def open(cls, path, *, max_length, diagnostic_callback=None, expert_executor=None, **budgets):
        weights = Qwen4ExpWeights.open(path)
        try:
            return cls._build(weights, max_length=max_length, diagnostic_callback=diagnostic_callback,
                              expert_executor=expert_executor, **budgets)
        except Exception:
            weights.close()
            raise

    @classmethod
    def from_index(cls, index, *, max_length, diagnostic_callback=None, expert_executor=None, **budgets):
        return cls._build(Qwen4ExpWeights(index), max_length=max_length,
                          diagnostic_callback=diagnostic_callback, expert_executor=expert_executor, **budgets)

    @classmethod
    def _build(cls, weights, *, max_length, diagnostic_callback, expert_executor,
               expert_promote_after=3, qsa_fast_decode=False,
               hc_fast_decode=False, router_fast_decode=False, gdn_fast_decode=False,
               hc_graph_decode=False, hc_native_read=False, group_after_admission=False,
               elementwise_fast_decode=False, expert_victim_policy='sort',
               expert_cpu_backend='native', group_mixed_decode=False,
               expert_cache_slots=False, overlap_shared=False, gdn_graph_decode=False,
               device_expert_plan=False, pre_moe_graph=False, shared_graph_decode=False,
                              defer_shared=False, **budgets):
        expert_promote_after = integer(expert_promote_after, 'expert_promote_after', 1)
        if expert_cpu_backend not in ('native', 'ggml', 'ggml_pool'):
            raise ValueError('expert_cpu_backend must be native, ggml or ggml_pool')
        for name, value in (('qsa_fast_decode', qsa_fast_decode),
                            ('hc_fast_decode', hc_fast_decode), ('router_fast_decode', router_fast_decode),
                            ('gdn_fast_decode', gdn_fast_decode), ('hc_graph_decode', hc_graph_decode),
                            ('hc_native_read', hc_native_read),
                            ('elementwise_fast_decode', elementwise_fast_decode),
                            ('overlap_shared', overlap_shared),
                            ('gdn_graph_decode', gdn_graph_decode),
                            ('device_expert_plan', device_expert_plan),
                            ('pre_moe_graph', pre_moe_graph),
                            ('shared_graph_decode',shared_graph_decode),
                                                        ('defer_shared',defer_shared),
                            ('group_mixed_decode', group_mixed_decode),
                            ('group_after_admission', group_after_admission),
                            ('expert_cache_slots', expert_cache_slots)):
            if not isinstance(value, bool):
                raise TypeError(f'{name} must be bool')
        if pre_moe_graph and (BACKEND_NAME != 'cupy' or not hc_native_read or not hc_fast_decode or
                not router_fast_decode or not gdn_fast_decode or hc_graph_decode or gdn_graph_decode):
            raise ValueError('Pre-MoE graph requires native HC/router/GDN and no nested graph mode')
        if diagnostic_callback is not None and not callable(diagnostic_callback):
            raise TypeError('diagnostic_callback must be callable')
        index, h = weights.index, weights.headers
        if device_expert_plan:
            if BACKEND_NAME != 'cupy' or expert_cpu_backend != 'ggml_pool':
                raise ValueError('Device expert plans require CuPy and ggml_pool')
            if (len(h.recurrent) > 64 or index.n_experts > 512 or index.top_k > 10 or
                    max(h.width, index.d_ff) > 4096):
                raise ValueError('Unsupported bounded device expert plan geometry')
            per_device = index.n_experts*24+index.top_k*(32+4*(2*h.width+index.d_ff))+4*h.width+4
            per_host = 4*(h.width+index.top_k+1+index.top_k*h.width)
            budgets['activation_bytes'] = integer(budgets.get('activation_bytes', 64*1024**2),
                'activation_bytes') + len(h.recurrent)*2*per_device
            budgets['staging_bytes'] = integer(budgets.get('staging_bytes', 64*1024**2),
                'staging_bytes') + len(h.recurrent)*2*per_host
        if os.environ.get('MINI_LLM_BOUND_RECURRENT_STATE','0')=='1':
            if not pre_moe_graph:
                raise ValueError('Bound recurrent state requires the pre-MoE graph')
            r=h.mixer
            retained=4*sum(h.recurrent)*((r.conv_kernel_size-1)*r.qkv_channels+r.n_value_heads*r.head_dim**2)
            # Old bound buffers can survive until the next request's first
            # recapture. Reserve this bounded overlap before allocating state.
            budgets['activation_bytes']=integer(budgets.get('activation_bytes',64*1024**2),'activation_bytes')+retained
        if os.environ.get('MINI_LLM_NATIVE_MOE_TIMELINE','0')=='1':
            if not device_expert_plan or os.environ.get('MINI_LLM_NATIVE_MOE_HANDOFF','0')!='1':
                raise ValueError('Native timeline requires native device handoffs')
            trace_bytes=len(h.recurrent)*2048*9*8*2+64*1024
            budgets['staging_bytes']=integer(budgets.get('staging_bytes',64*1024**2),'staging_bytes')+trace_bytes
        from mini_llm.ops.native_expert_staging import stage_limit
        if stage_limit():
            if (not device_expert_plan or os.environ.get('MINI_LLM_NATIVE_MOE_SERVICE','0')!='1' or
                    os.environ.get('MINI_LLM_NATIVE_MOE_HANDOFF','0')!='1' or
                    os.environ.get('MINI_LLM_EXPERT_GPU_Q8','0')!='1'):
                raise ValueError('Native staging requires device plans, native service/handoff and GPU Q8')
            # Shared16MiB device + pinned arena, conservatively charged to each
            # memory budget; per-layer Q8/term scratch has its own reservation.
            budgets['staging_bytes']=integer(budgets.get('staging_bytes',64*1024**2),'staging_bytes')+32*1024**2
            per_stage=index.top_k*(4*(h.width+index.d_ff)+24+index.d_ff//32*36)+h.width//32*36+256
            budgets['activation_bytes']=integer(budgets.get('activation_bytes',64*1024**2),'activation_bytes')+len(h.recurrent)*per_stage
        if expert_executor is not None:
            if (expert_executor.d_model,expert_executor.d_ff,expert_executor.compute_dtype,
                    expert_executor.cpu_packed_backend,expert_executor.model_id) != (
                    index.d_model,index.d_ff,'float32',expert_cpu_backend,index.model_id):
                raise ValueError('Borrowed executor geometry/precision/backend/model identity mismatch')
            if getattr(expert_executor, 'device_expert_plan', False) != device_expert_plan:
                raise ValueError('Borrowed executor device-plan configuration mismatch')
            if expert_executor.host_cache.budget_bytes:
                raise ValueError('Full expert host-dequant cache is forbidden')
            budgets['cache_bytes'] = expert_executor.cache.device_budget_bytes
            budgets['cpu_workers'] = expert_executor.cpu_workers
        slot_stride = 0
        if expert_cache_slots:
            if BACKEND_NAME != 'cupy':
                raise RuntimeError('Expert cache slots require CuPy')
            if expert_executor is not None:
                raise ValueError('Configure slots on the borrowed expert executor explicitly')
            if not budgets.get('cache_bytes', 0):
                raise ValueError('Expert cache slots require a positive cache budget')
            layer_bundle_bytes=[sum(index.descriptors[f'blk.{layer}.ffn_{p}_exps.weight'].nbytes
                              // index.n_experts for p in ('gate', 'up', 'down'))
                          for layer in range(len(h.recurrent))]
            largest = max(layer_bundle_bytes)
            slot_stride = (largest + 255) // 256 * 256
            # Account for pinned allocator rounding plus other transfer scratch.
            budgets['staging_bytes'] = max(integer(budgets.get('staging_bytes', 64*1024**2),
                'staging_bytes'), 2 * (1 << (slot_stride-1).bit_length()))
        graph_reservation = len(h.recurrent)*2*1024**2 if shared_graph_decode else 0
        if shared_graph_decode and BACKEND_NAME!='cupy':
            raise RuntimeError('Shared graph decode requires CuPy')
        if gdn_graph_decode or pre_moe_graph:
            if BACKEND_NAME != 'cupy':
                raise RuntimeError('GDN graph decode requires CuPy')
            graph_reservation += sum(h.recurrent) * 16 * 1024**2
        if hc_graph_decode:
            if BACKEND_NAME != 'cupy':
                raise RuntimeError('HC graph decode requires CuPy')
            graph_reservation += (2 * len(h.recurrent) + 1) * 2 * 1024**2
        qsa_block_graph = os.environ.get('MINI_LLM_QSA_BLOCK_GRAPH','0') == '1'
        if qsa_block_graph and (not qsa_fast_decode or not hc_native_read or not router_fast_decode or
                not shared_graph_decode or not device_expert_plan or
                os.environ.get('MINI_LLM_QSA_DEFER_VALIDATION','0') != '1' or
                os.environ.get('MINI_LLM_NATIVE_MOE_SERVICE','0') != '1'):
            raise ValueError('QSA block capture requires native HC/router/service, shared capture and deferred validation')
        qsa_front_graph = qsa_fast_decode and os.environ.get('MINI_LLM_QSA_FRONTEND_GRAPH','0')=='1'
        if qsa_block_graph and not qsa_front_graph:
            raise ValueError('QSA block capture requires capture-ready frontend projections')
        if qsa_block_graph:
            graph_reservation += (len(h.recurrent)-sum(h.recurrent))*8*1024**2
        if qsa_front_graph:
            if BACKEND_NAME != 'cupy':
                raise RuntimeError('QSA frontend graph requires CuPy')
            graph_reservation += (len(h.recurrent)-sum(h.recurrent))*2*1024**2
        if device_expert_plan and os.environ.get('MINI_LLM_NATIVE_MOE_GRAPH','0')=='1':
            graph_reservation += len(h.recurrent)*1024**2
        if graph_reservation:
            budgets['graph_workspace_bytes'] = max(
                integer(budgets.get('graph_workspace_bytes', 0), 'graph_workspace_bytes'),
                graph_reservation)
        hc_bf16=os.environ.get('MINI_LLM_HC_BF16_WEIGHTS','0')=='1'
        if hc_bf16:
            if not hc_native_read or not hc_fast_decode or hc_graph_decode:
                raise ValueError('BF16 HC storage requires native HC read')
            weights.configure_hc_bf16()
        plan = weights.memory_plan(max_length, **budgets)
        weights.approve(plan)
        slot_sizes=None
        if os.environ.get('MINI_LLM_SIZED_EXPERT_SLOTS','0')=='1':
            if not expert_cache_slots:
                raise ValueError('Sized slots require the fixed-slot expert cache')
            from mini_llm.ops.expert_slot_layout import sized_slot_layout
            slot_sizes=sized_slot_layout(plan['cache_bytes'],layer_bundle_bytes,index.n_experts)
        if gdn_graph_decode or pre_moe_graph:
            for i, recurrent in enumerate(h.recurrent):
                if recurrent and any(
                        weights.store.tensors[f'blk.{i}.{name}.weight'].quant_type not in {'F32', 'F16', 'BF16'}
                        and f'blk.{i}.{name}.weight' not in weights.device_names
                        for name in ('attn_qkv', 'attn_gate', 'ssm_beta', 'ssm_alpha', 'ssm_out')):
                    raise ValueError('GDN capture requires every recurrent projection on the GPU')
        if shared_graph_decode and any(weights.store.tensors[f'blk.{i}.{name}.weight'].quant_type not in
                {'F32','F16','BF16'} and f'blk.{i}.{name}.weight' not in weights.device_names
                for i in range(len(h.recurrent))
                for name in ('ffn_gate_shexp','ffn_up_shexp','ffn_down_shexp')):
            raise ValueError('Shared capture requires every shared projection on GPU')
        if qsa_front_graph and any(weights.store.tensors[f'blk.{i}.{name}.weight'].quant_type not in
                {'F32','F16','BF16'} and f'blk.{i}.{name}.weight' not in weights.device_names
                for i, recurrent in enumerate(h.recurrent) if not recurrent
                for name in ('attn_q','attn_k','attn_v','indexer.k_proj')):
            raise ValueError('QSA frontend capture requires every input projection on GPU')
        executor = expert_executor
        try:
            if executor is None:
                executor = HeterogeneousExpertExecutor(index.provider, model_id=index.model_id,
                    d_model=index.d_model, d_ff=index.d_ff, compute_dtype='float32',
                    cpu_packed_backend=expert_cpu_backend, cpu_workers=plan['cpu_workers'],
                    device_budget_bytes=plan['cache_bytes'], host_dequant_budget_bytes=0,
                    promote_after=expert_promote_after, group_after_admission=group_after_admission,
                    expert_victim_policy=expert_victim_policy,
                    group_mixed_decode=group_mixed_decode, device_expert_plan=device_expert_plan,
                    expert_slot_stride_bytes=slot_stride, expert_slot_sizes=slot_sizes,
                                        slot_staging_budget_bytes=plan['staging_bytes'])
            dense, project = weights.dense, weights.projection
            def recurrent_project(name):
                return weights.projection(name, capturable=gdn_graph_decode or pre_moe_graph)
            def ple_project(name):
                capture_ready = os.environ.get('MINI_LLM_TOKEN_GRAPH_PREP','0') == '1'
                base = weights.projection(name, capturable=capture_ready)
                if not capture_ready:
                    return base
                return lambda x: base(x.reshape(1,-1)).reshape(-1)
            def shared_project(name):
                base=weights.projection(name,capturable=shared_graph_decode)
                if not shared_graph_decode:
                    return base
                return lambda x:base(x.reshape(-1,x.shape[-1])).reshape(x.shape[:-1]+(-1,))
            def qsa_project(name):
                base=weights.projection(name,capturable=qsa_front_graph)
                if not qsa_front_graph:
                    return base
                return lambda x:base(x.reshape(1,-1)).reshape(-1)
            def hc(stem, inject=True):
                # HC copies these small matrices. No host dictionary retains
                # the constructor inputs after this call returns.
                return Qwen4ExpHyperConnection(dense(stem+'norm.weight'),
                    weights.hc_matrix(stem+'down.weight'),weights.hc_matrix(stem+'up.weight'),
                    weights.hc_matrix(stem+'inject.weight') if inject else None, epsilon=h.mixer.rms_eps,
                    fast_decode=hc_fast_decode, graph_decode=hc_graph_decode,
                    native_read=hc_native_read,packed_bf16=hc_bf16)
            blocks = []
            for i, rec in enumerate(h.recurrent):
                p = f'blk.{i}.'
                attn_hc, ffn_hc = hc(p+'hc_attn_'), hc(p+'hc_ffn_')
                ple = None
                if i in h.ple_layers:
                    ple = Qwen4ExpPLE.from_store(weights.store,
                        key=ple_project(p+'ple_key.weight'), value=ple_project(p+'ple_value.weight'),
                        gamma_key=dense(p+'ple_norm_key.weight').reshape(4,h.width),
                        gamma_query=dense(p+'ple_norm_query.weight').reshape(4,h.width),
                        gamma_conv=dense(p+'ple_norm_conv.weight').reshape(4,h.width),
                        conv=dense(p+'ple_conv1d.weight'))
                if rec:
                    qkv_projection = recurrent_project(p+'attn_qkv.weight')
                    z_projection = recurrent_project(p+'attn_gate.weight')
                    paired = (weights.paired_projection(qkv_projection,z_projection)
                              if os.environ.get('MINI_LLM_Q8_RECURRENT_PAIR','0') == '1' else None)
                    mixer = Qwen4ExpRecurrentMixerInference(h.mixer,
                        qkv=qkv_projection, z=z_projection, qkv_z_projection=paired,
                        beta=recurrent_project(p+'ssm_beta.weight'),alpha=recurrent_project(p+'ssm_alpha.weight'),
                        out=recurrent_project(p+'ssm_out.weight'),conv_weight=dense(p+'ssm_conv1d.weight'),
                        norm_weight=dense(p+'ssm_norm.weight'),ssm_a=dense(p+'ssm_a'),
                        dt_bias=dense(p+'ssm_dt.bias'), fast_coefficients=elementwise_fast_decode,
                        graph_decode=gdn_graph_decode)
                    mixer.mixer._decode_fastpath_enabled = gdn_fast_decode
                else:
                    mixer = Qwen4ExpAttention(h.attention,
                        query_gate=qsa_project(p+'attn_q.weight'),key=qsa_project(p+'attn_k.weight'),
                        value=qsa_project(p+'attn_v.weight'),output=(qsa_project(p+'attn_output.weight')
                                                    if qsa_block_graph else project(p+'attn_output.weight')),
                        index_query=project(p+'indexer.q_proj.weight'),index_key=qsa_project(p+'indexer.k_proj.weight'),
                        gamma_q=dense(p+'attn_q_norm.weight'),gamma_k=dense(p+'attn_k_norm.weight'),
                        gamma_index_q=dense(p+'indexer.q_norm.weight'),gamma_index_k=dense(p+'indexer.k_norm.weight'),
                        fast_decode=qsa_fast_decode)
                router = RouterInference(h.width,index.n_experts,index.top_k,
                    renormalize_selected=True,normalization_floor=2.0**-14)
                router.W_router = dense(p+'ffn_gate_inp.weight').T
                router.b_router = xp.zeros(index.n_experts,dtype=xp.float32)
                # Both paths retain full softmax; fusion is explicitly switchable.
                router._decode_fastpath_enabled = router_fast_decode
                router._capturable_logits = (pre_moe_graph and rec) or (qsa_block_graph and not rec)
                experts = ExpertsInference(h.width,index.d_ff,index.n_experts,
                    expert_executor=executor,layer_id=i)
                moe = Qwen4ExpMoE(router,experts,shared_project(p+'ffn_gate_shexp.weight'),
                    shared_project(p+'ffn_up_shexp.weight'),shared_project(p+'ffn_down_shexp.weight'),
                    dense(p+'ffn_gate_inp_shexp.weight'),scale=h.expert_scale,
                    fast_elementwise=elementwise_fast_decode, overlap_shared=overlap_shared,
                    shared_graph_decode=shared_graph_decode, defer_shared=defer_shared)
                blocks.append(Qwen4ExpLayer(attn_hc,mixer,ffn_hc,moe,rec,ple))
            result = cls.from_components(width=h.width,vocab=h.vocab,blocks=blocks,
                final_hc=hc('output_hc_',False),embedding=weights.embedding,
                output=project('output.weight'),max_length=plan['max_length'],
                diagnostic_callback=diagnostic_callback)
            result.weights, result.index = weights, index
            result.expert_executor, result._owns_executor = executor, expert_executor is None
            result.plan = plan
            if pre_moe_graph:
                from .qwen4exp_pre_moe_graph import Qwen4ExpPreMoEGraph
                result._pre_moe_segments = {i:Qwen4ExpPreMoEGraph(b.attention_hc,b.mixer,b.ffn_hc,b.moe.router)
                    for i,b in enumerate(blocks) if b.recurrent}
            else:
                result._pre_moe_segments = {}
            result._qsa_block_segments = {}
            result._token_graph = None
            result._shared_graph_owners=tuple(b.moe for b in blocks) if shared_graph_decode else ()
            result._qsa_frontend_owners=tuple(b.mixer for b in blocks if not b.recurrent) if qsa_front_graph else ()
            result._qsa_validation_status = None
            result._qsa_validation_views = {}
            if os.environ.get('MINI_LLM_QSA_DEFER_VALIDATION','0')=='1':
                if not qsa_fast_decode or BACKEND_NAME!='cupy':
                    raise ValueError('Deferred QSA validation requires fast CuPy QSA')
                result._qsa_validation_status = xp.zeros(len(blocks),dtype=xp.uint32)
                result._qsa_validation_views = {i:result._qsa_validation_status[i:i+1].reshape(())
                                                for i,b in enumerate(blocks) if not b.recurrent}
            result._gdn_graph_owners = tuple(b.mixer for b in blocks if b.recurrent) if gdn_graph_decode else ()
            result._hc_graph_owners = (tuple(hc for b in blocks for hc in
                (b.attention_hc, b.ffn_hc)) + (result.final_hc,)) if hc_graph_decode else ()
            m = weights.store.metadata
            result.stop_token_ids = frozenset(integer(m[k],k) for k in (
                'tokenizer.ggml.eos_token_id','tokenizer.ggml.eot_token_id',
                'tokenizer.ggml.eom_token_id') if k in m)
            if any(t >= h.vocab for t in result.stop_token_ids):
                raise ValueError('Generation stop ID exceeds vocabulary')
            return result
        except Exception:
            if expert_executor is None and executor is not None:
                executor.close()
            weights.device.clear()
            raise

    @classmethod
    def from_components(cls, *, width, vocab, blocks, final_hc, embedding, output,
                        max_length, diagnostic_callback=None):
        """Explicit operator injection for small synthetic integration fixtures."""
        result = cls()
        result.config = SimpleNamespace(hidden_size=integer(width,'width',1),
            vocab_size=integer(vocab,'vocab',1),num_hidden_layers=len(blocks),
            max_position_embeddings=integer(max_length,'max_length',1))
        result.blocks, result.final_hc = tuple(blocks), final_hc
        result.embedding, result.output = embedding, output
        result.diagnostic_callback = diagnostic_callback
        result._owner, result._closed = object(), False
        result._states = weakref.WeakSet()
        result._call_lock = threading.Lock()
        result.weights, result.expert_executor, result.plan = None, None, None
        result._owns_executor = False
        result.stop_token_ids = frozenset()
        return result

    def _open(self):
        if self._closed:
            raise RuntimeError('Model is closed')

    def _orphaned_bound_state_bytes(self):
        # A graph can outlive the request object while retaining its GPU arrays.
        # Credit those actual allocations once, not a second time for live state.
        live=set()
        for state in self._states:
            for recurrence in state.recurrent.values():
                if recurrence is not None:
                    for array in (recurrence.conv.history,recurrence.delta.memory):
                        live.add((array.data.ptr,array.nbytes))
            for qsa in state.qsa.values():
                for array in (qsa.keys, qsa.values, qsa.raw_index, qsa.pooled):
                    live.add((array.data.ptr,array.nbytes))
        retained=set()
        segments = tuple(getattr(self,'_pre_moe_segments',{}).values()) + tuple(
            getattr(self,'_qsa_block_segments',{}).values()) + (
                (self._token_graph,) if getattr(self, '_token_graph', None) is not None else ())
        for segment in segments:
            graph=getattr(segment,'graph',None)
            if graph is not None:
                for array in getattr(graph,'bound_arrays',lambda:())():
                    if array.dtype == xp.float32:
                        retained.add((array.data.ptr,array.nbytes))
        return sum(size for pointer,size in retained-live)

    def memory_plan(self, batch_size=1, max_length=None):
        self._open()
        if integer(batch_size,'batch_size',1) != 1:
            raise NotImplementedError('Only text batch_size=1 is supported')
        length = integer(max_length,'max_length',1)
        if length > self.config.max_position_embeddings:
            raise ValueError('Requested max_length exceeds approved capacity')
        if self.weights is None:
            return dict(max_length=length,state_bytes=0)
        p = self.plan
        resident = 0
        if BACKEND_NAME == 'cupy':
            # Static placement is committed after _build; cache capacity is not
            # resident until admitted, and only live sessions own state storage.
            resident = (p['device_static_bytes'] + getattr(self.expert_executor,
                            'resident_device_bytes', self.expert_executor.cache.used_bytes) +
                        sum(s.reserved_bytes for s in self._states) +
                        sum(hc.graph_statistics()['pool_reserved_bytes']
                            for hc in getattr(self, '_hc_graph_owners', ())) +
                        sum(m.graph_statistics()['pool_reserved_bytes']
                            for m in getattr(self, '_gdn_graph_owners', ())) +
                        sum(s.graph_statistics()['pool_reserved_bytes']
                            for s in getattr(self, '_pre_moe_segments', {}).values()) +
                        sum(s.graph_statistics()['pool_reserved_bytes']
                            for s in getattr(self, '_qsa_block_segments', {}).values()) +
                        (self._token_graph.graph_statistics()['pool_reserved_bytes']
                            if getattr(self, '_token_graph', None) is not None else 0) +
                        sum(s.graph_statistics()['pool_reserved_bytes']
                            for s in getattr(self,'_shared_graph_owners',())) +
                        sum(s._frontend_graph.statistics()['pool_reserved_bytes']
                            for s in getattr(self,'_qsa_frontend_owners',()) if s._frontend_graph is not None) +
                        sum(p.native_handoff.graph.statistics()['pool_reserved_bytes']
                            for p in getattr(self.expert_executor,'_device_expert_plans',{}).values()
                            if p.native_handoff is not None and p.native_handoff.graph is not None))
            resident+=self._orphaned_bound_state_bytes()
        return self.weights.memory_plan(length, resident_device_bytes=resident,
                    host_expert_bytes=p.get('host_expert_bytes',0), **{key:p[key] for key in (
            'host_budget_bytes','static_device_budget_bytes','device_budget_bytes',
            'cache_bytes','headroom_bytes','staging_bytes','activation_bytes','cpu_workers',
            'graph_workspace_bytes')})

    def prepare_host_experts(self,*,pin=False,reserve_bytes=4*1024**3,progress=None):
        """Commit the approved packed RAM complement between serialized requests."""
        self._open()
        if self.weights is None or not self.plan.get('host_expert_bytes',0):
            raise ValueError('Host expert source requires an approved host_expert_bytes budget')
        if not self._call_lock.acquire(blocking=False):
            raise RuntimeError('Host source setup cannot race model execution')
        try:
            from mini_llm.ops.expert_residency import ExpertKey
            index=self.weights.index
            e=self.expert_executor
            keys=[ExpertKey(index.model_id,l,x) for l in range(index.n_layers)
                  for x in range(index.n_experts) if ExpertKey(index.model_id,l,x) not in e.cache.resident]
            budget=self.plan['host_expert_bytes']
            page=os.sysconf('SC_PAGE_SIZE')
            sizes={l:sum(index.descriptors[f'blk.{l}.ffn_{p}_exps.weight'].nbytes//index.n_experts
                    for p in ('gate','up','down')) for l in range(index.n_layers)}
            if sum((sizes[k.layer_id]+page-1)//page*page for k in keys)>budget:
                keys.sort(key=lambda k:(-getattr(e.cache._usage.get(k),'hits',0),k.layer_id,k.expert_id))
            return e.prepare_resident_host_source(keys,budget_bytes=budget,pin=pin,
                reserve_bytes=reserve_bytes,progress=progress)
        finally:
            self._call_lock.release()

    def _retire_orphaned_state_graphs(self):
        if BACKEND_NAME != 'cupy' or self._states:
            return
        if not self._call_lock.acquire(blocking=False):
            raise RuntimeError('Cannot retire request graphs during forward')
        try:
            if self._states:
                return
            segments = tuple(getattr(self, '_pre_moe_segments', {}).values()) + tuple(
                getattr(self, '_qsa_block_segments', {}).values()) + (
                    (self._token_graph,) if getattr(self, '_token_graph', None) is not None else ())
            retired = False
            for segment in segments:
                graph = getattr(segment, 'graph', None)
                if graph is not None and graph.bound_arrays():
                    graph.close()
                    segment.graph = None
                    retired = True
            if retired:
                # Only now can the default allocator return the old request's
                # arrays, before the next physical-capacity admission check.
                xp.get_default_memory_pool().free_all_blocks()
        finally:
            self._call_lock.release()

    def create_generation_state(self, batch_size=1, max_length=None):
        self._open()
        if max_length is None:
            raise ValueError('Explicit max_length is required; metadata context is not allocated by default')
        self._retire_orphaned_state_graphs()
        plan = self.memory_plan(batch_size,max_length)
        # One approved aggregate state reservation. Multiple smaller independent
        # sessions can coexist, but cannot silently exceed that reservation.
        live = sum(s.reserved_bytes for s in self._states)
        if self.plan is not None and live+plan['state_bytes'] > self.plan['state_bytes']:
            raise MemoryError('Live generation states exceed approved aggregate state reservation')
        rec, qsa, ple = {}, {}, {}
        for i,b in enumerate(self.blocks):
            if b.recurrent:
                rec[i] = b.mixer.create_state(1)
            else:
                qsa[i] = b.mixer.create_state(max_length)
            if b.ple is not None:
                ple[i] = b.ple.fresh_state()
        state = Qwen4ExpGenerationState(self._owner,int(max_length),rec,qsa,ple,
                                        plan['state_bytes'])
        self._states.add(state)
        return state

    def _validate(self, state):
        self._open()
        if not isinstance(state,Qwen4ExpGenerationState) or state.owner is not self._owner:
            raise ValueError('Generation state owner mismatch')
        if state.failed:
            raise RuntimeError('Partial forward failed; discard state and create a fresh state')
        if state not in self._states:
            raise ValueError('Unregistered generation state')
        integer(state.length,'state.length')
        if state.batch_size != 1 or not 0 <= state.length <= state.max_length <= self.config.max_position_embeddings:
            raise ValueError('Invalid generation state capacity/batch')
        for i,b in enumerate(self.blocks):
            if b.recurrent:
                s = state.recurrent[i]
                if s.conv.length != state.length or s.delta.length != state.length:
                    raise ValueError('Recurrent state positions disagree')
            else:
                s = state.qsa[i]
                if s.length != state.length or s.max_length != state.max_length:
                    raise ValueError('QSA state position/capacity disagrees')
                b.mixer._state(s)

    def _diagnostic(self, layer, stage, value):
        if self.diagnostic_callback is not None:
            self.diagnostic_callback(layer,stage,value.copy())

    def step(self, token, state):
        return self._step(token, state, project_logits=True)

    def _step(self, token, state, *, project_logits):
        self._validate(state)
        token = integer(token,'token')
        if token >= self.config.vocab_size:
            raise ValueError('Token ID exceeds vocabulary')
        if not state.has_capacity():
            raise ValueError('Generation state capacity exceeded')
        if not self._call_lock.acquire(blocking=False):
            raise RuntimeError('Calls sharing this model/executor must execute serially')
        try:
            # Revalidate after acquiring the execution lock.
            self._validate(state)
            try:
                deferred_status = (getattr(self,'_qsa_validation_status',None)
                                   if self.diagnostic_callback is None else None)
                if deferred_status is not None:
                    deferred_status.fill(0)
                if (project_logits and deferred_status is not None and
                        os.environ.get('MINI_LLM_TOKEN_GRAPH','0') == '1'):
                    from .qwen4exp_token_graph import Qwen4ExpTokenGraph
                    token_graph = getattr(self, '_token_graph', None)
                    if Qwen4ExpTokenGraph.is_eligible(self, state):
                        if token_graph is None:
                            token_graph = Qwen4ExpTokenGraph(self)
                            self._token_graph = token_graph
                        logits = token_graph.forward(token, state)
                        invalid = np.flatnonzero(asnumpy(deferred_status))
                        if invalid.size:
                            raise ValueError(f'QSA hidden/projections must be finite (first layer {int(invalid[0])})')
                        self.expert_executor.check_native_service()
                        state.length += 1
                        return logits
                    if token_graph is not None:
                        token_graph.close()
                        self._token_graph = None
                streams = self.final_hc.initialize(self.embedding(token))
                chain_tails = set()
                for i,b in enumerate(self.blocks):
                    if i in chain_tails:
                        continue
                    if b.ple is not None:
                        streams,state.ple[i] = b.ple.step(token,streams,state.ple[i])
                    if (not b.recurrent and deferred_status is not None and
                            os.environ.get('MINI_LLM_QSA_BLOCK_GRAPH','0') == '1' and
                            os.environ.get('MINI_LLM_QSA_BLOCK_REPLAY','1') == '1'):
                        qsa = state.qsa[i]
                        eligible = (state.length + 1 <= b.mixer.config.index_topk + b.mixer.config.block_size - 1 and
                            sum(a.nbytes for a in (qsa.keys, qsa.values, qsa.raw_index, qsa.pooled)) <= 6*1024**2)
                        qsa_segment = self._qsa_block_segments.get(i)
                        if not eligible and qsa_segment is not None:
                            qsa_segment.close()
                            del self._qsa_block_segments[i]
                            qsa_segment = None
                        if eligible and qsa_segment is None:
                            plan = b.moe.experts.expert_executor._device_expert_plans.get(i)
                            handoff = getattr(plan, 'native_handoff', None)
                            if (handoff is not None and handoff.capture_enabled and
                                    handoff.service is not None and handoff.stage is None):
                                from .qwen4exp_qsa_block_graph import Qwen4ExpQSABlockGraph
                                qsa_segment = Qwen4ExpQSABlockGraph(b, plan, self._qsa_validation_views[i])
                                self._qsa_block_segments[i] = qsa_segment
                        if qsa_segment is not None:
                            streams = qsa_segment.forward(streams, qsa, state.length)
                            continue
                    segment = (getattr(self,'_pre_moe_segments',{}).get(i)
                               if self.diagnostic_callback is None else None)
                    if (segment is not None and b.ple is None and
                            os.environ.get('MINI_LLM_RECURRENT_BLOCK_GRAPH','0') == '1' and
                            not getattr(segment, 'complete_block', False)):
                        executor = b.moe.experts.expert_executor
                        plan = getattr(executor, '_device_expert_plans', {}).get(i)
                        handoff = getattr(plan, 'native_handoff', None)
                        if (plan is not None and executor.cache._admission_limit == 0 and
                                handoff is not None and handoff.capture_enabled and
                                handoff.service is not None and handoff.stage is None):
                            from .qwen4exp_recurrent_block_graph import Qwen4ExpRecurrentBlockGraph
                            segment = Qwen4ExpRecurrentBlockGraph(segment, b, plan)
                            self._pre_moe_segments[i] = segment
                    if segment is not None and getattr(segment, 'complete_block', False):
                        if (os.environ.get('MINI_LLM_RECURRENT_CHAIN_GRAPH','0') == '1' and
                                not hasattr(segment, 'members')):
                            members = [segment]
                            for j in range(i+1, min(i+3, len(self.blocks))):
                                following = self._pre_moe_segments.get(j)
                                if (not getattr(following, 'complete_block', False) or
                                        hasattr(following, 'members')):
                                    break
                                members.append(following)
                            if len(members) > 1:
                                from .qwen4exp_recurrent_chain_graph import Qwen4ExpRecurrentChainGraph
                                segment = Qwen4ExpRecurrentChainGraph(members)
                                self._pre_moe_segments[i] = segment
                        if hasattr(segment, 'members'):
                            layers = tuple(m.plan.layer for m in segment.members)
                            streams = segment.forward(streams, tuple(state.recurrent[j] for j in layers))
                            chain_tails.update(layers[1:])
                        else:
                            streams = segment.forward(streams, state.recurrent[i])
                        continue
                    routing = None
                    if segment is not None:
                        streams,hidden,gains,rw,ri = segment.forward(streams,state.recurrent[i])
                        routing = rw,ri
                    else:
                        hidden,gains = b.attention_hc.read(streams)
                        self._diagnostic(i,'hc_attention_read',hidden)
                        if b.recurrent:
                            mixed = b.mixer.forward(hidden[None,None,:],state.recurrent[i])[0,0]
                        else:
                            if deferred_status is None:
                                mixed = b.mixer.step(hidden,state.qsa[i],position=state.length)
                            else:
                                mixed = b.mixer.step(hidden,state.qsa[i],position=state.length,
                                    validation_status=self._qsa_validation_views[i])
                        streams = b.attention_hc.write(streams,mixed,gains)
                        self._diagnostic(i,'post_mixer',streams)
                        hidden,gains = b.ffn_hc.read(streams)
                        self._diagnostic(i,'hc_ffn_read',hidden)
                    moe = b.moe(hidden,routing=routing) if routing is not None else b.moe(hidden)
                    streams = b.ffn_hc.write(streams,moe,gains)
                    self._diagnostic(i,'post_moe',streams)
                logits = None
                if project_logits or self.diagnostic_callback is not None:
                    hidden = self.final_hc.collapse(streams)
                    self._diagnostic(-1,'hc_output_read',hidden)
                    logits = self.output(hidden).reshape(1,-1)
                    if logits.shape != (1,self.config.vocab_size) or logits.dtype != xp.float32:
                        raise ValueError('Output must be FP32 [1,vocab]')
                    self._diagnostic(-1,'logits',logits)
                if deferred_status is not None:
                    invalid = np.flatnonzero(asnumpy(deferred_status))
                    if invalid.size:
                        raise ValueError(f'QSA hidden/projections must be finite (first layer {int(invalid[0])})')
                executor=getattr(self,'expert_executor',None)
                if isinstance(executor,HeterogeneousExpertExecutor):
                    executor.check_native_service()
            except Exception:
                # Operators can already have committed recurrent/PLE history.
                # Never retry or pretend this request remains resumable.
                state.failed = True
                raise
            state.length += 1
            return logits
        finally:
            self._call_lock.release()

    def _tokens(self, sequence):
        tokens = np.asarray(asnumpy(sequence))
        if tokens.ndim == 2 and tokens.shape[0] == 1:
            tokens = tokens[0]
        if tokens.ndim != 1 or not tokens.size or tokens.dtype.kind not in 'iu':
            raise ValueError('Expected a nonempty integer sequence [T] or [1,T]')
        if np.any(tokens < 0) or np.any(tokens >= self.config.vocab_size):
            raise ValueError('Token ID out of range')
        return tokens

    def prefill(self, sequence, state):
        self._validate(state)
        tokens = self._tokens(sequence)
        if not state.has_capacity(tokens.size):
            raise ValueError('Generation state capacity exceeded')
        # The output head does not feed back into any persistent state.
        for token in tokens[:-1]:
            self._step(int(token), state, project_logits=False)
        return self.step(int(tokens[-1]), state)

    def decode_one(self, token_ids, state):
        tokens = self._tokens(token_ids)
        if tokens.size != 1:
            raise ValueError('decode_one expects [1] or [1,1]')
        return self.step(int(tokens[0]),state)

    def forward(self, token_ids):
        tokens = self._tokens(token_ids)
        state = self.create_generation_state(1,int(tokens.size))
        return xp.stack([self.step(int(t),state)[0] for t in tokens],axis=0)[None]

    def greedy(self, prompt, *, max_new_tokens):
        count = integer(max_new_tokens,'max_new_tokens')
        tokens = self._tokens(prompt)
        if not count:
            return []
        state = self.create_generation_state(1,int(tokens.size)+count-1)
        logits = self.prefill(tokens,state)
        generated = []
        for i in range(count):
            token = int(asnumpy(xp.argmax(logits[0])))
            generated.append(token)
            if token in self.stop_token_ids or i+1 == count:
                break
            logits = self.step(token,state)
        return generated

    def close(self):
        if self._closed:
            return
        if not self._call_lock.acquire(blocking=False):
            raise RuntimeError('Cannot close during forward')
        try:
            if getattr(self, '_token_graph', None) is not None:
                self._token_graph.close()
                self._token_graph = None
            segments = tuple(getattr(self, '_pre_moe_segments', {}).values()) + tuple(
                getattr(self, '_qsa_block_segments', {}).values())
            for segment in segments:
                if getattr(segment, 'complete_block', False):
                    segment.close()
            if self._owns_executor:
                self.expert_executor.close()
            if (getattr(self, '_hc_graph_owners', ()) or getattr(self, '_gdn_graph_owners', ()) or
                    getattr(self, '_pre_moe_segments', {}) or getattr(self,'_shared_graph_owners',()) or
                                        getattr(self,'_qsa_frontend_owners',()) or
                                                            getattr(self,'_qsa_validation_status',None) is not None):
                # Graphs can have run on their bound non-default streams.
                xp.cuda.runtime.deviceSynchronize()
            for block in self.blocks:
                if block.ple is not None and hasattr(block.ple, 'close'):
                    block.ple.close()
            self.blocks = ()
            self._hc_graph_owners = ()
            self._gdn_graph_owners = ()
            self._pre_moe_segments = {}
            self._qsa_block_segments = {}
            self._shared_graph_owners = ()
            self._qsa_frontend_owners = ()
            self._qsa_validation_views = {}
            self._qsa_validation_status = None
            self.embedding = self.output = self.final_hc = None
            if self.weights is not None:
                self.weights.close()
            if self._owns_executor:
                self.expert_executor = None
            self._closed = True
        finally:
            self._call_lock.release()

    def __enter__(self):
        self._open()
        return self

    def __exit__(self,*exc):
        self.close()
