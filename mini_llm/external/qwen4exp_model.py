"""First Qwen4Exp text runtime: FP32 operators, sparse experts, explicit sessions.

Synthetic integration validation is not independent full-checkpoint parity.
"""
from dataclasses import dataclass, field
from types import SimpleNamespace
import threading
import weakref

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
    def __init__(self, router, experts, gate, up, down, shared_gate, *, scale=1.0):
        self.router, self.experts = router, experts
        self.gate, self.up, self.down = gate, up, down
        self.shared_gate, self.scale = shared_gate, np.float32(scale)

    def __call__(self, hidden):
        x = hidden.reshape(1,1,-1)
        weights, ids = self.router.forward(x)
        routed = self.experts.forward(x, weights*self.scale, ids).reshape(-1)
        gate = self.gate(hidden)
        shared = self.down((gate*_sigmoid(gate))*self.up(hidden))
        return routed + _sigmoid(hidden @ self.shared_gate)*shared


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
               expert_cache_slots=False, **budgets):
        expert_promote_after = integer(expert_promote_after, 'expert_promote_after', 1)
        for name, value in (('qsa_fast_decode', qsa_fast_decode),
                            ('hc_fast_decode', hc_fast_decode), ('router_fast_decode', router_fast_decode),
                            ('gdn_fast_decode', gdn_fast_decode), ('hc_graph_decode', hc_graph_decode),
                            ('hc_native_read', hc_native_read),
                            ('group_after_admission', group_after_admission),
                            ('expert_cache_slots', expert_cache_slots)):
            if not isinstance(value, bool):
                raise TypeError(f'{name} must be bool')
        if diagnostic_callback is not None and not callable(diagnostic_callback):
            raise TypeError('diagnostic_callback must be callable')
        index, h = weights.index, weights.headers
        if expert_executor is not None:
            if (expert_executor.d_model,expert_executor.d_ff,expert_executor.compute_dtype,
                    expert_executor.cpu_packed_backend,expert_executor.model_id) != (
                    index.d_model,index.d_ff,'float32','native',index.model_id):
                raise ValueError('Borrowed executor geometry/precision/backend/model identity mismatch')
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
            largest = max(sum(index.descriptors[f'blk.{layer}.ffn_{p}_exps.weight'].nbytes
                              // index.n_experts for p in ('gate', 'up', 'down'))
                          for layer in range(len(h.recurrent)))
            slot_stride = (largest + 255) // 256 * 256
            # Account for pinned allocator rounding plus other transfer scratch.
            budgets['staging_bytes'] = max(integer(budgets.get('staging_bytes', 64*1024**2),
                'staging_bytes'), 2 * (1 << (slot_stride-1).bit_length()))
        if hc_graph_decode:
            if BACKEND_NAME != 'cupy':
                raise RuntimeError('HC graph decode requires CuPy')
            graph_reservation = (2 * len(h.recurrent) + 1) * 2 * 1024**2
            budgets['graph_workspace_bytes'] = max(
                integer(budgets.get('graph_workspace_bytes', 0), 'graph_workspace_bytes'),
                graph_reservation)
        plan = weights.memory_plan(max_length, **budgets)
        weights.approve(plan)
        executor = expert_executor
        try:
            if executor is None:
                executor = HeterogeneousExpertExecutor(index.provider, model_id=index.model_id,
                    d_model=index.d_model, d_ff=index.d_ff, compute_dtype='float32',
                    cpu_packed_backend='native', cpu_workers=plan['cpu_workers'],
                    device_budget_bytes=plan['cache_bytes'], host_dequant_budget_bytes=0,
                    promote_after=expert_promote_after, group_after_admission=group_after_admission,
                    expert_slot_stride_bytes=slot_stride, slot_staging_budget_bytes=plan['staging_bytes'])
            dense, project = weights.dense, weights.projection
            def hc(stem, inject=True):
                # HC copies these small matrices. No host dictionary retains
                # the constructor inputs after this call returns.
                return Qwen4ExpHyperConnection(dense(stem+'norm.weight'),
                    dense(stem+'down.weight'),dense(stem+'up.weight'),
                    dense(stem+'inject.weight') if inject else None, epsilon=h.mixer.rms_eps,
                    fast_decode=hc_fast_decode, graph_decode=hc_graph_decode,
                    native_read=hc_native_read)
            blocks = []
            for i, rec in enumerate(h.recurrent):
                p = f'blk.{i}.'
                attn_hc, ffn_hc = hc(p+'hc_attn_'), hc(p+'hc_ffn_')
                ple = None
                if i in h.ple_layers:
                    ple = Qwen4ExpPLE.from_store(weights.store,
                        key=project(p+'ple_key.weight'), value=project(p+'ple_value.weight'),
                        gamma_key=dense(p+'ple_norm_key.weight').reshape(4,h.width),
                        gamma_query=dense(p+'ple_norm_query.weight').reshape(4,h.width),
                        gamma_conv=dense(p+'ple_norm_conv.weight').reshape(4,h.width),
                        conv=dense(p+'ple_conv1d.weight'))
                if rec:
                    mixer = Qwen4ExpRecurrentMixerInference(h.mixer,
                        qkv=project(p+'attn_qkv.weight'), z=project(p+'attn_gate.weight'),
                        beta=project(p+'ssm_beta.weight'),alpha=project(p+'ssm_alpha.weight'),
                        out=project(p+'ssm_out.weight'),conv_weight=dense(p+'ssm_conv1d.weight'),
                        norm_weight=dense(p+'ssm_norm.weight'),ssm_a=dense(p+'ssm_a'),
                        dt_bias=dense(p+'ssm_dt.bias'))
                    mixer.mixer._decode_fastpath_enabled = gdn_fast_decode
                else:
                    mixer = Qwen4ExpAttention(h.attention,
                        query_gate=project(p+'attn_q.weight'),key=project(p+'attn_k.weight'),
                        value=project(p+'attn_v.weight'),output=project(p+'attn_output.weight'),
                        index_query=project(p+'indexer.q_proj.weight'),index_key=project(p+'indexer.k_proj.weight'),
                        gamma_q=dense(p+'attn_q_norm.weight'),gamma_k=dense(p+'attn_k_norm.weight'),
                        gamma_index_q=dense(p+'indexer.q_norm.weight'),gamma_index_k=dense(p+'indexer.k_norm.weight'),
                        fast_decode=qsa_fast_decode)
                router = RouterInference(h.width,index.n_experts,index.top_k,
                    renormalize_selected=True,normalization_floor=2.0**-14)
                router.W_router = dense(p+'ffn_gate_inp.weight').T
                router.b_router = xp.zeros(index.n_experts,dtype=xp.float32)
                # Both paths retain full softmax; fusion is explicitly switchable.
                router._decode_fastpath_enabled = router_fast_decode
                experts = ExpertsInference(h.width,index.d_ff,index.n_experts,
                    expert_executor=executor,layer_id=i)
                moe = Qwen4ExpMoE(router,experts,project(p+'ffn_gate_shexp.weight'),
                    project(p+'ffn_up_shexp.weight'),project(p+'ffn_down_shexp.weight'),
                    dense(p+'ffn_gate_inp_shexp.weight'),scale=h.expert_scale)
                blocks.append(Qwen4ExpLayer(attn_hc,mixer,ffn_hc,moe,rec,ple))
            result = cls.from_components(width=h.width,vocab=h.vocab,blocks=blocks,
                final_hc=hc('output_hc_',False),embedding=weights.embedding,
                output=project('output.weight'),max_length=plan['max_length'],
                diagnostic_callback=diagnostic_callback)
            result.weights, result.index = weights, index
            result.expert_executor, result._owns_executor = executor, expert_executor is None
            result.plan = plan
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
                            for hc in getattr(self, '_hc_graph_owners', ())))
        return self.weights.memory_plan(length, resident_device_bytes=resident, **{key:p[key] for key in (
            'host_budget_bytes','static_device_budget_bytes','device_budget_bytes',
            'cache_bytes','headroom_bytes','staging_bytes','activation_bytes','cpu_workers',
            'graph_workspace_bytes')})

    def create_generation_state(self, batch_size=1, max_length=None):
        self._open()
        if max_length is None:
            raise ValueError('Explicit max_length is required; metadata context is not allocated by default')
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
                streams = self.final_hc.initialize(self.embedding(token))
                for i,b in enumerate(self.blocks):
                    if b.ple is not None:
                        streams,state.ple[i] = b.ple.step(token,streams,state.ple[i])
                    hidden,gains = b.attention_hc.read(streams)
                    self._diagnostic(i,'hc_attention_read',hidden)
                    if b.recurrent:
                        mixed = b.mixer.forward(hidden[None,None,:],state.recurrent[i])[0,0]
                    else:
                        mixed = b.mixer.step(hidden,state.qsa[i],position=state.length)
                    streams = b.attention_hc.write(streams,mixed,gains)
                    self._diagnostic(i,'post_mixer',streams)
                    hidden,gains = b.ffn_hc.read(streams)
                    self._diagnostic(i,'hc_ffn_read',hidden)
                    streams = b.ffn_hc.write(streams,b.moe(hidden),gains)
                    self._diagnostic(i,'post_moe',streams)
                logits = None
                if project_logits or self.diagnostic_callback is not None:
                    hidden = self.final_hc.collapse(streams)
                    self._diagnostic(-1,'hc_output_read',hidden)
                    logits = self.output(hidden).reshape(1,-1)
                    if logits.shape != (1,self.config.vocab_size) or logits.dtype != xp.float32:
                        raise ValueError('Output must be FP32 [1,vocab]')
                    self._diagnostic(-1,'logits',logits)
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
            if self._owns_executor:
                self.expert_executor.close()
            if getattr(self, '_hc_graph_owners', ()):
                # Graphs can have run on their bound non-default streams.
                xp.cuda.runtime.deviceSynchronize()
            self.blocks = ()
            self._hc_graph_owners = ()
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
