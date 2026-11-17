"""Synthetic combined graph/lifecycle tests; NOT full-model numerical parity."""
from dataclasses import replace
import gc
import math
import os
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

from mini_llm.backend import BACKEND_NAME, asnumpy, xp
from mini_llm.external.gguf_store import GGUFFormat, TensorDescriptor
from mini_llm.external.qwen4exp_gguf import Qwen4ExpGGUFIndex
from mini_llm.external.qwen4exp_model import Qwen4ExpGGUFModel
from mini_llm.external.qwen4exp_weights import Qwen4ExpWeights, inspect_headers
from mini_llm.ops.quantized_weight import BLOCK_FORMATS, PackedGGUFWeight
from mini_llm.ops.expert_residency import HeterogeneousExpertExecutor
from mini_llm.external import qwen4exp_weights as storage

pytestmark = pytest.mark.skipif(BACKEND_NAME != 'numpy', reason='Synthetic CPU baseline')


class TinyStore:
    """Read-only byte owner with actual packed layouts, no real model payloads."""
    def __init__(self):
        d, ff, e, rank = 256, 64, 4, 3
        self.closed = False
        self.paths = (Path('synthetic.gguf'),)
        self.tensors, self.payload, self.reads = {}, {}, []
        self.metadata = {'general.architecture':'qwen4exp'}
        values = dict(block_count=2,embedding_length=d,context_length=16,expert_count=e,
            expert_used_count=2,expert_feed_forward_length=ff,expert_shared_feed_forward_length=ff,
            full_attention_interval=2)
        values.update({'hyper_connection.count':4,'hyper_connection.low_rank':rank,
            'ssm.group_count':1,'ssm.time_step_rank':2,'ssm.state_size':4,'ssm.conv_kernel':4,
            'ssm.inner_size':8,'attention.head_count':2,'attention.head_count_kv':1,
            'attention.key_length':4,'attention.value_length':4,'rope.dimension_count':4,
            'rope.dimension_sections':[1,1,0,0],'rope.freq_base':10000.,
            'attention.indexer.head_count':2,'attention.indexer.key_length':4,
            'attention.indexer.top_k':5,'attention.compress_ratios':[0,4],
            'attention.layer_norm_rms_epsilon':1e-5,'expert_weights_scale':.75})
        self.metadata.update({'qwen4exp.'+k:v for k,v in values.items()})
        self.metadata['tokenizer.ggml.eos_token_id'] = 7
        self.rng = np.random.default_rng(209)
        def add(name,shape,kind='F32',norm=False):
            if kind in BLOCK_FORMATS:
                qk,qb = BLOCK_FORMATS[kind]
                raw = self.rng.integers(0,256,(math.prod(shape)//qk,qb),np.uint8)
                raw[:,:2] = np.frombuffer(np.float16(1/1024).tobytes(),np.uint8)
                raw = raw.reshape(-1)
            else:
                arr = self.rng.normal(0,.035,shape).astype(np.float32)
                if norm:
                    arr += np.float32(1)
                if name.endswith('ssm_a'):
                    arr = -np.exp(arr)
                if kind == 'BF16':
                    raw = (arr.view(np.uint32)>>16).astype('<u2').view(np.uint8).reshape(-1)
                else:
                    raw = arr.astype('<f4' if kind=='F32' else '<f2').view(np.uint8).reshape(-1)
                qk,qb = 1,4 if kind=='F32' else 2
            raw.flags.writeable = False
            row_bytes = shape[-1]//qk*qb
            bs = shape[:-1]+(row_bytes,)
            self.tensors[name] = TensorDescriptor(name,0,shape,shape[::-1],GGUFFormat(
                0,kind,qk,qb),0,0,raw.size,row_bytes,bs,
                tuple(math.prod(bs[j+1:]) for j in range(len(bs))))
            self.payload[name] = raw
        self.add = add
        add('token_embd.weight',(8,d),'IQ4_XS')
        add('output.weight',(8,d),'IQ4_NL')
        def hc(stem,inject=True):
            add(stem+'norm.weight',(4*d,),norm=True)
            add(stem+'down.weight',(rank,4*d),'BF16')
            add(stem+'up.weight',(4*d,rank),'BF16')
            if inject:
                add(stem+'inject.weight',(4,4*d),'BF16')
        hc('output_hc_',False)
        for i in range(2):
            p = f'blk.{i}.'
            hc(p+'hc_attn_'); hc(p+'hc_ffn_')
            add(p+'ffn_gate_inp.weight',(e,d),'BF16')
            add(p+'ffn_gate_inp_shexp.weight',(d,),'BF16')
            for part in ('gate','up'):
                add(p+f'ffn_{part}_shexp.weight',(ff,d),'IQ4_NL')
                add(p+f'ffn_{part}_exps.weight',(e,ff,d),'IQ3_S')
            add(p+'ffn_down_shexp.weight',(d,ff),'IQ4_NL')
            add(p+'ffn_down_exps.weight',(e,d,ff),'Q2_0')
            if i == 0:
                for name,shape in {'attn_qkv.weight':(16,d),'attn_gate.weight':(8,d),
                    'ssm_alpha.weight':(2,d),'ssm_beta.weight':(2,d),'ssm_out.weight':(d,8),
                    'ssm_conv1d.weight':(16,4),'ssm_norm.weight':(4,),
                    'ssm_a':(2,),'ssm_dt.bias':(2,)}.items():
                    add(p+name,shape,norm=name=='ssm_norm.weight')
            else:
                for name,shape in {'attn_q.weight':(16,d),'attn_k.weight':(4,d),
                    'attn_v.weight':(4,d),'attn_output.weight':(d,8),
                    'indexer.q_proj.weight':(8,d),'indexer.k_proj.weight':(4,d),
                    'attn_q_norm.weight':(4,),'attn_k_norm.weight':(4,),
                    'indexer.q_norm.weight':(4,),'indexer.k_norm.weight':(4,)}.items():
                    add(p+name,shape,norm='norm' in name)

    def tensor_bytes(self,name):
        self.reads.append(('tensor',name))
        return self.payload[name]

    def packed_weight(self,name):
        t = self.tensors[name]
        return PackedGGUFWeight(self.tensor_bytes(name),t.shape,t.quant_type)

    def rows(self,name,first,last,*,outer=()):
        self.reads.append(('rows',name,first,last,outer))
        t = self.tensors[name]
        return self.payload[name].reshape(t.byte_shape)[outer+(slice(first,last),slice(None))]

    def close(self):
        self.closed = True


@pytest.fixture
def source(monkeypatch,tmp_path):
    monkeypatch.setenv('MINI_LLM_NATIVE_BUILD_DIR',str(tmp_path/'native'))
    return TinyStore()


def factory(store,**kwargs):
    return Qwen4ExpGGUFModel.from_index(Qwen4ExpGGUFIndex(store,model_id='tiny'),
        max_length=16,cpu_workers=1,**kwargs)


def decoded(store,name):
    t = store.tensors[name]
    raw = store.payload[name]
    if t.quant_type in BLOCK_FORMATS:
        return PackedGGUFWeight(raw,t.shape,t.quant_type).dequantize_reference().astype(np.float64)
    if t.quant_type == 'BF16':
        return (raw.view('<u2').astype(np.uint32)<<16).view(np.float32).reshape(t.shape).astype(np.float64)
    return raw.view('<f4' if t.quant_type=='F32' else '<f2').reshape(t.shape).astype(np.float64)


def combined_oracle(store,tokens,headers):
    """Layer-major FP64 composition: independent HC/MoE + existing FP64 operator oracles."""
    from test_0207_qwen4exp_mixer import oracle as recurrent_oracle
    from test_0208_qwen4exp_attention import oracle as qsa_oracle
    w = lambda name: decoded(store,name)
    sig = lambda x: np.exp(-np.logaddexp(0.,-x))
    eps = headers.mixer.rms_eps
    def read(x,p,inject=True):
        normal = x/np.sqrt(np.mean(x*x,axis=-1,keepdims=True)+eps)*w(p+'norm.weight').reshape(4,-1)
        flat = normal.reshape(len(tokens),-1)
        lo = (flat@w(p+'down.weight').T)/4
        gate = sig((lo*sig(lo))@w(p+'up.weight').T).reshape(x.shape)
        mixed = np.mean(normal*gate,axis=1)
        gains = 2*sig((flat@w(p+'inject.weight').T)/4) if inject else None
        return mixed,gains
    x = np.repeat(w('token_embd.weight')[tokens,None,:],4,axis=1)
    for i in range(2):
        p = f'blk.{i}.'
        hidden,gains = read(x,p+'hc_attn_')
        if i == 0:
            names = dict(qkv='attn_qkv.weight',z='attn_gate.weight',beta='ssm_beta.weight',
                alpha='ssm_alpha.weight',out='ssm_out.weight',conv_weight='ssm_conv1d.weight',
                norm_weight='ssm_norm.weight',ssm_a='ssm_a',dt_bias='ssm_dt.bias')
            ws = {k:w(p+v) for k,v in names.items()}
            out = recurrent_oracle(headers.mixer,ws,hidden[None],np.zeros((1,3,16)),
                                   np.zeros((1,2,4,4)))[0][0]
        else:
            names = dict(query_gate='attn_q.weight',key='attn_k.weight',value='attn_v.weight',
                output='attn_output.weight',index_query='indexer.q_proj.weight',index_key='indexer.k_proj.weight',
                gamma_q='attn_q_norm.weight',gamma_k='attn_k_norm.weight',
                gamma_index_q='indexer.q_norm.weight',gamma_index_k='indexer.k_norm.weight')
            # All four cells retained: independent oracle's tie policy cannot change membership.
            out = np.stack([row[0] for row in qsa_oracle(
                hidden,headers.attention,{k:w(p+v) for k,v in names.items()})])
        x = x+gains[:,:,None]*out[:,None,:]
        hidden,gains = read(x,p+'hc_ffn_')
        rows = []
        for row in hidden:
            logits = w(p+'ffn_gate_inp.weight')@row
            prob = np.exp(logits-logits.max()); prob /= prob.sum()
            ids = np.argsort(-prob)[:2]
            route = prob[ids]/max(prob[ids].sum(),2**-14)*headers.expert_scale
            routed = np.zeros(256)
            for e,rw in sorted(zip(ids,route)):
                g = w(p+'ffn_gate_exps.weight')[e]@row
                u = w(p+'ffn_up_exps.weight')[e]@row
                routed += rw*(w(p+'ffn_down_exps.weight')[e]@((g*sig(g))*u))
            g = w(p+'ffn_gate_shexp.weight')@row
            shared = w(p+'ffn_down_shexp.weight')@((g*sig(g))*(w(p+'ffn_up_shexp.weight')@row))
            rows.append(routed+sig(w(p+'ffn_gate_inp_shexp.weight')@row)*shared)
        x += gains[:,:,None]*np.array(rows)[:,None,:]
    return read(x,'output_hc_',False)[0]@w('output.weight').T


def test_combined_factory_graph_against_synthetic_fp64(source):
    index = Qwen4ExpGGUFIndex(source,model_id='tiny')
    h = inspect_headers(index)
    tokens = np.array([1,3,2,5])
    expected = combined_oracle(source,tokens,h)
    traces = []
    with factory(source,diagnostic_callback=lambda *args:traces.append(args)) as model:
        got = model.forward(tokens)[0]
        error = np.max(np.abs(got.astype(np.float64)-expected))
        print(f'synthetic combined FP32 vs FP64 max_abs={error:.9g}; atol=2e-5 rtol=2e-4')
        np.testing.assert_allclose(got,expected,atol=2e-5,rtol=2e-4)
        assert model.blocks[1].mixer.config.tie_policy == 'ggml_cpu'
        assert all(not b.moe.experts.experts for b in model.blocks)
        assert model.expert_executor.host_cache.used_bytes == 0
        assert model.expert_executor.counters['native_cpu_experts'] == 16
        assert len(traces) == 4*(2*4+2)
        assert all(a.dtype==np.float32 for _,_,a in traces)
    assert not source.closed


def test_sequence_incremental_reset_isolation_capacity(source):
    with factory(source) as model:
        seq = [1,2,3,4]
        a = model.create_generation_state(1,4)
        b = model.create_generation_state(1,4)
        first = model.step(1,a).copy()
        model.step(6,b)
        incremental = model.prefill(seq[1:],a)
        del a,b
        gc.collect()
        fresh = model.create_generation_state(1,4)
        np.testing.assert_array_equal(model.step(1,fresh),first)
        model.prefill(seq[1:],fresh)
        assert fresh.length == 4 and fresh.remaining_capacity()==0
        with pytest.raises(ValueError,match='capacity'):
            model.step(1,fresh)
        assert not fresh.failed
        del fresh
        gc.collect()
        s = model.create_generation_state(1,4)
        np.testing.assert_array_equal(model.prefill(np.array([seq]),s),incremental)
        assert s.recurrent[0].conv.length==s.qsa[1].length==s.length==4
        with pytest.raises(ValueError,match='Explicit'):
            model.create_generation_state()
        with pytest.raises(NotImplementedError,match='batch'):
            model.create_generation_state(2,1)
        with pytest.raises(MemoryError,match='aggregate'):
            model.create_generation_state(1,16)
        with factory(source) as other:
            with pytest.raises(ValueError,match='owner'):
                other.step(1,s)


def test_invalid_prefill_no_mutation_decode_and_poison(source):
    with factory(source) as model:
        s = model.create_generation_state(1,4)
        for bad in ([1,99],[1,-1],[1.,2.]):
            with pytest.raises(ValueError):
                model.prefill(bad,s)
            assert s.length==0 and not s.failed
        assert model.decode_one(np.array([[1]]),s).shape==(1,8)
        def fail(hidden):
            raise RuntimeError('injected post-recurrent failure')
        model.blocks[0].moe = fail
        with pytest.raises(RuntimeError,match='injected'):
            model.step(2,s)
        assert s.failed and s.length==1 and s.recurrent[0].conv.length==2
        with pytest.raises(RuntimeError,match='Partial'):
            model.step(2,s)


@pytest.mark.parametrize('defect',['shape','missing','ratio','hc','extra','norm_format'])
def test_bad_headers_reject_before_payload(source,defect):
    name = 'blk.1.attn_q.weight'
    if defect=='shape':
        source.tensors[name] = replace(source.tensors[name],shape=(2,3))
    elif defect=='missing':
        del source.tensors[name]
    elif defect=='ratio':
        source.metadata['qwen4exp.attention.compress_ratios'] = [0,8]
    elif defect=='hc':
        source.metadata['qwen4exp.hyper_connection.count'] = 3
    elif defect=='extra':
        source.tensors['output_norm.weight'] = source.tensors[name]
    else:
        name = 'blk.0.hc_attn_norm.weight'
        source.tensors[name] = replace(source.tensors[name],format=GGUFFormat(0,'IQ4_NL',32,18))
    with pytest.raises((ValueError,NotImplementedError)):
        factory(source)
    assert not source.reads


def test_plan_before_payload_accounts_expansion_and_temporary(source):
    weights = Qwen4ExpWeights(Qwen4ExpGGUFIndex(source))
    with pytest.raises(RuntimeError,match='approved'):
        weights.dense('output_hc_down.weight')
    with pytest.raises(MemoryError,match='host budget'):
        factory(source,host_budget_bytes=1)
    assert not source.reads
    p = weights.memory_plan(8,cpu_workers=1)
    assert p['dense_fp32_bytes'] > sum(t.nbytes for t in source.tensors.values() if t.quant_type in ('F32','BF16','F16'))
    assert p['constructor_temporary_bytes']>0 and p['state_bytes']>0
    assert p['host_reserved_bytes']<=8*1024**3
    assert p['device_static_bytes']==0 and p['canonical_mmap_bytes']>0
    with pytest.raises(ValueError,match='cpu_workers'):
        weights.memory_plan(8,cpu_workers=0)
    with pytest.raises(ValueError,match='cap'):
        weights.memory_plan(8,cpu_workers=10000)


def test_native_projections_never_full_decode_and_selected_rows_only(source,monkeypatch):
    original = PackedGGUFWeight.dequantize_reference
    def bounded(self):
        assert self.shape==(1,256), 'Only one token embedding row may reference-decode'
        return original(self)
    monkeypatch.setattr(PackedGGUFWeight,'dequantize_reference',bounded)
    native = storage.packed_cpu_linear
    calls = []
    def track(x,w,**kwargs):
        calls.append((w.quant_type,w.shape,kwargs))
        assert np.shares_memory(w.data,source.payload['output.weight']) or not w.data.flags.writeable
        return native(x,w,**kwargs)
    monkeypatch.setattr(storage,'packed_cpu_linear',track)
    with factory(source) as model:
        state = model.create_generation_state(1,2)
        got = model.prefill([1,2],state)
        assert np.isfinite(got).all()
        assert len(calls)==14 and all(c[2]=={'backend':'native'} for c in calls)
        assert not model.expert_executor.cache.resident
    rows = [r for r in source.reads if r[0]=='rows']
    assert [r[2:4] for r in rows if r[1]=='token_embd.weight']==[(1,2),(2,3)]
    assert all(r[4] for r in rows if '_exps' in r[1])
    assert all(not p.flags.writeable for p in source.payload.values())


def test_borrowed_executor_close_ownership(source):
    index = Qwen4ExpGGUFIndex(source,model_id='tiny')
    executor = HeterogeneousExpertExecutor(index.provider,model_id='tiny',d_model=256,d_ff=64,
        compute_dtype='float32',cpu_workers=1,cpu_packed_backend='native')
    model = Qwen4ExpGGUFModel.from_index(index,max_length=4,expert_executor=executor)
    model.close(); model.close()
    assert executor._pool is not None and not source.closed
    with pytest.raises(RuntimeError,match='closed'):
        model.create_generation_state(1,1)
    executor.close()


def test_ple_new_state_before_attention_and_diagnostic_copies(source):
    events = []
    class PLE:
        def fresh_state(self):
            return 0
        def step(self,token,x,state):
            events.append(('ple',state))
            return x+np.float32(.1),state+1
    with factory(source,diagnostic_callback=lambda i,s,x:events.append((s,x.copy()))) as model:
        model.blocks[0].ple = PLE()
        state = model.create_generation_state(1,2)
        model.prefill([1,2],state)
        assert state.ple[0]==2
        assert events[0]==('ple',0) and events[1][0]=='hc_attention_read'
        assert [event[1] for event in events if event[0]=='ple']==[0,1]


def test_greedy_generation_uses_tokenizer_eos_not_ple_eos(source):
    source.metadata['qwen4exp.ple.eos_token_id'] = 0
    with factory(source) as model:
        assert model.stop_token_ids==frozenset({7})
        def output(hidden):
            logits = np.zeros(8,np.float32); logits[7]=10
            return logits
        model.output = output
        assert model.greedy([1,2],max_new_tokens=4)==[7]
        assert model.greedy([1],max_new_tokens=0)==[]


def test_actual_original_factory_opt_in():
    path = os.environ.get('QWEN4EXP_MODEL_PATH')
    if not path:
        pytest.skip('Set QWEN4EXP_MODEL_PATH explicitly to run local original-shard probe')
    token = int(os.environ.get('QWEN4EXP_PROBE_TOKEN','1'))
    with Qwen4ExpGGUFModel.open(path,max_length=2,cpu_workers=1) as model:
        state = model.create_generation_state(1,2)
        logits = model.prefill([token],state)
        assert logits.shape==(1,model.config.vocab_size) and np.isfinite(asnumpy(logits)).all()
        model.decode_one(np.array([int(asnumpy(xp.argmax(logits[0])))]),state)
        assert state.length==2
        print('Actual checkpoint smoke only; independent llama.cpp logits parity NOT tested')
