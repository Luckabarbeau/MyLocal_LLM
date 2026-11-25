"""WP7 native complete loop; checkpoint gate is bounded by caller to 180 seconds."""
from contextlib import ExitStack
import ctypes as C
import gc
import json
import os
from pathlib import Path

import numpy as np
import pytest

from mini_llm.native import Runtime, NativeError, HCState, RecurrentLayer, MoE
from mini_llm.native.decoder import NativeDecoder
from test_native_engine_moe import libraries, selected_library, fixture as moe_fixture, native_setup
from test_native_engine_hc import upload, make_weights
from test_native_engine_recurrent import fixture as rec_fixture, reference as rec_reference
from test_0228_iq_gpu_integration import packed
from test_0201_gguf_store import write_gguf


def forbidden(*args, **kwargs):
    raise AssertionError('per-layer Python execution/upload inside native decoder step')


@pytest.fixture
def cp():
    from mini_llm.backend import BACKEND_NAME
    if BACKEND_NAME != 'cupy':
        pytest.skip('explicit CuPy reference required')
    import cupy
    return cupy


def small_setup(stack, tmp_path, libraries, *, bad_last=False):
    from mini_llm.external.qwen4exp_mixer import Qwen4ExpMixerSpec
    path = tmp_path/'moe.gguf'; data = moe_fixture(path)
    r, e, store, leases, weights, first, _ = native_setup(stack, path, data, libraries, (), budget=512*2**20)
    a, ar, _ = make_weights(stack, tmp_path, r, e, 256, 5)
    f, fr, _ = make_weights(stack, tmp_path, r, e, 256, 5, seed=5)
    final, final_ref, _ = make_weights(stack, tmp_path, r, e, 256, 5, inject=False, seed=6)
    spec = Qwen4ExpMixerSpec(256, 2, 2, 8, 4, 1e-6)
    tensors, values = upload(stack, tmp_path, r, e, rec_fixture(spec))
    embedding = packed('IQ4_XS', 7, 256, 41)
    output = packed('IQ4_NL', 7, 256, 42)
    path = tmp_path/'ends.gguf'
    write_gguf(path, [('embedding', (256,7),23,0,embedding.data.tobytes()),
        ('output',(256,7),20,1024,output.data.tobytes())])
    ends = stack.enter_context(r.gguf(path))
    emb = stack.enter_context(ends.lease('embedding'))
    out = stack.enter_context(e.upload(stack.enter_context(ends.lease('output'))))
    layers = []
    for i in range(48):
        branch = stack.enter_context(e.recurrent_layer(spec,4,*tensors,arithmetic='reference_fp32'))
        moe = first if i == 0 else stack.enter_context(e.moe(*weights,*leases[5:],
            d_model=256,d_ff=320,gpu_experts=(),threads=1,**libraries))
        layers.append((a,branch,f,moe,None))
    if bad_last:
        # Finite metadata, nonfinite shared scalar caught only at layer 47 execution.
        bad = np.zeros(256,np.float32); bad[0] = np.nan
        us,_ = upload(stack,tmp_path,r,e,[bad])
        layers[-1] = (a,layers[-1][1],f,stack.enter_context(e.moe(
            *weights[:4],us[0],*leases[5:],d_model=256,d_ff=320,
            gpu_experts=(),threads=1,**libraries)),None)
    hc = stack.enter_context(e.hc_state(256,5))
    decoder = stack.enter_context(e.decoder(emb,out,final,hc,layers,capacity=4,trace=True))
    return decoder, (r,e,store), (ar,fr,final_ref,spec,values,data,embedding,output)


def small_reference(cp, refs, stack):
    from mini_llm.external.qwen4exp_model import Qwen4ExpGGUFModel, Qwen4ExpLayer, Qwen4ExpMoE
    from mini_llm.ops.router_inference import RouterInference
    from mini_llm.ops.experts_inference import ExpertsInference
    from mini_llm.ops.expert_residency import HeterogeneousExpertExecutor, ExpertKey
    from mini_llm.ops.quantized_linear import packed_to_device, quantized_linear
    ar,fr,final,spec,values,data,embedding,output = refs
    router = RouterInference(256,512,10,renormalize_selected=True,normalization_floor=2**-14)
    router._decode_fastpath_enabled = False
    router.W_router = cp.asarray(data['router'].T.copy()); router.b_router = cp.zeros(512,cp.float32)
    executor = HeterogeneousExpertExecutor(lambda key:data['expert'](key.expert_id),model_id='wp7',
        d_model=256,d_ff=320,compute_dtype='float32',cpu_workers=1,cpu_packed_backend='ggml_pool',
        group_after_admission=True,group_mixed_decode=True)
    stack.callback(executor.close)
    executor._prepare = lambda layer,i,x:(ExpertKey('wp7',layer,i),None,data['expert'](i),None)
    callbacks = []
    for w in data['shared']:
        dev = packed_to_device(w)
        callbacks.append(lambda x,w=dev:quantized_linear(x.reshape(1,-1),w).reshape(-1))
    moe = Qwen4ExpMoE(router,ExpertsInference(256,320,512,10,expert_executor=executor,layer_id=0),
        *callbacks,cp.asarray(data['scalar']))
    mixer = rec_reference(spec,[cp.asarray(a) for a in values])
    output_device = packed_to_device(output)
    traces = {}
    model = Qwen4ExpGGUFModel.from_components(width=256,vocab=7,
        blocks=[Qwen4ExpLayer(ar,mixer,fr,moe,True)]*48,final_hc=final,
        embedding=lambda t:cp.asarray(embedding.dequantize_rows_reference([t])[0]),
        output=lambda x:quantized_linear(x[None],output_device),max_length=4,
        diagnostic_callback=lambda i,s,x:traces.__setitem__((i,s),cp.asnumpy(x).copy()))
    return model,traces


def test_synthetic_all48_reference_replay_lifetime(cp,libraries,tmp_path,monkeypatch):
    with ExitStack() as stack:
        decoder, (r,e,store), refs = small_setup(stack,tmp_path,libraries)
        model,traces = small_reference(cp,refs,stack)
        baseline = e.stats.device_bytes
        for replay in range(2):
            state = model.create_generation_state(max_length=4)
            for t, token in enumerate((1,2)):
                expected = cp.asnumpy(model.step(token,state))
                with monkeypatch.context() as patch:
                    for obj,name in ((e,'upload'),(HCState,'mix'),(RecurrentLayer,'step'),(MoE,'submit')):
                        patch.setattr(obj,name,forbidden)
                    got = decoder.step(token)
                assert decoder.state == (t+1,False)
                for i in range(48):
                    for stage in ('hc_attention_read','post_mixer','hc_ffn_read','post_moe'):
                        np.testing.assert_allclose(decoder.trace_for_test(i,stage),traces[i,stage],rtol=2e-4,atol=2e-4,
                            err_msg=f'first boundary layer={i} stage={stage}')
                np.testing.assert_allclose(got,expected,rtol=2e-4,atol=2e-4)
            decoder.reset()
            assert decoder.state == (0,False)
            assert e.stats.device_bytes == baseline
        store.close(); e.close()
        assert np.isfinite(decoder.step(1)).all()  # Retained native engine/maps/weights.


def test_last_layer_failure_global_position(libraries,tmp_path):
    with ExitStack() as stack:
        decoder,_,_ = small_setup(stack,tmp_path,libraries,bad_last=True)
        for replay in range(2):
            with pytest.raises(NativeError,match='MoE'):
                decoder.step(1)
            assert decoder.state == (0,True)
            with pytest.raises(NativeError,match='unresumable'):
                decoder.step(1)
            with pytest.raises(NativeError,match='unavailable'):
                decoder.trace_for_test(0,'post_moe')
            decoder.reset()
            assert decoder.state == (0,False)


@pytest.mark.skipif(os.environ.get('MINI_LLM_DECODER_CHECKPOINT')!='1',reason='explicit bounded full checkpoint gate')
def test_actual_all48_two_steps_reset(cp,libraries,monkeypatch,tmp_path):
    """Sequential owners keep total GPU use bounded; no forced native intermediates."""
    from mini_llm.external.qwen4exp_model import Qwen4ExpGGUFModel
    inventory = json.loads(Path('docs/native_engine/WP2_GGUF_INVENTORY.json').read_text())
    path = inventory['shards'][0]['path']
    tokens = (1,7)

    native, snapshots = [], []
    with NativeDecoder.open(Runtime(),path,**libraries,capacity=4,physical_pages=1,threads=2,trace=True) as decoder:
        print('WP7 plan',decoder.memory_plan,flush=True)
        for replay in range(2):
            for t,token in enumerate(tokens):
                got = decoder.step(token)
                assert decoder.state == (t+1,False)
                if replay == 0:
                    native.append(got)
                    snapshots.append({(i,s):decoder.trace_for_test(i,s) for i in range(48) for s in
                        ('hc_attention_read','post_mixer','hc_ffn_read','post_moe')})
                else:
                    np.testing.assert_array_equal(got,native[t])
            decoder.reset()
    gc.collect(); cp.get_default_memory_pool().free_all_blocks()
    import subprocess
    import sys
    destination = tmp_path/'reference.npz'
    subprocess.run([sys.executable,'-m','tests.native_decoder_reference',path,str(destination)],
                   check=True,timeout=180,env=dict(os.environ))
    with np.load(destination,allow_pickle=False) as reference:
        for t,token in enumerate(tokens):
            expected = reference[f'{t}_logits']
            for i in range(48):
                for s in ('hc_attention_read','post_mixer','hc_ffn_read','post_moe'):
                    got,ref = snapshots[t][i,s],reference[f'{t}_{i}_{s}']
                    absolute = float(np.max(np.abs(got-ref)))
                    relative = float(np.linalg.norm((got-ref).astype(np.float64))/max(np.linalg.norm(ref.astype(np.float64)),1e-30))
                    print(f'WP7 token={t} layer={i} stage={s} maxabs={absolute:.9g} relL2={relative:.9g}',flush=True)
                    np.testing.assert_allclose(got,ref,rtol=2e-4,atol=2e-4,err_msg=f'first boundary token={t} layer={i} stage={s}')
            np.testing.assert_allclose(native[t],expected,rtol=2e-4,atol=2e-4)


def test_factory_budget_preflight(libraries,monkeypatch):
    inventory = json.loads(Path('docs/native_engine/WP2_GGUF_INVENTORY.json').read_text())
    path = inventory['shards'][0]['path']
    r = Runtime()
    monkeypatch.setattr('mini_llm.native.Engine.upload',forbidden)
    with pytest.raises(ValueError,match='dense GPU budget'):
        NativeDecoder.open(r,path,**libraries,dense_budget_bytes=1)
    with pytest.raises(ValueError,match='GPU allocation budget'):
        NativeDecoder.open(r,path,**libraries,budget_bytes=1)
    with pytest.raises(ValueError,match='host page budget'):
        NativeDecoder.open(r,path,**libraries,host_budget_bytes=12*70000)
    with pytest.raises(ValueError,match='mapped shard budget'):
        NativeDecoder.open(r,path,**libraries,mapped_budget_bytes=1)
