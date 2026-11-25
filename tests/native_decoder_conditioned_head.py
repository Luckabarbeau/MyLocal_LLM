"""Bounded final-HC/head diagnostic using the decoder's existing CUDA launch.

The internal C++ symbol is test-only, not a new public binding or new math.
"""
from contextlib import ExitStack
import ctypes as C
import json
import os
from pathlib import Path
import sys

import cupy as cp
import numpy as np

from mini_llm.native import Runtime
from mini_llm.external.qwen4exp_hyperconnection import Qwen4ExpHyperConnection
from mini_llm.ops.quantized_linear import DevicePackedGGUFWeight, quantized_linear
from native_decoder_conditioned import preflight, compare, sha, SOURCE, TARGET, BUDGET
from native_decoder_divergence import actual_path
from native_moe_checkpoint_helpers import KINDS
from test_native_engine_chaining import checkpoint_upload
from test_native_engine_hc import upload, read as hc_read


def main(root):
    admission = preflight()
    target = root/'head'; target.mkdir(exist_ok=False)
    saved, checks = {}, {}
    library = C.CDLL(os.environ['MINI_LLM_NATIVE_LIBRARY'])
    launch = getattr(library, '_Z26ml_vocab_projection_launchP11CUstream_stPKvjmmmPKfPfPi')
    launch.argtypes = [C.c_void_p, C.c_void_p, C.c_uint32, C.c_uint64, C.c_uint64,
                       C.c_uint64, C.c_void_p, C.c_void_p, C.c_void_p]
    launch.restype = C.c_int
    with ExitStack() as stack:
        runtime=Runtime(); engine=stack.enter_context(runtime.engine(budget_bytes=BUDGET))
        store=stack.enter_context(runtime.gguf(actual_path()))
        pairs=[checkpoint_upload(stack,engine,store,'output_hc_'+s) for s in
               ('norm.weight','down.weight','up.weight')]
        eps=store.metadata['qwen4exp.attention.layer_norm_rms_epsilon']
        weights=stack.enter_context(engine.hc_weights(*(p[0] for p in pairs),
            epsilon=eps,arithmetic='reference_fp32'))
        reference=Qwen4ExpHyperConnection(*(p[1] for p in pairs),epsilon=eps)
        hc=stack.enter_context(engine.hc_state(2560,320))
        lease=stack.enter_context(store.lease('output.weight')); d=lease.descriptor
        # One shared immutable packed upload, never a full decoded vocabulary.
        packed=cp.asarray(np.frombuffer(lease.read(0,d.nbytes),np.uint8))
        weight=DevicePackedGGUFWeight(packed,d.shape,KINDS[d.type_id])
        for lineage in ('native','python'):
            for t,folder in enumerate((SOURCE,SOURCE/'token1')):
                with np.load(folder/f'{lineage}.npz',allow_pickle=False) as capture:
                    streams=capture['47_post_moe']
                us,_=upload(stack,target,runtime,engine,[streams])
                hc.initialize(us[0]); hc.collapse(weights)
                collapsed=hc_read(hc,'hidden',2560)
                prefix=f'{lineage}_{t}_'
                compare(saved,checks,prefix+'collapse',collapsed,reference.collapse(cp.asarray(streams)))
                x=cp.asarray(collapsed); out=cp.empty(d.shape[0],cp.float32); status=cp.zeros(1,cp.int32)
                assert launch(cp.cuda.get_current_stream().ptr,packed.data.ptr,d.type_id,
                    d.shape[0],d.shape[1],d.row_bytes,x.data.ptr,out.data.ptr,status.data.ptr)==0
                cp.cuda.get_current_stream().synchronize(); assert int(status.get()[0])==0
                compare(saved,checks,prefix+'vocab',out,quantized_linear(x[None],weight)[0])
                saved[prefix+'collapsed_identical_input']=collapsed
                hc.reset()
        memory=dict(native_peak_bytes=engine.stats.peak_bytes,cupy_reserved_bytes=cp.get_default_memory_pool().total_bytes())
        assert sum(memory.values()) <= TARGET
    np.savez(target/'evidence.npz',**saved)
    report=dict(preflight=admission,memory=memory,checks=checks,
        passed=all(not c['violations'] for c in checks.values()),evidence_sha256=sha(target/'evidence.npz'),
        source_sha256=sha(__file__),native_library_sha256=sha(os.environ['MINI_LLM_NATIVE_LIBRARY']),
        output_weight=dict(shape=d.shape,type_id=d.type_id,nbytes=d.nbytes),
        note='Same collapsed F32 input to both existing GPU projections; no model-quality claim.')
    with (target/'report.json').open('x') as f: json.dump(report,f,indent=2)
    print(json.dumps(report,indent=2),flush=True)
    assert report['passed']


if __name__=='__main__':
    main(Path(sys.argv[1]))
