"""Small linked-GGML routing/expert graph and deterministic combination tests."""
from pathlib import Path
import subprocess
import sys
import numpy as np
import pytest

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'scripts'))
from probe_0224_isolated_moe import compile_driver, normalized_route, sum_routes
from probe_0218_hc_identity import COPY
from mini_llm.ops.quantized_weight import BLOCK_FORMATS


def test_router_normalization_and_expert_id_combination():
    p,ids,w=normalized_route(np.array([-1.,3.,2.,0.],np.float32),2)
    assert set(ids)=={1,2} and float(w.sum())==pytest.approx(1.)
    y=np.array([[1.,2.],[3.,4.]],np.float32)
    np.testing.assert_allclose(sum_routes(y,np.array([.25,.75],np.float32),np.array([8,2])),[2.5,3.5])


def test_linked_selected_expert_graph(tmp_path):
    if not (COPY/'build-cpu/bin/libggml-cpu.so').exists():pytest.skip('pinned CPU reference required')
    binary=compile_driver(tmp_path)
    rng=np.random.default_rng(224);d=f=256;e=16;k=4
    (tmp_path/'spec.txt').write_text('256 256 16 4 21 21 20 20 20 20')
    x=rng.normal(0,.1,d).astype(np.float32);x.tofile(tmp_path/'input.bin')
    def dense(name,shape):
        a=rng.normal(0,.05,shape).astype(np.float32)
        (a.view(np.uint32)>>16).astype('<u2').tofile(tmp_path/(name+'.bin'))
    def packed(name,fmt,shape):
        qk,qb=BLOCK_FORMATS[fmt];raw=rng.integers(0,256,(int(np.prod(shape))//qk,qb),np.uint8)
        raw[:,:2]=np.frombuffer(np.float16(.001).tobytes(),np.uint8)
        raw.tofile(tmp_path/(name+'.bin'))
    dense('router',(e,d))
    subprocess.run([str(binary),str(tmp_path),'route'],check=True,capture_output=True,timeout=30)
    ids=np.fromfile(tmp_path/'ref_ids.i32',np.int32);w=np.fromfile(tmp_path/'ref_weights.f32',np.float32)
    assert len(set(ids))==k and w.sum()==pytest.approx(1,abs=1e-6)
    w.tofile(tmp_path/'weights.bin');np.arange(k,dtype=np.int32).tofile(tmp_path/'local_ids.bin')
    for name,fmt,shape in [('gate','IQ3_S',(k,f,d)),('up','IQ3_S',(k,f,d)),('down','IQ4_NL',(k,d,f)),
                           ('shared_gate','IQ4_NL',(f,d)),('shared_up','IQ4_NL',(f,d)),('shared_down','IQ4_NL',(d,f))]:
        packed(name,fmt,shape)
    dense('shared_router',(1,d))
    subprocess.run([str(binary),str(tmp_path),'experts'],check=True,capture_output=True,timeout=30)
    y=np.fromfile(tmp_path/'ref_final.f32',np.float32)
    assert y.shape==(d,) and np.isfinite(y).all()
    parts=np.fromfile(tmp_path/'ref_down.f32',np.float32).reshape(k,d)
    shared=np.fromfile(tmp_path/'ref_shared.f32',np.float32)
    np.testing.assert_allclose(y,sum_routes(parts,w,ids)+shared,rtol=1e-6,atol=1e-8)
