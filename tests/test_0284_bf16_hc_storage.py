"""Exact BF16 storage versus the same checkpoint values expanded to FP32."""
import numpy as np
import pytest
from mini_llm.backend import xp,asnumpy
from mini_llm.external.qwen4exp_hyperconnection import Qwen4ExpHyperConnection as HC
from mini_llm.ops.hyperconnection_read_gpu import read_gpu

pytestmark=pytest.mark.skipif(xp.__name__!='cupy',reason='Actual CuPy required')


def storage(value):
    bits=asnumpy(value).view(np.uint32)
    bits=bits+np.uint32(0x7fff)+((bits>>16)&1)
    packed=(bits>>16).astype(np.uint16)
    return xp.asarray(packed),xp.asarray((packed.astype(np.uint32)<<16).view(np.float32))


@pytest.mark.parametrize('d,rank',[(1,1),(7,3),(257,33),(2560,320)])
@pytest.mark.parametrize('inject',[False,True])
def test_exact_reads_writes_capture(d,rank,inject):
    from mini_llm.ops.inference_graph import CapturedArrayInference
    rng=np.random.default_rng(284)
    gamma=xp.asarray(rng.uniform(.5,1.5,(4,d)),dtype=xp.float32)
    pairs=[storage(rng.normal(0,.04,shape).astype(np.float32)) for shape in
           [(rank,4*d),(4*d,rank)]+([(4,4*d)] if inject else [])]
    packed=HC(gamma,*(p[0] for p in pairs),epsilon=1e-5,fast_decode=True,native_read=True,packed_bf16=True)
    expanded=HC(gamma,*(p[1] for p in pairs),epsilon=1e-5,fast_decode=True,native_read=True)
    h=xp.asarray(rng.normal(size=(1,4,d)),dtype=xp.float32)
    graph=CapturedArrayInference(lambda value,unused:packed.read(value)[0],h,())
    for i in range(3):
        h*=xp.float32(-.9)
        a,g=packed.read(h); b,gb=expanded.read(h)
        np.testing.assert_array_equal(asnumpy(a),asnumpy(b))
        np.testing.assert_array_equal(asnumpy(graph.forward(h,())),asnumpy(b))
        if inject:
            np.testing.assert_array_equal(asnumpy(g),asnumpy(gb))
            np.testing.assert_array_equal(asnumpy(packed.write(h,a,g)),asnumpy(expanded.write(h,b,gb)))
        else:
            assert g is gb is None
            np.testing.assert_array_equal(asnumpy(packed.collapse(h)),asnumpy(b))
    assert packed.down.nbytes*2==expanded.down.nbytes
    # Construction owns storage; later input mutation must not change it.
    before=asnumpy(packed.down)
    pairs[0][0].fill(0)
    np.testing.assert_array_equal(asnumpy(packed.down),before)
    with pytest.raises(ValueError,match='single-token'):
        packed.read(xp.repeat(h,2,axis=0))


def test_header_budget_and_zero_copy_host_bytes():
    from types import SimpleNamespace
    from test_0209_qwen4exp_model import TinyStore
    from mini_llm.external.qwen4exp_weights import Qwen4ExpWeights
    source=TinyStore()
    index=SimpleNamespace(store=source,d_model=256,d_ff=64,n_experts=4,top_k=2,n_layers=2)
    weights=Qwen4ExpWeights(index)
    before=weights.memory_plan(16)
    weights.configure_hc_bf16()
    assert source.reads==[]
    after=weights.memory_plan(16)
    saving=sum(source.tensors[n].nbytes for n in weights.hc_bf16_names)
    assert after['hc_bf16_saved_bytes']==saving>0
    assert before['device_reserved_bytes']-after['device_reserved_bytes']==saving
    weights.approve(after)
    name='blk.0.hc_attn_down.weight'
    raw=weights.hc_matrix(name)
    assert raw.dtype==xp.uint16
    np.testing.assert_array_equal(asnumpy(raw).reshape(-1),source.payload[name].view('<u2'))
    with pytest.raises(ValueError,match='unapproved'):
        weights.configure_hc_bf16()


def test_bf16_invalid_configuration():
    gamma=xp.ones((4,7),xp.float32)
    down=xp.zeros((3,28),xp.uint16); up=xp.zeros((28,3),xp.uint16)
    with pytest.raises(ValueError,match='native'):
        HC(gamma,down,up,epsilon=1e-5,packed_bf16=True)
    with pytest.raises(TypeError,match='uint16'):
        HC(gamma,down.astype(xp.float32),up,epsilon=1e-5,packed_bf16=True,fast_decode=True,native_read=True)
