"""Capture parity for HC->GDN->HC->router, with explicit recurrent sessions."""
import numpy as np
import pytest
from mini_llm.backend import BACKEND_NAME, xp

pytestmark = pytest.mark.skipif(BACKEND_NAME != 'cupy', reason='CUDA segmented replay')


def test_pre_moe_boundaries_and_sessions_exact():
    from test_0265_gdn_segment_graph import mixers
    from mini_llm.external.qwen4exp_hyperconnection import Qwen4ExpHyperConnection
    from mini_llm.external.qwen4exp_pre_moe_graph import Qwen4ExpPreMoEGraph
    from mini_llm.ops.router_inference import RouterInference
    mixer, _, rng = mixers()
    def rand(shape):
        return xp.asarray(rng.normal(0,.1,shape).astype(np.float32))
    def hc():
        return Qwen4ExpHyperConnection(xp.ones((4,64),xp.float32),rand((16,256)),
            rand((256,16)),rand((4,256)),epsilon=1e-5,fast_decode=True,native_read=True)
    attention, ffn = hc(), hc()
    router = RouterInference(64,32,10,renormalize_selected=True,normalization_floor=2**-14)
    router.W_router,router.b_router=rand((64,32)),xp.zeros(32,xp.float32)
    router._decode_fastpath_enabled=True
    router._capturable_logits=True
    segment = Qwen4ExpPreMoEGraph(attention,mixer,ffn,router)
    reference = [mixer.create_state(1),mixer.create_state(1)]
    captured = [mixer.create_state(1),mixer.create_state(1)]
    for token in range(32):
        session=token%2
        streams=rand((4,64))
        hidden,gains=attention.read(streams)
        mixed=mixer.forward(hidden[None,None,:],reference[session])[0,0]
        residual=attention.write(streams,mixed,gains)
        hidden,gains=ffn.read(residual)
        weights,ids=router.forward(hidden.reshape(1,1,64))
        expected=(residual,hidden,gains,weights,ids)
        actual=segment.forward(streams,captured[session])
        for got,want in zip(actual,expected):
            np.testing.assert_array_equal(got.get(),want.get())
        for attr,field in (('conv','history'),('delta','memory')):
            a,b=getattr(captured[session],attr),getattr(reference[session],attr)
            assert a.length==b.length==token//2+1
            np.testing.assert_array_equal(getattr(a,field).get(),getattr(b,field).get())
    assert segment.graph_statistics()['calls']==32
    assert segment.graph_statistics()['pool_reserved_bytes']<=16*1024**2


def test_tuple_fresh_vs_borrowed_graph_outputs():
    from mini_llm.ops.inference_graph import CapturedArrayInference
    x=xp.arange(16,dtype=xp.float32)
    graph=CapturedArrayInference(lambda a,s:(a+1,a.astype(xp.int64)),x,())
    owned=graph.forward(x,())
    saved=tuple(a.get() for a in owned)
    borrowed=graph.forward(x*2,(),borrow_output=True)
    graph.forward(x*3,(),borrow_output=True)
    for a,b in zip(owned,saved):
        np.testing.assert_array_equal(a.get(),b)
    np.testing.assert_array_equal(borrowed[0].get(),(x*3+1).get())
