"""Shared expert graph: same FP32 projection path, overlap and output lifetime."""
from types import SimpleNamespace
import numpy as np
import pytest
from mini_llm.backend import BACKEND_NAME,xp

pytestmark=pytest.mark.skipif(BACKEND_NAME!='cupy',reason='CUDA shared graph')


@pytest.mark.parametrize('overlap',[False,True])
@pytest.mark.parametrize('fast',[False,True])
def test_shared_graph_exact_repeated_outputs(overlap,fast):
    from mini_llm.external.qwen4exp_model import Qwen4ExpMoE
    from mini_llm.ops.inference_matmul import matmul_inference
    rng=np.random.default_rng(273)
    def rand(shape): return xp.asarray(rng.normal(0,.1,shape).astype(np.float32))
    def projection(weight):
        return lambda x:matmul_inference(x.reshape(1,-1),weight).reshape(-1)
    gate,up,down=projection(rand((64,96))),projection(rand((64,96))),projection(rand((96,64)))
    router=SimpleNamespace(forward=lambda x:(xp.ones((1,1,1),xp.float32),xp.zeros((1,1,1),xp.int64)))
    experts=SimpleNamespace(forward=lambda x,w,i:x*xp.float32(.5))
    args=(router,experts,gate,up,down,rand((64,)))
    reference=Qwen4ExpMoE(*args,fast_elementwise=fast,overlap_shared=overlap,shared_graph_decode=True)
    reference._shared_graph_enabled=False
    captured=Qwen4ExpMoE(*args,fast_elementwise=fast,overlap_shared=overlap,shared_graph_decode=True)
    saved=[]
    for i in range(16):
        x=rand((64,))
        expected=reference(x)
        actual=captured(x)
        np.testing.assert_array_equal(actual.get(),expected.get())
        saved.append((actual,expected.get()))
    for actual,expected in saved:
        np.testing.assert_array_equal(actual.get(),expected)
    assert captured.graph_statistics()['calls']==16
    assert captured.graph_statistics()['pool_reserved_bytes']<=2*1024**2
