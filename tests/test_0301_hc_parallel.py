"""Parallel stream HC retains the existing reduction order, not just tolerance."""
import numpy as np
import pytest
from mini_llm.backend import xp, asnumpy
from mini_llm.ops.hyperconnection_read_gpu import read_gpu
from test_0284_bf16_hc_storage import storage

pytestmark=pytest.mark.skipif(xp.__name__!='cupy',reason='Actual CuPy required')


@pytest.mark.parametrize('d,rank',[(1,1),(7,3),(257,33),(2560,320)])
@pytest.mark.parametrize('packed',[False,True])
@pytest.mark.parametrize('inject',[False,True])
def test_bit_exact_and_capture(monkeypatch,d,rank,packed,inject):
    from mini_llm.ops.inference_graph import CapturedArrayInference
    rng=np.random.default_rng(301)
    gamma=xp.asarray(rng.uniform(.5,1.5,(4,d)),dtype=xp.float32)
    weights=[xp.asarray(rng.normal(0,.04,s),dtype=xp.float32) for s in
             [(rank,4*d),(4*d,rank),(4,4*d)]]
    if packed: weights=[storage(w)[0] for w in weights]
    down,up,gain=weights
    gain=gain if inject else None
    h=xp.asarray(rng.normal(size=(1,4,d)),dtype=xp.float32)
    def execute(value): return read_gpu(value,gamma,down,up,gain,1e-5)
    monkeypatch.setenv('MINI_LLM_HC_PARALLEL_READ','1')
    graph=CapturedArrayInference(lambda value,unused:execute(value)[0],h,())
    try:
        for _ in range(3):
            h*=xp.float32(-.9)
            monkeypatch.setenv('MINI_LLM_HC_PARALLEL_READ','0')
            expected=execute(h)
            monkeypatch.setenv('MINI_LLM_HC_PARALLEL_READ','1')
            actual=execute(h)
            for a,b in zip(actual,expected):
                if b is None: assert a is None
                else: np.testing.assert_array_equal(asnumpy(a).view(np.uint32),asnumpy(b).view(np.uint32))
            np.testing.assert_array_equal(asnumpy(graph.forward(h,())).view(np.uint32),asnumpy(expected[0]).view(np.uint32))
    finally: graph.close()
