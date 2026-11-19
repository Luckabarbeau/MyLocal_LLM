"""QSA frontend replay: runtime positions, exact caches and transactional checks."""
import numpy as np
import pytest
from mini_llm.backend import BACKEND_NAME, asnumpy, xp
from mini_llm.external.qwen4exp_attention import Qwen4ExpAttention
from test_0230_qsa_gpu_decode import fixture

pytestmark = pytest.mark.skipif(BACKEND_NAME != 'cupy', reason='QSA CUDA frontend')


@pytest.mark.parametrize('heads,dim', [(4,8),(24,256)])
def test_dynamic_positions_reset_fallback_and_exact_state(monkeypatch, heads, dim):
    monkeypatch.setenv('MINI_LLM_QSA_FRONTEND_GRAPH','1')
    c, weights, rng = fixture(heads=heads, dim=dim)
    reference = Qwen4ExpAttention(c, **weights, fast_decode=True)
    reference._frontend_graph_enabled = False
    candidate = Qwen4ExpAttention(c, **weights, fast_decode=True)
    stream = xp.cuda.Stream(non_blocking=True)
    stream.synchronize()
    # Constructor weights are produced on the default stream.
    xp.cuda.get_current_stream().synchronize()
    with stream:
        a,b = reference.create_state(24),candidate.create_state(24)
        for repeat in range(2):
            for t in range(24):
                x=xp.asarray(rng.normal(size=7),dtype=xp.float32)
                expected=reference.step(x,a,position=t)
                actual=candidate.step(x,b,position=t)
                np.testing.assert_array_equal(actual.get(stream=stream),expected.get(stream=stream))
            for name in ('keys','values','raw_index','pooled'):
                np.testing.assert_array_equal(getattr(a,name).get(stream=stream),
                                              getattr(b,name).get(stream=stream))
            assert a.length==b.length==24
            reference.reset(a)
            candidate.reset(b)
        # Width >16 uses the unchanged sparse path; two requests reuse one graph.
        assert candidate._frontend_graph.statistics()['calls']==32
        assert candidate._frontend_graph.statistics()['pool_reserved_bytes']<=2*1024**2
        before={name:getattr(b,name).copy() for name in ('keys','values','raw_index','pooled')}
        with pytest.raises(ValueError,match='finite'):
            candidate.step(xp.full(7,xp.nan,xp.float32),b,position=0)
        assert b.length==0
        for name,value in before.items():
            np.testing.assert_array_equal(getattr(b,name).get(stream=stream),value.get(stream=stream))
    stream.synchronize()
