"""Input-only captured operands do not trigger post-graph copies or mutation."""
import numpy as np
import pytest
from mini_llm.backend import BACKEND_NAME, xp
from mini_llm.ops.inference_graph import CapturedArrayInference

pytestmark = pytest.mark.skipif(BACKEND_NAME != 'cupy', reason='CUDA graph wrapper')


@pytest.mark.parametrize('copy_back',[False,True])
def test_graph_operand_copy_contract(monkeypatch,copy_back):
    x=xp.arange(32,dtype=xp.float32)
    operand=xp.full(32,2,xp.float32)
    graph=CapturedArrayInference(lambda x, arrays:x+arrays[0],x,(operand,),
                                copy_back_mutable=copy_back)
    copies=[]
    copyto=xp.copyto
    def tracked(target,source,*a,**kw):
        copies.append(target.data.ptr)
        return copyto(target,source,*a,**kw)
    monkeypatch.setattr(xp,'copyto',tracked)
    saved=[]
    for i in range(8):
        operand.fill(i)
        copies.clear()
        actual=graph.forward(x,(operand,))
        assert len(copies)==2+int(copy_back)
        assert (operand.data.ptr in copies)==copy_back
        np.testing.assert_array_equal(actual.get(),x.get()+i)
        np.testing.assert_array_equal(operand.get(),np.full(32,i,np.float32))
        saved.append((actual,actual.get()))
    for actual,expected in saved:
        np.testing.assert_array_equal(actual.get(),expected)
    assert graph.statistics()['copy_back_mutable']==copy_back
