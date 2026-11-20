"""Shared-memory lookup changes no packed bytes, integer dots or FP32 sums."""
import numpy as np
import pytest
from mini_llm.backend import BACKEND_NAME
from mini_llm.ops import mixed_q8_gpu as gpu,q8_1_gpu as q8
from test_0281_mixed_q8_gpu import weight,reference
from test_0281_mixed_q8_gpu import test_mixed_plan_all_checkpoint_triples
from test_0256_ggml_packed_cpu import selected_library

cp=pytest.importorskip('cupy')
pytestmark=pytest.mark.skipif(BACKEND_NAME!='cupy',reason='Actual CUDA kernels')


@pytest.fixture(autouse=True)
def shared_grid(monkeypatch): monkeypatch.setenv('MINI_LLM_Q8_SHARED_GRID','1')


@pytest.mark.parametrize('fmt',['IQ2_S','IQ3_S','IQ3_XXS'])
@pytest.mark.parametrize('k',[256,768,2560])
@pytest.mark.parametrize('n',[1,13])
def test_exact_strata_rows_and_capture(fmt,k,n):
    from mini_llm.ops.inference_graph import CapturedArrayInference
    _,w=weight(fmt,n=n,k=k)
    x=cp.asarray(np.random.default_rng(307).normal(0,.1,(2,k)),dtype=cp.float32)
    expected=reference(q8.quantize_q8_1(x),w)
    np.testing.assert_array_equal(gpu.linear(x,w).get().view(np.uint32),expected.get().view(np.uint32))
    graph=CapturedArrayInference(lambda a,unused:gpu.linear(a,w),x,())
    try:
        for scale in (0,.7,-.5):
            x*=cp.float32(scale)
            np.testing.assert_array_equal(graph.forward(x,()).get().view(np.uint32),
                reference(q8.quantize_q8_1(x),w).get().view(np.uint32))
    finally: graph.close()
