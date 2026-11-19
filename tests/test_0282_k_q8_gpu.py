"""K/Q8 GPU arithmetic: pinned Strata dots and independent FP64 baseline."""
import numpy as np
import pytest
from test_0281_mixed_q8_gpu import (
    cp, gpu, q8, weight, reference, pytestmark,
    test_exact_strata_parts_rows_and_fp64_baseline as check_arithmetic,
)


@pytest.mark.parametrize('fmt',['Q4_K','Q5_K','Q6_K','Q8_0'])
@pytest.mark.parametrize('k',[256,768,2560])
def test_k_parts_rows_fp64(fmt,k):
    check_arithmetic(fmt,k)


@pytest.mark.parametrize('fmt',['Q4_K','Q5_K','Q6_K','Q8_0'])
def test_zero_capture_lifetime_and_layout(fmt):
    from mini_llm.ops.inference_graph import CapturedArrayInference
    _,w=weight(fmt)
    x=cp.zeros((1,768),cp.float32)
    np.testing.assert_array_equal(gpu.linear(x,w).get(),np.zeros((1,13),np.float32))
    graph=CapturedArrayInference(lambda x,unused:gpu.linear(x,w),x,())
    for i in range(4):
        x[:]=cp.arange(768,dtype=cp.float32)*(i-.5)
        y=graph.forward(x,())
        np.testing.assert_array_equal(y.get(),reference(q8.quantize_q8_1(x),w).get())
    saved=y.get()
    gpu.linear(x*2,w)
    np.testing.assert_array_equal(y.get(),saved)
    with pytest.raises(ValueError,match='contiguous'):
        gpu.linear(cp.zeros((1,1536),cp.float32)[:,::2],w)
    with pytest.raises(ValueError,match='width'):
        gpu.linear(cp.zeros((1,256),cp.float32),w)


@pytest.mark.parametrize('fmt',gpu.FORMATS)
@pytest.mark.parametrize('shape',[(2,768),(2,3,768)])
def test_opt_in_linear_dispatch(monkeypatch,fmt,shape):
    from mini_llm.ops.quantized_linear import quantized_linear,packed_to_device
    host,_=weight(fmt)
    w=packed_to_device(host)
    x=cp.asarray(np.random.default_rng(282).normal(0,.1,shape).astype(np.float32))
    monkeypatch.setenv('MINI_LLM_LINEAR_GPU_Q8','1')
    got=quantized_linear(x,w)
    expected_q8=reference(q8.quantize_q8_1(x.reshape(-1,768)),w).reshape(got.shape)
    np.testing.assert_array_equal(got.get(),expected_q8.get())
    # Host/reference calls must remain unquantized even with the flag enabled.
    host_x=x.get()
    expected=(host_x.reshape(-1,768)@host.dequantize_reference().T).reshape(got.shape)
    np.testing.assert_array_equal(quantized_linear(host_x,host),expected)
    monkeypatch.setenv('MINI_LLM_LINEAR_GPU_Q8','0')
    ordinary=quantized_linear(x,w)
    monkeypatch.setenv('MINI_LLM_LINEAR_GPU_Q8','1')
    half=quantized_linear(x.astype(cp.float16),w)
    monkeypatch.setenv('MINI_LLM_LINEAR_GPU_Q8','0')
    np.testing.assert_array_equal(half.get(),quantized_linear(x.astype(cp.float16),w).get())
    assert ordinary.dtype==cp.float32 and got.shape==shape[:-1]+(13,)
