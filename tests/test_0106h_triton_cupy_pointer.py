"""0106H: validate quantized CuPy-to-Triton pointer ABI without a CUDA device.

Triton's JIT recognizes .data_ptr()/.dtype, unlike bare CuPy ndarrays. These
fake arrays exercise the *actual launch callsites* even in CPU-only CI.
RTX 5080 numerical execution remains covered by 0106B/0106C/0106G GPU tests.
"""

from types import SimpleNamespace

import numpy as np
import pytest

from mini_llm.ops import quantized_linear as ql
from mini_llm.ops.quantized_weight import BLOCK_FORMATS, PackedGGUFWeight


class FakeCuPyArray:
    def __init__(self, array):
        self.numpy_data = np.ascontiguousarray(array)
        self.shape = self.numpy_data.shape
        self.ndim = self.numpy_data.ndim
        self.dtype = self.numpy_data.dtype
        self.size = self.numpy_data.size
        self.nbytes = self.numpy_data.nbytes
        self.flags = self.numpy_data.flags
        self.data = SimpleNamespace(ptr=self.numpy_data.ctypes.data)

    def reshape(self, new_shape):
        return FakeCuPyArray(self.numpy_data.reshape(new_shape))


class FakeKernel:
    def __init__(self):
        self.calls = []

    def __getitem__(self, grid):
        def run(*args, **kwargs):
            self.calls.append((grid, args, kwargs))
            for tensor in args[:self.pointers]:
                assert isinstance(tensor, ql._TritonCuPyPointer)
                assert tensor.data_ptr() == tensor.array.data.ptr
                assert hasattr(tensor, 'dtype')
        return run


@pytest.fixture
def gpu_stub(monkeypatch):
    stub = SimpleNamespace(
        ndarray=FakeCuPyArray,
        uint8=np.uint8,
        asarray=lambda x, dtype=None: FakeCuPyArray(np.asarray(x, dtype=dtype)),
        empty=lambda shape, dtype: FakeCuPyArray(np.empty(shape, dtype=dtype)),
    )
    tl_stub = SimpleNamespace(float16='tl.fp16', float32='tl.fp32',
                              bfloat16='tl.bf16', uint8='tl.uint8')
    monkeypatch.setattr(ql, 'BACKEND_NAME', 'cupy')
    monkeypatch.setattr(ql, 'xp', stub)
    monkeypatch.setattr(ql, 'tl', tl_stub)
    monkeypatch.setattr(ql, 'triton', object())
    gemv, swiglu = FakeKernel(), FakeKernel()
    gemv.pointers, swiglu.pointers = 3, 4
    monkeypatch.setattr(ql, '_packed_linear_decode_kernel', gemv, raising=False)
    monkeypatch.setattr(ql, '_packed_gate_up_swiglu_kernel', swiglu, raising=False)
    return stub, gemv, swiglu


def _weight(kind, rows, cols):
    qk, qb = BLOCK_FORMATS[kind]
    data = np.zeros((rows * cols // qk) * qb, dtype=np.uint8)
    return PackedGGUFWeight(data, (rows, cols), kind)


@pytest.mark.parametrize('dtype', [np.float16, np.float32])
def test_quantized_linear_wraps_all_gpu_pointers(gpu_stub, dtype):
    stub, gemv, _ = gpu_stub
    x = stub.asarray(np.zeros((2, 256), dtype=dtype))
    packed = ql.packed_to_device(_weight('Q5_K', 7, 256))
    y = ql.quantized_linear(x, packed)
    assert y.shape == (2, 7)
    assert len(gemv.calls) == 1
    grid, args, _ = gemv.calls[0]
    assert grid == (2, 7)
    assert [t.dtype for t in args[:3]] == [
        'tl.fp16' if dtype is np.float16 else 'tl.fp32',
        'tl.uint8',
        'tl.fp16' if dtype is np.float16 else 'tl.fp32',
    ]
    assert args[0].array is x
    assert args[1].array is packed.data


@pytest.mark.parametrize('gfmt,ufmt', [('Q4_K', 'Q5_K'), ('Q5_K', 'Q4_K'), ('Q3_K', 'Q5_K')])
def test_fused_swiglu_wraps_every_gpu_pointer(gpu_stub, gfmt, ufmt):
    stub, _, fused = gpu_stub
    x = stub.asarray(np.zeros((1, 256), dtype=np.float16))
    gate = ql.packed_to_device(_weight(gfmt, 8, 256))
    up = ql.packed_to_device(_weight(ufmt, 8, 256))
    out = ql.quantized_swiglu(x, gate, up)
    assert out.shape == (1, 8)
    assert len(fused.calls) == 1
    grid, args, _ = fused.calls[0]
    assert grid == (1, 8)
    assert [t.dtype for t in args[:4]] == [
        'tl.fp16', 'tl.uint8', 'tl.uint8', 'tl.fp16',
    ]
    assert args[1].array is gate.data
    assert args[2].array is up.data


def test_reject_numpy_host_operand_before_triton_launch(gpu_stub):
    _, gemv, _ = gpu_stub
    packed = ql.packed_to_device(_weight('Q5_K', 4, 256))
    with pytest.raises((ValueError, TypeError), match='CuPy'):
        ql.quantized_linear(np.zeros((1, 256), dtype=np.float16), packed)
    assert not gemv.calls


def test_pointer_adapter_rejects_unsupported_dtypes(gpu_stub):
    stub, _, _ = gpu_stub
    with pytest.raises(TypeError, match='Unsupported'):
        ql._triton_cupy_pointer(stub.asarray(np.zeros(3, dtype=np.int32)))
