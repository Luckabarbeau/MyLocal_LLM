"""0105B: explicit CUDA checks; never skip or pretend NumPy tested GPU kernels."""
import os

from mini_llm.production_profile_0105 import ENVIRONMENT


def test_gpu_contract_environment_is_exact():
    from mini_llm.backend import BACKEND_NAME
    assert BACKEND_NAME == "cupy", "0105 GPU gate must run in its own CuPy process"
    for key, expected in ENVIRONMENT.items():
        assert os.environ.get(key) == expected, f"unexpected GPU profile value for {key}"


def test_gpu_contract_device_and_sparse_architecture():
    import cupy as cp
    import triton
    from mini_llm.model_registry import registry
    assert cp.cuda.runtime.getDeviceCount() > 0
    assert triton is not None
    config = registry.config("model_b_348m_sparse16")
    assert config.estimated_parameter_count() == 348_261_376
    sparse = config.dual_resolution
    assert sparse.sparse_retrieval_stride == 16
    assert tuple(sparse.sparse_retrieval_layers) == tuple(range(2, 18))
    assert sparse.block_size == 128
