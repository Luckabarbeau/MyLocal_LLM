"""0103A immutable architectural and execution-report contracts.

Production architecture is checked without allocating the full 348M weights.
Numerical tests are delegated to the original 0102O/0102Z regressions.
"""

from dataclasses import asdict
import json

import pytest

from mini_llm.config import ModelConfig
from mini_llm.model_contracts import ArchitectureSpec, ExecutionReport, UnverifiedExecutionPath
from mini_llm.model_registry import registry


def test_registered_production_model_is_sparse_stride16_not_matrix_only():
    config = registry.config("model_b_348m_sparse16")
    assert config.estimated_parameter_count() == 348_261_376
    spec = ArchitectureSpec.from_config(config)
    assert spec.family == "dual_resolution_matrix_pool_sparse"
    assert (spec.layers, spec.hidden_size, spec.ffn_size) == (20, 1024, 4096)
    assert spec.vocabulary_size == 32768
    assert spec.context_length == 65536
    assert (spec.block_size, spec.summaries_per_block) == (128, 4)
    assert (spec.local_window, spec.long_local_window) == (512, 4096)
    assert (spec.sparse_stride, spec.sparse_head) == (16, 1)
    assert spec.sparse_layers == tuple(range(2, 18))
    assert {o.name for o in spec.operators} >= {
        "completed_block_matrix_pool", "compressed_global_attention",
        "sparse_stride_history",
    }
    # The schema must be framework-neutral and JSON-serializable, allowing
    # future CuPy/PyTorch comparisons without importing either backend.
    assert json.loads(json.dumps(asdict(spec)))["sparse_stride"] == 16


def test_sparse_head_changes_no_learned_parameter_count():
    base = ModelConfig.dual_resolution_matrix_pool_1024_20l()
    sparse = ModelConfig.dual_resolution_matrix_pool_sparse_1024_20l()
    assert base.estimated_parameter_count() == sparse.estimated_parameter_count()
    assert sparse.estimated_parameter_count() == 348_261_376


def test_unknown_or_unverified_fast_path_is_not_benchmarked_as_fast():
    for report in (
        ExecutionReport(backend="numpy", device="cpu", dtype="float32"),
        ExecutionReport(backend="cupy", device="cuda", dtype="bfloat16",
                        active_optimizations=("cuda-graph-requested",)),
        ExecutionReport(backend="cupy", device="cuda", dtype="bfloat16",
                        acceleration_verified=True,
                        fallbacks=("cuda-graph-initialization-failed",)),
    ):
        with pytest.raises(UnverifiedExecutionPath):
            report.require_fast()
    ExecutionReport(backend="cupy", device="cuda", dtype="bfloat16",
                    acceleration_verified=True).require_fast()
