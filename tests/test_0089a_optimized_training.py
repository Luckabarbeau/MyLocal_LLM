"""CPU contract tests for the 0089A optimized-production probe."""

from __future__ import annotations

import importlib.util
from pathlib import Path

import numpy as np

from mini_llm.config import ModelConfig


ROOT = Path(__file__).parents[1]


def _load(name, relative):
    path = ROOT / relative
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


profile = _load("profile_0089a_training", "benchmarks/profile_0087a_training.py")
comparison = _load("compare_0089a_training", "benchmarks/compare_0089a_training.py")


def test_triton_scopes_share_the_current_attention_range():
    assert profile._scope_label("attention.triton_mixed.forward") == "attention_local"
    assert profile._scope_label("attention.triton_mixed.backward") == "attention_local"
    assert profile._scope_label("attention.local.forward") == "attention_local"


def test_fixture_hash_includes_dtype_shape_and_content():
    base = np.arange(12, dtype=np.int32).reshape(3, 4)
    assert profile._array_sha256(base) == profile._array_sha256(base.copy())
    assert profile._array_sha256(base) != profile._array_sha256(base.reshape(2, 6))
    assert profile._array_sha256(base) != profile._array_sha256(base.view(np.float32))
    changed = base.copy()
    changed[0, 0] = 99
    assert profile._array_sha256(base) != profile._array_sha256(changed)


def test_head_report_uses_all_rows_rule_for_executed_substitution():
    config = ModelConfig.ensemble_1024_20l_dense_historical_hybrid_64k()
    memory_length = int(config.memory_context.distant_memory_length)

    mixed = profile._active_head_patterns(
        config,
        {"history_valid_starts": np.asarray([memory_length, 0, 0])},
    )
    assert mixed["batch_all_rows_have_valid_history"] is False
    assert mixed["counts_by_executed_active_local_heads"] == {"15": 20}
    assert mixed["layers"][4]["executed_historical_skip_heads"] == []
    assert mixed["layers"][4]["historical_heads_with_any_valid_source"] == [
        10,
        11,
        12,
        13,
        14,
        15,
    ]

    all_valid = profile._active_head_patterns(
        config,
        {"history_valid_starts": np.asarray([0, 0, 0])},
    )
    assert all_valid["batch_all_rows_have_valid_history"] is True
    assert all_valid["layers"][0]["active_local_head_count"] == 13
    assert all_valid["layers"][2]["active_local_head_count"] == 11
    assert all_valid["layers"][4]["active_local_head_count"] == 9
    assert all_valid["layers"][8]["active_local_head_count"] == 15
    assert all_valid["layers"][4]["executed_historical_skip_heads"] == [
        10,
        11,
        12,
        13,
        14,
        15,
    ]


def _report(backend, *, loss=2.5, median_ms=100.0, input_hash="input"):
    summary = {
        "count": 3,
        "median_ms": median_ms,
        "p10_ms": median_ms * 0.99,
        "p90_ms": median_ms * 1.01,
        "min_ms": median_ms * 0.98,
        "max_ms": median_ms * 1.02,
    }
    return {
        "schema_version": "0089A-v1",
        "status": "measured",
        "safety": {
            "optimizer_constructed": False,
            "optimizer_update": False,
            "checkpoint_written": False,
            "weights_modified": False,
        },
        "model": {
            "checkpoint": "/checkpoint",
            "parameter_count": 338_134_308,
            "layers": 20,
            "d_model": 1024,
            "d_ff": 4096,
            "q_heads": 16,
            "kv_heads": 4,
            "dtype": "bfloat16",
            "activation_checkpoint": "none",
            "lm_head_chunk_tokens": 8192,
        },
        "runtime": {
            "attention_backend": backend,
            "attention_strict": backend == "triton",
            "effective_flags": {
                "MINI_LLM_TRAIN_ATTN_BACKEND": backend,
                "MINI_LLM_TRAIN_ATTN_STRICT": "1" if backend == "triton" else None,
                "MINI_LLM_BF16_MIXED_CONTEXT": "0",
            },
        },
        "workload": {
            "total_computed_rows_per_sample": 6912,
            "total_supervised_rows_per_sample": 3489,
            "batches": [
                {
                    "compact_working_length": 2304,
                    "computed_rows": 6912,
                    "supervised_rows": 3489,
                    "router_gumbel_seed": 42,
                    "fixture_hashes": {
                        "inputs": input_hash,
                        "targets": "targets",
                        "loss_mask": "mask",
                        "history_valid_starts": "history",
                    },
                }
            ],
        },
        "timing": {
            "gpu_event": summary,
            "host_wall_outer_synchronized": summary,
        },
        "last_loss_device_resident_until_boundary": loss,
    }


def test_comparison_accepts_matched_safe_reports():
    result = comparison.compare_reports(
        _report("grouped", median_ms=300.0),
        _report("triton", loss=2.5001, median_ms=200.0),
    )
    assert result["status"] == "valid"
    assert result["parity"]["accepted"] is True
    assert result["timing"]["speedup_grouped_over_triton"] == 1.5
    assert result["timing"]["triton_throughput"][
        "computed_rows_per_second"
    ] == 34_560.0


def test_comparison_rejects_unmatched_fixture_or_loss():
    result = comparison.compare_reports(
        _report("grouped"),
        _report("triton", loss=2.7, input_hash="different"),
    )
    assert result["status"] == "invalid"
    assert result["matched_batch_identity"] is None
    assert result["parity"]["accepted"] is False
    assert result["timing"]["speedup_grouped_over_triton"] is None
