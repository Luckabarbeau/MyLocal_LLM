"""CPU-only, hand-derived routing locality and cold cache replay contracts."""

from copy import deepcopy
from functools import lru_cache
from itertools import product
import json

import pytest

from mini_llm.diagnostics import routing_analysis
from mini_llm.diagnostics.routing_analysis import analyze_trace, main


IDENTITY = {
    "model_id": "tiny-model", "quantization": "q5_k",
    "expert_mapping": "global-layer-v1", "workload": "batched-prefill",
}


def _route(dispatch, layer, start, ids):
    return {"dispatch_id": dispatch, "layer_id": layer, "token_start": start,
            "expert_ids": ids,
            "routing_weights": [[1.0 / len(row)] * len(row) for row in ids]}


def _snapshot(routes):
    return {"schema_version": 1, "identity": dict(IDENTITY), "mode": "detailed",
            "routes": routes, "executions": [], "dropped_routes": 7}


def _sequence(ids):
    return _snapshot([_route(i, 0, i, [[expert]]) for i, expert in enumerate(ids)])


def _policies(result, capacity):
    return next(row["policies"] for row in result["replay_curves"]
                if row["capacity_slots"] == capacity)


def test_hand_derived_lru_belady_and_static_cold_curves():
    # ABCABC, two slots: LRU misses every time; Belady keeps A and B,
    # bypasses C, then hits A and B. A/B static-hot also has two cold misses.
    result = analyze_trace(_sequence([0, 1, 2, 0, 1, 2]),
                           windows=(), capacities=(0, 1, 2, 3))
    expected = {0: (0, 0), 1: (0, 1), 2: (0, 2), 3: (3, 3)}
    for capacity, (lru_hits, optimal_hits) in expected.items():
        policies = _policies(result, capacity)
        assert policies["lru"]["hits"] == lru_hits
        assert policies["belady"]["hits"] == optimal_hits
        for metrics in policies.values():
            assert metrics["requests"] == 6
            assert metrics["hits"] + metrics["misses"] == 6
            assert metrics["resident_slots"] <= capacity
    assert _policies(result, 2)["static_hot"]["hits"] == 2
    assert _policies(result, 2)["static_hot"]["misses"] == 4
    assert _policies(result, 2)["lru"]["evictions"] == 4
    assert "in-sample optimistic oracle" in result["policy_definitions"]["static_hot"]


def test_belady_can_bypass_for_cpu_fallback():
    # Mandatory-admission paging would miss all three at capacity one.
    policies = _policies(analyze_trace(_sequence([0, 1, 0]), capacities=(1,)), 1)
    assert policies["lru"]["hits"] == 0
    assert policies["belady"]["hits"] == 1
    assert "bypass" in routing_analysis.analyze_trace(
        _sequence([]), windows=(), capacities=())["policy_definitions"]["belady"]


def test_belady_matches_exhaustive_optional_admission_optimum():
    for sequence in product(range(3), repeat=4):
        for capacity in (0, 1, 2):
            @lru_cache(None)
            def optimum(index, resident):
                if index == len(sequence):
                    return 0
                key = sequence[index]
                if key in resident:
                    return 1 + optimum(index + 1, resident)
                choices = [optimum(index + 1, resident)]  # bypass
                if capacity:
                    if len(resident) < capacity:
                        choices.append(optimum(index + 1, tuple(sorted((*resident, key)))))
                    else:
                        for victim in resident:
                            replacement = tuple(sorted((set(resident) - {victim}) | {key}))
                            choices.append(optimum(index + 1, replacement))
                return max(choices)

            policies = _policies(analyze_trace(_sequence(sequence), windows=(),
                                               capacities=(capacity,)), capacity)
            assert policies["belady"]["hits"] == optimum(0, ())
            assert all(stats["hits"] <= policies["belady"]["hits"]
                       for stats in policies.values())


def test_capacity_zero_and_empty_trace_are_json_serializable():
    for trace in (_sequence([1, 1, 2]), _sequence([])):
        result = analyze_trace(trace, capacities=(0,))
        json.dumps(result, allow_nan=False)
        for stats in _policies(result, 0).values():
            assert stats["hits"] == stats["resident_slots"] == stats["admissions"] == 0
            assert stats["misses"] == len(trace["routes"])
        if not trace["routes"]:
            assert all(not width["layers"] for width in result["token_windows"])
            assert all(stats["hit_rate"] == stats["miss_rate"] == 0.0
                       for stats in _policies(result, 0).values())


def test_adaptive_uses_existing_policy_and_non_none_materializer(monkeypatch):
    original = routing_analysis.AdaptiveExpertCache
    calls = []

    class RecordingCache(original):
        def request(self, key, size_bytes, materialize):
            sentinel = materialize()
            assert sentinel is not None
            calls.append((key.model_id, key.layer_id, key.expert_id, size_bytes))
            return super().request(key, size_bytes, lambda: sentinel)

    monkeypatch.setattr(routing_analysis, "AdaptiveExpertCache", RecordingCache)
    result = analyze_trace(_sequence([0, 0, 0, 0]), windows=(), capacities=(1,))
    adaptive = _policies(result, 1)["adaptive"]
    assert calls == [(IDENTITY["model_id"], 0, 0, 1)] * 4
    assert adaptive["hits"] == 1  # third request promotes; only fourth is a hit
    assert adaptive["misses"] == 3
    assert adaptive["admissions"] == 1
    assert adaptive["cpu_fallbacks"] == 2


def test_identity_preserved_without_mutation():
    trace = _sequence([0, 1, 0])
    trace["identity"]["revision"] = "retained-extra-field"
    before = deepcopy(trace)
    result = analyze_trace(trace, capacities=(0, 2, 4, 8))
    assert trace == before
    assert result["identity"] == trace["identity"]
    result["identity"]["model_id"] = "different"
    assert trace == before
    assert "not VRAM bytes" in result["capacity_unit"]
    assert "retained" in result["scope"]
    assert result["cache_key"] == ["layer_id", "expert_id"]


def test_actual_record_order_not_dispatch_id_sort_order():
    trace = _snapshot([_route(20, 0, 0, [[0]]), _route(10, 0, 1, [[1]]),
                       _route(30, 0, 2, [[0]])])
    result = analyze_trace(trace, windows=(), capacities=(1,))
    assert _policies(result, 1)["lru"]["hits"] == 0


def test_dispatch_groups_sorted_distinct_experts_not_token_assignments():
    trace = _snapshot([_route(0, 0, 0, [[2, 0], [2, 2]]),
                       _route(1, 0, 2, [[0]])])
    result = analyze_trace(trace, windows=(2,), capacities=(1,))
    # Actual requests are 0,2,0, NOT 2,0,2,2,0 nor 2,0,0.
    assert result["retained"]["requests"] == 3
    assert _policies(result, 1)["lru"]["hits"] == 0
    window = result["token_windows"][0]["layers"][0]["windows"][0]
    assert window["selections"] == 4  # duplicates still count as selections
    assert window["experts"] == [
        {"expert_id": 0, "selections": 1, "selection_share": 0.25},
        {"expert_id": 2, "selections": 3, "selection_share": 0.75},
    ]
    assert any("Batched prefill" in note for note in result["limitations"])


def test_global_cache_separates_identical_ids_in_different_layers():
    trace = _snapshot([_route(0, 8, 0, [[0]]), _route(1, 2, 0, [[0]]),
                       _route(2, 8, 1, [[0]])])
    result = analyze_trace(trace, windows=(2,), capacities=(1, 2))
    assert result["retained"]["unique_experts"] == 2
    assert result["retained"]["layer_tokens"] == [
        {"layer_id": 2, "tokens": 1}, {"layer_id": 8, "tokens": 2}]
    assert _policies(result, 1)["lru"]["hits"] == 0
    assert _policies(result, 2)["lru"]["hits"] == 1
    assert [row["layer_id"] for row in result["token_windows"][0]["layers"]] == [2, 8]


def test_window_frequency_coverage_crosses_dispatch_boundaries_and_reports_tail():
    trace = _snapshot([_route(0, 0, 0, [[0, 1], [0, 2], [0, 1]]),
                       _route(1, 0, 3, [[1, 2], [2, 2]])])
    result = analyze_trace(trace, windows=(2, 5), capacities=())
    windows = result["token_windows"][0]["layers"][0]["windows"]
    assert [row["observed_tokens"] for row in windows] == [2, 2, 1]
    assert [row["partial"] for row in windows] == [False, False, True]
    assert windows[0]["static_coverage_slots"] == {
        "50": 1, "75": 2, "90": 3, "95": 3, "99": 3}
    assert windows[1]["static_coverage_slots"] == windows[0]["static_coverage_slots"]
    assert windows[2]["missing_tokens"] == 1
    assert set(windows[2]["static_coverage_slots"].values()) == {1}
    whole = result["token_windows"][1]["layers"][0]["windows"][0]
    assert whole["selections"] == 10
    assert whole["static_coverage_slots"] == {"50": 2, "75": 3, "90": 3, "95": 3, "99": 3}
    assert [expert["selections"] for expert in whole["experts"]] == [3, 3, 4]


def test_default_window_sizes_and_large_partial_tail():
    trace = _snapshot([_route(0, 0, 0, [[7]] * 8200)])
    result = analyze_trace(trace, capacities=(0,))
    assert [row["window_size"] for row in result["token_windows"]] == [128, 512, 2048, 8192]
    for row in result["token_windows"]:
        windows = row["layers"][0]["windows"]
        assert len(windows) == 8200 // row["window_size"] + 1
        assert sum(window["observed_tokens"] for window in windows) == 8200
        assert windows[-1]["observed_tokens"] == 8
        assert windows[-1]["partial"] is True


def test_retained_nonzero_start_and_gaps_are_not_claimed_complete():
    trace = _snapshot([_route(5, 3, 127, [[0]]), _route(8, 3, 129, [[1]])])
    windows = analyze_trace(trace, windows=(128,), capacities=())["token_windows"][0]["layers"][0]["windows"]
    assert [row["token_start"] for row in windows] == [0, 128]
    assert all(row["partial"] and row["missing_tokens"] == 127 for row in windows)


@pytest.mark.parametrize("policy,expected", [("lru", 2), ("lfu", 3), ("recency_frequency", 3)])
def test_online_policy_frequency_and_recency_contract(policy, expected):
    # After AAAB C, frequency protects A whereas LRU has evicted it.
    result = analyze_trace(_sequence([0, 0, 0, 1, 2, 0]), windows=(), capacities=(2,))
    assert _policies(result, 2)[policy]["hits"] == expected


@pytest.mark.parametrize("field,value", [
    ("schema_version", 2), ("schema_version", True), ("schema_version", 1.0),
    ("mode", "summary"), ("mode", "unknown"), ("routes", None),
    ("identity", {}), ("executions", {}),
])
def test_rejects_invalid_snapshot_contract(field, value):
    trace = _sequence([0])
    trace[field] = value
    with pytest.raises(ValueError):
        analyze_trace(trace)


@pytest.mark.parametrize("field,value", [
    ("dispatch_id", -1), ("dispatch_id", True), ("layer_id", "1"),
    ("token_start", 0.5), ("expert_ids", [[-1]]), ("expert_ids", [[True]]),
    ("expert_ids", [[1.0]]), ("expert_ids", [[[1]]]), ("expert_ids", [[]]),
    ("expert_ids", [[0], [1, 2]]), ("routing_weights", []),
    ("routing_weights", [[0.5, 0.5]]), ("routing_weights", [[-0.1]]),
    ("routing_weights", [[float("nan")]]), ("routing_weights", [[float("inf")]]),
    ("routing_weights", [[float("-inf")]]), ("routing_weights", [[True]]),
    ("routing_weights", [["1.0"]]), ("schema_version", 2),
    ("mode", "summary"), ("identity", {**IDENTITY, "workload": "different"}),
])
def test_rejects_malformed_or_mixed_route_records(field, value):
    trace = _sequence([0])
    trace["routes"][0][field] = value
    with pytest.raises(ValueError):
        analyze_trace(trace)


def test_rejects_mixed_execution_schema_and_merged_snapshots():
    trace = _sequence([0])
    trace["executions"] = [{"schema_version": 2}]
    with pytest.raises(ValueError, match="mixed schema"):
        analyze_trace(trace)
    with pytest.raises(ValueError, match="one snapshot"):
        analyze_trace([_sequence([0]), _sequence([1])])


@pytest.mark.parametrize("field,value", [("dispatch_id", 0), ("token_start", 0)])
def test_rejects_duplicate_dispatch_or_overlapping_layer_offsets(field, value):
    trace = _sequence([0, 1])
    trace["routes"][1][field] = value
    with pytest.raises(ValueError):
        analyze_trace(trace)


@pytest.mark.parametrize("kwargs", [
    {"windows": (0,)}, {"windows": (-1,)}, {"windows": (True,)},
    {"windows": (1.5,)}, {"capacities": (-1,)}, {"capacities": (True,)},
    {"capacities": (2.0,)}, {"capacities": None},
])
def test_rejects_malformed_window_sizes_or_capacities(kwargs):
    with pytest.raises(ValueError):
        analyze_trace(_sequence([0]), **kwargs)


def test_weights_need_not_be_normalized_and_executions_do_not_add_requests():
    trace = _sequence([0])
    trace["routes"][0]["routing_weights"] = [[0.0]]
    trace["executions"] = [{"layer_id": 99, "expert_id": 99}]
    result = analyze_trace(trace, windows=(1,), capacities=(1,))
    assert result["retained"]["requests"] == 1
    assert result["token_windows"][0]["layers"][0]["windows"][0]["selections"] == 1


def test_cli_json_output(tmp_path):
    trace = tmp_path / "trace.json"
    output = tmp_path / "analysis.json"
    trace.write_text(json.dumps(_sequence([0, 1, 0])), encoding="utf-8")
    main(["--trace", str(trace), "--output", str(output),
          "--windows", "2", "--capacities", "0", "2"])
    result = json.loads(output.read_text(encoding="utf-8"))
    assert result["identity"] == IDENTITY
    assert _policies(result, 2)["lru"]["hits"] == 1
    assert result["token_windows"][0]["window_size"] == 2
