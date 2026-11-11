"""CPU-safe CLI contracts and bounded generation-only report regression tests."""

import builtins
import importlib.util
import json
import math
import os
from pathlib import Path
import runpy
import subprocess
import sys
from types import SimpleNamespace

import pytest


ROOT = Path(__file__).resolve().parents[1]
BENCHMARK = ROOT / "benchmarks/benchmark_0106r_generation.py"


def _python(*arguments, timeout=30):
    return subprocess.run(
        [sys.executable, "-B", *map(str, arguments)],
        cwd=ROOT,
        env=dict(os.environ, MINI_LLM_BACKEND="numpy", OPENBLAS_NUM_THREADS="1",
                 PYTHONDONTWRITEBYTECODE="1"),
        capture_output=True, text=True, timeout=timeout,
    )


@pytest.fixture
def benchmark(monkeypatch):
    # The executable adds ROOT to sys.path; keep that change request-local too.
    monkeypatch.setattr(sys, "path", list(sys.path))
    spec = importlib.util.spec_from_file_location("generation_benchmark_test", BENCHMARK)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@pytest.mark.parametrize("help_requested", [False, True], ids=["import", "help"])
def test_import_and_help_do_not_import_package_or_initialize_backend(help_requested):
    code = """
import importlib.abc
import runpy
import sys

forbidden = {'mini_llm', 'numpy', 'cupy', 'torch'}
class NoBackend(importlib.abc.MetaPathFinder):
    def find_spec(self, fullname, path=None, target=None):
        if fullname.split('.')[0] in forbidden:
            raise AssertionError('unexpected backend/package import: ' + fullname)
sys.meta_path.insert(0, NoBackend())
path, mode = sys.argv[1:]
sys.argv = [path, '--help'] if mode == 'help' else [path]
if mode == 'help':
    try:
        runpy.run_path(path, run_name='__main__')
    except SystemExit as exc:
        assert exc.code == 0, exc.code
    else:
        raise AssertionError('--help did not exit')
else:
    runpy.run_path(path)
assert not any(name.split('.')[0] in forbidden for name in sys.modules)
"""
    result = _python("-I", "-S", "-c", code, BENCHMARK,
                     "help" if help_requested else "import")
    assert result.returncode == 0, result.stdout + result.stderr
    if help_requested:
        for option in ("--synthetic", "--max-new-tokens", "--requests", "--context",
                       "--prefill-chunk-tokens", "--ram-reserve-mib", "--output"):
            assert option in result.stdout


@pytest.mark.parametrize("option,value", [
    ("--max-new-tokens", "0"), ("--max-new-tokens", "257"),
    ("--requests", "0"), ("--requests", "5"),
    ("--context", "1"), ("--context", "513"),
    ("--prefill-chunk-tokens", "0"), ("--prefill-chunk-tokens", "129"),
    ("--cpu-workers", "-1"), ("--cpu-workers", "5"),
    ("--promote-after", "0"), ("--trace-records", "0"),
    ("--expert-cache-mib", "-1"), ("--ram-reserve-mib", "-1"),
    ("--max-rss-mib", "0"), ("--device-budget-mib", "0"),
    ("--device-reserve-mib", "-1"),
])
def test_invalid_bounds_rejected_before_observers_or_output(
        benchmark, monkeypatch, tmp_path, capsys, option, value):
    def unexpected_observer(*args):
        pytest.fail("invalid CLI options must not load observers or backends")

    monkeypatch.setattr(benchmark, "load_observer", unexpected_observer)
    output = tmp_path / "not-created" / "report.json"
    with pytest.raises(SystemExit) as exc:
        benchmark.main(["--synthetic", "--output", str(output), option, value])
    assert exc.value.code == 2
    assert "Invalid bounded generation/resource options" in capsys.readouterr().err
    assert not output.parent.exists()


def test_conflicting_prompts_rejected_before_observers(benchmark, monkeypatch, tmp_path, capsys):
    monkeypatch.setattr(benchmark, "load_observer", lambda *args: pytest.fail("observer loaded"))
    output = tmp_path / "report.json"
    with pytest.raises(SystemExit) as exc:
        benchmark.main(["--synthetic", "--output", str(output), "--prompt", "ab",
                        "--prompt-file", str(tmp_path / "missing.txt")])
    assert exc.value.code == 2
    assert "Choose --prompt or --prompt-file, not both" in capsys.readouterr().err
    assert not output.exists()


def test_existing_report_is_never_overwritten(benchmark, monkeypatch, tmp_path, capsys):
    monkeypatch.setattr(benchmark, "load_observer", lambda *args: pytest.fail("observer loaded"))
    output = tmp_path / "report.json"
    original = b"existing report must survive unchanged\n"
    output.write_bytes(original)
    with pytest.raises(SystemExit) as exc:
        benchmark.main(["--synthetic", "--output", str(output)])
    assert exc.value.code == 2
    assert "Refusing to overwrite existing report" in capsys.readouterr().err
    assert output.read_bytes() == original
    assert list(tmp_path.iterdir()) == [output]


@pytest.mark.parametrize("snapshot,reason", [
    ({"available_bytes": None, "rss_bytes": 0}, "mandatory available_bytes"),
    ({"available_bytes": 0, "rss_bytes": 0}, "available memory"),
    ({"available_bytes": 64 * 1024**3, "rss_bytes": 13 * 1024**3}, "RSS"),
], ids=["unknown-availability", "insufficient-headroom", "rss-over-budget"])
def test_host_preflight_fails_closed_without_backend_imports(
        benchmark, monkeypatch, tmp_path, snapshot, reason):
    memory = SimpleNamespace(**runpy.run_path(str(ROOT / "mini_llm/diagnostics/host_memory.py")))
    guard = memory.HostMemoryGuard(sample_fn=lambda: snapshot)
    checks, observers, budgets = [], [], []
    real_check = guard.check

    def check(loading=False):
        checks.append(loading)
        return real_check(loading=loading)

    def make_guard(reserve, max_rss):
        budgets.append((reserve, max_rss))
        return guard

    def load_observer(name, relative):
        observers.append((name, relative))
        assert (name, relative) == ("host_memory_0106r", "mini_llm/diagnostics/host_memory.py")
        return SimpleNamespace(HostMemoryGuard=make_guard, MemoryBudgetError=memory.MemoryBudgetError)

    real_import = builtins.__import__
    attempts = []

    def guarded_import(name, *args, **kwargs):
        if name.split(".")[0] in {"mini_llm", "numpy", "cupy", "torch"}:
            attempts.append(name)
            raise AssertionError("backend import before host preflight: " + name)
        return real_import(name, *args, **kwargs)

    monkeypatch.setattr(guard, "check", check)
    monkeypatch.setattr(benchmark, "load_observer", load_observer)
    output = tmp_path / "report.json"
    with monkeypatch.context() as imports:
        imports.setattr(builtins, "__import__", guarded_import)
        with pytest.raises(memory.MemoryBudgetError, match=reason) as exc:
            benchmark.main(["--synthetic", "--output", str(output)])
    report = json.loads(output.read_text())
    assert attempts == []
    assert checks == [True]
    assert len(observers) == 1
    assert budgets == [(16384 * 1024**2, 12288 * 1024**2)]
    assert report["patch"] == "0106R"
    assert report["status"] == "host_memory_guard_stopped"
    assert report["failed"] is True
    assert report["requests"] == []
    assert report["failure"] == {
        "stage": "host_preflight", "type": "MemoryBudgetError", "message": str(exc.value),
    }
    assert report["last_host_observation"] == snapshot
    assert not {"backend", "model_path", "host_preflight", "load_wall_ms"} & report.keys()
    assert list(tmp_path.iterdir()) == [output]


@pytest.mark.parametrize("values,expected", [
    ([], dict(count=0, mean_ms=None, p50_ms=None, p95_ms=None, max_ms=None, calls_per_s=None)),
    ([0, 0], dict(count=2, mean_ms=0, p50_ms=0, p95_ms=0, max_ms=0, calls_per_s=None)),
    ([8, 1, 3], dict(count=3, mean_ms=4, p50_ms=3, p95_ms=7.5, max_ms=8, calls_per_s=250)),
])
def test_latency_summary_uses_forward_call_count_and_interpolated_percentiles(benchmark, values, expected):
    original = list(values)
    actual = benchmark.latency_summary(values)
    assert actual.keys() == expected.keys()
    for key, value in expected.items():
        if value is None:
            assert actual[key] is None
        else:
            assert actual[key] == pytest.approx(value)
    assert values == original


@pytest.fixture(scope="module")
def synthetic_run(tmp_path_factory):
    output = tmp_path_factory.mktemp("0106r-generation") / "report.json"
    result = _python(
        BENCHMARK, "--synthetic", "--context", "32", "--max-new-tokens", "4",
        "--requests", "2", "--prefill-chunk-tokens", "2", "--cpu-workers", "0",
        "--ram-reserve-mib", "0", "--max-rss-mib", "512",
        "--device-budget-mib", "64", "--device-reserve-mib", "0",
        "--expert-cache-mib", "1", "--output", output, timeout=90,
    )
    assert result.returncode == 0, result.stdout + result.stderr
    return json.loads(output.read_text())


def test_synthetic_cli_completes_two_identical_greedy_requests_with_fresh_state(synthetic_run):
    report = synthetic_run
    assert report["patch"] == "0106R"
    assert report["status"] == "generation_completed"
    assert report["failed"] is False
    assert "failure" not in report
    assert report["backend"] == "numpy"
    assert report["reference_or_full_model_hash_run"] is False
    assert report["warm_request_ids_equal_first"] is True
    assert report["prompt_token_ids"] == [256, 32, 97, 10, 256]
    assert report["prompt"] == "ab a\nab"
    assert report["host_final"]["rss_bytes"] > 0
    assert report["load_wall_ms"] > 0
    for key, expected in {"synthetic": True, "context": 32, "max_new_tokens": 4,
                          "requests": 2, "prefill_chunk_tokens": 2, "cpu_workers": 0}.items():
        assert report["configuration"][key] == expected
    first, warm = report["requests"]
    assert [request["request_id"] for request in (first, warm)] == [0, 1]
    assert [request["label"] for request in (first, warm)] == ["first_use", "warm"]
    assert first["generated_token_ids"] == warm["generated_token_ids"]
    assert first["text"] == warm["text"]
    for request in (first, warm):
        assert request["finish_reason"] == "length"
        assert request["prompt_tokens"] == 5
        assert request["completion_tokens"] == len(request["generated_token_ids"]) == 4
        assert all(type(token) is int and 0 <= token < 260 for token in request["generated_token_ids"])
        assert request["forwarded_decode_tokens"] == 3
        # The final sampled token is not forwarded; warm state must not append to first-use state.
        assert request["state_length"] == request["prompt_tokens"] + request["forwarded_decode_tokens"] == 8


def test_synthetic_routing_and_stage_counts_are_phase_local_without_replay(synthetic_run):
    for request in synthetic_run["requests"]:
        assert set(request["routing"]) == {"prefill", "decode"}
        phases = request["stages"]["phases"]
        assert set(phases) == {"prefill", "decode"}
        for phase, tokens in (("prefill", 5), ("decode", 3)):
            trace = request["routing"][phase]
            assert trace["tokens_by_layer"] == {str(layer): tokens for layer in range(4)}
            assert trace["identity"]["workload"] == f"0106r/request-{request['request_id']}/{phase}"
            assert trace["mode"] == "summary"
            assert trace["counters"]["layer_dispatches"] == 3 * 4
            assert trace["counters"]["routed_assignments"] == tokens * 4 * 2
            for stage in ("norm1", "mixer", "norm2", "moe"):
                assert phases[phase][stage]["calls"] == 3 * 4


def test_synthetic_memory_observes_state_release_exactly_once(synthetic_run):
    expected = ["before_state", "after_state"]
    for phase in ("prefill", "decode"):
        expected.extend(["before_" + phase, "after_" + phase] * 3)
    expected.append("after_state_release")
    for request in synthetic_run["requests"]:
        labels = [observation["label"] for observation in request["memory"]]
        assert labels == expected
        assert labels.count("after_state_release") == 1
        assert all(observation["rss_bytes"] > 0 for observation in request["memory"])


def test_synthetic_timing_denominators_count_actual_work(synthetic_run):
    for request in synthetic_run["requests"]:
        timing = request["timing"]
        prefill, decode, samples = (timing[key] for key in
                                    ("prefill_chunk_ms", "decode_forward_ms", "sample_ms"))
        assert len(prefill) == 3
        assert len(decode) == request["forwarded_decode_tokens"] == 3
        assert len(samples) == request["completion_tokens"] == 4
        assert all(math.isfinite(value) and value >= 0 for value in prefill + decode + samples)
        assert timing["prefill_forward_ms"] == pytest.approx(sum(prefill))
        assert timing["prefill_tokens_per_s"] == pytest.approx(5 * 1000 / sum(prefill))
        summary = timing["decode"]
        assert summary["count"] == 3
        assert summary["mean_ms"] == pytest.approx(sum(decode) / 3)
        assert summary["calls_per_s"] == pytest.approx(3 * 1000 / sum(decode))
        ordered = sorted(decode)
        assert summary["p50_ms"] == pytest.approx(ordered[1])
        assert summary["p95_ms"] == pytest.approx(ordered[1] + .9 * (ordered[2] - ordered[1]))
        assert summary["max_ms"] == pytest.approx(max(decode))
        wall = timing["request_wall_ms"]
        assert wall >= sum(prefill) + sum(decode)
        assert timing["completion_tokens_per_s_end_to_end"] == pytest.approx(4 * 1000 / wall)
        assert 0 <= timing["first_sampled_token_ms"] <= wall
        assert timing["memory_observation_ms"] >= 0
        if request["text"]:
            assert timing["first_sampled_token_ms"] <= timing["first_visible_text_ms"] <= wall
        else:
            assert timing["first_visible_text_ms"] is None
