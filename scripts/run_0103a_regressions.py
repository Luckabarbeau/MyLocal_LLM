#!/usr/bin/env python3
"""0103A baseline regression gates (no training and no kernel changes).

Each suite runs in a fresh interpreter.  Backend selection and some legacy
runtime toggles are import-time global state; mixing them in one pytest process
can produce order-dependent results and misleading benchmark conclusions.
"""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parents[1]

# Exact, versioned selectors intentionally avoid running unbounded historical
# research benchmarks or allocating the full 348M model on a CPU.
CPU_SUITES = {
    "public_api": (
        "tests/test_model_api_0103_0.py",
        "tests/test_canonical_model_loading_0103c.py",
        "tests/test_inference_policy_0103d.py",
        "tests/test_baseline_contract_0103a.py",
        "tests/test_regression_runner_0103a.py",
    ),
    "generation_sessions_and_serving": (
        "tests/test_generation_session_0103e.py",
        "tests/test_remote_serving_0103f.py",
        "tests/test_0088a_openai_server.py",
        "tests/test_0088c_serving_sampling.py",
        "tests/test_generation_benchmark.py",
    ),
    "matrix_pool_numerics": (
        "tests/test_matrix_pool_dual_resolution_0102o.py",
        "tests/test_matrix_pool_metadata_0102r.py",
    ),
    "stride16_and_incremental_decode": (
        "tests/test_sparse_stride_history_0102z.py",
    ),
    "checkpoint_compatibility": (
        "tests/test_checkpoint.py",
    ),
    "runtime_profile_isolated": (
        "tests/test_runtime_defaults.py",
    ),
}

GPU_SUITES = {
    "sparse_triton_parity": (
        "tests/test_sparse_stride_history_0102z.py::test_triton_parity_and_strided_kv_gradient",
        "tests/test_sparse_stride_history_0102z.py::test_model_b_local_sparse_triton_matches_reference",
    ),
    "matrix_pool_triton_parity": (
        "tests/test_matrix_pool_dual_resolution_0102o.py::test_production_width_triton_mixed_window_and_global_parity",
    ),
    "native_cuda_graph_contract": (
        "tests/test_native_dual_cuda_graph_decode.py::test_model_b_native_graph_activates_and_matches_fallback",
    ),
}


def environment_for(mode: str, inherited: dict[str, str]) -> dict[str, str]:
    """Pin backend, without changing the parent's environment or fast flags."""
    if mode not in {"cpu", "gpu"}:
        raise ValueError(f"Invalid backend tier: {mode}")
    env = dict(inherited)
    env["MINI_LLM_BACKEND"] = "numpy" if mode == "cpu" else "cupy"
    env["PYTHONPATH"] = str(ROOT) + os.pathsep + env.get("PYTHONPATH", "")
    # Do not silently inherit a custom profile from an unrelated experiment.
    # The exact test suite uses explicit profiles, and CUDA kernel opt-ins
    # are set by individual tests using monkeypatch when required.
    for name in tuple(env):
        if name.startswith("MINI_LLM_") and name != "MINI_LLM_BACKEND":
            del env[name]
    return env


def suites_for(mode: str) -> tuple[tuple[str, str, tuple[str, ...]], ...]:
    if mode not in {"cpu", "gpu", "all"}:
        raise ValueError(f"Invalid regression mode: {mode}")
    items = []
    if mode in {"cpu", "all"}:
        items.extend(("cpu", key, paths) for key, paths in CPU_SUITES.items())
    if mode in {"gpu", "all"}:
        items.extend(("gpu", key, paths) for key, paths in GPU_SUITES.items())
    return tuple(items)


def gpu_preflight(env: dict[str, str]) -> None:
    """Do not report success when missing CuPy would merely skip GPU tests."""
    result = subprocess.run(
        [sys.executable, "-c",
         "import cupy as cp; assert cp.cuda.runtime.getDeviceCount()>0; "
         "import triton; print(cp.cuda.runtime.getDeviceProperties(0)['name'])"],
        cwd=ROOT, env=env, capture_output=True, text=True,
    )
    if result.returncode:
        raise RuntimeError(
            "GPU regression requires working CuPy, Triton, and a visible CUDA device.\n"
            + result.stdout + result.stderr
        )


def gpu_memory_headroom(env: dict[str, str]) -> dict[str, float]:
    """Read device memory in a fresh process, without importing CuPy here.

    Query before each GPU suite: a concurrently running training job can
    change available VRAM between individual test processes.
    """
    snippet = (
        "import json, cupy as cp; "
        "free, total = cp.cuda.runtime.memGetInfo(); "
        "print(json.dumps({'free_gib':round(free/2**30,3), "
        "'total_gib':round(total/2**30,3)}))"
    )
    result = subprocess.run(
        [sys.executable, "-c", snippet], cwd=ROOT, env=env,
        capture_output=True, text=True,
    )
    if result.returncode:
        raise RuntimeError("CUDA memory query failed: " + result.stderr.strip())
    return json.loads(result.stdout.strip())


def run(mode: str, *, report: Path | None = None, extra: tuple[str, ...] = (),
        dry_run: bool = False) -> int:
    suites = suites_for(mode)
    results = []
    started = time.time()
    gpu_checked = False
    for tier, name, selectors in suites:
        env = environment_for(tier, os.environ)
        command = [sys.executable, "-m", "pytest", "-q", *selectors, *extra]
        if dry_run:
            print(f"[{tier}/{name}] {' '.join(command)}", flush=True)
            continue
        if tier == "gpu" and not gpu_checked:
            try:
                gpu_preflight(env)
            except RuntimeError as exc:
                print(f"GPU preflight FAILED: {exc}", file=sys.stderr, flush=True)
                results.append({"tier": tier, "suite": "gpu_preflight", "status": "failed",
                                "reason": str(exc)})
                break
            gpu_checked = True
        memory = None
        if tier == "gpu":
            try:
                memory = gpu_memory_headroom(env)
            except (RuntimeError, ValueError) as exc:
                print(f"GPU memory query FAILED: {exc}", file=sys.stderr, flush=True)
                results.append({"tier": tier, "suite": name,
                                "status": "blocked", "reason": str(exc)})
                break
            print(f"GPU headroom before {name}: "
                  f"{memory['free_gib']:.2f}/{memory['total_gib']:.2f} GiB free",
                  flush=True)
            if memory["free_gib"] < 2.0:
                reason = (f"Only {memory['free_gib']:.2f} GiB GPU memory is free. "
                          "Stop the training/serving process using the GPU, then "
                          "rerun the GPU regression gate. No parity result was obtained.")
                print(f"GPU regression BLOCKED: {reason}", file=sys.stderr,
                      flush=True)
                results.append({"tier": tier, "suite": name,
                                "status": "blocked", "reason": reason,
                                "gpu_memory": memory})
                break
        print(f"\n=== {tier}/{name} ===", flush=True)
        t0 = time.monotonic()
        proc = subprocess.run(command, cwd=ROOT, env=env)
        seconds = round(time.monotonic() - t0, 3)
        state = "passed" if proc.returncode == 0 else "failed"
        results.append({"tier": tier, "suite": name, "status": state,
                        "returncode": proc.returncode, "seconds": seconds,
                        "selectors": selectors,
                        **({"gpu_memory": memory} if memory is not None else {})})
        if proc.returncode:
            print(f"Regression gate FAILED: {tier}/{name}", file=sys.stderr, flush=True)
            break

    if report is not None and not dry_run:
        payload = {
            "schema": "0103A-regression-v1",
            "model": "model_b_348m_sparse16",
            "started_utc_epoch": started,
            "mode": mode,
            "complete": len(results) == len(suites) and
                        all(row["status"] == "passed" for row in results),
            "suites": results,
            "note": "Test-pass statuses are correctness gates, NOT proof of fast inference. "
                    "The CUDA Graph GPU test asserts capture but does not measure throughput.",
        }
        report.parent.mkdir(parents=True, exist_ok=True)
        report.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
        print(f"Report: {report}", flush=True)
    return 0 if dry_run or (len(results) == len(suites) and
                           all(row["status"] == "passed" for row in results)) else 1


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mode", choices=("cpu", "gpu", "all"), default="cpu")
    parser.add_argument("--report", type=Path, help="Optional JSON validation record")
    parser.add_argument("--list", action="store_true", help="Print suite commands only")
    args = parser.parse_args()
    return run(args.mode, report=args.report, dry_run=args.list)


if __name__ == "__main__":
    raise SystemExit(main())
