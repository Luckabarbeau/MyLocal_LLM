#!/usr/bin/env python3
"""Probe independent native CuPy workers; the coordinator uses only stdlib."""

import argparse
import json
import os
from pathlib import Path
import subprocess
import sys


ROOT = Path(__file__).resolve().parents[1]
TIMEOUT_SECONDS = 120
PROBE = r'''
import importlib.metadata as metadata
import json
import os
import sys

# Inspect Triton's distribution without importing a driver or framework.
providers = metadata.packages_distributions().get("triton", [])
if not providers:
    raise RuntimeError("No installed distribution provides triton")
triton_versions = {name: metadata.version(name) for name in providers}

import cupy as cp

expected = sys.argv[1]
device = int(sys.argv[2])
runtime = "hip" if cp.cuda.runtime.is_hip else "cuda"
if runtime != expected:
    raise RuntimeError(f"Expected {expected} CuPy runtime, found {runtime}")
count = cp.cuda.runtime.getDeviceCount()
if not 0 <= device < count:
    raise RuntimeError(f"Device {device} outside visible device count {count}")
with cp.cuda.Device(device):
    props = cp.cuda.runtime.getDeviceProperties(device)
    a = cp.ones((16, 16), dtype=cp.float32)
    result = a @ a
    cp.cuda.get_current_stream().synchronize()
    error = float(cp.max(cp.abs(result - 16.0)).item())
    if not bool(cp.all(cp.isfinite(result)).item()) or error != 0.0:
        raise RuntimeError(f"FP32 matmul failed: max absolute error {error}")
    def text(value):
        return value.decode(errors="replace") if isinstance(value, bytes) else value
    print(json.dumps({
        "python": sys.executable,
        "backend": os.environ["MINI_LLM_BACKEND"],
        "runtime": runtime,
        "cupy_version": cp.__version__,
        "triton_distributions": triton_versions,
        "device": device,
        "visible_device_count": count,
        "device_name": text(props.get("name", "")),
        "architecture": text(props.get("gcnArchName", "")),
        "runtime_version": cp.cuda.runtime.runtimeGetVersion(),
        "matmul_max_abs_error": error,
    }), flush=True)
'''


def load_workers(path):
    with path.open(encoding="utf-8") as source:
        config = json.load(source)
    workers = config.get("workers") if isinstance(config, dict) else None
    if not isinstance(workers, list) or not workers:
        raise ValueError("config must contain a nonempty workers list")
    names = set()
    for worker in workers:
        if not isinstance(worker, dict):
            raise ValueError("each worker must be an object")
        name = worker.get("name")
        if not isinstance(name, str) or not name.strip() or name in names:
            raise ValueError("worker names must be nonempty, unique strings")
        names.add(name)
        python = worker.get("python")
        if not isinstance(python, str) or not python.strip():
            raise ValueError(f"{name}: python must be an interpreter path")
        if worker.get("expected_runtime") not in ("cuda", "hip"):
            raise ValueError(f"{name}: expected_runtime must be cuda or hip")
        device = worker.get("device", 0)
        if type(device) is not int or device < 0:
            raise ValueError(f"{name}: device must be a nonnegative integer")
        env = worker.get("env", {})
        if not isinstance(env, dict) or any(
            not isinstance(key, str) or not key or "=" in key or "\0" in key
            or not isinstance(value, str) or "\0" in value
            for key, value in env.items()
        ):
            raise ValueError(f"{name}: env must map environment names to strings")
        if env.get("MINI_LLM_BACKEND", "cupy") != "cupy":
            raise ValueError(f"{name}: MINI_LLM_BACKEND must be cupy")
    return workers


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True, type=Path, help="JSON worker configuration")
    args = parser.parse_args()
    try:
        workers = load_workers(args.config)
    except (OSError, ValueError) as exc:
        parser.error(str(exc))

    results = []
    for worker in workers:
        interpreter = Path(worker["python"]).expanduser()
        if not interpreter.is_absolute():
            interpreter = ROOT / interpreter
        env = os.environ.copy()
        env.update(worker.get("env", {}))
        env["MINI_LLM_BACKEND"] = "cupy"
        record = {"name": worker["name"], "ok": False}
        try:
            completed = subprocess.run(
                [str(interpreter), "-u", "-c", PROBE,
                 worker["expected_runtime"], str(worker.get("device", 0))],
                cwd=ROOT, env=env, capture_output=True, text=True,
                timeout=TIMEOUT_SECONDS, check=False,
            )
            record.update(returncode=completed.returncode,
                          stdout=completed.stdout, stderr=completed.stderr)
            if completed.returncode == 0:
                record["probe"] = json.loads(completed.stdout.strip().splitlines()[-1])
                record["ok"] = True
        except subprocess.TimeoutExpired as exc:
            def decoded(value):
                return value.decode(errors="replace") if isinstance(value, bytes) else value or ""
            record.update(error=f"Timed out after {TIMEOUT_SECONDS} seconds",
                          stdout=decoded(exc.stdout), stderr=decoded(exc.stderr))
        except (OSError, ValueError, IndexError) as exc:
            record["error"] = str(exc)
        results.append(record)
    ok = all(record["ok"] for record in results)
    print(json.dumps({"ok": ok, "workers": results}, indent=2))
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
