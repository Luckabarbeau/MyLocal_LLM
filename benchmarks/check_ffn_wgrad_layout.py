#!/usr/bin/env python3
"""Check FFN down-wgrad layout; default is CPU/mocks only.

Use --gpu explicitly for HIP BF16 numerical and full ExpertFFN checks. No timing
claims are made here. References use the actual rounded BF16 operands, not their
pre-cast random sources. Only normwise tolerances are imposed.
"""
import argparse
from contextlib import contextmanager
import json
import os
from pathlib import Path
import sys
from types import SimpleNamespace
from unittest.mock import patch

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
OPTION = "MINI_LLM_HIP_FFN_TRANSPOSE_WGRAD"


@contextmanager
def option(value):
    with patch.dict(os.environ, {OPTION: str(value)}):
        yield


def report(record):
    print(json.dumps(record, allow_nan=False), flush=True)


def dispatch_checks(experts):
    """Record exact transpose/matmul call sequence without allocating a GPU."""
    calls = []

    class Operand:
        def __init__(self, name, shape, dtype="bfloat16"):
            self.name, self.shape, self.dtype = name, shape, dtype
            self.ndim = len(shape)

        @property
        def T(self):
            calls.append(["transpose", self.name])
            return Operand(self.name + ".T", self.shape[::-1], self.dtype)

        def __matmul__(self, other):
            calls.append(["matmul", self.name, other.name])
            return Operand("product", (4096, 1024), self.dtype)

    cases = [
        ("default_off", None, "cupy", True, (8192, 4096), (8192, 1024), "bfloat16", "bfloat16", False),
        ("short", "1", "cupy", True, (1024, 4096), (1024, 1024), "bfloat16", "bfloat16", False),
        ("8192", "1", "cupy", True, (8192, 4096), (8192, 1024), "bfloat16", "bfloat16", True),
        ("16384", "1", "cupy", True, (16384, 4096), (16384, 1024), "bfloat16", "bfloat16", True),
        ("odd", "1", "cupy", True, (8193, 4096), (8193, 1024), "bfloat16", "bfloat16", True),
        ("cuda", "1", "cupy", False, (8192, 4096), (8192, 1024), "bfloat16", "bfloat16", False),
        ("cpu", "1", "numpy", True, (8192, 4096), (8192, 1024), "bfloat16", "bfloat16", False),
        ("route_fp32", "1", "cupy", True, (8192, 4096), (8192, 1024), "bfloat16", "float32", False),
        ("h_fp32", "1", "cupy", True, (8192, 4096), (8192, 1024), "float32", "bfloat16", False),
        ("width", "1", "cupy", True, (8192, 2048), (8192, 1024), "bfloat16", "bfloat16", False),
        ("dy_width", "1", "cupy", True, (8192, 4096), (8192, 512), "bfloat16", "bfloat16", False),
        ("rows", "1", "cupy", True, (8192, 4096), (8193, 1024), "bfloat16", "bfloat16", False),
        ("rank_h", "1", "cupy", True, (1, 8192, 4096), (8192, 1024), "bfloat16", "bfloat16", False),
        ("rank_dy", "1", "cupy", True, (8192, 4096), (1, 8192, 1024), "bfloat16", "bfloat16", False),
    ]
    for name, enabled, backend, hip, hs, ds, hd, dd, swapped in cases:
        calls.clear()
        fake_xp = SimpleNamespace(cuda=SimpleNamespace(runtime=SimpleNamespace(is_hip=hip)))
        with patch.dict(os.environ), patch.object(experts, "BACKEND_NAME", backend), patch.object(experts, "xp", fake_xp):
            if enabled is None:
                os.environ.pop(OPTION, None)
            else:
                os.environ[OPTION] = enabled
            experts._down_projection_wgrad(Operand("h", hs, hd), Operand("dy", ds, dd))
        expected = ([["transpose", "dy"], ["matmul", "dy.T", "h"], ["transpose", "product"]]
                    if swapped else [["transpose", "h"], ["matmul", "h.T", "dy"]])
        assert calls == expected, (name, calls, expected)
    report({"check": "mock_dispatch", "cases": len(cases), "passed": True})


def cpu_check(experts):
    h = np.arange(15, dtype=np.float32).reshape(5, 3) / 10
    dy = np.arange(10, dtype=np.float32).reshape(5, 2) / 10
    with patch.object(experts, "BACKEND_NAME", "numpy"), option(1):
        actual = experts._down_projection_wgrad(h, dy)
    assert np.array_equal(actual, h.T @ dy)
    report({"check": "tiny_cpu_fallback", "passed": True})


def gpu_checks(experts):
    import cupy as cp
    from mini_llm.backend import RandomStream, resolve_dtype

    if not cp.cuda.runtime.is_hip:
        raise RuntimeError("--gpu requires HIP, not CUDA")
    bf16 = resolve_dtype("bfloat16")
    rng = cp.random.RandomState(9700)

    def random(shape, std=0.1, dtype=bf16):
        return (rng.standard_normal(shape, dtype=cp.float32) * std).astype(dtype)

    def host(a):
        return cp.asnumpy(a.astype(cp.float32))

    def compare(label, actual, reference, tolerance):
        # Host FP64 norms avoid BF16 reduction and FP32 norm roundoff.
        a = np.asarray(actual, dtype=np.float64)
        r = np.asarray(reference, dtype=np.float64)
        delta = a - r
        absolute = float(np.linalg.norm(delta))
        magnitude = float(np.linalg.norm(r))
        relative = absolute / max(magnitude, 1e-300)
        report({"check": label, "absolute_l2": absolute, "reference_l2": magnitude,
                "relative_l2": relative, "max_absolute": float(np.max(np.abs(delta))),
                "tolerance_relative_l2": tolerance})
        assert np.isfinite(relative) and relative < tolerance, label

    for t, strided in [(1024, False), (8192, False), (16384, False), (8193, False), (8192, True)]:
        # Row slices retain a noncontiguous row stride in both operands.
        h = random((t * (2 if strided else 1), 4096))[::2 if strided else 1]
        dy = random((t * (2 if strided else 1), 1024))[::2 if strided else 1]
        label = f"T{t}_strided{strided}"
        if strided:
            assert not h.flags.c_contiguous and not dy.flags.c_contiguous
        original = h.T @ dy
        with option(1):
            candidate = experts._down_projection_wgrad(h, dy)
            repeated = experts._down_projection_wgrad(h, dy)
        assert candidate.dtype == original.dtype == np.dtype(bf16)
        assert bool(cp.array_equal(candidate, repeated)), label + " determinism"
        if t < 8192:
            assert bool(cp.array_equal(candidate, original)), label + " fallback"
        compare(label + "_vs_original_bf16", host(candidate), host(original), 0.001)
        reference = h.astype(cp.float32).T @ dy.astype(cp.float32)
        compare(label + "_vs_actual_bf16_fp32_reference", host(candidate), host(reference), 0.003)
        initial = random((4096, 1024), std=0.01, dtype=cp.float32)
        accumulated = initial.copy()
        expected = initial.copy()
        with option(1):
            for _ in range(2):
                accumulated += experts._down_projection_wgrad(h, dy)
                expected += original
        compare(label + "_two_accumulations", host(accumulated), host(expected), 0.001)
        assert accumulated.dtype == cp.float32
        del h, dy, original, candidate, repeated, reference, initial, accumulated, expected
        cp.get_default_memory_pool().free_all_blocks()

    # Reuse identical parameters, x, dY, and nonzero starting grads for both
    # settings; run sequentially and keep comparison snapshots on the host.
    expert = experts.ExpertFFN(1024, 4096, 0.02, 0.02, RandomStream(9701), dtype="bfloat16")
    x = random((8192, 1024))
    dy = random((8192, 1024), std=0.001)
    initial_grads = [host(random(p.data.shape, std=0.0001, dtype=cp.float32))
                     for p in expert.parameters()]
    baseline = None
    for enabled in (0, 1):
        for p, initial in zip(expert.parameters(), initial_grads):
            p.grad[...] = cp.asarray(initial)
        snapshots = []
        with option(enabled):
            for call in range(2):
                y, cache = expert.forward(x)
                loss = float(cp.sum(y.astype(cp.float32) * dy.astype(cp.float32), dtype=cp.float64).item())
                dx = expert.backward(dy, cache)
                snapshots.append((host(dx), [host(p.grad) for p in expert.parameters()], loss))
                del y, cache, dx
        if baseline is None:
            baseline = snapshots
        else:
            for call, ((dx, grads, loss), (ref_dx, ref_grads, ref_loss)) in enumerate(zip(snapshots, baseline)):
                compare(f"expert_call{call}_dx", dx, ref_dx, 0.001)
                for name, grad, ref_grad in zip(("gate", "up", "down"), grads, ref_grads):
                    compare(f"expert_call{call}_{name}_grad", grad, ref_grad, 0.001)
                report({"check": f"expert_call{call}_loss_probe", "actual": loss,
                        "reference": ref_loss, "absolute_error": abs(loss - ref_loss)})
                assert loss == ref_loss
    report({"check": "hip_suite", "seed": 9700, "passed": True})


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--gpu", action="store_true", help="explicitly execute HIP GPU checks")
    args = parser.parse_args()
    # Select the backend before importing any mini_llm modules.
    os.environ["MINI_LLM_BACKEND"] = "cupy" if args.gpu else "numpy"
    from mini_llm.ops import experts
    dispatch_checks(experts)
    cpu_check(experts)
    if args.gpu:
        gpu_checks(experts)
    report({"check": "complete", "gpu_executed": args.gpu, "passed": True})


if __name__ == "__main__":
    try:
        main()
    except Exception as exc:
        report({"passed": False, "error_type": type(exc).__name__, "error": str(exc)})
        raise
