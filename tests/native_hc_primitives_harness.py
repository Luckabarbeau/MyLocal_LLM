"""Standalone WP4 CUDA test; no runtime/CMake/binding integration.

Run: MINI_LLM_BACKEND=cupy .venv/bin/python tests/native_hc_primitives_harness.py
Builds a temporary test-only C shim and shared library with bounded nvcc.
"""
import ctypes as ct
import os
from pathlib import Path
import subprocess
import sys
import tempfile

import cupy as cp
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from mini_llm.backend import xp
from mini_llm.external.qwen4exp_hyperconnection import Qwen4ExpHyperConnection

SHIM = r'''
#include "hyperconnection.h"
extern "C" int hc_read(cudaStream_t s, int d, int r, float e,
 const float* h, const float* g, const float* dn, const float* up,
 const float* inj, float* n, float* low, float* mix, float* gain, int* st) {
 return ml_hc_read_launch(s,d,r,e,h,g,dn,up,inj,n,low,mix,gain,st);
}
extern "C" int hc_collapse(cudaStream_t s, int d, int r, float e,
 const float* h, const float* g, const float* dn, const float* up,
 float* n, float* low, float* mix, int* st) {
 return ml_hc_collapse_launch(s,d,r,e,h,g,dn,up,n,low,mix,st);
}
extern "C" int hc_write(cudaStream_t s, int d, const float* h,
 const float* out, const float* gain, float* updated, int* st) {
 return ml_hc_write_launch(s,d,h,out,gain,updated,st);
}
'''


def ptr(a):
    return None if a is None else a.data.ptr


def oracle(h, gamma, down, up, inject, epsilon):
    """Independent FP64 graph, not CUDA reduction emulation."""
    h, gamma, down, up, inject = [cp.asnumpy(a).astype(np.float64)
                                 for a in (h, gamma, down, up, inject)]
    n = (h / np.sqrt(np.mean(h * h, axis=-1, keepdims=True) + float(epsilon))) * gamma
    sigmoid = lambda a: np.where(a >= 0, 1 / (1 + np.exp(-np.abs(a))),
                                np.exp(-np.abs(a)) / (1 + np.exp(-np.abs(a))))
    a = down @ n.reshape(-1) * 0.25
    low = a * sigmoid(a)
    mix = np.sum(n * sigmoid(up @ low).reshape(h.shape), axis=0) * 0.25
    gains = 2 * sigmoid(inject @ n.reshape(-1) * 0.25)
    return mix, gains


MAX_ERROR = 0.0


def close(actual, expected, label, exact=False):
    global MAX_ERROR
    actual = cp.asnumpy(actual)
    if isinstance(expected, cp.ndarray):
        expected = cp.asnumpy(expected)
    error = float(np.max(np.abs(actual - expected)))
    MAX_ERROR = max(MAX_ERROR, error)
    if exact:
        np.testing.assert_array_equal(actual, expected, err_msg=label)
    else:
        np.testing.assert_allclose(actual, expected, atol=2e-6, rtol=2e-6, err_msg=label)


class Case:
    def __init__(self, lib, d, rank, kind="random", seed=410):
        self.lib, self.d, self.rank = lib, d, rank
        self.stream = cp.cuda.Stream(non_blocking=True)
        rng = np.random.default_rng(seed)
        with self.stream:
            def arr(shape, scale):
                return cp.asarray(rng.normal(0, scale, shape).astype(np.float32))
            self.h = arr((4, d), 0.7)
            self.gamma = cp.asarray(rng.uniform(0.3, 1.7, (4, d)).astype(np.float32))
            self.down = arr((rank, 4 * d), 0.2 / np.sqrt(d))
            self.up = arr((4 * d, rank), 0.4 / np.sqrt(rank))
            self.inject = arr((4, 4 * d), 0.3 / np.sqrt(d))
            self.epsilon = np.float32(1e-5)
            if kind == "exact":
                self.h.fill(1)
                self.gamma[:] = cp.asarray([[2], [4], [6], [8]], dtype=cp.float32)
                self.down.fill(0)
                self.up.fill(0)
                self.inject.fill(0)
                self.epsilon = np.float32(3)
            elif kind == "saturated":
                self.h.fill(1)
                self.gamma.fill(1)
                self.down.fill(4000)
                self.up[:] = cp.asarray([[4000], [-4000], [4000], [-4000]], dtype=cp.float32)
                self.inject[:] = cp.asarray([[4000], [-4000], [4000], [-4000]], dtype=cp.float32)
            elif kind == "zero":
                self.h.fill(0)
            self.n = cp.empty_like(self.h)
            self.low = cp.empty(rank, dtype=cp.float32)
            self.mixed = cp.empty(d, dtype=cp.float32)
            self.gains = cp.empty(4, dtype=cp.float32)
            self.updated = cp.empty_like(self.h)
            self.status = cp.zeros(1, dtype=cp.int32)
        self.stream.synchronize()

    def read(self, **overrides):
        args = dict(d=self.d, rank=self.rank, epsilon=self.epsilon,
                    h=self.h, gamma=self.gamma, down=self.down, up=self.up,
                    inject=self.inject, n=self.n, low=self.low, mixed=self.mixed,
                    gains=self.gains, status=self.status)
        args.update(overrides)
        return self.lib.hc_read(self.stream.ptr, args.pop("d"), args.pop("rank"),
                                args.pop("epsilon"), *map(ptr, args.values()))

    def collapse(self):
        return self.lib.hc_collapse(self.stream.ptr, self.d, self.rank, self.epsilon,
            *map(ptr, (self.h, self.gamma, self.down, self.up, self.n, self.low,
                       self.mixed, self.status)))

    def write(self, output):
        return self.lib.hc_write(self.stream.ptr, self.d,
            *map(ptr, (self.h, output, self.gains, self.updated, self.status)))

    def finish(self, expected_status=0):
        self.stream.synchronize()
        assert int(self.status.get()[0]) == expected_status


def numerical(lib, d, rank, kind):
    c = Case(lib, d, rank, kind)
    hc = Qwen4ExpHyperConnection(c.gamma, c.down, c.up, c.inject, epsilon=c.epsilon)
    inputs = (c.h, c.gamma, c.down, c.up, c.inject)
    snapshots = [a.copy() for a in inputs]
    for block in range(2):
        expected, gains = hc.read(c.h)
        mathematical, math_gains = oracle(*inputs, c.epsilon)
        cp.cuda.get_current_stream().synchronize()
        assert c.read() == 0
        c.finish()
        close(c.mixed, expected, "Python FP32 read")
        close(c.gains, gains, "Python FP32 gains")
        close(c.mixed, mathematical, "FP64 read")
        close(c.gains, math_gains, "FP64 gains")
        if block == 0 and kind == "exact":
            close(c.mixed, np.full(d, 1.25), "converted gamma/grouped RMS", exact=True)
            close(c.gains, np.ones(4), "zero injection", exact=True)
        if block == 0 and kind == "saturated":
            close(c.gains, [2, 0, 2, 0], "saturated gains", exact=True)
        with c.stream:
            output = cp.tanh(c.mixed) * cp.float32(block + 0.75)
            saved_output, saved_gains = output.copy(), c.gains.copy()
            expected_write = c.h + c.gains[:, None] * output[None, :]
        assert c.write(output) == 0
        c.finish()
        close(c.updated, expected_write, "original residual write", exact=True)
        close(output, saved_output, "output immutable", exact=True)
        close(c.gains, saved_gains, "gains immutable", exact=True)
        for a, saved in zip(inputs, snapshots):
            close(a, saved, "read/write inputs immutable", exact=True)
        # Collapse consumes no injection; compare both explicit and null-inject read.
        expected_collapse = hc.collapse(c.updated)
        cp.cuda.get_current_stream().synchronize()
        original = c.h
        c.h = c.updated
        assert c.collapse() == 0
        c.finish()
        close(c.mixed, expected_collapse, "final collapse")
        saved_collapse = c.mixed.copy()
        cp.cuda.get_current_stream().synchronize()
        assert c.read(inject=None, gains=None) == 0
        c.finish()
        close(c.mixed, saved_collapse, "null injection read equals collapse", exact=True)
        # Advance only after testing immutable original h; keep output disjoint.
        c.h, c.updated = c.updated, original
        inputs = (c.h, c.gamma, c.down, c.up, c.inject)
        snapshots = [a.copy() for a in inputs]
        cp.cuda.get_current_stream().synchronize()


def validation_and_errors(lib):
    c = Case(lib, 3, 2)
    invalid = [{"d": 0}, {"d": 2**29}, {"rank": 0}, {"rank": 2**31 - 1},
               *({"epsilon": e} for e in (0, -1, np.nan, np.inf)),
               *({name: None} for name in ("h", "gamma", "down", "up", "n", "low",
                                            "mixed", "status", "inject", "gains"))]
    for args in invalid:
        assert c.read(**args) == 1, args  # cudaErrorInvalidValue
        c.finish()
    assert lib.hc_write(c.stream.ptr, 0, *map(ptr, (c.h, c.mixed, c.gains, c.updated, c.status))) == 1
    c.finish()
    # Positive subnormal epsilon is supported: no fast-math flush-to-zero.
    assert c.read(epsilon=np.nextafter(np.float32(0), np.float32(1))) == 0
    c.finish()
    for name in ("h", "gamma", "down", "up", "inject"):
        for bad in (np.nan, np.inf, -np.inf):
            c = Case(lib, 3, 2)
            with c.stream:
                getattr(c, name).reshape(-1)[0] = bad
            assert c.read() == 0
            c.finish(1)
            # Clean subsequent arithmetic never clears a previous error.
            with c.stream:
                getattr(c, name).reshape(-1)[0] = 0
            assert c.read() == 0
            c.finish(1)
    # Finite source data can overflow before sigmoid masks an infinite logit.
    for name in ("h", "gamma", "down", "up", "inject"):
        c = Case(lib, 3, 2)
        with c.stream:
            c.h.fill(1)
            c.gamma.fill(2)
            c.down.fill(1)
            getattr(c, name).fill(np.finfo(np.float32).max)
        assert c.read() == 0
        c.finish(1)
    for source in ("h", "mixed", "gains"):
        c = Case(lib, 3, 2)
        assert c.read() == 0
        with c.stream:
            c.mixed.fill(1)
            getattr(c, source).reshape(-1)[0] = np.inf
        assert c.write(c.mixed) == 0
        c.finish(1)
    c = Case(lib, 3, 2)
    with c.stream:
        c.mixed.fill(np.finfo(np.float32).max)
        c.gains.fill(2)
    assert c.write(c.mixed) == 0
    c.finish(1)


def main():
    assert xp is cp, "Run with MINI_LLM_BACKEND=cupy (no silent backend substitute)"
    with tempfile.TemporaryDirectory(prefix="wp4-hc-") as tmp:
        shim, so = Path(tmp) / "shim.cu", Path(tmp) / "hc.so"
        shim.write_text(SHIM)
        command = [os.environ.get("NVCC", "/usr/local/cuda/bin/nvcc"),
                   "-std=c++17", "-O2", "--fmad=false", "-ftz=false", "-prec-div=true",
                   "-prec-sqrt=true", "-arch=sm_120", "--shared", "-Xcompiler", "-fPIC",
                   "-I", str(ROOT / "mini_llm/native_cpp"),
                   str(ROOT / "mini_llm/native_cpp/hyperconnection.cu"), str(shim), "-o", str(so)]
        print("BUILD:", " ".join(command), flush=True)
        subprocess.run(command, check=True, timeout=120)
        lib = ct.CDLL(str(so))
        lib.hc_read.argtypes = [ct.c_void_p, ct.c_int, ct.c_int, ct.c_float] + [ct.c_void_p] * 10
        lib.hc_collapse.argtypes = [ct.c_void_p, ct.c_int, ct.c_int, ct.c_float] + [ct.c_void_p] * 8
        lib.hc_write.argtypes = [ct.c_void_p, ct.c_int] + [ct.c_void_p] * 5
        for fn in (lib.hc_read, lib.hc_collapse, lib.hc_write):
            fn.restype = ct.c_int
        for device in range(cp.cuda.runtime.getDeviceCount()):
            with cp.cuda.Device(device):
                for d, rank, kind in [(1, 1, "exact"), (1, 1, "saturated"),
                                       (7, 5, "zero"), (3, 2, "random"),
                                       (33, 35, "random"), (257, 257, "random"),
                                       (2560, 32, "random"), (2560, 320, "random")]:
                    numerical(lib, d, rank, kind)
                validation_and_errors(lib)
                print(f"PASS device={device}: paired blocks/collapse, FP32/FP64, "
                      "nondefault stream, immutability, host validation, sticky finite errors", flush=True)
        print(f"PASS max_abs_error={MAX_ERROR:.9g}; rtol=atol=2e-6; no runtime integration")


if __name__ == "__main__":
    main()
