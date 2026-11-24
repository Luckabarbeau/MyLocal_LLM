"""Bounded standalone WP4 PLE checks; no runtime, codebook, CMake or binding.

MINI_LLM_BACKEND=numpy .venv/bin/python tests/native_ple_primitives_harness.py
Append --hash-only for a host compiler only. CUDA shim exists only in a temp dir.
"""
import argparse
import ctypes as ct
import os
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from mini_llm.backend import asnumpy, xp
from mini_llm.external.qwen4exp_ple import PLEConfig, Qwen4ExpPLE

SRC = ROOT / "mini_llm/native_cpp"
D = 2560


class Config(ct.Structure):
    _fields_ = [("multipliers", ct.c_uint64 * 3), ("offsets", ct.c_uint64 * 16),
                ("vocabs", ct.c_uint64 * 16), ("eos", ct.c_int32),
                ("table_rows", ct.c_uint64)]


HASH_SHIM = r'''
#include "ple_hash.h"
extern "C" int hash_rows(const MlPleHashConfig* c, int32_t t,
 const int32_t* p, uint64_t* r, int32_t* n) {
 return ml_ple_hash_rows(*c,t,p,r,n);
}
'''
CUDA_SHIM = r'''
#include "ple.h"
extern "C" int apply(cudaStream_t s, float e, const float* h, const float* k,
 const float* v, const float* gk, const float* gq, const float* gc,
 const float* w, const float* hist, float* kn, float* qn, float* u,
 float* gates, float* z, float* out, float* next, int* st) {
 return ml_ple_apply_launch(s,e,h,k,v,gk,gq,gc,w,hist,kn,qn,u,gates,z,out,next,st);
}
extern "C" int project(cudaStream_t s, const float* e, const float* kw,
 const float* vw, float* k, float* v, int* st) {
 return ml_ple_project_launch(s,e,kw,vw,k,v,st);
}
'''


def hash_checks():
    with tempfile.TemporaryDirectory(prefix="ple_hash_") as tmp:
        shim, libpath = Path(tmp)/"shim.cpp", Path(tmp)/"hash.so"
        shim.write_text(HASH_SHIM)
        subprocess.run([os.environ.get("CXX", "c++"), "-std=c++17", "-shared",
                        "-fPIC", "-O2", "-Wall", "-Wextra", "-Werror", "-I", str(SRC),
                        str(shim), str(SRC/"ple_hash.cpp"), "-o", str(libpath)],
                       check=True, timeout=60)
        lib = ct.CDLL(str(libpath))
        lib.hash_rows.argtypes = [ct.POINTER(Config), ct.c_int32,
                                 ct.c_void_p, ct.c_void_p, ct.c_void_p]
        lib.hash_rows.restype = ct.c_int
        rng = np.random.default_rng(416)
        cases = 0
        for multipliers in ((2**64-1, 20109073645365, 8052911324071),
                            (0, 0, 0), (2**63+1, 2**64-3, 2**64-5)):
            vocabs = tuple(100003 + 2*h for h in range(16))
            offsets = tuple(h*200003 for h in range(16))
            c = PLEConfig(multipliers, offsets, vocabs, 248044, 4000000)
            raw = Config((ct.c_uint64*3)(*c.multipliers),
                         (ct.c_uint64*16)(*c.offsets),
                         (ct.c_uint64*16)(*c.vocabs), c.eos, c.table_rows)
            previous = (-1, -1)
            tokens = [0, 7, 9, c.eos, 3, 248046, 2**31-1, c.eos, c.eos, 11]
            tokens += rng.integers(0, 2**31, size=200).tolist()
            for token in tokens:
                p = np.array(previous, dtype=np.int32)
                rows = np.full(16, 99, dtype=np.uint64)
                nxt = np.full(2, -99, dtype=np.int32)
                assert lib.hash_rows(ct.byref(raw), token, p.ctypes.data,
                                     rows.ctypes.data, nxt.ctypes.data) == 1
                np.testing.assert_array_equal(rows, c.rows(token, previous))
                np.testing.assert_array_equal(nxt, (previous[1], token))
                np.testing.assert_array_equal(p, previous)
                previous = tuple(nxt.tolist())
                cases += 1
            # A missing immediate predecessor cuts an otherwise valid older token.
            for previous in ((123, -1), (123, c.eos), (-1, 123), (c.eos, 123)):
                p = np.array(previous, dtype=np.int32)
                assert lib.hash_rows(ct.byref(raw), 17, p.ctypes.data,
                                     rows.ctypes.data, nxt.ctypes.data) == 1
                np.testing.assert_array_equal(rows, c.rows(17, previous))
                cases += 1
            # Validate before writing, including overflow-safe offset/vocab range.
            for field, value in (("table_rows", 0), ("eos", -1)):
                saved = getattr(raw, field)
                setattr(raw, field, value)
                rows.fill(99); nxt.fill(-99)
                assert lib.hash_rows(ct.byref(raw), 1, p.ctypes.data,
                                     rows.ctypes.data, nxt.ctypes.data) == 0
                np.testing.assert_array_equal(rows, np.full(16, 99))
                np.testing.assert_array_equal(nxt, [-99, -99])
                setattr(raw, field, saved)
            raw.offsets[0] = 2**64-1
            assert lib.hash_rows(ct.byref(raw), 1, p.ctypes.data,
                                 rows.ctypes.data, nxt.ctypes.data) == 0
        print(f"PASS exact host hash/state: {cases} wrap/EOS/missing/random cases")


def nvcc_path():
    return os.environ.get("NVCC") or shutil.which("nvcc") or \
        ("/usr/local/cuda/bin/nvcc" if Path("/usr/local/cuda/bin/nvcc").exists() else None)


def cuda_checks():
    import cupy as cp
    compiler = nvcc_path()
    if compiler is None:
        raise RuntimeError("nvcc unavailable; use --hash-only or set NVCC")
    with tempfile.TemporaryDirectory(prefix="ple_cuda_") as tmp:
        shim, libpath = Path(tmp)/"shim.cu", Path(tmp)/"ple.so"
        shim.write_text(CUDA_SHIM)
        capability = cp.cuda.Device().compute_capability
        subprocess.run([compiler, "-std=c++17", "-shared", "-Xcompiler", "-fPIC",
                        "-O2", "--fmad=false", f"-arch=sm_{capability}",
                        "-I", str(SRC), str(shim), str(SRC/"ple.cu"),
                        "-o", str(libpath)], check=True, timeout=180)
        lib = ct.CDLL(str(libpath))
        lib.apply.argtypes = [ct.c_void_p, ct.c_float] + [ct.c_void_p]*16
        lib.apply.restype = ct.c_int
        lib.project.argtypes = [ct.c_void_p]*7
        lib.project.restype = ct.c_int
        stream = cp.cuda.Stream(non_blocking=True)
        rng = np.random.default_rng(417)
        max_error = 0.0

        def close(a, b, label, exact=False):
            nonlocal max_error
            a = cp.asnumpy(a)
            b = cp.asnumpy(b) if isinstance(b, cp.ndarray) else np.asarray(asnumpy(b))
            error = float(np.max(np.abs(a-b)))
            max_error = max(max_error, error)
            if exact:
                np.testing.assert_array_equal(a, b, err_msg=label)
            else:
                np.testing.assert_allclose(a, b, atol=8e-6, rtol=8e-6, err_msg=label)

        def random(shape):
            return rng.normal(0, 0.4, shape).astype(np.float32)

        # Real dimensions; ~125 MiB dense weights, no giant embedding table.
        emb = random((D,))
        kw = random((4*D, D)) / np.float32(np.sqrt(D))
        vw = random((D, D)) / np.float32(np.sqrt(D))
        with stream:
            de, dk, dv = map(cp.asarray, (emb, kw, vw))
            pk, pv = cp.empty((4,D), cp.float32), cp.empty(D, cp.float32)
            st = cp.zeros(1, cp.int32)
        assert lib.project(stream.ptr, *[a.data.ptr for a in (de,dk,dv,pk,pv,st)]) == 0
        stream.synchronize()
        assert st.get()[0] == 0
        close(pk, (kw@emb).reshape(4,D), "key projection")
        close(pv, vw@emb, "value projection")
        del de, dk, dv, kw, vw

        for kind in ("random", "zero", "signed", "tiny", "saturated", "tap0", "tap3", "tap6", "current"):
            h, key, value = random((4,D)), random((4,D)), random((D,))
            gammas = [rng.uniform(-1.5, 1.5, (4,D)).astype(np.float32) for _ in range(3)]
            weights, history = random((4,D,4)), random((9,4,D))
            if kind == "zero":
                h.fill(0); key.fill(0); value.fill(0)
            if kind in ("signed", "tiny", "saturated"):
                h.fill(1); key[:] = np.array([1,-1,0,1], np.float32)[:,None]
                gammas[0].fill(1); gammas[1].fill(1)
                if kind == "tiny": gammas[0] *= np.float32(1e-10)
                if kind == "saturated": gammas[0] *= np.float32(1000)
            if kind.startswith("tap") or kind == "current":
                weights.fill(0)
                weights[..., {"tap0":0,"tap3":1,"tap6":2,"current":3}[kind]] = 1
            ref = object.__new__(Qwen4ExpPLE)
            ref._closed = False
            ref.epsilon = xp.float32(1e-5)
            ref.gamma_key, ref.gamma_query, ref.gamma_conv = map(xp.asarray, gammas)
            ref.conv = xp.asarray(weights.reshape(4*D,4))
            ref.key = lambda e: xp.asarray(key.reshape(-1))
            ref.value = lambda e: xp.asarray(value)
            rh, rhist = xp.asarray(h), xp.asarray(history)
            with stream:
                inputs = [cp.asarray(a) for a in (h,key,value,*gammas,weights,history)]
                kn, qn, u, z, out = [cp.empty((4,D), cp.float32) for _ in range(5)]
                gates, nxt = cp.empty(4, cp.float32), cp.empty((9,4,D), cp.float32)
                st = cp.zeros(1, cp.int32)
            original = [a.copy() for a in (h,key,value,*gammas,weights,history)]
            # 12 steps exceed history length and cross conceptual EOS boundaries.
            # Hash/state tested separately; apply API deliberately has no EOS input.
            for step in range(12 if kind == "random" else 1):
                expected, expected_hist = ref.apply_embedding(rh, xp.zeros(D, xp.float32), rhist)
                args = [*inputs,kn,qn,u,gates,z,out,nxt,st]
                assert lib.apply(stream.ptr, 1e-5, *[a.data.ptr for a in args]) == 0
                stream.synchronize()
                assert st.get()[0] == 0
                close(out, expected, f"{kind}/{step} residual")
                close(nxt, expected_hist, f"{kind}/{step} normalized history")
                rk = ref._norm(xp.asarray(key), ref.gamma_key)
                rq = ref._norm(rh, ref.gamma_query)
                score = xp.sum(rk*rq, axis=-1, dtype=xp.float32)*xp.float32(1/np.sqrt(D))
                from mini_llm.external.qwen4exp_hyperconnection import _sigmoid
                expected_gate = _sigmoid(xp.sign(score)*xp.sqrt(xp.maximum(xp.abs(score), xp.float32(1e-6))))
                close(gates, expected_gate, f"{kind} signed-sqrt gate")
                close(kn, rk, "key RMS"); close(qn, rq, "query RMS")
                close(u, expected_gate[:,None]*xp.asarray(value), "gated value")
                close(z, expected_hist[-1], "normalized z")
                close(nxt[:8], cp.asnumpy(inputs[-1])[1:], "oldest-first history shift", exact=True)
                for a, saved in zip(inputs, original): close(a, saved, "input immutable", exact=True)
                rh, rhist = expected, expected_hist
                # Ping-pong explicitly; no output aliases an input on the next call.
                inputs[0], out = out, inputs[0]
                inputs[-1], nxt = nxt, inputs[-1]
                original[0], original[-1] = cp.asnumpy(inputs[0]), cp.asnumpy(inputs[-1])
            # Host invalid epsilon must not launch or touch status/output.
            before = out.copy(); cp.cuda.get_current_stream().synchronize()
            assert lib.apply(stream.ptr, 0, *[a.data.ptr for a in [*inputs,kn,qn,u,gates,z,out,nxt,st]]) != 0
            stream.synchronize(); close(out, before, "invalid args no write", exact=True)
            # Nonfinite input and sticky status (reset only by the caller).
            with stream: inputs[-1][2,0,0] = cp.nan
            args = [*inputs,kn,qn,u,gates,z,out,nxt,st]
            assert lib.apply(stream.ptr, 1e-5, *[a.data.ptr for a in args]) == 0
            stream.synchronize(); assert st.get()[0] == 1
            with stream: inputs[-1][2,0,0] = 0
            assert lib.apply(stream.ptr, 1e-5, *[a.data.ptr for a in args]) == 0
            stream.synchronize(); assert st.get()[0] == 1
        print(f"PASS CUDA D=2560 projections/gates/RMS/conv/history/status; max abs error={max_error:.9g}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--hash-only", action="store_true")
    args = parser.parse_args()
    hash_checks()
    if not args.hash_only:
        cuda_checks()
