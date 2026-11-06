"""On-demand, local C++ GGUF CPU kernels for packed-weight MoE decoding.

No admin rights, external libraries, GGML checkout, or Python extension build
are required. The caller explicitly opts in with cpu_packed_backend='native'
or 'auto'. The compiler is invoked once and cached under ~/.cache/mylocal_llm.
The C functions release Python's GIL, so 0106D CPU workers can overlap with
CuPy GPU execution. Inputs/outputs are transient FP32 activations only; the
packed GGUF expert weights are never unpacked into dense matrices.
"""
import ctypes
import hashlib
import os
from pathlib import Path
import platform
import shutil
import subprocess
import sys
import threading
import warnings

import numpy as np

from mini_llm.ops.quantized_weight import PackedGGUFWeight

_FORMAT = {"Q8_0": 0, "Q4_0": 1, "Q4_K": 2, "Q3_K": 3, "Q5_K": 4}
_LOCK = threading.Lock()
_LIB = None
_FAILURE = None
_WARNED = False


def _load_native():
    global _LIB, _FAILURE
    with _LOCK:
        if _LIB is not None:
            return _LIB
        if _FAILURE is not None:
            raise RuntimeError("Packed CPU native backend unavailable") from _FAILURE
        try:
            source = Path(__file__).with_name("native_packed_cpu.cpp")
            cpp_bytes = source.read_bytes()
            compiler = os.environ.get("CXX") or shutil.which("g++") or shutil.which("clang++")
            if not compiler:
                raise RuntimeError("No local C++ compiler found (g++ or clang++)")
            fingerprint = hashlib.sha256(cpp_bytes + platform.machine().encode() +
                                         sys.platform.encode() + compiler.encode()).hexdigest()[:20]
            cache_dir = Path(os.environ.get("MINI_LLM_NATIVE_BUILD_DIR",
                    str(Path.home() / ".cache" / "mylocal_llm" / "native")))
            cache_dir.mkdir(parents=True, exist_ok=True)
            library = cache_dir / f"packed_cpu_{fingerprint}.so"
            if not library.exists():
                import uuid
                temporary = library.with_name(library.name + "." + uuid.uuid4().hex + ".tmp.so")
                try:
                    process = subprocess.run(
                        [compiler, "-O3", "-std=c++17", "-shared", "-fPIC",
                         str(source), "-o", str(temporary)],
                        check=False, capture_output=True, text=True,
                    )
                    if process.returncode:
                        raise RuntimeError(f"C++ compiler failed: {process.stderr[-1500:]}")
                    os.replace(temporary, library)
                finally:
                    temporary.unlink(missing_ok=True)
            loaded = ctypes.CDLL(str(library))
            ptr = ctypes.c_void_p
            integer = ctypes.c_int
            loaded.mylocal_packed_gemv.argtypes = [ptr, integer, integer, integer,
                                                    ptr, integer, ptr]
            loaded.mylocal_packed_gemv.restype = integer
            loaded.mylocal_packed_expert.argtypes = [ptr, integer, ptr, integer,
                                                      ptr, integer, integer, integer,
                                                      ptr, integer, ptr]
            loaded.mylocal_packed_expert.restype = integer
            _LIB = loaded
            return loaded
        except Exception as exc:
            _FAILURE = exc
            raise


def _backend(mode):
    global _WARNED
    if mode not in ("reference", "auto", "native"):
        raise ValueError("cpu_packed_backend must be 'reference', 'auto' or 'native'")
    if mode == "reference":
        return None
    try:
        return _load_native()
    except Exception as exc:
        if mode == "native":
            raise RuntimeError("Requested native packed CPU backend could not be built") from exc
        if not _WARNED:
            warnings.warn("Native packed CPU backend unavailable; using reference decoder: "
                          f"{exc}", RuntimeWarning, stacklevel=3)
            _WARNED = True
        return None


def packed_cpu_linear(x: np.ndarray, weight: PackedGGUFWeight, *, backend="native"):
    """x @ packed_weight.T; returns None if 'auto' cannot load native backend."""
    if not isinstance(weight, PackedGGUFWeight) or len(weight.shape) != 2:
        raise TypeError("Expected a packed [N,K] GGUF matrix")
    lib = _backend(backend)
    if lib is None:
        return None
    x = np.asarray(x)
    if x.ndim not in (2, 3) or x.shape[-1] != weight.shape[1]:
        raise ValueError("Expected [M,K] or [B,T,K] activations matching weight")
    if x.dtype not in (np.float16, np.float32, np.float64):
        raise TypeError("CPU activation must be FP16, FP32 or FP64")
    input32 = np.ascontiguousarray(x.reshape(-1, weight.shape[1]), dtype=np.float32)
    output32 = np.empty((input32.shape[0], weight.shape[0]), np.float32)
    rc = lib.mylocal_packed_gemv(
        weight.data.ctypes.data, _FORMAT[weight.quant_type], weight.shape[0],
        weight.shape[1], input32.ctypes.data, input32.shape[0], output32.ctypes.data,
    )
    if rc:
        raise RuntimeError(f"Native packed GEMV failed with error code {rc}")
    return output32.astype(x.dtype, copy=False).reshape(x.shape[:-1] + (weight.shape[0],))


def packed_cpu_expert(x: np.ndarray, weights: tuple, *, backend="native"):
    """Execute complete packed MoE SwiGLU expert without dense CPU matrices.

    Returns None only if backend='auto' and compiler/native backend unavailable.
    Host provider/mmap keeps ownership of all three byte buffers while called.
    """
    if len(weights) != 3 or not all(isinstance(w, PackedGGUFWeight) for w in weights):
        raise TypeError("Expected three PackedGGUFWeight projections")
    gate, up, down = weights
    if gate.shape != up.shape or down.shape != (gate.shape[1], gate.shape[0]):
        raise ValueError("Invalid packed expert projection shapes")
    lib = _backend(backend)
    if lib is None:
        return None
    x = np.asarray(x)
    if x.ndim != 2 or x.shape[1] != gate.shape[1]:
        raise ValueError("Expert activations must be [M,d_model]")
    if x.dtype not in (np.float16, np.float32, np.float64):
        raise TypeError("Expert activations must be FP16, FP32 or FP64")
    input32 = np.ascontiguousarray(x, dtype=np.float32)
    output32 = np.empty((x.shape[0], gate.shape[1]), np.float32)
    rc = lib.mylocal_packed_expert(
        gate.data.ctypes.data, _FORMAT[gate.quant_type],
        up.data.ctypes.data, _FORMAT[up.quant_type],
        down.data.ctypes.data, _FORMAT[down.quant_type],
        gate.shape[1], gate.shape[0], input32.ctypes.data,
        input32.shape[0], output32.ctypes.data,
    )
    if rc:
        raise RuntimeError(f"Native packed expert failed with error code {rc}")
    return output32.astype(x.dtype, copy=False)
