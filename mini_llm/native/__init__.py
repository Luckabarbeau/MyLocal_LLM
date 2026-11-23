"""WP1 synthetic CUDA transport only: no weights, tokenizer, or Qwen inference."""
from __future__ import annotations

import ctypes as C
import os
import threading
import struct
from dataclasses import dataclass


class NativeError(RuntimeError):
    def __init__(self, status, message):
        self.status = status
        super().__init__(message)


class Stats(C.Structure):
    _fields_ = [(name, C.c_uint64) for name in (
        "device_bytes", "pinned_bytes", "peak_bytes", "sessions", "budget_bytes"
    )]


def _integer(value, name, maximum):
    if isinstance(value, bool) or not isinstance(value, int) or not 0 <= value <= maximum:
        raise ValueError(f"{name} must be an integer in [0, {maximum}]")
    return value


class _TensorRequirement(C.Structure):
    _fields_ = [("name", C.c_char_p), ("rank", C.c_uint64),
                ("ggml_shape", C.c_uint64 * 4), ("type_id", C.c_uint32)]


class _GGUFTensor(C.Structure):
    _fields_ = [("shard_index", C.c_uint64), ("rank", C.c_uint64),
                ("ggml_shape", C.c_uint64 * 4)] + [
        (name, C.c_uint64) for name in ("relative_offset", "data_offset", "nbytes",
                                       "row_bytes", "block_elements", "block_bytes")
    ] + [("type_id", C.c_uint32)]


@dataclass(frozen=True)
class GGUFTensorDescriptor:
    name: str
    shard_index: int
    ggml_shape: tuple
    type_id: int
    relative_offset: int
    data_offset: int
    nbytes: int
    row_bytes: int
    block_elements: int
    block_bytes: int

    @property
    def shape(self):
        return self.ggml_shape[::-1]

    @property
    def byte_shape(self):
        return self.shape[:-1] + (self.row_bytes,)

    @property
    def byte_strides(self):
        import math
        return tuple(math.prod(self.byte_shape[i + 1:]) for i in range(len(self.shape)))


def _gguf_value(raw, kind):
    def read(kind, pos):
        if kind == 8:
            size = struct.unpack_from("<Q", raw, pos)[0]
            pos += 8
            return raw[pos:pos + size].decode("utf-8"), pos + size
        if kind == 9:
            subtype, count = struct.unpack_from("<IQ", raw, pos)
            pos += 12
            values = []
            for _ in range(count):
                value, pos = read(subtype, pos)
                values.append(value)
            return values, pos
        fmt = "<" + {0: "B", 1: "b", 2: "H", 3: "h", 4: "I", 5: "i",
                      6: "f", 7: "?", 10: "Q", 11: "q", 12: "d"}[kind]
        return struct.unpack_from(fmt, raw, pos)[0], pos + struct.calcsize(fmt)
    return read(kind, 0)[0]


class Runtime:
    """Load an explicitly built shared library; never builds or falls back to CPU."""
    def __init__(self, library=None):
        path = library or os.environ.get("MINI_LLM_NATIVE_LIBRARY")
        if not path:
            raise RuntimeError("Set MINI_LLM_NATIVE_LIBRARY to the built CUDA shared library")
        self._lib = C.CDLL(os.fspath(path))
        self._lock = threading.RLock()
        h, u, i = C.c_uint64, C.c_uint64, C.c_int32
        signatures = {
            "ml_gguf_validate_schema": [h, C.POINTER(_TensorRequirement), u],
                        "ml_gguf_open": [C.POINTER(C.c_char_p), u, C.POINTER(h)],
                        "ml_gguf_destroy": [h],
                        "ml_gguf_counts": [h, C.POINTER(u), C.POINTER(u)],
                        "ml_gguf_tensor_get": [h, u, C.POINTER(C.c_char_p), C.POINTER(_GGUFTensor)],
                        "ml_gguf_metadata_get": [h, u, C.POINTER(C.c_char_p), C.POINTER(C.c_uint32),
                                                 C.POINTER(C.c_void_p), C.POINTER(u), C.POINTER(u)],
                        "ml_engine_create": [C.c_int, u, u, C.POINTER(h)],
            "ml_engine_destroy": [h],
            "ml_engine_stats": [h, C.POINTER(Stats)],
            "ml_session_create": [h, u, C.POINTER(h)],
            "ml_session_destroy": [h],
            "ml_session_reset": [h],
            "ml_session_step": [h, i, C.POINTER(i), C.POINTER(u)],
        }
        for name, args in signatures.items():
            fn = getattr(self._lib, name)
            fn.argtypes, fn.restype = args, C.c_int
        self._lib.ml_last_error.argtypes = []
        self._lib.ml_last_error.restype = C.c_char_p

    def _call(self, name, *args):
        with self._lock:
            status = getattr(self._lib, name)(*args)
            if status:
                raise NativeError(status, self._lib.ml_last_error().decode("utf-8", "replace"))

    def gguf(self, path):
        # Reuse established filename discovery, not Python header/tensor parsing.
        from mini_llm.external.gguf_store import _discover
        paths, _ = _discover(path)
        encoded = [os.fsencode(p) for p in paths]
        handle = C.c_uint64()
        self._call("ml_gguf_open", (C.c_char_p * len(paths))(*encoded), len(paths), C.byref(handle))
        store = GGUFStore(self, handle.value)
        try:
            store.paths = paths
            store._snapshot()
        except BaseException:
            store.close()
            raise
        return store

    def engine(self, *, device=0, budget_bytes=12 * 64, max_sessions=64):
        _integer(device, "device", 2**31 - 1)
        _integer(budget_bytes, "budget_bytes", 2**64 - 1)
        _integer(max_sessions, "max_sessions", 2**64 - 1)
        handle = C.c_uint64()
        self._call("ml_engine_create", device, budget_bytes, max_sessions, C.byref(handle))
        return Engine(self, handle.value)


class _Owned:
    def __init__(self, runtime, handle):
        self._runtime, self._handle = runtime, handle

    def _require_open(self):
        if not self._handle:
            raise NativeError(1, "object is closed")
        return self._handle

    def close(self):
        with self._runtime._lock:
            if self._handle:
                self._runtime._call(self._destroy, self._handle)
                self._handle = 0

    def __enter__(self):
        self._require_open()
        return self

    def __exit__(self, *args):
        self.close()

    def __del__(self):
        try:
            self.close()
        except Exception:
            pass


class GGUFStore(_Owned):
    """Read-only native mmap owner; detached metadata/descriptors, no payload API.

    Source files must remain immutable for the lifetime of the mapping.
    """
    _destroy = "ml_gguf_destroy"

    def validate_qwen4exp(self):
        """Validate native headers using the existing Python architecture authority.

        No payload access, allocation on GPU, or persistent validated-model handle.
        Always refresh from the live owner; mutable detached snapshots are not trusted.
        """
        from types import SimpleNamespace
        from mini_llm.external.gguf_store import _reader_api, GGUFFormat
        from mini_llm.external.qwen4exp_gguf import Qwen4ExpGGUFIndex
        from mini_llm.external.qwen4exp_weights import inspect_headers

        with self._runtime._lock:
            self._require_open()
            self._snapshot()
            _, formats = _reader_api()
            descriptors = {}
            for name, d in self.tensors.items():
                fmt = GGUFFormat(d.type_id, formats[d.type_id].name,
                                 d.block_elements, d.block_bytes)
                descriptors[name] = SimpleNamespace(shape=d.shape, quant_type=fmt.name, format=fmt)
            view = SimpleNamespace(closed=False, metadata=self.metadata, tensors=descriptors)
            headers = inspect_headers(Qwen4ExpGGUFIndex(view))
            # inspect_headers has resolved every accepted grouped-gamma alias and
            # validated storage restrictions. Bind that exact schema to the owner.
            schema = (_TensorRequirement * len(headers.shapes))(*(
                _TensorRequirement(name.encode(), len(self.tensors[name].ggml_shape),
                                   (C.c_uint64 * 4)(*self.tensors[name].ggml_shape),
                                   self.tensors[name].type_id)
                for name in headers.shapes))
            self._runtime._call("ml_gguf_validate_schema", self._handle, schema, len(schema))
            return headers

    def _snapshot(self):
        self.tensors, self.metadata, self.shard_metadata = {}, {}, [{} for _ in self.paths]
        with self._runtime._lock:
            nt, nm = C.c_uint64(), C.c_uint64()
            self._runtime._call("ml_gguf_counts", self._require_open(), C.byref(nt), C.byref(nm))
            for index in range(nt.value):
                name, d = C.c_char_p(), _GGUFTensor()
                self._runtime._call("ml_gguf_tensor_get", self._handle, index, C.byref(name), C.byref(d))
                name = name.value.decode("utf-8")
                self.tensors[name] = GGUFTensorDescriptor(
                    name, d.shard_index, tuple(d.ggml_shape[:d.rank]), d.type_id,
                    d.relative_offset, d.data_offset, d.nbytes, d.row_bytes,
                    d.block_elements, d.block_bytes)
            for index in range(nm.value):
                key, kind, ptr = C.c_char_p(), C.c_uint32(), C.c_void_p()
                size, shard = C.c_uint64(), C.c_uint64()
                self._runtime._call("ml_gguf_metadata_get", self._handle, index, C.byref(key),
                                    C.byref(kind), C.byref(ptr), C.byref(size), C.byref(shard))
                key = key.value.decode("utf-8")
                value = _gguf_value(C.string_at(ptr, size.value), kind.value)
                self.shard_metadata[shard.value][key] = value
                self.metadata.setdefault(key, value)


class Engine(_Owned):
    _destroy = "ml_engine_destroy"

    def session(self, capacity=128):
        _integer(capacity, "capacity", 2**64 - 1)
        handle = C.c_uint64()
        self._runtime._call("ml_session_create", self._require_open(), capacity, C.byref(handle))
        return Session(self, handle.value)

    @property
    def stats(self):
        result = Stats()
        self._runtime._call("ml_engine_stats", self._require_open(), C.byref(result))
        return result


class Session(_Owned):
    _destroy = "ml_session_destroy"

    def __init__(self, engine, handle):
        super().__init__(engine._runtime, handle)
        self._engine = engine  # Strong Python ownership, in addition to native shared ownership.

    def step(self, token):
        _integer(token, "token", 2**31 - 1)
        result, position = C.c_int32(), C.c_uint64()
        self._runtime._call("ml_session_step", self._require_open(), token,
                            C.byref(result), C.byref(position))
        return result.value, position.value

    def reset(self):
        self._runtime._call("ml_session_reset", self._require_open())


__all__ = ["Runtime", "Engine", "Session", "Stats", "NativeError", "GGUFStore", "GGUFTensorDescriptor"]
