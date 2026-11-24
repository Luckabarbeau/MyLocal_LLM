"""Native exact-byte GGUF uploads and synthetic CUDA transport; no model inference."""
from __future__ import annotations

import ctypes as C
import os
import threading
import struct
import sys
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
            'ml_moe_create': [h, C.c_int, C.c_int, C.POINTER(h), C.POINTER(h), C.POINTER(i), u, C.c_int, C.c_char_p, C.c_char_p, C.c_char_p, C.POINTER(h)],
            'ml_moe_submit': [h, C.POINTER(C.c_float), u, C.c_float, C.POINTER(h)],
            'ml_moe_reset': [h], 'ml_moe_destroy': [h],
            'ml_moe_job_wait': [h], 'ml_moe_job_destroy': [h],
                        'ml_moe_input_readback_for_test': [h, C.POINTER(C.c_float), u],
            'ml_moe_job_readback_for_test': [h, C.POINTER(C.c_float), u, C.POINTER(i), C.POINTER(C.c_float)],
                    'ml_cpu_experts_create': [h, C.c_int, C.c_int, C.c_int, C.c_int, C.c_char_p, C.c_char_p, C.POINTER(h)],
                    'ml_cpu_experts_destroy': [h],
                    'ml_cpu_experts_register': [h, C.POINTER(h), C.POINTER(u), C.POINTER(C.c_int32)],
                    'ml_cpu_experts_submit': [h, C.POINTER(C.c_int32), u, C.POINTER(C.c_float), u, C.POINTER(h)],
                    'ml_cpu_job_wait': [h],
                    'ml_cpu_job_readback_for_test': [h, C.POINTER(C.c_float), u],
                    'ml_cpu_job_destroy': [h],
            'ml_ple_weights_create': [h, h, C.POINTER(h), C.POINTER(u), C.POINTER(u), C.POINTER(u), i, C.c_float, C.c_int, C.POINTER(h)],
            'ml_ple_weights_destroy': [h],
            'ml_ple_create': [h, u, C.POINTER(h)],
            'ml_ple_destroy': [h],
            'ml_ple_reset': [h],
            'ml_ple_step': [h, h, i, u, C.c_int],
            'ml_ple_metadata': [h, C.POINTER(i), C.POINTER(u), C.POINTER(u)],
            'ml_ple_readback_for_test': [h, C.c_int, C.POINTER(C.c_float), u],
            "ml_gguf_lease_create": [h, C.c_char_p, C.POINTER(h)],
            "ml_payload_lease_destroy": [h],
            "ml_payload_lease_descriptor": [h, C.POINTER(_GGUFTensor)],
            "ml_payload_lease_copy": [h, u, u, C.c_void_p, u, C.POINTER(u)],
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
                        "ml_engine_upload": [h, h, C.POINTER(h)],
                        "ml_uploaded_destroy": [h],
                        "ml_uploaded_descriptor": [h, C.POINTER(_GGUFTensor)],
                        "ml_uploaded_query": [h, C.POINTER(C.c_int)],
                        "ml_uploaded_wait": [h],
                        "ml_uploaded_readback_for_test": [h, u, u, C.c_void_p, u, C.POINTER(u)],
            "ml_recurrent_create": [h] + [C.c_int]*5 + [u, C.POINTER(h), C.c_float, C.c_int, C.POINTER(h)],
            "ml_recurrent_destroy": [h],
            "ml_recurrent_reset": [h],
            "ml_recurrent_position": [h, C.POINTER(u)],
            "ml_recurrent_step": [h, h, u, C.POINTER(u)],
            "ml_recurrent_readback_for_test": [h, C.c_int, C.POINTER(C.c_float), u],
            "ml_hc_weights_create": [h]*5 + [C.c_float, C.c_int, C.POINTER(h)],
                        "ml_hc_weights_destroy": [h],
                        "ml_hc_create": [h, C.c_int, C.c_int, C.POINTER(h)],
                        "ml_hc_destroy": [h],
                        "ml_hc_reset": [h],
                        "ml_hc_initialize": [h, h],
                        "ml_hc_read": [h, h, C.POINTER(u)],
                        "ml_hc_write": [h, u, h],
                                                "ml_hc_mix": [h, h, h, C.c_int, u, C.POINTER(u)],
                                                                        "ml_hc_mix_scaled": [h, h, h, C.c_int, u, C.c_float, C.POINTER(u)],
                        "ml_hc_collapse": [h, h],
                        "ml_hc_readback_for_test": [h, C.c_int, C.POINTER(C.c_float), u],
                        "ml_qsa_layer_create": [h, u, C.POINTER(h), C.c_int, C.c_float, C.c_float, C.POINTER(h)],
            "ml_qsa_layer_destroy": [h],
            "ml_qsa_layer_reset": [h],
            "ml_qsa_layer_position": [h, C.POINTER(u)],
            "ml_qsa_layer_step": [h, h, u, C.POINTER(u)],
            "ml_qsa_layer_import": [h, u, h, h, h, h],
            "ml_qsa_layer_readback_for_test": [h, C.c_int, C.c_void_p, u],
            "ml_qsa_frontend_create": [h, C.POINTER(h)],
            "ml_qsa_frontend_destroy": [h],
            "ml_qsa_frontend_set_n": [h, u],
            "ml_qsa_frontend_reset": [h],
            "ml_qsa_frontend_run": [h]*11 + [C.c_float],
            "ml_qsa_frontend_from_hidden": [h]*13 + [u, C.c_float],
            "ml_qsa_frontend_hidden_readback_for_test": [h] + [C.POINTER(C.c_float), u]*5,
            "ml_qsa_frontend_readback_for_test": [h] + [C.POINTER(C.c_float), u]*6
                + [C.POINTER(C.c_int), C.POINTER(u)],
            "ml_qsa_projection_create": [h, h, C.POINTER(h)],
            "ml_qsa_projection_destroy": [h],
            "ml_qsa_projection_run": [h, u, C.POINTER(C.c_float)],
            "ml_qsa_projection_readback_for_test": [h, C.POINTER(C.c_float), u],
            "ml_qsa_attention_create": [h, C.POINTER(h)],
            "ml_qsa_attention_destroy": [h],
            "ml_qsa_attention_run": [h]*10 + [C.c_int, C.c_float, u, C.POINTER(C.c_float)],
            "ml_qsa_attention_readback_for_test": [h, C.POINTER(C.c_float), u,
                                                    C.POINTER(C.c_float), u],
            "ml_qsa_create": [h, u, C.POINTER(h)],
            "ml_qsa_destroy": [h],
            "ml_qsa_set_n": [h, u],
            "ml_qsa_reset": [h],
            "ml_qsa_run": [h, h, h, h, C.c_int, u, C.POINTER(C.c_float)],
            "ml_qsa_readback_for_test": [h, C.POINTER(C.c_float), u,
                                         C.POINTER(C.c_int64), u, C.POINTER(u)],
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
    """Read-only native mmap owner with independent shared-ownership payload leases.

    Source files must remain immutable for the lifetime of the mapping.
    """
    _destroy = "ml_gguf_destroy"

    def lease(self, name):
        """Retain one native tensor independently of this store's Python lifetime."""
        if not isinstance(name, str) or "\x00" in name:
            raise ValueError("name must be a string without NUL")
        with self._runtime._lock:
            handle = C.c_uint64()
            self._runtime._call("ml_gguf_lease_create", self._require_open(),
                                name.encode("utf-8"), C.byref(handle))
            if not handle.value:
                raise NativeError(4, "native lease creation returned a null handle")
            return PayloadLease(self._runtime, handle.value, name)

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


class PayloadLease(_Owned):
    """Native shared mmap ownership; reads return detached bytes, never raw views.

    Source shards must remain immutable until all leases and pending uploads retire.
    Upload consumers retain this same native shared lease, independently of Python.
    """
    _destroy = "ml_payload_lease_destroy"

    def __init__(self, runtime, handle, name):
        super().__init__(runtime, handle)
        self._name = name

    @property
    def descriptor(self):
        with self._runtime._lock:
            d = _GGUFTensor()
            self._runtime._call("ml_payload_lease_descriptor", self._require_open(), C.byref(d))
            if not 1 <= d.rank <= 4:
                raise NativeError(4, "native lease returned an invalid descriptor")
            return GGUFTensorDescriptor(
                self._name, d.shard_index, tuple(d.ggml_shape[:d.rank]), d.type_id,
                d.relative_offset, d.data_offset, d.nbytes, d.row_bytes,
                d.block_elements, d.block_bytes)

    def read(self, offset, size):
        """Copy an explicit tensor-relative byte range into caller-owned host memory."""
        _integer(offset, "offset", 2**64 - 1)
        _integer(size, "size", 2**64 - 1)
        with self._runtime._lock:
            d = self.descriptor
            if offset > d.nbytes or size > d.nbytes - offset:
                raise ValueError("payload range out of bounds")
            if size > sys.maxsize:
                raise ValueError("payload size exceeds host address space")
            buffer = (C.c_ubyte * size)()
            copied = C.c_uint64()
            self._runtime._call("ml_payload_lease_copy", self._require_open(), offset, size,
                                buffer, size, C.byref(copied))
            if copied.value != size:
                raise NativeError(4, "native payload copy returned an invalid byte count")
            return bytes(buffer)


class Engine(_Owned):
    _destroy = "ml_engine_destroy"

    def moe(self, router, shared_gate, shared_up, shared_down, scalar_gate,
            gate_bank, up_bank, down_bank, *, d_model, d_ff, gpu_experts,
            bridge_library, ggml_library, threads=4,
            arithmetic="cpu_ggml_q8_gpu_reference_fp32"):
        """Experimental static native owner; explicit mixed arithmetic, no fallback."""
        weights = (router, shared_gate, shared_up, shared_down, scalar_gate)
        banks = (gate_bank, up_bank, down_bank)
        for value, kind in [(v, UploadedTensor) for v in weights] + [(v, PayloadLease) for v in banks]:
            if not isinstance(value, kind) or value._runtime is not self._runtime:
                raise ValueError("MoE sources must belong to this Runtime")
        gpu_experts = tuple(gpu_experts)
        for value in gpu_experts:
            _integer(value, "GPU expert ID", 511)
        for name, value, maximum in (("d_model", d_model, 6144), ("d_ff", d_ff, 12288), ("threads", threads, 256)):
            _integer(value, name, maximum)
        with self._runtime._lock:
            out = C.c_uint64()
            self._runtime._call("ml_moe_create", self._require_open(), d_model, d_ff,
                (C.c_uint64 * 5)(*(v._require_open() for v in weights)),
                (C.c_uint64 * 3)(*(v._require_open() for v in banks)),
                (C.c_int32 * len(gpu_experts))(*gpu_experts), len(gpu_experts), threads,
                os.fsencode(bridge_library), os.fsencode(ggml_library), arithmetic.encode(), C.byref(out))
            return MoE(self._runtime, out.value, d_model)

    def cpu_experts(self, *, d_model, d_ff, threads, capacity=10,
                    bridge_library, ggml_library, arithmetic="ggml_q8"):
        """Explicit GGML Q8 CPU row pool, not reference_fp32 or GPU staging."""
        if arithmetic != "ggml_q8":
            raise ValueError("CPU experts require explicit ggml_q8 arithmetic")
        for name, value, maximum in (("d_model", d_model, 65536), ("d_ff", d_ff, 65536),
                                     ("threads", threads, 256), ("capacity", capacity, 512)):
            _integer(value, name, maximum)
        paths = []
        for path in (bridge_library, ggml_library):
            encoded = os.fsencode(path)
            if not encoded or b"\0" in encoded:
                raise ValueError("explicit library path without NUL required")
            paths.append(encoded)
        with self._runtime._lock:
            handle = C.c_uint64()
            self._runtime._call("ml_cpu_experts_create", self._require_open(), d_model,
                                d_ff, threads, capacity, *paths, C.byref(handle))
            return CPUExperts(self._runtime, handle.value, d_model, capacity)

    def upload(self, lease):
        """Upload exact stored bytes. Pageable CUDA copies may block the host.

        The returned owner is independent of public engine/store/lease close.
        No decoding, casting, or application-managed pinned staging is performed.
        """
        if not isinstance(lease, PayloadLease) or lease._runtime is not self._runtime:
            raise ValueError("lease must be a PayloadLease from the same Runtime")
        with self._runtime._lock:
            descriptor = lease.descriptor
            handle = C.c_uint64()
            self._runtime._call("ml_engine_upload", self._require_open(),
                                lease._require_open(), C.byref(handle))
            if not handle.value:
                raise NativeError(4, "native upload returned a null handle")
            try:
                return UploadedTensor(self._runtime, handle.value, descriptor.name)
            except BaseException:
                self._runtime._call("ml_uploaded_destroy", handle.value)
                raise

    def ple_weights(self, table, config, key, value, gamma_key, gamma_query,
                    gamma_conv, conv, *, epsilon, arithmetic):
        """Retain native sparse table lease and immutable uploads; no Python PLE math."""
        from mini_llm.external.qwen4exp_ple import PLEConfig
        if not isinstance(config, PLEConfig):
            raise TypeError('expected PLEConfig')
        if arithmetic != 'reference_fp32':
            raise ValueError('PLE requires explicit reference_fp32')
        with self._runtime._lock:
            items = [table, key, value, gamma_key, gamma_query, gamma_conv, conv]
            for j, item in enumerate(items):
                if not isinstance(item, PayloadLease if j == 0 else UploadedTensor) or item._runtime is not self._runtime:
                    raise TypeError('expected same-runtime table lease/uploads')
            if table.descriptor.shape != (config.table_rows, 160):
                raise ValueError('PLE config/table row count mismatch')
            handle = C.c_uint64()
            self._runtime._call('ml_ple_weights_create', self._require_open(), table._require_open(),
                (C.c_uint64*6)(*(x._require_open() for x in items[1:])),
                (C.c_uint64*3)(*config.multipliers), (C.c_uint64*16)(*config.offsets),
                (C.c_uint64*16)(*config.vocabs), config.eos, epsilon, 0, C.byref(handle))
            return PLEWeights(self._runtime, handle.value)

    def hc_weights(self, gamma, down, up, inject=None, *, epsilon, arithmetic):
        """Bind immutable F32/BF16 HC weights; BF16 expands exactly on GPU.

        Explicit reference_fp32 only. Converted gamma is flat[4D] or [4,D];
        down[R,4D], up[4D,R], optional inject[4,4D]. No CPU decoding/fallback.
        """
        if arithmetic != 'reference_fp32':
            raise ValueError('HC arithmetic must be explicit reference_fp32')
        with self._runtime._lock:
            handles = []
            for tensor in (gamma, down, up, inject):
                if tensor is None and len(handles) == 3:
                    handles.append(0)
                    continue
                if not isinstance(tensor, UploadedTensor) or tensor._runtime is not self._runtime:
                    raise ValueError('weights must be UploadedTensor from the same Runtime')
                handles.append(tensor._require_open())
            handle = C.c_uint64()
            self._runtime._call('ml_hc_weights_create', self._require_open(), *handles,
                                epsilon, 0, C.byref(handle))
            return HCWeights(self._runtime, handle.value)

    def hc_state(self, width, rank_capacity):
        """Allocate stable HC streams/scratch. No implicit model geometry or weights."""
        _integer(width, 'width', (2**31-1)//4)
        _integer(rank_capacity, 'rank_capacity', 2**31-5)
        handle = C.c_uint64()
        self._runtime._call('ml_hc_create', self._require_open(), width, rank_capacity, C.byref(handle))
        return HCState(self._runtime, handle.value)

    def recurrent_layer(self, spec, capacity, qkv, z, beta, alpha, out,
                        conv_weight, norm_weight, ssm_a, dt_bias, *, arithmetic):
        """Bounded batch-one native recurrent step; converted checkpoint order only."""
        from mini_llm.external.qwen4exp_mixer import Qwen4ExpMixerSpec
        if not isinstance(spec, Qwen4ExpMixerSpec):
            raise TypeError('spec must be Qwen4ExpMixerSpec')
        if arithmetic != 'reference_fp32':
            raise ValueError('explicit reference_fp32 arithmetic required')
        _integer(capacity, 'capacity', 2**64-1)
        dims = (spec.d_model, spec.n_key_heads, spec.n_value_heads,
                spec.head_dim, spec.conv_kernel_size)
        for value in dims:
            _integer(value, 'dimension', 2**31-1)
        with self._runtime._lock:
            handles = []
            for tensor in (qkv, z, beta, alpha, out, conv_weight, norm_weight, ssm_a, dt_bias):
                if not isinstance(tensor, UploadedTensor) or tensor._runtime is not self._runtime:
                    raise ValueError('inputs must be UploadedTensor from the same Runtime')
                handles.append(tensor._require_open())
            handle = C.c_uint64()
            self._runtime._call('ml_recurrent_create', self._require_open(), *dims, capacity,
                                (C.c_uint64*9)(*handles), spec.rms_eps, 0, C.byref(handle))
            return RecurrentLayer(self._runtime, handle.value)

    def qsa_layer(self, capacity, query_gate, key, value, index_query, index_key,
                  output, gamma_q, gamma_k, gamma_iq, gamma_ik, *,
                  policy, epsilon=1e-6, attention_scale=1/16):
        """Own a complete native FP32 QSA attention layer and append-only cache.

        Fixed geometry: hidden/output 2560, Q24/KV2 x 256, index Q4 x 128.
        capacity is 4096 or 16384. Explicit converted gammas, no model lookup.
        """
        _integer(capacity, "capacity", 16384)
        if policy not in ("ggml_cpu", "strata"):
            raise ValueError("explicit policy must be ggml_cpu or strata")
        with self._runtime._lock:
            handles = []
            for tensor in (query_gate, key, value, index_query, index_key, output,
                           gamma_q, gamma_k, gamma_iq, gamma_ik):
                if not isinstance(tensor, UploadedTensor) or tensor._runtime is not self._runtime:
                    raise ValueError("inputs must be UploadedTensor from the same Runtime")
                handles.append(tensor._require_open())
            handle = C.c_uint64()
            self._runtime._call("ml_qsa_layer_create", self._require_open(), capacity,
                                (C.c_uint64 * 10)(*handles), int(policy == "strata"),
                                epsilon, attention_scale, C.byref(handle))
            return QSALayer(self._runtime, handle.value)

    def qsa_frontend(self):
        """Allocate fixed projected-input QSA normalization/RoPE/pooling storage."""
        handle = C.c_uint64()
        self._runtime._call("ml_qsa_frontend_create", self._require_open(), C.byref(handle))
        return QSAFrontend(self._runtime, handle.value)

    def qsa_selection(self, capacity):
        """Allocate a bounded GPU index scoring/selection workspace, not attention."""
        _integer(capacity, "capacity", 2**31 - 4)
        handle = C.c_uint64()
        self._runtime._call("ml_qsa_create", self._require_open(), capacity, C.byref(handle))
        return QSASelection(self._runtime, handle.value)

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


class MoE(_Owned):
    """Static placement owner. Python submits one row, never dispatches experts."""
    _destroy = "ml_moe_destroy"

    def __init__(self, runtime, handle, d_model):
        super().__init__(runtime, handle)
        self.d_model = d_model

    def reset(self):
        with self._runtime._lock:
            self._runtime._call("ml_moe_reset", self._require_open())

    def input_readback_for_test(self):
        """Detached diagnostic only; HC chaining does not consume this copy."""
        import numpy as np
        with self._runtime._lock:
            out = np.empty(self.d_model, np.float32)
            self._runtime._call('ml_moe_input_readback_for_test', self._require_open(),
                out.ctypes.data_as(C.POINTER(C.c_float)), out.size)
            return out

    def submit(self, x, *, scale=1.0):
        import numpy as np
        if (not isinstance(x, np.ndarray) or x.dtype != np.float32 or
                x.shape not in ((self.d_model,), (1, self.d_model)) or not x.flags.c_contiguous):
            raise ValueError("input must be contiguous FP32 [D] or [1,D]")
        with self._runtime._lock:
            out = C.c_uint64()
            self._runtime._call("ml_moe_submit", self._require_open(),
                x.ctypes.data_as(C.POINTER(C.c_float)), x.size, scale, C.byref(out))
            return MoEJob(self._runtime, out.value, self.d_model)


class MoEJob(_Owned):
    _destroy = "ml_moe_job_destroy"

    def __init__(self, runtime, handle, d_model):
        super().__init__(runtime, handle)
        self.d_model = d_model

    def wait(self):
        with self._runtime._lock:
            self._runtime._call("ml_moe_job_wait", self._require_open())

    def readback_for_test(self):
        """Detached output, ascending route IDs, and once-scaled route weights."""
        import numpy as np
        with self._runtime._lock:
            output = np.empty(self.d_model, np.float32)
            ids, weights = np.empty(10, np.int32), np.empty(10, np.float32)
            self._runtime._call("ml_moe_job_readback_for_test", self._require_open(),
                output.ctypes.data_as(C.POINTER(C.c_float)), output.size,
                ids.ctypes.data_as(C.POINTER(C.c_int32)), weights.ctypes.data_as(C.POINTER(C.c_float)))
            return output, ids, weights


class CPUExperts(_Owned):
    """Native registered CPU dispatch; source leases are retained in C++, not Python."""
    _destroy = "ml_cpu_experts_destroy"
    arithmetic = "ggml_q8"

    def __init__(self, runtime, handle, d_model, capacity):
        super().__init__(runtime, handle)
        self.d_model, self.capacity = d_model, capacity

    def register(self, gate, up, down, *, offsets=(0, 0, 0)):
        """Register three leases and tensor-relative byte offsets to whole matrices."""
        leases, offsets = (gate, up, down), tuple(offsets)
        if len(offsets) != 3:
            raise ValueError("three subtensor byte offsets required")
        for offset in offsets:
            _integer(offset, "offset", 2**64 - 1)
        if any(not isinstance(v, PayloadLease) or v._runtime is not self._runtime for v in leases):
            raise ValueError("sources must be PayloadLeases from the same Runtime")
        with self._runtime._lock:
            handles = (C.c_uint64 * 3)(*(v._require_open() for v in leases))
            result = C.c_int32(-1)
            self._runtime._call("ml_cpu_experts_register", self._require_open(), handles,
                                (C.c_uint64 * 3)(*offsets), C.byref(result))
            return result.value

    def submit(self, x, indices):
        """Copy one FP32 row into a native-owned job; no Python expert dispatch."""
        import numpy as np
        if (not isinstance(x, np.ndarray) or x.dtype != np.float32 or
                x.shape not in ((self.d_model,), (1, self.d_model)) or not x.flags.c_contiguous):
            raise ValueError("input must be contiguous FP32 [D] or [1,D]")
        indices = tuple(indices)
        if not 0 < len(indices) <= self.capacity:
            raise ValueError("expert count exceeds pool capacity")
        for index in indices:
            _integer(index, "registered index", 2**31 - 1)
        with self._runtime._lock:
            handle = C.c_uint64()
            self._runtime._call("ml_cpu_experts_submit", self._require_open(),
                (C.c_int32 * len(indices))(*indices), len(indices),
                x.ctypes.data_as(C.POINTER(C.c_float)), x.size, C.byref(handle))
            return CPUExpertJob(self._runtime, handle.value, len(indices), self.d_model)


class CPUExpertJob(_Owned):
    """Native input/output owner. close drains, readback is explicit diagnostics."""
    _destroy = "ml_cpu_job_destroy"

    def __init__(self, runtime, handle, count, d_model):
        super().__init__(runtime, handle)
        self.shape = (count, d_model)

    def wait(self):
        with self._runtime._lock:
            self._runtime._call("ml_cpu_job_wait", self._require_open())

    def readback_for_test(self):
        import numpy as np
        with self._runtime._lock:
            handle = self._require_open()
            output = np.empty(self.shape, np.float32)
            self._runtime._call("ml_cpu_job_readback_for_test", handle,
                output.ctypes.data_as(C.POINTER(C.c_float)), output.size)
            return output


class UploadedTensor(_Owned):
    """Owned exact-byte device payload. Native ownership retains the engine.

    query()/wait() retire the source lease only on successful completion. close()
    drains work before freeing; no raw device pointer is available to Python.
    """
    _destroy = "ml_uploaded_destroy"

    def __init__(self, runtime, handle, name):
        super().__init__(runtime, handle)
        self._name = name

    @property
    def descriptor(self):
        with self._runtime._lock:
            d = _GGUFTensor()
            self._runtime._call("ml_uploaded_descriptor", self._require_open(), C.byref(d))
            if not 1 <= d.rank <= 4:
                raise NativeError(4, "native upload returned an invalid descriptor")
            return GGUFTensorDescriptor(
                self._name, d.shard_index, tuple(d.ggml_shape[:d.rank]), d.type_id,
                d.relative_offset, d.data_offset, d.nbytes, d.row_bytes,
                d.block_elements, d.block_bytes)

    def query(self):
        with self._runtime._lock:
            ready = C.c_int(-1)
            self._runtime._call("ml_uploaded_query", self._require_open(), C.byref(ready))
            if ready.value not in (0, 1):
                raise NativeError(4, "native upload returned invalid completion state")
            return bool(ready.value)

    def wait(self):
        self._runtime._call("ml_uploaded_wait", self._require_open())

    def readback_for_test(self, offset, size, destination):
        """Wait and copy an explicit byte range to a writable contiguous buffer.

        Test-only synchronous D2H; caller supplies storage, not a device pointer.
        """
        _integer(offset, "offset", 2**64 - 1)
        _integer(size, "size", 2**64 - 1)
        with self._runtime._lock:
            d = self.descriptor
            if offset > d.nbytes or size > d.nbytes - offset:
                raise ValueError("upload range out of bounds")
            view = memoryview(destination)
            try:
                if view.readonly or not view.c_contiguous or view.nbytes < size:
                    raise ValueError("destination must be writable, contiguous, and large enough")
                buffer = (C.c_ubyte * view.nbytes).from_buffer(view)
                copied = C.c_uint64()
                self._runtime._call("ml_uploaded_readback_for_test", self._require_open(),
                                    offset, size, buffer, view.nbytes, C.byref(copied))
                if copied.value != size:
                    raise NativeError(4, "native readback returned an invalid byte count")
                return copied.value
            finally:
                view.release()


class PLEWeights(_Owned):
    """Immutable native table lease/weight owner, independent of public closes."""
    _destroy = 'ml_ple_weights_destroy'

    def state(self, capacity):
        _integer(capacity, 'capacity', 2**64-1)
        handle = C.c_uint64()
        self._runtime._call('ml_ple_create', self._require_open(), capacity, C.byref(handle))
        return PLEState(self._runtime, handle.value)


class PLEState(_Owned):
    """Single-sequence transactional history. Tokens only; no Python hash/row math."""
    _destroy = 'ml_ple_destroy'

    def reset(self):
        self._runtime._call('ml_ple_reset', self._require_open())

    @property
    def metadata(self):
        predecessors, rows, position = (C.c_int32*2)(), (C.c_uint64*16)(), C.c_uint64()
        self._runtime._call('ml_ple_metadata', self._require_open(), predecessors, rows, C.byref(position))
        return tuple(predecessors), tuple(rows), position.value

    def step(self, streams, token, position):
        _integer(token, 'token', 2**31-1)
        _integer(position, 'position', 2**64-1)
        with self._runtime._lock:
            if not isinstance(streams, (UploadedTensor, HCState)) or streams._runtime is not self._runtime:
                raise TypeError('expected same-runtime uploaded F32 streams or HC state')
            self._runtime._call('ml_ple_step', self._require_open(), streams._require_open(),
                                token, position, int(isinstance(streams, HCState)))

    def readback_for_test(self, which, destination):
        names = ('streams', 'history', 'embedding', 'gates')
        if which not in names:
            raise ValueError('unknown PLE diagnostic')
        view = memoryview(destination)
        try:
            if view.readonly or not view.c_contiguous or view.itemsize != 4 or view.format not in ('f', '@f', '=f'):
                raise ValueError('expected writable contiguous native F32')
            buffer = (C.c_float*(view.nbytes//4)).from_buffer(view)
            self._runtime._call('ml_ple_readback_for_test', self._require_open(), names.index(which), buffer, len(buffer))
        finally:
            view.release()


class HCWeights(_Owned):
    """Native immutable HC weight owner, independent of public upload lifetime."""
    _destroy = 'ml_hc_weights_destroy'


class HCState(_Owned):
    """Native HC transaction state. No Python numerical intermediates.

    read returns a scalar one-use snapshot ticket; write pairs that ticket with
    its original streams/gains. Any native mutating failure requires reset and
    initialize. Final collapse is terminal until reset. Readback is diagnostics.
    """
    _destroy = 'ml_hc_destroy'

    def _input(self, value, cls):
        if not isinstance(value, cls) or value._runtime is not self._runtime:
            raise ValueError('input owner must be from the same Runtime')
        return value._require_open()

    def reset(self):
        self._runtime._call('ml_hc_reset', self._require_open())

    def initialize(self, embedding_or_streams):
        with self._runtime._lock:
            self._runtime._call('ml_hc_initialize', self._require_open(),
                                self._input(embedding_or_streams, UploadedTensor))

    def read(self, weights):
        with self._runtime._lock:
            ticket = C.c_uint64()
            self._runtime._call('ml_hc_read', self._require_open(),
                                self._input(weights, HCWeights), C.byref(ticket))
            return ticket.value

    def write(self, snapshot, block_output):
        _integer(snapshot, 'snapshot', 2**64-1)
        with self._runtime._lock:
            self._runtime._call('ml_hc_write', self._require_open(), snapshot,
                                self._input(block_output, UploadedTensor))

    def mix(self, branch, weights, position, *, scale=1.0):
        """One native HC read -> QSA/recurrent/MoE -> paired HC write.

        Only scalar completion returns. MoE scale affects routed experts only.
        On execution failure reset both owners; PLE is a preceding operation.
        """
        _integer(position, 'position', 2**64-1)
        kinds = (QSALayer, RecurrentLayer, MoE)
        if not isinstance(branch, kinds):
            raise ValueError('branch must be native QSALayer, RecurrentLayer or MoE')
        kind = next(i for i, cls in enumerate(kinds) if isinstance(branch, cls))
        with self._runtime._lock:
            value = C.c_uint64()
            self._runtime._call('ml_hc_mix_scaled', self._require_open(),
                                self._input(weights, HCWeights), self._input(branch, kinds),
                                kind, position, scale, C.byref(value))
            return value.value

    def collapse(self, weights):
        with self._runtime._lock:
            self._runtime._call('ml_hc_collapse', self._require_open(), self._input(weights, HCWeights))

    def readback_for_test(self, which, destination):
        names = ('streams', 'hidden', 'gains')
        if which not in names:
            raise ValueError('unknown HC diagnostic')
        view = memoryview(destination)
        try:
            if view.readonly or not view.c_contiguous or view.itemsize != 4 or view.format != 'f':
                raise ValueError('expected writable contiguous native F32 buffer')
            buffer = (C.c_float * (view.nbytes // 4)).from_buffer(view)
            self._runtime._call('ml_hc_readback_for_test', self._require_open(),
                                names.index(which), buffer, len(buffer))
        finally:
            view.release()


class QSALayer(_Owned):
    """Native complete attention step; no residual/HC/MLP or Python computation.

    Only scalar completion crosses the host boundary during step. Diagnostic
    readback is explicit and never used by step. Weights retained independently
    of public upload/engine handles. CUDA failures can require reset.
    """
    _destroy = "ml_qsa_layer_destroy"

    @property
    def position(self):
        value = C.c_uint64()
        self._runtime._call("ml_qsa_layer_position", self._require_open(), C.byref(value))
        return value.value

    def reset(self):
        self._runtime._call("ml_qsa_layer_reset", self._require_open())

    def step(self, hidden, position):
        _integer(position, "position", 2**64 - 1)
        if not isinstance(hidden, UploadedTensor) or hidden._runtime is not self._runtime:
            raise ValueError("hidden must be UploadedTensor from the same Runtime")
        with self._runtime._lock:
            value = C.c_uint64()
            self._runtime._call("ml_qsa_layer_step", self._require_open(), hidden._require_open(),
                                position, C.byref(value))
            return value.value

    def import_reference_state(self, length, raw_index, keys, values, pooled):
        """Explicit import into empty state; exact F32 shapes and finite checked.

        raw_index[length,128], keys/values[length,512], pooled[length//4,128].
        length >= 4. Caller establishes reference provenance and semantic
        consistency with this layer's weights, gamma, RoPE and epsilon.
        """
        _integer(length, "length", 16384)
        with self._runtime._lock:
            handles = []
            for tensor in (raw_index, keys, values, pooled):
                if not isinstance(tensor, UploadedTensor) or tensor._runtime is not self._runtime:
                    raise ValueError("state must be UploadedTensor from the same Runtime")
                handles.append(tensor._require_open())
            self._runtime._call("ml_qsa_layer_import", self._require_open(), length, *handles)

    def readback_for_test(self, which, destination):
        """Explicit diagnostics only: output/raw/keys/values/pooled/ids/scores.

        Copies committed prefixes (no future capacity); output/IDs/scores require
        a successful step. Destination is writable contiguous F32 (IDs int64).
        """
        names = ("output", "raw", "keys", "values", "pooled", "ids", "scores")
        if which not in names:
            raise ValueError("unknown diagnostic")
        view = memoryview(destination)
        try:
            integer = which == "ids"
            if (view.readonly or not view.c_contiguous or view.itemsize != (8 if integer else 4)
                    or view.format not in (("l", "q") if integer else ("f",))):
                raise ValueError("expected writable contiguous native F32 or int64 IDs")
            buffer = (C.c_char * view.nbytes).from_buffer(view)
            self._runtime._call("ml_qsa_layer_readback_for_test", self._require_open(),
                                names.index(which), buffer, view.nbytes // view.itemsize)
        finally:
            view.release()


class RecurrentLayer(_Owned):
    """Engine-owned recurrence, single native dispatch; diagnostics are never step inputs."""
    _destroy = 'ml_recurrent_destroy'

    @property
    def position(self):
        value = C.c_uint64()
        self._runtime._call('ml_recurrent_position', self._require_open(), C.byref(value))
        return value.value

    def reset(self):
        self._runtime._call('ml_recurrent_reset', self._require_open())

    def step(self, hidden, position):
        _integer(position, 'position', 2**64-1)
        if not isinstance(hidden, UploadedTensor) or hidden._runtime is not self._runtime:
            raise ValueError('hidden must be UploadedTensor from the same Runtime')
        with self._runtime._lock:
            value = C.c_uint64()
            self._runtime._call('ml_recurrent_step', self._require_open(), hidden._require_open(),
                                position, C.byref(value))
            return value.value

    def readback_for_test(self, which, destination):
        names = ('output', 'history', 'matrix')
        if which not in names:
            raise ValueError('unknown diagnostic')
        view = memoryview(destination)
        try:
            if view.readonly or not view.c_contiguous or view.itemsize != 4 or view.format != 'f':
                raise ValueError('expected writable contiguous native F32')
            buffer = (C.c_float * (view.nbytes//4)).from_buffer(view)
            self._runtime._call('ml_recurrent_readback_for_test', self._require_open(),
                                names.index(which), buffer, view.nbytes//4)
        finally:
            view.release()


class QSAFrontend(_Owned):
    """Native hidden or projected-input frontend; no attention or state commits.

    Native runs retain inputs and engine through synchronized status completion.
    n includes the current token. set_n uploads a stable device scalar; reset
    invalidates outputs and sets n=0 without reallocating. No device pointers.
    """
    _destroy = "ml_qsa_frontend_destroy"

    def set_n(self, n):
        _integer(n, "n", 2**64 - 1)
        self._runtime._call("ml_qsa_frontend_set_n", self._require_open(), n)

    def reset(self):
        self._runtime._call("ml_qsa_frontend_reset", self._require_open())

    def run(self, qg, k, raw, iq, first_raw, partial,
            gamma_q, gamma_k, gamma_iq, gamma_ik, *, epsilon=1e-6):
        """Consume ten same-engine F32 UploadedTensors; leave results on GPU.

        Shapes: [24,512], [2,256], [128], [4,128], [128], [3,128],
        [256], [256], [128], [128]. Gamma is already converted (not 1+gamma).
        Every native run attempt invalidates the previous output snapshot.
        """
        with self._runtime._lock:
            handles = []
            for tensor in (qg, k, raw, iq, first_raw, partial,
                           gamma_q, gamma_k, gamma_iq, gamma_ik):
                if not isinstance(tensor, UploadedTensor) or tensor._runtime is not self._runtime:
                    raise ValueError("inputs must be UploadedTensor from the same Runtime")
                handles.append(tensor._require_open())
            self._runtime._call("ml_qsa_frontend_run", self._require_open(), *handles, epsilon)

    def from_hidden(self, hidden, query_gate, key, value, index_query, index_key,
                    gamma_q, gamma_k, gamma_iq, gamma_ik, first_raw, partial, n,
                    *, epsilon=1e-6):
        """Five native reference_fp32 matvecs then the same norm/RoPE/pool kernel.

        All tensors must already be uploaded. Hidden is F32[2560]; weights have
        Python shapes [12288,2560], [512,2560], [512,2560], [512,2560],
        [128,2560]. No Python numerical intermediate or activation quantization.
        Scratch is allocated once, budgeted and owned by this workspace.
        """
        _integer(n, "n", 2**64 - 1)
        with self._runtime._lock:
            handles = []
            for tensor in (hidden, query_gate, key, value, index_query, index_key,
                           gamma_q, gamma_k, gamma_iq, gamma_ik, first_raw, partial):
                if not isinstance(tensor, UploadedTensor) or tensor._runtime is not self._runtime:
                    raise ValueError("inputs must be UploadedTensor from the same Runtime")
                handles.append(tensor._require_open())
            self._runtime._call("ml_qsa_frontend_from_hidden", self._require_open(),
                                *handles, n, epsilon)

    def hidden_readback_for_test(self, qg, k, v, raw, iq):
        """Explicit diagnostic only: qg[24,512], K/V[2,256], raw[128], IQ[4,128].

        Raw and V are retained native outputs for a later state integration;
        no state/cache commit is performed here. Valid only after from_hidden.
        """
        views = [memoryview(x) for x in (qg, k, v, raw, iq)]
        try:
            arguments = []
            for view in views:
                if (view.readonly or not view.c_contiguous or view.itemsize != 4
                        or view.format != "f"):
                    raise ValueError("expected writable contiguous native F32 buffers")
                buffer = (C.c_float * (view.nbytes // 4)).from_buffer(view)
                arguments.extend((buffer, len(buffer)))
            self._runtime._call("ml_qsa_frontend_hidden_readback_for_test",
                                self._require_open(), *arguments)
        finally:
            for view in views:
                view.release()

    def readback_for_test(self, q, k, gate, iq, dead, pool):
        """Explicit diagnostic D2H into six writable contiguous native F32 buffers.

        Returns (actual_device_n, pool_ready). Shapes are [24,256], [2,256],
        [24,256], [4,128], [128], [128]. pool is unchanged when not ready;
        it must not be treated as a newly completed pool in that case.
        """
        views = [memoryview(x) for x in (q, k, gate, iq, dead, pool)]
        try:
            arguments = []
            for view in views:
                if (view.readonly or not view.c_contiguous or view.itemsize != 4
                        or view.format != "f"):
                    raise ValueError("expected writable contiguous native F32 buffers")
                buffer = (C.c_float * (view.nbytes // 4)).from_buffer(view)
                arguments.extend((buffer, len(buffer)))
            ready, n = C.c_int(), C.c_uint64()
            self._runtime._call("ml_qsa_frontend_readback_for_test", self._require_open(),
                                *arguments, C.byref(ready), C.byref(n))
            return n.value, bool(ready.value)
        finally:
            for view in views:
                view.release()


class QSASelection(_Owned):
    """Stable engine-owned GPU scores, IDs and n parameter; no production integration.

    set_n explicitly uploads a host scalar. run synchronizes and checks scalar
    status only, leaving scores/IDs on GPU. Newly completed pools must already
    be committed in the supplied pooled upload. No CuPy dependency.
    """
    _destroy = "ml_qsa_destroy"

    def attention(self):
        """Allocate selected GQA context/gating storage, retaining native selection."""
        handle = C.c_uint64()
        self._runtime._call("ml_qsa_attention_create", self._require_open(), C.byref(handle))
        return QSAAttention(self._runtime, handle.value)

    def set_n(self, n):
        _integer(n, "n", 2**64 - 1)
        self._runtime._call("ml_qsa_set_n", self._require_open(), n)

    def reset(self):
        self._runtime._call("ml_qsa_reset", self._require_open())

    def run(self, query, pooled, dead, *, policy, repeats=1):
        """Return CUDA-event milliseconds/iteration (scores+selection only).

        repeats > 1 is a microbenchmark mode; input uploads, n upload, scalar
        status readback, diagnostics and host synchronization are not timed.
        Both policies are explicit: ggml_cpu or strata. No CPU fallback.
        """
        if policy not in ("ggml_cpu", "strata"):
            raise ValueError("unknown tie policy")
        _integer(repeats, "repeats", 10000)
        with self._runtime._lock:
            handles = []
            for tensor in (query, pooled, dead):
                if not isinstance(tensor, UploadedTensor) or tensor._runtime is not self._runtime:
                    raise ValueError("inputs must be UploadedTensor from the same Runtime")
                handles.append(tensor._require_open())
            elapsed = C.c_float()
            self._runtime._call("ml_qsa_run", self._require_open(), *handles,
                                int(policy == "strata"), repeats, C.byref(elapsed))
            return elapsed.value

    def readback_for_test(self, scores, ids):
        """Explicit diagnostic D2H to caller-owned contiguous F32/int64 buffers.

        Returns n. Capacities are element counts; no NumPy dependency in binding.
        """
        views = [memoryview(scores), memoryview(ids)]
        try:
            for view, size, formats in zip(views, (4, 8), (("f",), ("q", "l"))):
                if (view.readonly or not view.c_contiguous or view.itemsize != size
                        or view.format not in formats):
                    raise ValueError("expected writable contiguous native F32/int64 buffers")
            sb = (C.c_float * (views[0].nbytes // 4)).from_buffer(views[0])
            ib = (C.c_int64 * (views[1].nbytes // 8)).from_buffer(views[1])
            n = C.c_uint64()
            self._runtime._call("ml_qsa_readback_for_test", self._require_open(),
                                sb, len(sb), ib, len(ib), C.byref(n))
            return n.value
        finally:
            for view in views:
                view.release()


class QSAAttention(_Owned):
    """Experimental selected FP32 GQA, no projection or cache mutation.

    q and separate sigmoid gate: [24,256]; current k/v: [2,256]; previous
    keys/values: [selection.capacity,2,256]. All are exact F32 uploads.
    Runs synchronize and check source/arithmetic status; no selected-ID readback.
    """
    _destroy = "ml_qsa_attention_destroy"

    def projection(self, weight, *, arithmetic="reference_fp32"):
        """Create a bounded packed-weight GPU consumer of this gated snapshot."""
        if arithmetic != "reference_fp32":
            raise ValueError("only reference_fp32 arithmetic is supported")
        if not isinstance(weight, UploadedTensor) or weight._runtime is not self._runtime:
            raise ValueError("weight must be UploadedTensor from the same Runtime")
        with self._runtime._lock:
            handle = C.c_uint64()
            self._runtime._call("ml_qsa_projection_create", self._require_open(),
                                weight._require_open(), C.byref(handle))
            return QSAProjection(self._runtime, handle.value)

    def run(self, q, gate, k, v, keys, values, *, scale=1/16, repeats=1,
            selection_inputs=None, policy="ggml_cpu"):
        """CUDA ms/iteration, attention-only or combined selection+attention.

        selection_inputs=(index_query, pooled, dead) reruns selection per timed
        iteration. Otherwise use the last valid device selection. Uploads and
        final status/synchronization/readbacks are excluded from event timing.
        """
        if policy not in ("ggml_cpu", "strata"):
            raise ValueError("unknown tie policy")
        _integer(repeats, "repeats", 10000)
        tensors = (q, gate, k, v, keys, values)
        if selection_inputs is not None:
            if len(selection_inputs) != 3:
                raise ValueError("selection_inputs requires three uploads")
            tensors += tuple(selection_inputs)
        with self._runtime._lock:
            handles = []
            for tensor in tensors:
                if not isinstance(tensor, UploadedTensor) or tensor._runtime is not self._runtime:
                    raise ValueError("inputs must be UploadedTensor from the same Runtime")
                handles.append(tensor._require_open())
            if selection_inputs is None:
                handles += [0, 0, 0]
            elapsed = C.c_float()
            self._runtime._call("ml_qsa_attention_run", self._require_open(), *handles,
                                int(policy == "strata"), scale, repeats, C.byref(elapsed))
            return elapsed.value

    def readback_for_test(self, contexts, gated):
        """Copy valid snapshots into writable contiguous native F32 buffers."""
        views = [memoryview(contexts), memoryview(gated)]
        try:
            for view in views:
                if (view.readonly or not view.c_contiguous or view.itemsize != 4
                        or view.format != "f"):
                    raise ValueError("expected writable contiguous native F32 buffers")
            buffers = [(C.c_float * (view.nbytes // 4)).from_buffer(view) for view in views]
            self._runtime._call("ml_qsa_attention_readback_for_test", self._require_open(),
                                buffers[0], len(buffers[0]), buffers[1], len(buffers[1]))
        finally:
            for view in views:
                view.release()


class QSAProjection(_Owned):
    """Native reference_fp32 output projection; no CPU/activation-Q8 fallback."""
    _destroy = "ml_qsa_projection_destroy"

    def run(self, *, repeats=1):
        _integer(repeats, "repeats", 10000)
        elapsed = C.c_float()
        self._runtime._call("ml_qsa_projection_run", self._require_open(),
                            repeats, C.byref(elapsed))
        return elapsed.value

    def readback_for_test(self, output):
        view = memoryview(output)
        try:
            if (view.readonly or not view.c_contiguous or view.itemsize != 4
                    or view.format != "f"):
                raise ValueError("expected writable contiguous native F32 buffer")
            buffer = (C.c_float * (view.nbytes // 4)).from_buffer(view)
            self._runtime._call("ml_qsa_projection_readback_for_test",
                                self._require_open(), buffer, len(buffer))
        finally:
            view.release()


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


__all__ = ["Runtime", "Engine", "Session", "Stats", "NativeError", "GGUFStore", "GGUFTensorDescriptor", "PayloadLease", "UploadedTensor"]
