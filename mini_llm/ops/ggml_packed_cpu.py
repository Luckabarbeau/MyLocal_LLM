# SPDX-License-Identifier: LGPL-3.0-or-later
"""0256 opt-in GGML integer-dot CPU executor; no dense weight materialization.

Usage::

    cpu = GGMLPackedCPUExecutor('/absolute/path/to/libggml-cpu.so')
    y = cpu.linear(x_float32, packed_weight)
    y = cpu.expert(x_float32, (gate, up, down))  # unweighted FP32

Alternatively set MINI_LLM_GGML_CPU_LIBRARY explicitly before constructing cpu.
Only the small original bridge is compiled (CXX, MINI_LLM_NATIVE_BUILD_DIR follow
native_packed_cpu.py). No GGML linking, downloading or OpenMP. The synchronous
executor is thread-safe; the opt-in row pool owns persistent native workers.
The caller must not mutate packed storage while calls are in flight.

HeterogeneousExpertExecutor selects 'ggml' or 'ggml_pool' explicitly. Router
weighting remains outside CPU arithmetic; no dense expert copy is created.

ABI verified against ggml/include/{ggml.h,ggml-cpu.h}, src/ggml-common.h and
src/ggml-cpu/quants.h at the supplied reference revision below. GGML is a
separately selected MIT-licensed dependency (Copyright (c) 2022 Georgi Gerganov);
no GGML source or mathematics is included in this original LGPL project bridge.
Arbitrary incompatible GGML forks cannot be made safe by ctypes: use the pinned
build. Traits checks detect type mismatches, not arbitrary ABI changes.
"""
import ctypes as C
import hashlib
import os
from pathlib import Path
import platform
import shutil
import subprocess
import sys
import threading
import uuid

import numpy as np

from .quantized_weight import BLOCK_FORMATS, PackedGGUFWeight

REFERENCE_REVISION = '171e8846b4af9766c354064cb776cb34a50f053f'
# GGML type ID, activation type ID, exported dot suffix.
_FORMATS = {
    'IQ2_S': (22, 15, 'iq2_s_q8_K'),
    'IQ3_S': (21, 15, 'iq3_s_q8_K'),
    'IQ3_XXS': (18, 15, 'iq3_xxs_q8_K'),
    'IQ4_XS': (23, 15, 'iq4_xs_q8_K'),
    'IQ4_NL': (20, 8, 'iq4_nl_q8_0'),
    'Q2_0': (42, 8, 'q2_0_q8_0'),
}
SUPPORTED_FORMATS = frozenset(_FORMATS)


class _Traits(C.Structure):
    _fields_ = [('from_float', C.c_void_p), ('vec_dot', C.c_void_p),
                ('vec_dot_type', C.c_int), ('nrows', C.c_int64)]


_BUILD_LOCK = threading.Lock()


def _load_bridge():
    source = Path(__file__).with_suffix('.cpp')
    compiler = os.environ.get('CXX') or shutil.which('g++') or shutil.which('clang++')
    if not compiler:
        raise RuntimeError('GGML CPU bridge needs g++ or clang++; set CXX to a compiler executable')
    flags = ['-O3', '-std=c++17', '-shared', '-fPIC', '-pthread', '-ffp-contract=off']
    header = source.with_name('q2_cpu_simd.h')
    digest = hashlib.sha256(source.read_bytes() + header.read_bytes() + repr(
        (compiler, flags, platform.machine(), sys.platform)).encode()).hexdigest()[:20]
    cache = Path(os.environ.get('MINI_LLM_NATIVE_BUILD_DIR',
                              str(Path.home() / '.cache/mylocal_llm/native')))
    library = cache / f'ggml_packed_cpu_{digest}.so'
    with _BUILD_LOCK:
        cache.mkdir(parents=True, exist_ok=True)
        if not library.exists():
            temporary = library.with_name(library.name + '.' + uuid.uuid4().hex + '.tmp.so')
            try:
                try:
                    result = subprocess.run([compiler, *flags, str(source), '-o', str(temporary)],
                                            capture_output=True, text=True, timeout=60)
                except (OSError, subprocess.TimeoutExpired) as exc:
                    raise RuntimeError('GGML CPU bridge compiler failed or exceeded 60s; '
                                       'check CXX and MINI_LLM_NATIVE_BUILD_DIR') from exc
                if result.returncode:
                    raise RuntimeError(f'GGML CPU bridge build failed: {result.stderr[-3000:]}')
                os.replace(temporary, library)
            finally:
                temporary.unlink(missing_ok=True)
    lib = C.CDLL(str(library))
    projection = [C.c_void_p, C.c_size_t, C.c_int, C.c_void_p, C.c_void_p]
    tail = [C.c_int, C.c_int, C.c_void_p, C.c_int, C.c_void_p]
    lib.mylocal_ggml_linear.argtypes = projection + tail
    lib.mylocal_ggml_linear.restype = C.c_int
    lib.mylocal_ggml_expert.argtypes = projection * 3 + tail
    lib.mylocal_ggml_expert.restype = C.c_int
    lib.mylocal_ggml_pool_create.argtypes = [C.c_int] * 4
    lib.mylocal_ggml_pool_create.restype = C.c_void_p
    lib.mylocal_ggml_pool_set_spin.argtypes = [C.c_void_p, C.c_int]
    lib.mylocal_ggml_pool_set_spin.restype = C.c_int
    lib.mylocal_ggml_pool_configure.argtypes = [C.c_void_p, C.c_int, C.c_void_p, C.c_int]
    lib.mylocal_ggml_pool_configure.restype = C.c_int
    lib.mylocal_ggml_pool_set_task_mode.argtypes = [C.c_void_p,C.c_int]
    lib.mylocal_ggml_pool_set_task_mode.restype = C.c_int
    lib.mylocal_ggml_pool_register.argtypes = [C.c_void_p,C.c_void_p,C.c_void_p]
    lib.mylocal_ggml_pool_register.restype = C.c_int
    lib.mylocal_ggml_pool_submit_registered.argtypes = [C.c_void_p,C.c_void_p,C.c_int,C.c_void_p,C.c_void_p]
    lib.mylocal_ggml_pool_submit_registered.restype = C.c_int
    lib.mylocal_ggml_pool_submit.argtypes = [C.c_void_p, C.c_void_p, C.c_int,
                                           C.c_void_p, C.c_void_p]
    lib.mylocal_ggml_pool_submit.restype = C.c_int
    lib.mylocal_ggml_pool_wait.argtypes = [C.c_void_p]
    lib.mylocal_ggml_pool_wait.restype = None
    lib.mylocal_ggml_pool_destroy.argtypes = [C.c_void_p]
    lib.mylocal_ggml_pool_destroy.restype = None
    lib.mylocal_q2_avx2_available.argtypes = []
    lib.mylocal_q2_avx2_available.restype = C.c_int
    return lib


class GGMLPackedCPUExecutor:
    """Thread-safe synchronous FP32 linear/expert operations on packed CPU weights.

    library_path overrides the environment. provenance records resolved path and
    actual SHA256; reference_revision is the ABI target, NOT a verified library
    commit. Keep this object alive for the duration of all worker calls.
    """

    def __init__(self, library_path=None, *, q2_simd=None):
        selected = library_path if library_path is not None else os.environ.get('MINI_LLM_GGML_CPU_LIBRARY')
        if not selected:
            raise RuntimeError('Explicitly set MINI_LLM_GGML_CPU_LIBRARY or pass library_path '
                               'pointing to the pinned libggml-cpu.so')
        path = Path(selected).expanduser().resolve(strict=True)
        sha = hashlib.sha256()
        with path.open('rb') as stream:
            for chunk in iter(lambda: stream.read(1024 * 1024), b''):
                sha.update(chunk)
        try:
            self._ggml = C.CDLL(str(path))
            # GGML dot kernels decode half scales through tables initialized here.
            initialize = self._ggml.ggml_cpu_init
            initialize.argtypes, initialize.restype = [], None
            initialize()
            traits = self._ggml.ggml_get_type_traits_cpu
            traits.argtypes, traits.restype = [C.c_int], C.POINTER(_Traits)
            self._functions = {}
            for fmt, (type_id, activation, suffix) in _FORMATS.items():
                t = traits(type_id)
                if not t or t.contents.vec_dot_type != activation or not t.contents.vec_dot:
                    raise RuntimeError(f'Incompatible GGML CPU traits for {fmt}')
                dot = getattr(self._ggml, 'ggml_vec_dot_' + suffix)
                quant = getattr(self._ggml, 'quantize_row_q8_K' if activation == 15 else 'quantize_row_q8_0')
                # Resolve symbols once; native row loops call these directly.
                self._functions[fmt] = (activation, C.cast(quant, C.c_void_p).value,
                                        C.cast(dot, C.c_void_p).value)
        except (OSError, AttributeError) as exc:
            raise RuntimeError(f'Cannot load pinned GGML CPU symbols from {path}: {exc}') from exc
        self._bridge = _load_bridge()
        if q2_simd is None:
            q2_simd = os.environ.get('MINI_LLM_GGML_Q2_SIMD', '0') == '1'
        if not isinstance(q2_simd, bool):
            raise TypeError('q2_simd must be bool')
        if q2_simd:
            if not self._bridge.mylocal_q2_avx2_available():
                raise RuntimeError('Q2_0 SIMD requires x86 AVX2 and F16C')
            activation, quant, _ = self._functions['Q2_0']
            self._functions['Q2_0'] = (activation, quant,
                C.cast(self._bridge.mylocal_q2_avx2_dot, C.c_void_p).value)
        self.provenance = {'library_path': str(path), 'library_sha256': sha.hexdigest(),
                           'reference_revision': REFERENCE_REVISION,
                           'library_revision_verified': False,
                           'bridge_path': self._bridge._name,
                           'q2_dot': 'avx2_q8_0_scalar_float_order' if q2_simd else 'ggml',
                           'activation_quantizers': ['quantize_row_q8_K', 'quantize_row_q8_0']}

    def _projection(self, weight):
        if not isinstance(weight, PackedGGUFWeight) or len(weight.shape) != 2:
            raise TypeError('Expected PackedGGUFWeight [N,K] matrix')
        if weight.quant_type not in _FORMATS:
            raise ValueError(f'Unsupported GGML CPU format: {weight.quant_type}')
        n, k = weight.shape
        if max(n, k) > 2**31 - 1:
            raise ValueError('GGML dimensions must fit int32')
        activation, quant, dot = self._functions[weight.quant_type]
        if activation == 15 and k % 256:
            raise ValueError('Q8_K activation requires last K multiple of 256')
        qk, qb = BLOCK_FORMATS[weight.quant_type]
        if (weight.data.dtype != np.uint8 or not weight.data.flags.c_contiguous or
                weight.data.nbytes != n * (k // qk) * qb):
            raise ValueError('Invalid packed weight storage')
        return (weight.data.ctypes.data, (k // qk) * qb, activation, quant, dot)

    @staticmethod
    def _input(x, k, *, expert=False):
        x = np.asarray(x)
        if x.dtype != np.float32:
            raise TypeError('GGML CPU activations must be FP32')
        if x.ndim not in ((2,) if expert else (2, 3)) or x.shape[-1] != k:
            raise ValueError('Expected matching [M,K] activations (linear also accepts [B,T,K])')
        rows = int(np.prod(x.shape[:-1]))
        if rows > 2**31 - 1:
            raise ValueError('Activation row count must fit int32')
        return np.ascontiguousarray(x.reshape(rows, k)), x.shape[:-1]

    @staticmethod
    def _check(rc):
        if rc:
            raise RuntimeError(f'GGML CPU bridge failed: code={rc} (1=arguments, 2=allocation, 3=exception)')

    def linear(self, x, weight):
        """Return x @ weight.T using GGML Q8 activations, always FP32."""
        args = self._projection(weight)
        n, k = weight.shape
        x32, leading = self._input(x, k)
        out = np.empty((x32.shape[0], n), np.float32)
        if len(x32):
            self._check(self._bridge.mylocal_ggml_linear(
                *args, n, k, x32.ctypes.data, len(x32), out.ctypes.data))
        return out.reshape(leading + (n,))

    def expert(self, x, weights):
        """Return down(SiLU(gate(x))*up(x)); no routing weights applied."""
        if len(weights) != 3:
            raise ValueError('Expected (gate, up, down)')
        args = [self._projection(w) for w in weights]
        gate, up, down = weights
        if gate.shape != up.shape or down.shape != (gate.shape[1], gate.shape[0]):
            raise ValueError('Invalid packed expert projection shapes')
        f, d = gate.shape
        x32, leading = self._input(x, d, expert=True)
        out = np.empty((x32.shape[0], d), np.float32)
        if len(x32):
            self._check(self._bridge.mylocal_ggml_expert(
                *args[0], *args[1], *args[2], d, f, x32.ctypes.data, len(x32), out.ctypes.data))
        return out.reshape(leading + (d,))


class _Projection(C.Structure):
    _fields_ = [('weight', C.c_void_p), ('stride', C.c_size_t), ('qtype', C.c_int),
                ('quantize', C.c_void_p), ('dot', C.c_void_p)]


class _PoolExpert(C.Structure):
    _fields_ = [('gate', _Projection), ('up', _Projection), ('down', _Projection)]


class _RowBatch:
    def __init__(self, pool, weights, output):
        self.pool, self.weights, self.output = pool, weights, output
        self.complete = False

    def result(self):
        self.pool._finish(self)
        return self.output


class GGMLPackedCPURowPool:
    """Bounded singleton-decode pool; one outstanding batch, unweighted [E,D].

    submit quantizes the shared FP32 input once per required Q8 type, then returns
    while persistent native workers process rows. A batch retains its immutable
    packed sources and output until result()/close() drains all workers. Busy
    submission is rejected, not queued. close is idempotent and drains; outputs
    remain valid across subsequent batches. Callers must not mutate sources or
    supplied output until completion. Optional bounded spinning trades CPU power
    for wake latency; sleeping remains the default. No affinity policy is imposed.
    """
    def __init__(self, executor, d_model, d_ff, workers, capacity=10, *, spin_us=None,
                 host_help=None, cpu_ids=None, scheduler=None):
        for name, value, maximum in (('d_model', d_model, 65536), ('d_ff', d_ff, 65536),
                                      ('workers', workers, 256), ('capacity', capacity, 512)):
            if isinstance(value, bool) or not isinstance(value, int) or not 0 < value <= maximum:
                raise ValueError(f'{name} must be a positive integer <= {maximum}')
        if d_model % 32 or d_ff % 32:
            raise ValueError('Row pool dimensions must be multiples of 32')
        if spin_us is None:
            spin_us = int(os.environ.get('MINI_LLM_GGML_POOL_SPIN_US', '0'))
        if isinstance(spin_us, bool) or not isinstance(spin_us, int) or not 0 <= spin_us <= 2000:
            raise ValueError('spin_us must be an integer in [0,2000]')
        if host_help is None:
            text = os.environ.get('MINI_LLM_GGML_POOL_HOST_HELP', '0')
            if text not in ('0', '1'):
                raise ValueError('MINI_LLM_GGML_POOL_HOST_HELP must be 0 or 1')
            host_help = text == '1'
        if not isinstance(host_help, bool):
            raise TypeError('host_help must be bool')
        if cpu_ids is None and os.environ.get('MINI_LLM_GGML_POOL_CPUS'):
            cpu_ids = tuple(int(v) for v in os.environ['MINI_LLM_GGML_POOL_CPUS'].split(','))
        if cpu_ids is not None:
            cpu_ids = tuple(cpu_ids)
            if (len(cpu_ids) != workers or len(set(cpu_ids)) != workers or
                    any(isinstance(v, bool) or not isinstance(v, int) or not 0 <= v < 1024 for v in cpu_ids)):
                raise ValueError('cpu_ids must contain one distinct valid CPU ID per worker')
        scheduler = os.environ.get('MINI_LLM_GGML_POOL_SCHEDULER','barrier') if scheduler is None else scheduler
        if scheduler not in ('barrier','tasks'):
            raise ValueError('scheduler must be barrier or tasks')
        self.scheduler = scheduler
        self.host_help, self.cpu_ids = host_help, cpu_ids
        self.spin_us = spin_us
        self.executor = executor
        self.d_model, self.d_ff, self.capacity = d_model, d_ff, capacity
        self._lock, self._pending = threading.Lock(), None
        self._registered = {}
        self._handle = executor._bridge.mylocal_ggml_pool_create(d_model, d_ff, capacity, workers)
        if not self._handle:
            raise RuntimeError('Cannot allocate bounded GGML row pool')
        executor._check(executor._bridge.mylocal_ggml_pool_set_spin(self._handle, spin_us))
        cpus = np.asarray(cpu_ids, dtype=np.int32) if cpu_ids is not None else None
        executor._check(executor._bridge.mylocal_ggml_pool_configure(
            self._handle, int(host_help), cpus.ctypes.data if cpus is not None else None,
            len(cpus) if cpus is not None else 0))
        executor._check(executor._bridge.mylocal_ggml_pool_set_task_mode(self._handle,int(scheduler=='tasks')))

    def register(self, key, weights):
        """Retain one immutable source bundle and its validated native metadata."""
        with self._lock:
            if not self._handle or self._pending is not None:
                raise RuntimeError('Registration requires an idle open row pool')
            weights = tuple(weights)
            existing = self._registered.get(key)
            if existing is not None:
                if len(weights) != 3 or any(a is not b for a,b in zip(weights,existing[1])):
                    raise ValueError('Registered canonical source changed')
                return existing[0]
            if len(weights) != 3 or tuple(w.shape for w in weights) != (
                    (self.d_ff,self.d_model),(self.d_ff,self.d_model),(self.d_model,self.d_ff)):
                raise ValueError('Invalid registered projection shapes')
            descriptor = _PoolExpert(*(_Projection(*self.executor._projection(w)) for w in weights))
            index = C.c_int()
            self.executor._check(self.executor._bridge.mylocal_ggml_pool_register(
                self._handle,C.byref(descriptor),C.byref(index)))
            self._registered[key] = index.value, weights
            return index.value

    def _submit_registered(self, x, indices, out):
        """Internal path for distinct owned staging slabs; registry retains sources."""
        with self._lock:
            if not self._handle or self._pending is not None:
                raise RuntimeError('Registered submit requires an idle open row pool')
            if (not isinstance(x,np.ndarray) or x.dtype != np.float32 or
                    x.shape != (1,self.d_model) or not x.flags.c_contiguous or
                    not isinstance(indices,np.ndarray) or indices.dtype != np.int32 or
                    indices.ndim != 1 or not indices.flags.c_contiguous or
                    not 0 < len(indices) <= self.capacity or
                    not isinstance(out,np.ndarray) or out.dtype != np.float32 or
                    out.shape != (len(indices),self.d_model) or not out.flags.c_contiguous or
                    not out.flags.writeable or np.shares_memory(x,out)):
                raise ValueError('Invalid registered input/index/output slab')
            batch = _RowBatch(self,None,out)
            self.executor._check(self.executor._bridge.mylocal_ggml_pool_submit_registered(
                self._handle,indices.ctypes.data,len(indices),x.ctypes.data,out.ctypes.data))
            self._pending = batch
            return batch

    def submit(self, x, experts, *, out=None):
        with self._lock:
            if not self._handle:
                raise RuntimeError('GGML row pool is closed')
            if self._pending is not None:
                raise RuntimeError('GGML row pool is busy; drain the previous batch')
            experts = tuple(tuple(ws) for ws in experts)
            if not 0 < len(experts) <= self.capacity:
                raise ValueError('Expert count exceeds bounded row pool capacity')
            x, _ = self.executor._input(x, self.d_model, expert=True)
            if x.shape[0] != 1:
                raise ValueError('Row pool accepts one shared activation row')
            descriptors = (_PoolExpert * len(experts))()
            for i, ws in enumerate(experts):
                if len(ws) != 3 or tuple(w.shape for w in ws) != (
                        (self.d_ff, self.d_model), (self.d_ff, self.d_model),
                        (self.d_model, self.d_ff)):
                    raise ValueError('Invalid row pool expert projection shapes')
                descriptors[i] = _PoolExpert(*(_Projection(*self.executor._projection(w)) for w in ws))
            if out is None:
                out = np.empty((len(experts), self.d_model), np.float32)
            if (not isinstance(out, np.ndarray) or out.dtype != np.float32 or
                    out.shape != (len(experts), self.d_model) or
                    not out.flags.c_contiguous or not out.flags.writeable):
                raise ValueError('Output must be writable contiguous FP32 [E,D]')
            if np.shares_memory(x, out) or any(np.shares_memory(out, w.data)
                                             for ws in experts for w in ws):
                raise ValueError('Output must not alias activations or packed sources')
            batch = _RowBatch(self, experts, out)
            self.executor._check(self.executor._bridge.mylocal_ggml_pool_submit(
                self._handle, descriptors, len(experts), x.ctypes.data, out.ctypes.data))
            self._pending = batch
            return batch

    def _finish(self, batch):
        with self._lock:
            if batch.complete:
                return
            if self._pending is not batch:
                raise RuntimeError('Batch does not belong to this active pool epoch')
            self.executor._bridge.mylocal_ggml_pool_wait(self._handle)
            batch.complete = True
            batch.weights = None
            self._pending = None

    def close(self):
        with self._lock:
            if self._handle:
                self.executor._bridge.mylocal_ggml_pool_destroy(self._handle)
                self._handle = None
                self._registered.clear()
                if self._pending is not None:
                    self._pending.complete = True
                    self._pending.weights = None
                    self._pending = None

    def __enter__(self):
        return self

    def __exit__(self, *args):
        self.close()

    def __del__(self):
        if getattr(self, '_handle', None):
            self.close()
