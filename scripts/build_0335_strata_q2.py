#!/usr/bin/env python3
"""Diagnostic-only Strata Q2 adapter builder; run through run_0290_isolated_benchmark.py.

No model/GGML imports. --self-test loads native code only in timeout-bounded children.
The archive digest authenticates the local source input, not an independent Git signature.
"""
from __future__ import annotations

import argparse
import fcntl
import hashlib
import json
import math
import os
from pathlib import Path
import shutil
import struct
import subprocess
import sys
import tarfile
import tempfile

from run_0290_isolated_benchmark import require_isolation

ROOT = Path(__file__).resolve().parents[1]
REFERENCE = ROOT / 'docs/qwen4exp/reference_0335'
SOURCE = REFERENCE / 'source'
BUILD = REFERENCE / 'build'
REVISION = '712fec6b4297c9cfd32ab43daaa2979a9aa01edf'
ARCHIVE_SHA256 = '26984c09b4d42476a8a3dee72d227d4583a07d0640572a1cb7e25b4f4d27b4f6'
# Exact transitive project-local include closure; never extract arbitrary archive paths.
INPUTS = (
    'src/kernels/cpu/q2_avx2.cpp',
    'src/kernels/cpu/q2_avx2_rows.inl',
    'include/strata/kernels/cpu/expert.hpp',
    'include/strata/kernels/cpu/expert_layout.hpp',
    'include/strata/kernels/cpu/native_expert.hpp',
    'include/strata/core/geometry.hpp',
    'include/strata/core/geometries/flash_next.hpp',
    'include/strata/core/geometry_checks.hpp',
    'include/strata/artifact/model_family.hpp',
)
COMMON = ['-std=c++20', '-O3', '-DNDEBUG', '-fPIC', '-fvisibility=hidden', '-march=x86-64']
WRAPPER_FLAGS = COMMON + ['-mno-avx', '-mno-avx2', '-mno-fma', '-mno-f16c']
KERNEL_FLAGS = COMMON + ['-mavx2', '-mfma', '-mf16c', '-DSTRATA_AVXVNNI=1']
LINK_FLAGS = ['-shared', '-Wl,-z,defs', '-Wl,-z,noexecstack']


def sha256(path):
    digest = hashlib.sha256()
    with path.open('rb') as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b''):
            digest.update(chunk)
    return digest.hexdigest()


def prepare_source():
    archive = REFERENCE / 'strata_712fec6.tar.gz'
    if sha256(archive) != ARCHIVE_SHA256:
        raise RuntimeError('Pinned archive SHA256 mismatch; refusing extraction/build')
    # A generated geometry override would change ActQ and is not part of this diagnostic.
    if (SOURCE / 'include/strata/core/geometry_selected.hpp').exists():
        raise RuntimeError('Unexpected geometry_selected.hpp; require pinned default Flash-Next geometry')
    with tarfile.open(archive, 'r:gz') as tar:
        for relative in INPUTS:
            member = tar.getmember(f'Strata-{REVISION}/{relative}')
            if not member.isfile() or member.size > 1024 * 1024:
                raise RuntimeError(f'Not a bounded regular archive member: {relative}')
            data = tar.extractfile(member).read()
            target = SOURCE / relative
            # Refuse symlinks in the destination, including parent directories.
            if SOURCE.is_symlink() or any(p.is_symlink() for p in (target, *target.parents) if p != ROOT.parent):
                raise RuntimeError(f'Symlink destination refused: {target}')
            if not target.resolve().is_relative_to(SOURCE.resolve()):
                raise RuntimeError(f'Extraction escaped source: {target}')
            if target.exists():
                if target.read_bytes() != data:
                    raise RuntimeError(f'Existing pinned source differs from archive: {relative}')
            else:
                target.parent.mkdir(parents=True, exist_ok=True)
                with target.open('xb') as stream:
                    stream.write(data)
    return {relative: sha256(SOURCE / relative) for relative in INPUTS}


def build():
    require_isolation()
    BUILD.mkdir(parents=True, exist_ok=True)
    with (BUILD / '.build.lock').open('a') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX)
        sources = prepare_source()
        selected = os.environ.get('CXX', 'g++')
        compiler = shutil.which(selected)
        if not compiler:
            raise RuntimeError(f'Compiler executable not found: {selected}')
        compiler = str(Path(compiler).resolve())
        version = subprocess.run([compiler, '--version'], check=True, capture_output=True,
                                 text=True, timeout=10).stdout
        adapter = ROOT / 'scripts/native_0335_strata_q2.cpp'
        identity = dict(schema=1, revision=REVISION, archive_sha256=ARCHIVE_SHA256,
                        sources=sources, adapter_sha256=sha256(adapter),
                        builder_sha256=sha256(Path(__file__)), compiler=compiler,
                        compiler_sha256=sha256(Path(compiler)), compiler_version=version,
                        wrapper_flags=WRAPPER_FLAGS, kernel_flags=KERNEL_FLAGS,
                        link_flags=LINK_FLAGS, geometry='flash_next',
                        arithmetic='pinned AVX2 legacy; STRATA_Q2_BITPLANE read at library load',
                        dispatch='AVX-VNNI only with CPUID + OS YMM state; STRATA_NO_AVXVNNI disables',
                        include=str(SOURCE / 'include'))
        key = hashlib.sha256(json.dumps(identity, sort_keys=True).encode()).hexdigest()
        library = BUILD / f'libstrata0335_q2_{key}.so'
        manifest = library.with_suffix('.json')
        if library.exists() and manifest.exists():
            previous = json.loads(manifest.read_text())
            if previous['identity'] == identity and previous['library_sha256'] == sha256(library):
                return library, manifest, True
            raise RuntimeError('Cached artifact/provenance mismatch; refusing overwrite')
        if library.exists() or manifest.exists():
            raise RuntimeError('Incomplete cached build; preserve artifacts for inspection')
        commands = []
        with tempfile.TemporaryDirectory(prefix='.compile-', dir=BUILD) as temporary:
            temp = Path(temporary)
            for source, flags, obj in ((adapter, WRAPPER_FLAGS, temp / 'adapter.o'),
                                       (SOURCE / INPUTS[0], KERNEL_FLAGS, temp / 'kernel.o')):
                command = [compiler, *flags, '-I', str(SOURCE / 'include'),
                           '-c', str(source), '-o', str(obj)]
                commands.append(command)
                subprocess.run(command, check=True, timeout=60)
            command = [compiler, *LINK_FLAGS, str(temp / 'adapter.o'), str(temp / 'kernel.o'),
                       '-o', str(temp / 'adapter.so')]
            commands.append(command)
            subprocess.run(command, check=True, timeout=30)
            os.replace(temp / 'adapter.so', library)
        manifest.write_text(json.dumps(dict(identity=identity, build_key=key, commands=commands,
                                            library=str(library), library_sha256=sha256(library)),
                                       indent=2, sort_keys=True) + '\n')
        return library, manifest, False


def smoke(library):
    """Small synthetic ABI/numerical check, deliberately not a benchmark harness."""
    require_isolation()
    import ctypes as C
    lib = C.CDLL(str(library))
    ptr, size, integer = C.c_void_p, C.c_size_t, C.c_int
    fp, qp = C.POINTER(C.c_float), C.POINTER(C.c_int8)
    for name in ('available', 'vnni', 'bitplane', 'max_n', 'max_rows'):
        fn = getattr(lib, 'strata0335_q2_' + name)
        fn.argtypes, fn.restype = [], integer
    lib.strata0335_q2_act_alignment.argtypes = []
    lib.strata0335_q2_act_alignment.restype = size
    lib.strata0335_q2_create.argtypes, lib.strata0335_q2_create.restype = [], ptr
    lib.strata0335_q2_destroy.argtypes, lib.strata0335_q2_destroy.restype = [ptr], None
    lib.strata0335_q2_quantize.argtypes = [ptr, fp, integer]
    lib.strata0335_q2_import_q8_0.argtypes = [ptr, ptr, size, integer]
    lib.strata0335_q2_import_q8_0.restype = integer
    lib.strata0335_q2_export.argtypes = [ptr, qp, size, fp, size]
    lib.strata0335_q2_rows.argtypes = [ptr, ptr, size, size, integer, integer, integer, fp, size]
    for name in ('quantize', 'export', 'rows'):
        getattr(lib, 'strata0335_q2_' + name).restype = integer
    if not lib.strata0335_q2_available():
        assert not lib.strata0335_q2_create()
        return dict(skipped='CPU lacks AVX2/FMA/F16C or OS YMM state')
    handle = lib.strata0335_q2_create()
    assert handle and handle % lib.strata0335_q2_act_alignment() == 0
    checked, max_error = 0, 0.0
    imported_checked, imported_max_error = 0, 0.0
    try:
        maximum = lib.strata0335_q2_max_n()
        for n in (32, 64, 128, 192, 640, maximum):
            # Exact scale 1: ties verify half-away codes, with unequal chunks below.
            values = [127.0 if j % 32 == 0 else (0.5 if j % 2 else -0.5) for j in range(n)]
            x = (C.c_float * n)(*values)
            assert lib.strata0335_q2_quantize(handle, x, n) == 0
            q, scales = (C.c_int8 * n)(), (C.c_float * (n // 32))()
            assert lib.strata0335_q2_export(handle, q, n, scales, n // 32) == 0
            assert list(q) == [127 if j % 32 == 0 else (1 if j % 2 else -1) for j in range(n)]
            assert all(s == 1.0 for s in scales)
            assert lib.strata0335_q2_export(handle, q, n - 1, scales, n // 32) == 1
            if n % 64:
                assert lib.strata0335_q2_rows(handle, x, C.sizeof(x), 18, n, 0, 1, x, n) == 1
                continue
            for general, imported in ((False, False), (True, False), (True, True)):
                if imported and lib.strata0335_q2_bitplane():
                    continue
                if general:
                    for j in range(n):
                        x[j] = math.sin(j * 0.71) * (1 + (j // 32) % 7)
                    assert lib.strata0335_q2_quantize(handle, x, n) == 0
                    assert lib.strata0335_q2_export(handle, q, n, scales, n // 32) == 0
                if imported:
                    expected_q = list(q)
                    packed = b''.join(struct.pack('<e', scales[c]) +
                                      struct.pack('<32b', *expected_q[c*32:(c+1)*32])
                                      for c in range(n // 32))
                    expected_scales = [struct.unpack_from('<e', packed, c*34)[0]
                                       for c in range(n // 32)]
                    blocks = C.create_string_buffer(packed)
                    assert lib.strata0335_q2_import_q8_0(handle, blocks, len(packed), n) == 0
                    assert lib.strata0335_q2_export(handle, q, n, scales, n // 32) == 0
                    assert list(q) == expected_q and list(scales) == expected_scales
                nr, stride = 7, n // 64 * 18 + 5
                raw = bytearray(nr * stride)
                reference, magnitudes = [], []
                for r in range(nr):
                    terms = []
                    for block in range(n // 64):
                        dw = (0.5, 1.0, 2.0)[(block + r) % 3]
                        offset = r * stride + block * 18
                        raw[offset:offset + 2] = struct.pack('<e', dw)
                        for b in range(16):
                            codes = [(b * 7 + t + r + block) % 4 for t in range(4)]
                            raw[offset + 2 + b] = sum(c << (2 * t) for t, c in enumerate(codes))
                            for t, code in enumerate(codes):
                                j = block * 64 + b * 4 + t
                                terms.append(dw * (code - 1) * q[j] * scales[j // 32])
                    reference.append(math.fsum(terms))
                    magnitudes.append(math.fsum(abs(v) for v in terms))
                weights = (C.c_uint8 * len(raw)).from_buffer_copy(raw)
                output = (C.c_float * nr)(*([-999.0] * nr))
                args = (handle, weights, len(raw), stride, n)
                assert lib.strata0335_q2_rows(*args, 0, nr, output, nr) == 0
                for r, expected in enumerate(reference):
                    error = abs(output[r] - expected)
                    tolerance = 0.0 if not general else 1e-5 + 2e-5 * magnitudes[r]
                    assert error <= tolerance, dict(n=n, row=r, actual=output[r], reference=expected,
                        absolute_error=error, relative_error=error / max(abs(expected), 1e-30),
                        tolerance=tolerance, dtype='float32', finite_difference_step=None)
                    if imported:
                        imported_max_error = max(imported_max_error, error)
                        imported_checked += 1
                    else:
                        max_error = max(max_error, error)
                        checked += 1
                ranged = (C.c_float * nr)(*([-999.0] * nr))
                assert lib.strata0335_q2_rows(*args, 1, 6, ranged, nr) == 0
                assert ranged[0] == ranged[6] == -999.0
                assert list(ranged)[1:6] == list(output)[1:6]
                assert lib.strata0335_q2_rows(handle, weights, 1, stride, n, 0, nr, output, nr) == 1
                assert lib.strata0335_q2_rows(*args, -1, nr, output, nr) == 1
                assert lib.strata0335_q2_rows(*args, 0, nr, output, nr - 1) == 1
        x = (C.c_float * 32)(*([0.0] * 32))
        assert lib.strata0335_q2_quantize(handle, x, 0) == 1
        assert lib.strata0335_q2_quantize(handle, x, 31) == 1
        assert lib.strata0335_q2_quantize(handle, x, maximum + 32) == 1
        x[0] = float('nan')
        assert lib.strata0335_q2_quantize(handle, x, 32) == 3
        assert lib.strata0335_q2_export(handle, q, len(q), scales, len(scales)) == 1
        x[0] = 0.0
        assert lib.strata0335_q2_quantize(handle, x, 32) == 0
        q0, s0 = (C.c_int8 * 32)(), (C.c_float * 1)()
        assert lib.strata0335_q2_export(handle, q0, 32, s0, 1) == 0
        assert not any(q0) and s0[0] == 0
        packed = struct.pack('<e32b', 1.0, *range(-16, 16))
        blocks = C.create_string_buffer(packed)
        if lib.strata0335_q2_bitplane():
            assert lib.strata0335_q2_import_q8_0(handle, blocks, 34, 32) == 1
            assert lib.strata0335_q2_export(handle, q0, 32, s0, 1) == 1
        else:
            # Zero, signed zero, smallest/largest subnormal, smallest normal,
            # a nontrivial mantissa, unity and the largest finite half.
            for half in (0, 0x8000, 1, 0x3ff, 0x400, 0x3555, 0x3c00, 0x7bff):
                payload = struct.pack('<H32b', half, *range(-16, 16))
                edge = C.create_string_buffer(payload)
                assert lib.strata0335_q2_import_q8_0(handle, edge, 34, 32) == 0
                assert lib.strata0335_q2_export(handle, q0, 32, s0, 1) == 0
                assert list(q0) == list(range(-16, 16))
                assert struct.pack('<f', s0[0]) == struct.pack('<f', struct.unpack('<e', payload[:2])[0])
            bad_inputs = [(blocks, 34, 0, 1), (blocks, 34, 31, 1),
                          (blocks, 34, maximum+32, 1), (blocks, 33, 32, 1),
                          (blocks, 35, 32, 1), (None, 34, 32, 1)]
            for half in (0xbc00, 0x8001, 0x7c00, 0xfc00, 0x7e00):
                bad_inputs.append((C.create_string_buffer(struct.pack('<H', half) + packed[2:]), 34, 32, 3))
            bad_inputs.append((C.create_string_buffer(packed[:2] + b'\x80' + packed[3:]), 34, 32, 3))
            for data, length, n, status in bad_inputs:
                assert lib.strata0335_q2_import_q8_0(handle, blocks, 34, 32) == 0
                assert lib.strata0335_q2_import_q8_0(handle, data, length, n) == status
                assert lib.strata0335_q2_export(handle, q0, 32, s0, 1) == 1
            assert lib.strata0335_q2_import_q8_0(None, blocks, 34, 32) == 1
    finally:
        lib.strata0335_q2_destroy(handle)
    return dict(checked_rows=checked, max_absolute_error=max_error,
                imported_rows=imported_checked, imported_max_absolute_error=imported_max_error,
                vnni=lib.strata0335_q2_vnni(),
                bitplane=lib.strata0335_q2_bitplane(), max_n=maximum, passed=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--self-test', action='store_true')
    parser.add_argument('--_smoke', type=Path, help=argparse.SUPPRESS)
    args = parser.parse_args()
    require_isolation()
    if args._smoke:
        print(json.dumps(smoke(args._smoke)))
        return
    library, manifest, cached = build()
    result = dict(library=str(library), provenance=str(manifest), cached=cached)
    if args.self_test:
        tests = []
        for bitplane, no_vnni in (('0', False), ('0', True), ('1', False)):
            env = dict(os.environ, STRATA_Q2_BITPLANE=bitplane)
            env.pop('STRATA_NO_AVXVNNI', None)
            if no_vnni:
                env['STRATA_NO_AVXVNNI'] = '1'
            child = subprocess.run([sys.executable, str(Path(__file__).resolve()), '--_smoke', str(library)],
                                   env=env, check=True, text=True, capture_output=True, timeout=20)
            tests.append(dict(bitplane_requested=bitplane, no_vnni=no_vnni,
                              result=json.loads(child.stdout)))
        result['self_tests'] = tests
    print(json.dumps(result, indent=2))


if __name__ == '__main__':
    try:
        main()
    except (RuntimeError, OSError, ValueError, KeyError, subprocess.SubprocessError, tarfile.TarError) as exc:
        print(f'Strata Q2 diagnostic refused/failed: {exc}', file=sys.stderr)
        if isinstance(exc, subprocess.CalledProcessError):
            print(exc.stdout or '', file=sys.stderr)
            print(exc.stderr or '', file=sys.stderr)
        sys.exit(1)
