"""Bounded same-input CPU expert A/B; diagnostic only, never model throughput.

Pinned Strata GGML singleton traits are compared in the SAME local row pool.
Strata Q2 ActQ rows use a neutral native team against local Q8_0 dot rows.
Neither leg represents the whole Strata engine or its optional dispatch settings.
"""
from __future__ import annotations
import argparse
import ctypes as C
import hashlib
import json
import os
from pathlib import Path
import resource
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT), str(ROOT / 'scripts')]
REFERENCE = ROOT / 'docs/qwen4exp/reference_0335'


def metrics(value, reference):
    import numpy as np
    a, b = np.asarray(value, np.float64), np.asarray(reference, np.float64)
    if a.shape != b.shape or not np.isfinite(a).all() or not np.isfinite(b).all():
        raise ValueError('Nonfinite or mismatched comparison')
    delta = a - b
    return dict(max_abs=float(np.max(np.abs(delta))),
                relative_l2=float(np.linalg.norm(delta) / max(np.linalg.norm(b), 1e-30)),
                reference_norm=float(np.linalg.norm(b)))


def paired_samples(execute, names, trials, iterations):
    """Alternate order; each execute returns ms/call from the same timing scope."""
    samples = {name: [] for name in names}
    for _ in range(4):
        for name in names:
            execute(name, 1)
    for trial in range(trials):
        for name in names if trial % 2 == 0 else names[::-1]:
            samples[name].append(execute(name, iterations))
    return samples


def summaries(samples):
    import numpy as np
    return {name: dict(median_ms=float(np.median(v)), min_ms=min(v), max_ms=max(v),
                       samples_ms=v) for name, v in samples.items()}


def configure_q2(library):
    lib = C.CDLL(str(library))
    P, Z, I = C.c_void_p, C.c_size_t, C.c_int
    lib.strata0335_q2_create.argtypes = []
    lib.strata0335_q2_create.restype = P
    lib.strata0335_q2_destroy.argtypes = [P]
    lib.strata0335_q2_destroy.restype = None
    for name, args in [('quantize', [P, P, I]), ('export', [P, P, Z, P, Z]),
                       ('rows', [P, P, Z, Z, I, I, I, P, Z]),
                                              ('import_q8_0', [P, P, Z, I])]:
        fn = getattr(lib, 'strata0335_q2_' + name)
        fn.argtypes, fn.restype = args, I
    return lib


def configure_team(library):
    lib = C.CDLL(str(library))
    P, Z, I = C.c_void_p, C.c_size_t, C.c_int
    lib.strata0335_row_team_create.argtypes = [I]
    lib.strata0335_row_team_create.restype = P
    lib.strata0335_row_team_destroy.argtypes = [P]
    lib.strata0335_row_team_destroy.restype = None
    for mode in ('dot', 'q2'):
        fn = getattr(lib, 'strata0335_row_team_run_' + mode)
        fn.argtypes = [P, P, P, P, P, I, I, Z, I, P, I, C.POINTER(C.c_double)]
        fn.restype = I
    return lib


def q2_comparison(experts, x, base, q2, team_lib, participants, trials, iterations):
    """Same hidden FP32 vector for every down matrix: teacher-forced projection."""
    import numpy as np
    matrices = [ws[2] for ws in experts]
    n, k = matrices[0].shape
    if any(w.quant_type != 'Q2_0' or w.shape != (n, k) for w in matrices):
        raise ValueError('Q2 down comparison requires uniform raw Q2_0 matrices')
    x = np.ascontiguousarray(x, dtype=np.float32)
    if x.shape != (k,):
        raise ValueError('Wrong teacher-forced hidden shape')
    stride = k // 64 * 18
    pointers = np.asarray([w.data.ctypes.data for w in matrices], dtype=np.uintp)
    sizes = np.asarray([w.data.nbytes for w in matrices], dtype=np.uintp)
    handle = q2.strata0335_q2_create()
    matched_handle = q2.strata0335_q2_create()
    team = team_lib.strata0335_row_team_create(participants)
    if not handle or not matched_handle or not team:
        if matched_handle:
            q2.strata0335_q2_destroy(matched_handle)
        if handle:
            q2.strata0335_q2_destroy(handle)
        if team:
            team_lib.strata0335_row_team_destroy(team)
        raise RuntimeError('Native team/ActQ allocation failed')
    try:
        if q2.strata0335_q2_quantize(handle, x.ctypes.data, k):
            raise RuntimeError('Strata ActQ quantization failed')
        # GGML Q8_0 is a different activation representation; preserve both.
        activation = np.empty(k // 32 * 34, np.uint8)
        quant_address = base._functions['Q2_0'][1]
        C.CFUNCTYPE(None, C.c_void_p, C.c_void_p, C.c_int64)(quant_address)(
            x.ctypes.data, activation.ctypes.data, k)
        if q2.strata0335_q2_import_q8_0(matched_handle, activation.ctypes.data, activation.nbytes, k):
            raise RuntimeError('Q8_0 import failed; require bitplane mode off')
        outputs = {name: np.empty((len(matrices), n), np.float32)
                   for name in ('local_q8_0', 'strata_actq', 'strata_same_q8_0')}
        rows_address = C.cast(q2.strata0335_q2_rows, C.c_void_p).value

        def execute(name, repeats):
            elapsed = C.c_double()
            mode = 'dot' if name == 'local_q8_0' else 'q2'
            fn = getattr(team_lib, 'strata0335_row_team_run_' + mode)
            address = base._functions['Q2_0'][2] if mode == 'dot' else rows_address
            act = (activation.ctypes.data if mode == 'dot' else
                   matched_handle if name == 'strata_same_q8_0' else handle)
            rc = fn(team, address, act, pointers.ctypes.data, sizes.ctypes.data,
                    len(matrices), n, stride, k, outputs[name].ctypes.data,
                    repeats, C.byref(elapsed))
            if rc:
                raise RuntimeError(f'Neutral native team failed {mode}: {rc}')
            return elapsed.value / repeats / 1e6

        samples = paired_samples(execute, tuple(outputs), trials, iterations)
        # Independent bounded FP64 baseline from exact checkpoint block layout.
        q = np.empty(k, np.int8)
        scales = np.empty(k // 32, np.float32)
        if q2.strata0335_q2_export(handle, q.ctypes.data, k, scales.ctypes.data, len(scales)):
            raise RuntimeError('ActQ export failed')
        actq = q.astype(np.float64) * np.repeat(scales.astype(np.float64), 32)
        blocks = activation.reshape(-1, 34)
        q8 = blocks[:, 2:].copy().view(np.int8).astype(np.float64).reshape(-1)
        q8 *= np.repeat(blocks[:, :2].copy().view('<f2').astype(np.float64).reshape(-1), 32)
        refs = {name: [] for name in ('fp32', 'q8_0', 'actq')}
        for w in matrices:
            packed = w.data.reshape(n, k // 64, 18)
            d = packed[:, :, :2].copy().view('<f2').astype(np.float64).reshape(n, k // 64)
            codes = ((packed[:, :, 2:, None] >> np.array([0, 2, 4, 6], np.uint8)) & 3)
            decoded = (codes.reshape(n, k // 64, 64).astype(np.float64) - 1) * d[:, :, None]
            decoded = decoded.reshape(n, k)
            for name, v in [('fp32', x.astype(np.float64)), ('q8_0', q8), ('actq', actq)]:
                refs[name].append(decoded @ v)
        refs = {name: np.stack(v) for name, v in refs.items()}
        result = dict(participants=participants, experts=len(experts), k=k, n=n,
                      timing_scope='same neutral native row scheduler; quantization excluded',
                      warm_weight_bytes=sum(w.data.nbytes for w in matrices),
                      input_sha256=hashlib.sha256(x.tobytes()).hexdigest(),
                      timing=summaries(samples),
                      arithmetic_difference=metrics(outputs['strata_actq'], outputs['local_q8_0']),
                      activation_difference=metrics(actq, q8),
                      local_vs_own_decoded_baseline=metrics(outputs['local_q8_0'], refs['q8_0']),
                      strata_vs_own_decoded_baseline=metrics(outputs['strata_actq'], refs['actq']),
                      strata_same_q8_vs_baseline=metrics(outputs['strata_same_q8_0'], refs['q8_0']),
                      same_q8_arithmetic_difference=metrics(outputs['strata_same_q8_0'], outputs['local_q8_0']),
                      local_vs_fp32_baseline=metrics(outputs['local_q8_0'], refs['fp32']),
                      strata_vs_fp32_baseline=metrics(outputs['strata_actq'], refs['fp32']))
        result['row_speedup_local_over_strata'] = (
            result['timing']['local_q8_0']['median_ms'] / result['timing']['strata_actq']['median_ms'])
        result['same_q8_row_speedup'] = (
            result['timing']['local_q8_0']['median_ms'] / result['timing']['strata_same_q8_0']['median_ms'])
        result['effective_packed_gb_per_second'] = {
            name: result['warm_weight_bytes'] / timing['median_ms'] / 1e6
            for name, timing in result['timing'].items()}
        # Arithmetic validation is distinct from agreement between activation contracts.
        for name in ('local_vs_own_decoded_baseline', 'strata_vs_own_decoded_baseline',
                     'strata_same_q8_vs_baseline'):
            if result[name]['relative_l2'] > 1e-5:
                raise AssertionError(f'Q2 row implementation disagrees with own baseline: {result[name]}')
        return result
    finally:
        team_lib.strata0335_row_team_destroy(team)
        q2.strata0335_q2_destroy(handle)
        q2.strata0335_q2_destroy(matched_handle)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--trials', type=int, default=7)
    parser.add_argument('--iterations', type=int, default=30)
    parser.add_argument('--participants', type=int, nargs='+', default=[16, 20])
    args = parser.parse_args()
    if not 3 <= args.trials <= 15 or not 5 <= args.iterations <= 100:
        parser.error('Require 3..15 trials and 5..100 iterations')
    if not args.participants or any(p not in (16, 20) for p in args.participants):
        parser.error('Only 16 and 20 participants are permitted')
    from run_0290_isolated_benchmark import require_isolation
    require_isolation()
    args.output.mkdir(parents=True, exist_ok=False)
    os.environ['MINI_LLM_BACKEND'] = 'numpy'
    # Set before NumPy import: independent small baselines must not oversubscribe.
    os.environ['OPENBLAS_NUM_THREADS'] = os.environ['OMP_NUM_THREADS'] = '1'
    import numpy as np
    from mini_llm.ops.ggml_packed_cpu import GGMLPackedCPUExecutor, GGMLPackedCPURowPool
    from mini_llm.external.qwen4exp_gguf import Qwen4ExpGGUFIndex
    from mini_llm.ops.expert_residency import ExpertKey
    from probe_0218_hc_identity import MODEL
    from build_0335_strata_q2 import build, sha256, REVISION, ARCHIVE_SHA256
    q2_path, q2_manifest, _ = build()
    team_source = ROOT / 'scripts/native_0335_row_team.cpp'
    team_flags = ['-O3', '-std=c++17', '-shared', '-pthread', '-fPIC', '-Wl,-z,defs']
    digest = hashlib.sha256(team_source.read_bytes() + repr(team_flags).encode()).hexdigest()
    team_path = REFERENCE / 'build' / f'librow_team_{digest}.so'
    if not team_path.exists():
        subprocess.run(['g++', *team_flags, str(team_source), '-o', str(team_path)], check=True, timeout=30)
    q2, team = configure_q2(q2_path), configure_team(team_path)
    pinned = REFERENCE / 'ggml-build/src/libggml-cpu.so'
    local_library = Path('/home/lucka/Documents/my_local_LLM/qwen4exp-references/llama-no-prefetch/build-cpu/bin/libggml-cpu.so')
    # Separate dependent SONAMEs (0.24 vs0.25) coexist; function pointers stay owned.
    local = GGMLPackedCPUExecutor(local_library, q2_simd=True, iq3_gather=True,
        iq2_gather=False, iq4_vnni=False, iq3xxs_gather=False, iq3xxs_pair=False)
    strata = GGMLPackedCPUExecutor(pinned, q2_simd=False, iq3_gather=False,
        iq2_gather=False, iq4_vnni=False, iq3xxs_gather=False, iq3xxs_pair=False)
    capture_path = ROOT / 'docs/qwen4exp/0224_isolated_moe_v1.json'
    captures = json.loads(capture_path.read_text())['cases']
    report = dict(schema=1, scope='warm singleton CPU operators, not engine throughput',
        production_modified=False, reference_math_modified=False,
        strata_revision=REVISION, archive_sha256=ARCHIVE_SHA256,
        local=local.provenance, strata_traits=strata.provenance,
        strata_traits_scope='source-pinned GGML; native Q2 ActQ compared separately; not installed engine dispatch proof',
        q2_manifest=str(q2_manifest.relative_to(ROOT)),
        row_team_sha256=sha256(team_path), row_team_source_sha256=sha256(team_source),
        capture_sha256=sha256(capture_path), affinity=sorted(os.sched_getaffinity(0)),
        participants=args.participants, trials=args.trials, iterations=args.iterations,
        expert_batches=[], q2_projection_batches=[])
    index = Qwen4ExpGGUFIndex.open(MODEL, model_id='0335')
    result_path = args.output / 'result.json'
    try:
        for case in captures:
            if case['position'] != 0 or case['layer'] not in (0, 1, 2, 17):
                continue
            layer = case['layer']
            vectors = ROOT / f'docs/qwen4exp/diagnostics_0224_v1/p0_l{layer}'
            x = np.ascontiguousarray(np.load(vectors / 'input.npy').reshape(1, -1), dtype=np.float32)
            experts = [index.provider(ExpertKey(index.model_id, layer, int(e))) for e in case['reference_ids']]
            formats = tuple(w.quant_type for w in experts[0])
            for participants in args.participants:
                with GGMLPackedCPURowPool(local, index.d_model, index.d_ff, participants - 1,
                        scheduler='tasks', host_help=True, spin_us=50, tasks_per_worker=3) as pool:
                    ids = {}
                    original = local._functions.copy()
                    try:
                        for name, executor in [('local', local), ('strata_traits', strata)]:
                            local._functions = original if name == 'local' else strata._functions
                            ids[name] = np.asarray([pool.register((name, i), ws)
                                                   for i, ws in enumerate(experts)], np.int32)
                    finally:
                        local._functions = original
                    for count in (3, 6, 10):
                        outputs = {name: np.empty((count, index.d_model), np.float32) for name in ids}
                        def execute(name, repeats):
                            start = time.perf_counter_ns()
                            for _ in range(repeats):
                                pool._submit_registered(x, ids[name][:count], outputs[name]).result()
                            return (time.perf_counter_ns() - start) / repeats / 1e6
                        samples = paired_samples(execute, tuple(ids), args.trials, args.iterations)
                        row = dict(layer=layer, formats=formats, experts=count, participants=participants,
                            expert_ids=case['reference_ids'][:count],
                            input_sha256=hashlib.sha256(x.tobytes()).hexdigest(),
                            warm_packed_bytes=sum(w.data.nbytes for ws in experts[:count] for w in ws),
                            timing_scope='same local native row pool; includes shared/hidden quantization, SwiGLU and Python ABI boundary',
                            timing=summaries(samples), output_difference=metrics(outputs['local'], outputs['strata_traits']),
                            bit_exact=bool(np.array_equal(outputs['local'].view(np.uint32), outputs['strata_traits'].view(np.uint32))))
                        row['local_over_strata_traits_speedup'] = row['timing']['local']['median_ms'] / row['timing']['strata_traits']['median_ms']
                        report['expert_batches'].append(row)
                        print(json.dumps(row), flush=True)
                if formats[2] == 'Q2_0':
                    hidden = np.fromfile(vectors / 'ref_swiglu.f32', np.float32).reshape(10, index.d_ff)[0]
                    for count in (3, 6, 10):
                        row = q2_comparison(experts[:count], hidden, local, q2, team,
                                            participants, args.trials, args.iterations)
                        row.update(layer=layer, expert_ids=case['reference_ids'][:count])
                        report['q2_projection_batches'].append(row)
                        print(json.dumps(row), flush=True)
            # Do not retain mapped expert bundles from every case.
            experts = None
            index.clear_source_cache()
            report['peak_rss_bytes'] = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss * 1024
            with result_path.open('w') as stream:
                json.dump(report, stream, indent=2, allow_nan=False)
    finally:
        index.clear_source_cache()
        index.store.close()
    report['completed'] = True
    report['peak_rss_bytes'] = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss * 1024
    result_path.write_text(json.dumps(report, indent=2, allow_nan=False) + '\n')


if __name__ == '__main__':
    main()
