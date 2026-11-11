"""Independent, source-pinned llama.cpp logits checks; never a production backend.

Default: a small packed-original/all-F32-equivalent pair for strict graph parity.
--model: characterize the installed GGUF, explicitly accounting for the reference
CPU's Q8 activation operands rather than inventing a strict cross-engine tolerance.
"""

import argparse
import hashlib
import json
import os
from pathlib import Path
import shutil
import struct
import subprocess
import sys
import time

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
REVISION = '171e8846b4af9766c354064cb776cb34a50f053f'
SOURCE = ROOT / 'benchmarks/reference_0106q_llama_logits.cpp'
STRICT = dict(rtol=3e-4, atol=2e-5)


def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for chunk in iter(lambda: stream.read(8 * 1024**2), b''):
            digest.update(chunk)
    return digest.hexdigest()


def file_identity(path):
    path = Path(path).resolve()
    before = path.stat()
    identity = dict(path=str(path), bytes=before.st_size, mtime_ns=before.st_mtime_ns,
                    inode=before.st_ino, device=before.st_dev, sha256=sha256(path))
    verify_input_unchanged(identity)
    return identity


def verify_input_unchanged(identity):
    stat = Path(identity['path']).stat()
    if (stat.st_size, stat.st_mtime_ns, stat.st_ino, stat.st_dev) != (
            identity['bytes'], identity['mtime_ns'], identity['inode'], identity['device']):
        raise ValueError('Reference model input changed during comparison')


def validate_loaded_libraries(metadata, provenance):
    loaded = metadata.get('loaded_reference_libraries', [])
    if not loaded:
        raise ValueError('Exporter did not identify loaded reference libraries')
    expected = {value['path']: value['sha256']
                for value in provenance['linked_library_files'].values()}
    for name in loaded:
        path = str(Path(name).resolve())
        if path not in expected or sha256(path) != expected[path]:
            raise ValueError(f'Loaded reference library does not match selected build: {path}')


def read_reference(path):
    """Bounded little-endian, identity-carrying exporter protocol; no pickle."""
    path = Path(path)
    with path.open('rb') as stream:
        header = stream.read(16)
    if len(header) != 16 or header[:8] != b'MLLQREF1':
        raise ValueError('Invalid independent reference header')
    rows, vocab = struct.unpack('<II', header[8:])
    if not 1 <= rows <= 64 or not 1 <= vocab <= 2000000:
        raise ValueError('Unbounded reference dimensions')
    if path.stat().st_size != 16 + rows * 4 + rows * vocab * 4:
        raise ValueError('Reference binary length disagrees with header')
    with path.open('rb') as stream:
        stream.seek(16)
        tokens = np.fromfile(stream, dtype='<i4', count=rows)
        logits = np.fromfile(stream, dtype='<f4', count=rows * vocab).reshape(rows, vocab)
    if not np.isfinite(logits).all() or np.any(tokens < 0) or np.any(tokens >= vocab):
        raise ValueError('Nonfinite logits or out-of-vocabulary reference IDs')
    return tokens, logits


def compare_logits(actual, reference, tokens, reference_tokens, tolerance=None):
    """Compare identities before values; characterize near-zero/cancellation errors."""
    actual, reference = np.asarray(actual), np.asarray(reference)
    ids, ref_ids = np.asarray(tokens), np.asarray(reference_tokens)
    if ids.ndim != 1 or ref_ids.ndim != 1 or ids.dtype.kind not in 'iu' or ref_ids.dtype.kind not in 'iu':
        raise ValueError('Token identities must be one-dimensional integer arrays')
    if not np.array_equal(ids, ref_ids):
        raise ValueError('Refusing to compare different token IDs')
    if (actual.ndim != 2 or actual.shape != reference.shape or actual.shape[0] != len(ids)
            or actual.shape[1] < 2 or actual.dtype != np.float32 or reference.dtype != np.float32):
        raise ValueError('Expected matched FP32 [positions,vocabulary] logits')
    if not np.isfinite(actual).all() or not np.isfinite(reference).all():
        raise ValueError('Nonfinite logits cannot establish numerical parity')
    a, r = actual.astype(np.float64), reference.astype(np.float64)
    delta = a - r
    error = np.abs(delta)
    worst = np.unravel_index(int(error.argmax()), error.shape)
    rows = []
    for n in range(len(ids)):
        ai = np.argsort(-a[n])[:min(5, a.shape[1])]
        ri = np.argsort(-r[n])[:min(5, r.shape[1])]
        denom = float(np.linalg.norm(a[n]) * np.linalg.norm(r[n]))
        rows.append(dict(position=n, input_token_id=int(ids[n]), actual_top_ids=ai.tolist(),
            reference_top_ids=ri.tolist(), top1_equal=bool(ai[0] == ri[0]),
            actual_top1_margin=float(a[n, ai[0]] - a[n, ai[1]]),
            reference_top1_margin=float(r[n, ri[0]] - r[n, ri[1]]),
            top5_overlap=len(set(ai) & set(ri)),
            cosine_similarity=float(np.dot(a[n], r[n]) / denom) if denom else None,
            max_abs_error=float(error[n].max()), rms_error=float(np.sqrt(np.mean(delta[n]**2)))))
    metric = dict(dtype='float32', reference_dtype='float32', shape=list(a.shape),
        max_abs_error=float(error.max()), mean_abs_error=float(error.mean()),
        rms_error=float(np.sqrt(np.mean(delta**2))),
        relative_l2_error=float(np.linalg.norm(delta) / max(np.linalg.norm(r), np.finfo(np.float64).tiny)),
        centered_rms_error=float(np.sqrt(np.mean((delta - delta.mean(axis=1, keepdims=True))**2))),
        worst_index=[int(v) for v in worst], actual=float(a[worst]), reference=float(r[worst]),
        relative_error_at_worst=float(error[worst] / max(abs(r[worst]), np.finfo(np.float64).tiny)),
        greedy_ids_equal=all(row['top1_equal'] for row in rows), positions=rows,
        strict_parity_passed=None)
    if tolerance is not None:
        limit = tolerance['atol'] + tolerance['rtol'] * np.abs(r)
        metric.update(tolerance=tolerance, max_tolerance_ratio=float((error / limit).max()),
                      strict_parity_passed=bool(np.all(error <= limit)))
    return metric


def logged_run(command, prefix, *, env=None, timeout=180):
    start = time.perf_counter()
    result = subprocess.run([str(v) for v in command], cwd=ROOT, env=env,
                            capture_output=True, text=True, timeout=timeout)
    Path(str(prefix) + '.stdout.txt').write_text(result.stdout)
    Path(str(prefix) + '.stderr.txt').write_text(result.stderr)
    if result.returncode:
        raise RuntimeError(f'Command exited {result.returncode}; logs: {prefix}\n{result.stderr[-5000:]}')
    return result, (time.perf_counter() - start) * 1000


def build_exporter(llama_root, lib_dir, work, expected_revision):
    llama_root, lib_dir = Path(llama_root).resolve(), Path(lib_dir).resolve()
    revision = subprocess.check_output(['git', '--no-pager', '-C', str(llama_root),
                                        'rev-parse', 'HEAD'], text=True).strip()
    if revision != expected_revision:
        raise ValueError(f'Reference source revision {revision} != expected {expected_revision}')
    subprocess.run(['git', '--no-pager', '-C', str(llama_root), 'diff', '--quiet',
                    'HEAD', '--', 'src', 'include', 'ggml'], check=True)
    compiler = shutil.which('c++')
    if compiler is None:
        raise RuntimeError('A local C++17 compiler is required')
    library = lib_dir / 'libllama.so'
    if not library.is_file():
        raise FileNotFoundError(library)
    info = lib_dir.parent / 'common/build-info.cpp'
    cache = lib_dir.parent / 'CMakeCache.txt'
    if not info.is_file() or not cache.is_file():
        raise ValueError('Selected build must provide common/build-info.cpp and CMakeCache.txt')
    if revision[:9] not in info.read_text():
        raise ValueError('Selected build-info does not match source revision')
    homes = [line.split('=', 1)[1] for line in cache.read_text().splitlines()
             if line.startswith('CMAKE_HOME_DIRECTORY:INTERNAL=')]
    if len(homes) != 1 or Path(homes[0]).resolve() != llama_root:
        raise ValueError('Selected build CMake source directory does not match checked source')
    executable = work / 'llama_logits'
    command = [compiler, '-std=c++17', '-O2', '-DNDEBUG', '-DLLAMA_SHARED', '-DGGML_SHARED',
        '-DGGML_BACKEND_SHARED', '-I' + str(llama_root / 'include'),
        '-I' + str(llama_root / 'ggml/include'), str(SOURCE), '-o', str(executable),
        '-L' + str(lib_dir), '-Wl,-rpath,' + str(lib_dir), '-lllama']
    logged_run(command, work / 'build', timeout=60)
    libraries = {path.name: dict(path=str(path.resolve()), sha256=sha256(path))
                 for path in [library, *sorted(lib_dir.glob('libggml*.so'))] if path.is_file()}
    return executable, dict(source_revision=revision, source_root=str(llama_root),
        source_sha256=sha256(SOURCE), executable_sha256=sha256(executable),
        compiler=compiler, compile_command=command, linked_library_files=libraries,
        generated_build_info=info.read_text(), build_info_path=str(info),
        cmake_cache_sha256=sha256(cache), selected_library_dir=str(lib_dir),
        loader_policy='LD_PRELOAD removed; actual loaded reference library paths and hashes verified',
        execution='CPU, zero GPU layers, FP32 KV, no flash attention or backend sampling',
        cuda_initialization_caveat='Existing CUDA-linked reference may enumerate/probe GPU devices; CPU-only execution is not a no-CUDA-initialization guarantee')


def export_local(args):
    from mini_llm.backend import BACKEND_NAME, asnumpy
    if BACKEND_NAME != 'numpy':
        raise RuntimeError('MyLocal reference export requires isolated NumPy backend')
    from mini_llm.external.qwen3next_gguf import Qwen3NextGGUFIndex
    from mini_llm.external.qwen3next_model import Qwen3NextGGUFModel
    ids = np.asarray([int(token) for token in args.tokens_file.read_text().split()], np.int32)
    if not 1 <= len(ids) <= 64 or len(ids) > args.context:
        raise ValueError('Invalid bounded token sequence')
    index = Qwen3NextGGUFIndex.open(args.model)
    with Qwen3NextGGUFModel.from_index(index, cpu_workers=2) as model:
        full = asnumpy(model.forward(ids[None]))[0]
        state = model.create_generation_state(1, args.context)
        incremental = []
        for token in ids:
            incremental.append(asnumpy(model.decode_one([int(token)], state))[0])
        np.savez(args.export_local, token_ids=ids, full_logits=full,
                 incremental_logits=np.stack(incremental))
        print(json.dumps(dict(backend='numpy', dtype='float32', layers=len(model.blocks),
                              state_length=state.length, memory_plan=model.memory_plan(1, args.context))))


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--llama-root', type=Path)
    parser.add_argument('--llama-lib-dir', type=Path)
    parser.add_argument('--expected-revision', default=REVISION)
    parser.add_argument('--model', type=Path)
    parser.add_argument('--token-ids', type=int, nargs='+', default=[750, 912, 2877, 11])
    parser.add_argument('--context', type=int, default=16)
    parser.add_argument('--threads', type=int, default=2)
    parser.add_argument('--output', type=Path)
    parser.add_argument('--export-local', type=Path, help=argparse.SUPPRESS)
    parser.add_argument('--tokens-file', type=Path, help=argparse.SUPPRESS)
    args = parser.parse_args(argv)
    if not 1 <= args.context <= 128 or not 1 <= args.threads <= 8:
        parser.error('Require context 1..128 and threads 1..8')
    if args.export_local:
        export_local(args)
        return
    if not args.llama_root or not args.output:
        parser.error('--llama-root and --output are required for comparison')
    args.output = args.output.resolve()
    work = args.output.with_suffix('')
    if args.output.exists() or work.exists():
        parser.error('Refusing to overwrite existing report/artifacts; choose a new --output')
    args.output.parent.mkdir(parents=True, exist_ok=True)
    work.mkdir()
    report = dict(patch='0106Q', status='running', strict_reference_parity_passed=None,
                  production_backend_replaced=False, comparisons={})
    def save():
        args.output.write_text(json.dumps(report, indent=2, allow_nan=False) + '\n')
    stage = 'fixture/inputs'
    try:
        if args.model:
            model_path = reference_path = args.model.resolve()
            ids = args.token_ids
            precision = 'Actual packed GGUF: llama.cpp CPU quantizes activation operands; cross-engine metrics are characterization, not strict FP32 parity'
        else:
            from benchmarks.fixture_0106q_qwen_reference import write_fixture
            fixture = write_fixture(work / 'fixture')
            model_path, reference_path, ids = fixture['model_path'], fixture['reference_path'], fixture['token_ids']
            report['fixture'] = fixture['metadata']
            precision = 'Identical dequantized weight values: packed original uses FP32 activations; reference GGUF contains all-F32 weights, avoiding Q8 activation conversion'
        if not 1 <= len(ids) <= min(64, args.context) or any(token < 0 or token > 2**31-1 for token in ids):
            raise ValueError('Token sequence exceeds explicit bounds')
        token_path = work / 'tokens.txt'
        token_path.write_text(' '.join(str(token) for token in ids) + '\n')
        report.update(model_path=str(model_path), reference_model_path=str(reference_path),
                      token_ids=ids, precision_contract=precision, context=args.context)
        stage = 'input_fingerprints'
        model_identity = file_identity(model_path)
        reference_identity = (model_identity if reference_path.resolve() == model_path.resolve()
                              else file_identity(reference_path))
        report['model_identity'], report['reference_model_identity'] = model_identity, reference_identity
        stage = 'build/source provenance'
        lib_dir = args.llama_lib_dir or args.llama_root / 'build-cuda/bin'
        executable, provenance = build_exporter(args.llama_root, lib_dir, work, args.expected_revision)
        report['reference_provenance'] = provenance
        reference_arrays = {}
        for mode, chunk in (('full', len(ids)), ('incremental', 1)):
            stage = 'independent_reference/' + mode
            output = work / f'reference_{mode}.bin'
            command = [executable, '--model', reference_path, '--tokens', token_path,
                '--output', output, '--chunk', chunk, '--threads', args.threads, '--context', args.context]
            env = dict(os.environ)
            env.pop('LD_PRELOAD', None)
            result, wall = logged_run(command, work / f'reference_{mode}', env=env)
            metadata = json.loads(result.stdout.strip().splitlines()[-1])
            validate_loaded_libraries(metadata, provenance)
            ref_ids, logits = read_reference(output)
            if not np.array_equal(ref_ids, ids):
                raise ValueError('Independent runtime returned different token identity')
            reference_arrays[mode] = logits
            report['reference_' + mode] = dict(command=[str(v) for v in command],
                run_wall_ms=wall, metadata=metadata, artifact_sha256=sha256(output))
        stage = 'mylocal/isolated_cpu_export'
        local_path = work / 'mylocal.npz'
        command = [sys.executable, str(Path(__file__).resolve()), '--export-local', local_path,
            '--model', model_path, '--tokens-file', token_path, '--context', args.context]
        result, wall = logged_run(command, work / 'mylocal', timeout=240,
            env=dict(os.environ, MINI_LLM_BACKEND='numpy', OPENBLAS_NUM_THREADS='1', OMP_NUM_THREADS='1'))
        report['mylocal'] = dict(metadata=json.loads(result.stdout.strip().splitlines()[-1]),
                                run_wall_ms=wall, artifact_sha256=sha256(local_path))
        with np.load(local_path, allow_pickle=False) as local:
            local_ids = local['token_ids']
            full, incremental = local['full_logits'], local['incremental_logits']
        verify_input_unchanged(model_identity)
        verify_input_unchanged(reference_identity)
        stage = 'logits_comparison'
        strict = None if args.model else STRICT
        for name, actual, reference in (
            ('local_full_vs_reference_full', full, reference_arrays['full']),
            ('local_incremental_vs_reference_incremental', incremental, reference_arrays['incremental']),
            ('local_full_vs_local_incremental', full, incremental),
            ('reference_full_vs_reference_incremental', reference_arrays['full'], reference_arrays['incremental']),
        ):
            report['comparisons'][name] = compare_logits(actual, reference, local_ids, np.asarray(ids), strict)
        greedy = all(metric['greedy_ids_equal'] for metric in report['comparisons'].values())
        report['greedy_agreement_all_comparisons'] = greedy
        if not args.model:
            report['strict_reference_parity_passed'] = all(
                metric['strict_parity_passed'] for metric in report['comparisons'].values())
            report['status'] = 'strict_fixture_verified' if report['strict_reference_parity_passed'] else 'strict_fixture_failed'
            save()
            if not report['strict_reference_parity_passed']:
                raise AssertionError('Independent F32 fixture parity failed; inspect saved metrics before edits')
        else:
            report['status'] = 'quantized_reference_characterized' if greedy else 'quantized_reference_greedy_difference'
            save()
            if not greedy:
                raise AssertionError('Matched-token greedy reference difference; no implementation defect inferred')
    except Exception as exc:
        report['failure'] = dict(stage=stage, type=type(exc).__name__, message=str(exc))
        if report['status'] == 'running':
            report['status'] = 'failed'
        save()
        raise
    save()
    print(json.dumps(dict(patch='0106Q', status=report['status'],
                         strict_reference_parity_passed=report['strict_reference_parity_passed'],
                         greedy_agreement=greedy, output=str(args.output)), indent=2))
    return report


if __name__ == '__main__':
    main()
