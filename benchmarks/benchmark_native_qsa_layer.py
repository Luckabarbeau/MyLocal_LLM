"""Bounded actual-weight full QSA step benchmark; run as a module from repo root.

Uses the unchanged Python reference and validated test state/readback helpers.
No full model is constructed. CuPy uses one decoded FP32 attention layer only.
"""
import argparse
from contextlib import ExitStack
from datetime import datetime, timezone
import gc
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
from time import perf_counter_ns
from types import SimpleNamespace

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'tests'))
from mini_llm.backend import BACKEND_NAME, xp, asnumpy, synchronize
from mini_llm.native import Runtime
from mini_llm.external.qwen4exp_attention import QSAConfig, Qwen4ExpAttention
from mini_llm.ops.quantized_weight import PackedGGUFWeight
from test_native_engine_qsa_layer import (NAMES, STEMS, NORMS, KINDS, hidden,
    seed_reference, import_state, read, compare_state, inputs)

LENGTHS = [128, 2051, 2052, 2053, 4096, 8192, 16384]
LIMIT = 512 * 2**20


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def decode(raw, d, start, count):
    width = d.ggml_shape[0]
    chunk = raw[start*d.row_bytes:(start+count)*d.row_bytes]
    if d.type_id == 0:
        return np.frombuffer(chunk, '<f4').reshape(count, width)
    if d.type_id == 30:
        return (np.frombuffer(chunk, '<u2').astype(np.uint32) << 16).view(np.float32).reshape(count, width)
    return PackedGGUFWeight(np.frombuffer(chunk, np.uint8),
                            (count, width), KINDS[d.type_id]).dequantize_reference()


def actual_weights(stack, store, engine, layer_index):
    tensors, callbacks, provenance = [], {}, []
    prefix = f'blk.{layer_index}.'
    for name, stem in zip(NAMES, STEMS + ['attn_output.weight']):
        lease = stack.enter_context(store.lease(prefix + stem))
        d = lease.descriptor
        raw = lease.read(0, d.nbytes)
        tensors.append(stack.enter_context(engine.upload(lease)))
        provenance.append(dict(name=prefix+stem, type_id=d.type_id,
            shape=list(d.ggml_shape), nbytes=d.nbytes, sha256=hashlib.sha256(raw).hexdigest()))
        rows, width = d.ggml_shape[1], d.ggml_shape[0]
        if BACKEND_NAME == 'cupy':
            # Bounded CPU decode tiles, one FP32 attention layer resident on GPU.
            matrix = xp.empty((rows, width), xp.float32)
            for start in range(0, rows, 32):
                count = min(32, rows-start)
                matrix[start:start+count] = xp.asarray(decode(raw, d, start, count))
            callbacks[name] = lambda x, matrix=matrix: matrix @ xp.asarray(x)
        else:
            def project(x, raw=raw, d=d, rows=rows):
                y = np.empty(rows, np.float32)
                for start in range(0, rows, 32):
                    count = min(32, rows-start)
                    y[start:start+count] = decode(raw, d, start, count) @ x
                return y
            callbacks[name] = project
    gammas = []
    for stem in NORMS:
        lease = stack.enter_context(store.lease(prefix+stem))
        raw = lease.read(0, lease.descriptor.nbytes)
        gammas.append(xp.asarray(np.frombuffer(raw, '<f4').copy()))
        tensors.append(stack.enter_context(engine.upload(lease)))
        provenance.append(dict(name=prefix+stem, type_id=0,
            nbytes=len(raw), sha256=hashlib.sha256(raw).hexdigest()))
    return tensors, callbacks, gammas, provenance


def host_state(state):
    return SimpleNamespace(length=state.length, **{name: asnumpy(getattr(state, name)).copy()
        for name in ['raw_index', 'keys', 'values', 'pooled']})


def seed(ref, capacity):
    # hidden(t) has exactly 11 distinct vectors. Memoize only setup projections;
    # each position still uses the reference norm/RoPE and chronological pooling.
    original = ref.projections.copy()
    for name in ['key', 'value', 'index_key']:
        project, rows = original[name]
        cache = {}
        def cached(x, project=project, cache=cache):
            key = x.tobytes()
            if key not in cache:
                cache[key] = project(x)
            return cache[key]
        ref.projections[name] = cached, rows
    try:
        state = seed_reference(ref, capacity, capacity-1)
        return host_state(state)
    finally:
        ref.projections = original


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--device', type=int, default=0)
    p.add_argument('--layer', type=int, choices=list(range(3,48,4)), default=3)
    p.add_argument('--samples', type=int, default=3)
    p.add_argument('--warmups', type=int, default=1)
    p.add_argument('--output', type=Path, required=True)
    args = p.parse_args()
    if not 1 <= args.samples <= 10 or not 0 <= args.warmups <= 3:
        p.error('samples must be 1..10, warmups 0..3')
    if BACKEND_NAME not in ('numpy', 'cupy'):
        p.error('use MINI_LLM_BACKEND=cupy or numpy')
    if os.environ.get('MINI_LLM_QSA_NATIVE_SELECT', '0') != '0':
        p.error('MINI_LLM_QSA_NATIVE_SELECT must be 0 for Python reference')
    if BACKEND_NAME == 'cupy':
        xp.cuda.Device(args.device).use()
        # Separate hard payload admissions sum to 512 MiB; sampled usage is
        # required to remain strictly below that limit.
        xp.get_default_memory_pool().set_limit(size=320*2**20)
    inventory_path = ROOT / 'docs/native_engine/WP2_GGUF_INVENTORY.json'
    inventory = json.loads(inventory_path.read_text())
    runtime = Runtime()
    source_paths = [Path(__file__), ROOT/'tests/test_native_engine_qsa_layer.py',
        ROOT/'mini_llm/native/__init__.py', ROOT/'mini_llm/backend.py']
    source_paths += list((ROOT/'mini_llm/native_cpp').glob('*'))
    source_paths += list((ROOT/'mini_llm/external').glob('qwen4exp*.py'))
    source_paths += [ROOT/'mini_llm/ops'/name for name in
        ['quantized_weight.py', 'rmsnorm_inference.py', 'rotary_inference.py']]
    sources = {str(f.relative_to(ROOT)): sha(f) for f in source_paths if f.is_file()}
    library = Path(os.environ['MINI_LLM_NATIVE_LIBRARY']).resolve()
    library_hash = sha(library)
    rows, memory_samples = [], []
    def memory(engine):
        pool = xp.get_default_memory_pool() if BACKEND_NAME == 'cupy' else None
        entry = dict(native_device_bytes=engine.stats.device_bytes,
            native_pinned_bytes=engine.stats.pinned_bytes,
            cupy_used_bytes=pool.used_bytes() if pool else 0,
            cupy_reserved_bytes=pool.total_bytes() if pool else 0)
        entry['combined_reserved_payload_bytes'] = entry['native_device_bytes'] + entry['cupy_reserved_bytes']
        assert entry['combined_reserved_payload_bytes'] < LIMIT, entry
        memory_samples.append(entry)
    def trim():
        gc.collect()
        if BACKEND_NAME == 'cupy':
            xp.get_default_memory_pool().free_all_blocks()
    with tempfile.TemporaryDirectory(prefix='wp3-layer-') as tmp, ExitStack() as stack:
        path = Path(tmp)
        engine = stack.enter_context(runtime.engine(device=args.device, budget_bytes=192*2**20))
        store = stack.enter_context(runtime.gguf(inventory['shards'][0]['path']))
        tensors, callbacks, gammas, provenance = actual_weights(stack, store, engine, args.layer)
        for policy in ['ggml_cpu', 'strata']:
            ref = Qwen4ExpAttention(QSAConfig(2560, tie_policy=policy), **callbacks,
                gamma_q=gammas[0], gamma_k=gammas[1], gamma_index_q=gammas[2], gamma_index_k=gammas[3])
            if policy == 'ggml_cpu':
                baseline = seed(ref, 16384)
                trim()
            for n in LENGTHS:
                capacity, start = (4096 if n <= 4096 else 16384), n-1
                xs = hidden(start)
                x = xp.asarray(xs)
                native_ms, reference_ms, errors = [], [], []
                with ExitStack() as case:
                    layer = case.enter_context(engine.qsa_layer(capacity, *tensors, policy=policy))
                    upload = case.enter_context(inputs(path, runtime, engine, {'h':xs}))[0]
                    for trial in range(args.warmups + args.samples):
                        layer.reset()
                        prior = SimpleNamespace(length=start, **{name: getattr(baseline, name)[:(start//4 if name=='pooled' else start)]
                            for name in ['raw_index', 'keys', 'values', 'pooled']})
                        # Import uploads are closed before timing and reference allocation.
                        with ExitStack() as imports:
                            import_state(imports, path, runtime, engine, layer, prior)
                            memory(engine)
                        state = ref.create_state(capacity)
                        for name in ['raw_index', 'keys', 'values', 'pooled']:
                            src = getattr(prior, name)
                            getattr(state, name)[:len(src)] = xp.asarray(src)
                        state.length = start
                        synchronize()
                        memory(engine)
                        before = perf_counter_ns()
                        committed = layer.step(upload, start)
                        native_time = (perf_counter_ns()-before)/1e6
                        assert committed == n
                        before = perf_counter_ns()
                        want = ref.step(x, state, position=start)
                        synchronize()
                        reference_time = (perf_counter_ns()-before)/1e6
                        # Diagnostics are deliberately outside both timed intervals.
                        got = read(layer, 'output', (2560,))
                        want_host = asnumpy(want)
                        np.testing.assert_allclose(got, want_host, rtol=2e-4, atol=2e-4)
                        compare_state(layer, host_state(state))
                        del want, state
                        # A separate replay proves exact selection and score agreement
                        # without charging diagnostics only to the reference timing.
                        state = ref.create_state(capacity)
                        for name in ['raw_index', 'keys', 'values', 'pooled']:
                            src = getattr(prior, name)
                            getattr(state, name)[:len(src)] = xp.asarray(src)
                        state.length = start
                        _, diag = ref.step(x, state, position=start, diagnostics=True)
                        np.testing.assert_array_equal(read(layer,'ids',(min(n,2051),),np.int64),asnumpy(diag['selected_ids']))
                        np.testing.assert_allclose(read(layer,'scores',(n,)),asnumpy(diag['cell_scores']),rtol=5e-5,atol=2e-4)
                        memory(engine)
                        if trial >= args.warmups:
                            native_ms.append(native_time)
                            reference_ms.append(reference_time)
                            errors.append(float(np.max(np.abs(got-want_host))))
                        del state, diag, _
                        trim()
                    rows.append(dict(n=n, capacity=capacity, policy=policy, validation='passed every trial including warmups',
                        native_wall_ms=native_ms, reference_wall_ms=reference_ms,
                        native_median_ms=float(np.median(native_ms)), reference_median_ms=float(np.median(reference_ms)),
                        max_output_abs=max(errors)))
                    print(json.dumps(rows[-1]), flush=True)
                del layer, upload, x
                trim()
        native_peak = engine.stats.peak_bytes
    assert sha(library) == library_hash, 'native library changed during benchmark'
    changed = [name for name, digest in sources.items() if sha(ROOT/name) != digest]
    result = dict(timestamp_utc=datetime.now(timezone.utc).isoformat(), argv=sys.argv,
        backend=BACKEND_NAME, numpy_version=np.__version__, cupy_version=xp.__version__ if BACKEND_NAME=='cupy' else None,
        device=args.device, gpu=subprocess.check_output(['nvidia-smi','--query-gpu=name,driver_version','--format=csv'],text=True),
        layer=args.layer, samples=args.samples, warmups=args.warmups, lengths=LENGTHS,
        timing='perf_counter_ns full API wall time; native includes scalar status/commit sync; reference includes stream sync; excludes imports, reset, setup, uploads, diagnostics',
        precision='FP32 projections, norms, RoPE, scoring, attention, gate, output; native packed weights vs same bytes decoded exactly to FP32 for reference; no TF32 request',
        reference_projection='CuPy FP32 dense GEMV decoded once outside timing' if BACKEND_NAME=='cupy' else 'NumPy CPU tiled decode plus GEMV inside timing (not matched hardware)',
        caveat='Python reference retains its host selection and per-block Python loop, even on CuPy. One-layer microbenchmark, not full model or model speedup evidence.',
        trajectory='tests.hidden(t): 0.2*sin(arange(2560)*0.03125+(t%11)*0.125); reference-operator seeded valid history; exact same pre-step arrays and hidden per trial',
        selection_native_reference_override=False, fast_decode=False, payload_limit_bytes=LIMIT,
        native_budget_bytes=192*2**20, native_peak_bytes=native_peak,
        combined_reserved_payload_peak_observed=max(m['combined_reserved_payload_bytes'] for m in memory_samples),
        cupy_pool_limit_bytes=320*2**20 if BACKEND_NAME=='cupy' else 0,
        checkpoint_inventory=str(inventory_path.relative_to(ROOT)), inventory_sha256=sha(inventory_path),
        checkpoint_shard=inventory['shards'][0]['path'], tensor_payloads=provenance,
        native_library=dict(path=str(library),sha256=library_hash), source_sha256=sources,
        source_files_changed_during_run=changed, results=rows, speedup_claim=None,
        environment={k:os.environ.get(k) for k in ['MINI_LLM_BACKEND','MINI_LLM_QSA_NATIVE_SELECT','OPENBLAS_NUM_THREADS','OMP_NUM_THREADS','NVIDIA_TF32_OVERRIDE']})
    args.output.write_text(json.dumps(result, indent=2)+'\n')
    assert not changed, f'sources changed during benchmark: {changed}'
    print(f'Wrote {args.output}: {len(rows)} validated cases', flush=True)


if __name__ == '__main__':
    main()
