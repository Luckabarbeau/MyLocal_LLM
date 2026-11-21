#!/usr/bin/env python3
"""Guarded actual-weight Q8 projection A/B/C, never a model benchmark.

The retained layer-0 FFN-normalized input is teacher-forced into the layer-0
GDN QKV/Z weights. It is NOT a captured GDN activation. Each timed graph holds
100 evaluations; CUDA events exclude setup and repeated Python launch gaps.
"""
import argparse
from contextlib import contextmanager
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import traceback

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT), str(ROOT / 'scripts')]
WEIGHT_LIMIT = 64 * 1024**2
INPUT = ROOT / 'docs/qwen4exp/diagnostics_0224_v1/p0_l0/input.npy'


def sha256(data):
    return hashlib.sha256(data).hexdigest()


@contextmanager
def retain_quantizations(q8):
    """Observe the unchanged codec and own all scratch used by a captured graph."""
    original = q8.quantize_q8_1
    retained = []
    def quantize(*args, **kwargs):
        result = original(*args, **kwargs)
        retained.append(result)
        return result
    q8.quantize_q8_1 = quantize
    try:
        yield retained
    finally:
        q8.quantize_q8_1 = original


def bit_checks(np, snapshots):
    reference = snapshots['A']
    result = {}
    for name, arrays in snapshots.items():
        rows = []
        for actual, expected in zip(arrays, reference):
            finite = bool(np.isfinite(actual).all())
            different = int(np.count_nonzero(actual.view(np.uint32) != expected.view(np.uint32)))
            rows.append(dict(finite=finite, differing_fp32_bits=different,
                             sha256=sha256(actual.tobytes())))
            if not finite or different:
                raise AssertionError(f'{name}: output is nonfinite or differs from A: {rows[-1]}')
        result[name] = rows
    return result


def run(args):
    from run_0290_isolated_benchmark import require_isolation
    require_isolation()  # Before CUDA imports, checkpoint mapping or payload access.
    args.output.mkdir(parents=True, exist_ok=False)
    os.environ['MINI_LLM_BACKEND'] = 'cupy'
    os.environ['MINI_LLM_Q8_SHARED_GRID'] = '1' if args.shared_grid else '0'
    import numpy as np
    import cupy as cp
    from probe_0218_hc_identity import MODEL
    from mini_llm.external.gguf_store import MultiShardTensorStore
    from mini_llm.external.qwen4exp_gguf import Qwen4ExpGGUFIndex
    from mini_llm.external.qwen4exp_weights import inspect_headers
    from mini_llm.ops.quantized_linear import packed_to_device
    from mini_llm.ops import mixed_q8_gpu as mixed
    from mini_llm.ops import q8_projection_bundle as bundle

    if cp.cuda.runtime.is_hip:
        raise RuntimeError('Native Q8_1 requires CUDA')
    with MultiShardTensorStore.open(MODEL) as store:
        index = Qwen4ExpGGUFIndex(store, model_id='0338_projection_pair')
        headers = inspect_headers(index)
        if not headers.recurrent[0]:
            raise ValueError('Layer 0 is not recurrent according to checkpoint metadata')
        names = ('blk.0.attn_qkv.weight', 'blk.0.attn_gate.weight')
        descriptors = tuple(store.tensors[name] for name in names)
        total = sum(desc.nbytes for desc in descriptors)
        if total > WEIGHT_LIMIT:
            raise ValueError(f'Packed GPU weights {total} exceed 64 MiB limit')
        for desc, width in zip(descriptors, (headers.mixer.qkv_channels, headers.mixer.inner_size)):
            if desc.quant_type not in mixed.FORMATS or desc.shape != (width, headers.width):
                raise ValueError(f'Unsupported GDN projection format/shape: {desc}')
        if INPUT.stat().st_size > 1024**2:
            raise ValueError('Retained input exceeds 1 MiB input-file bound')
        host_input = np.load(INPUT, allow_pickle=False)
        if (host_input.dtype != np.float32 or host_input.size != headers.width or
                host_input.shape[-1] != headers.width or not np.isfinite(host_input).all()):
            raise ValueError('Expected one finite FP32 retained input with model width')
        host_input = np.ascontiguousarray(host_input.reshape(1, headers.width))
        host_weights = tuple(store.packed_weight(name) for name in names)
        weight_manifest = [dict(name=desc.name, format=desc.quant_type, shape=desc.shape,
                                nbytes=desc.nbytes, row_bytes=desc.row_bytes,
                                shard_index=desc.shard_index, data_offset=desc.data_offset,
                                packed_sha256=sha256(w.data.tobytes()))
                           for desc, w in zip(descriptors, host_weights)]
        weights = tuple(packed_to_device(w) for w in host_weights)
        if sum(w.data.nbytes for w in weights) != total:
            raise AssertionError('Device packed-byte size changed during loading')
        x = cp.asarray(host_input)
        outputs = {name: tuple(cp.empty((1, w.shape[0]), cp.float32) for w in weights)
                   for name in ('A', 'B', 'C')}

        def a():
            return tuple(mixed.linear(x, w, out=y) for w, y in zip(weights, outputs['A']))

        def b():
            packed = mixed.q8.quantize_q8_1(x)
            return tuple(mixed.linear_packed(packed, w, out=y) for w, y in zip(weights, outputs['B']))

        def c():
            return bundle.linear_pair(x, *weights, out=outputs['C'])

        operations = dict(A=a, B=b, C=c)
        stream = cp.cuda.Stream(non_blocking=True)
        cp.cuda.get_current_stream().synchronize()
        validation = {}
        packed_snapshots = {}
        with stream:
            for name, operation in operations.items():
                with retain_quantizations(mixed.q8) as retained:
                    actual = operation()
                assert all(y is fixed for y, fixed in zip(actual, outputs[name]))
                stream.synchronize()
                packed_snapshots[name] = [p.get() for p in retained]
        expected_counts = dict(A=2, B=1, C=1)
        reference_packed = packed_snapshots['A'][0]
        for name, arrays in packed_snapshots.items():
            if len(arrays) != expected_counts[name]:
                raise AssertionError(f'{name}: unexpected number of quantizations')
            for packed in arrays:
                np.testing.assert_array_equal(packed, reference_packed)
        validation['uncaptured_outputs'] = bit_checks(np, {
            name: [y.get() for y in values] for name, values in outputs.items()})
        validation['packed_quantization'] = dict(byte_identical=True, counts=expected_counts,
                                                sha256=sha256(reference_packed.tobytes()))

        # Retain every scratch allocation referenced by a graph, as well as its
        # fixed input/output owners. Setup/compilation/capture are outside events.
        graphs, graph_scratch = {}, {}
        for name, operation in operations.items():
            with stream:
                for _ in range(5):
                    operation()
            stream.synchronize()
            with stream:
                with retain_quantizations(mixed.q8) as scratch:
                    stream.begin_capture()
                    try:
                        for _ in range(args.repeats):
                            operation()
                    finally:
                        graph = stream.end_capture()
            graphs[name], graph_scratch[name] = graph, scratch
            assert len(scratch) == expected_counts[name] * args.repeats
        pointers = dict(input=x.data.ptr,
                        outputs={name: [y.data.ptr for y in values] for name, values in outputs.items()})
        for label, value in (('retained', host_input), ('negated_same_address', -host_input),
                             ('restored', host_input)):
            with stream:
                x.set(value, stream=stream)
                for graph in graphs.values():
                    graph.launch(stream)
            stream.synchronize()
            validation[label + '_captured_outputs'] = bit_checks(np, {
                name: [y.get() for y in values] for name, values in outputs.items()})
            if x.data.ptr != pointers['input']:
                raise AssertionError('Input address changed')
            for name, values in outputs.items():
                assert [y.data.ptr for y in values] == pointers['outputs'][name]
        # Verify captured results against fresh, uncaptured calls, not just each other.
        captured = {name: [y.get() for y in values] for name, values in outputs.items()}
        with stream:
            for operation in operations.values():
                operation()
        stream.synchronize()
        for name in operations:
            for expected, actual in zip(captured[name], outputs[name]):
                np.testing.assert_array_equal(expected.view(np.uint32), actual.get().view(np.uint32))

        samples, orders = {name: [] for name in operations}, []
        graph_samples = {name: [] for name in operations}
        # Rotating forward/reverse orders balance position and drift without
        # randomizing inputs or including Python per-evaluation submission gaps.
        cycle = ('ABC', 'CBA', 'BCA', 'ACB', 'CAB', 'BAC')
        start, end = cp.cuda.Event(), cp.cuda.Event()
        with stream:
            for _ in range(10):
                for graph in graphs.values():
                    graph.launch(stream)
        stream.synchronize()
        for trial in range(args.trials):
            order = cycle[trial % len(cycle)]
            orders.append(order)
            for name in order:
                with stream:
                    start.record(stream)
                    graphs[name].launch(stream)
                    end.record(stream)
                end.synchronize()
                elapsed = float(cp.cuda.get_elapsed_time(start, end))
                graph_samples[name].append(elapsed)
                samples[name].append(elapsed / args.repeats)
        stream.synchronize()
        validation['post_timing_outputs'] = bit_checks(np, {
            name: [y.get() for y in values] for name, values in outputs.items()})
        medians = {name: float(np.median(values)) for name, values in samples.items()}
        stats = {name: dict(median_ms=medians[name], min_ms=float(min(values)),
                            max_ms=float(max(values)), samples_ms=values)
                 for name, values in samples.items()}
        ratios = {name + '_over_A': dict(
            ratio_of_medians=medians[name] / medians['A'],
            paired_trial_ratios=[value / base for value, base in zip(samples[name], samples['A'])])
                  for name in ('B', 'C')}
        paths = [Path(__file__), Path(mixed.__file__), Path(bundle.__file__), Path(mixed.q8.__file__),
                 Path(mixed.iq.__file__), ROOT / 'mini_llm/ops/quantized_linear.py',
                 ROOT / 'scripts/run_0290_isolated_benchmark.py']
        properties = cp.cuda.runtime.getDeviceProperties(x.device.id)
        gpu_name = properties['name']
        if isinstance(gpu_name, bytes):
            gpu_name = gpu_name.decode()
        manifest = dict(
            model=str(MODEL), shards=[dict(path=str(p), size=p.stat().st_size) for p in store.paths],
            layer=0, recurrent_verified=True,
            recurrent_metadata={key: store.metadata.get(key) for key in (
                'qwen4exp.attention.recurrent_layers', 'qwen4exp.full_attention_interval',
                'qwen4exp.attention.compress_ratios')},
            weights=weight_manifest, gpu_weight_bytes=total, gpu_weight_limit_bytes=WEIGHT_LIMIT,
            input=dict(path=str(INPUT.relative_to(ROOT)), file_sha256=sha256(INPUT.read_bytes()),
                       fp32_sha256=sha256(host_input.tobytes()), shape=host_input.shape, dtype='float32',
                       scope='Retained layer-0 FFN-normalized input, NOT actual GDN input'),
            arithmetic=dict(activation='Existing native Q8_1, unchanged',
                            quantizer_options=['--std=c++17', '--use_fast_math'],
                            projection_options=['--std=c++17', '--fmad=false'],
                            shared_grid=args.shared_grid, row_dot='Unchanged mixed_q8_gpu.row_dot<F>',
                            rows_per_cta=4),
            kernel_source_sha256=dict(mixed=sha256(mixed._SOURCE.encode()),
                                      pair=sha256(bundle._SOURCE.encode()),
                                      iq_base=sha256(mixed.iq._cuda_source('global').encode()),
                                      quantizer=sha256(mixed.q8._SOURCE.encode())),
            source_sha256={str(p.relative_to(ROOT)): sha256(p.read_bytes()) for p in paths},
            git_head=subprocess.check_output(['git', '--no-pager', 'rev-parse', 'HEAD'], cwd=ROOT,
                                              text=True).strip(),
            python=sys.version, executable=sys.executable, argv=sys.argv,
            numpy=np.__version__, cupy=cp.__version__, gpu_name=gpu_name, device=x.device.id,
            driver_version=cp.cuda.runtime.driverGetVersion(),
            runtime_version=cp.cuda.runtime.runtimeGetVersion(),
            cgroup=Path('/proc/self/cgroup').read_text(),
            environment={key: value for key, value in os.environ.items()
                         if key.startswith('MINI_LLM_') or key == 'CUDA_VISIBLE_DEVICES'})
        result = dict(
            scope='Warm captured actual-weight QKV/Z operator A/B/C; teacher-forced FFN input; NOT model throughput',
            accepted=True, manifest=manifest, validation=validation,
            arms=dict(A='Two mixed.linear calls', B='One quantization plus two mixed.linear_packed calls',
                      C='linear_pair: one quantization plus one paired projection kernel'),
            timing=dict(method='CUDA events around one graph containing repeats evaluations; no per-evaluation Python launches',
                        trials=args.trials, repeats=args.repeats, trial_orders=orders,
                        per_pair=stats, graph_total_samples_ms=graph_samples,
                        ratios=ratios, A_over_C_speedup=medians['A'] / medians['C'],
                        A_minus_C_ms=medians['A'] - medians['C']),
            limitations=['No full model or authentic GDN activation benchmark',
                         'Warm operator-only timings; no confidence interval or end-to-end speedup claim'])
        with (args.output / 'result.json').open('x') as output:
            json.dump(result, output, indent=2, allow_nan=False)
        print(json.dumps(dict(accepted=True, gpu_weight_bytes=total, weights=weight_manifest,
                              median_ms=medians, A_over_C_speedup=medians['A'] / medians['C'],
                              report=str(args.output / 'result.json')), indent=2), flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, required=True, help='New exclusive artifact directory')
    parser.add_argument('--trials', type=int, default=11)
    parser.add_argument('--repeats', type=int, default=100)
    parser.add_argument('--shared-grid', action='store_true')
    args = parser.parse_args()
    if not 1 <= args.trials <= 25 or not 1 <= args.repeats <= 200:
        parser.error('Require trials 1..25 and repeats 1..200')
    if args.output.exists():
        parser.error('Output exists; choose a fresh version')
    try:
        run(args)
    except Exception as exc:
        if args.output.is_dir():
            with (args.output / 'error.json').open('x') as output:
                json.dump(dict(accepted=False, error=str(exc), traceback=traceback.format_exc()), output, indent=2)
        raise


if __name__ == '__main__':
    main()
