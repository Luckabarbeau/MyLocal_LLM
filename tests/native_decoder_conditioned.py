"""Immutable, one-layer-at-a-time WP7 conditioned evidence; no production math."""
from contextlib import ExitStack
import argparse
import ctypes as C
import gc
import hashlib
import json
import os
from pathlib import Path
import sys
import time

import cupy as cp
import numpy as np

from mini_llm.native import Runtime
from mini_llm.external.qwen4exp_hyperconnection import Qwen4ExpHyperConnection
from mini_llm.external.qwen4exp_attention import Qwen4ExpAttention, QSAConfig
from mini_llm.external.qwen4exp_mixer import Qwen4ExpMixerSpec
from native_decoder_divergence import actual_path, libraries, snapshot, python_snapshot, metric
from native_moe_checkpoint_helpers import CheckpointMoE, POLICY
from test_native_engine_chaining import checkpoint_upload, REC_STEMS, STEMS, NORMS, NAMES
from test_native_engine_recurrent import reference as rec_reference
from test_native_engine_hc import upload, read as hc_read
from test_native_engine_moe_chaining import actual_ple

GPU = 'GPU-e961516f-d77d-813c-98e9-2f5403f71631'
SOURCE = Path('logs/wp7_divergence/actual_full')
BUDGET = 512 * 2**20
TARGET = 1024**3
RESERVE = 2 * 1024**3


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def preflight():
    if os.environ.get('CUDA_VISIBLE_DEVICES') != GPU:
        raise RuntimeError('exact GPU UUID mask required')
    assert cp.cuda.runtime.getDeviceCount() == 1 and cp.cuda.runtime.getDevice() == 0
    free, total = cp.cuda.runtime.memGetInfo()
    record = dict(gpu_mask=GPU, free_bytes=free, total_bytes=total,
                  target_bytes=TARGET, reserve_bytes=RESERVE, admitted=free >= TARGET + RESERVE)
    if not record['admitted']:
        raise RuntimeError(json.dumps(record))
    return record


def array(value):
    return cp.asnumpy(value) if isinstance(value, cp.ndarray) else np.asarray(value)


def compare(saved, report, key, got, want, exact=False):
    got, want = array(got), array(want)
    saved[key + '_native'] = got.copy()
    saved[key + '_python'] = want.copy()
    if not got.size:
        assert got.shape == want.shape
        report[key] = dict(empty=True, exact=True, violations=0)
        return
    row = metric(got, want)
    row['requires_exact'] = exact
    report[key] = row


def run_layer(index, root, captures):
    started = time.monotonic()
    admission = preflight()
    target = root / f'layer{index:02d}'
    target.mkdir(exist_ok=False)
    saved, checks, lineage_diags = {}, {}, {}
    with ExitStack() as stack:
        runtime = Runtime()
        engine = stack.enter_context(runtime.engine(budget_bytes=BUDGET))
        store = stack.enter_context(runtime.gguf(actual_path()))
        headers = store.validate_qwen4exp()
        spec = Qwen4ExpMixerSpec.from_metadata(store.metadata)
        def hc_weights(stem):
            pairs = [checkpoint_upload(stack, engine, store, stem + suffix) for suffix in
                     ('norm.weight', 'down.weight', 'up.weight', 'inject.weight')]
            native = stack.enter_context(engine.hc_weights(*(p[0] for p in pairs),
                epsilon=spec.rms_eps, arithmetic='reference_fp32'))
            return native, Qwen4ExpHyperConnection(*(p[1] for p in pairs), epsilon=spec.rms_eps)
        aw, ar = hc_weights(f'blk.{index}.hc_attn_')
        fw, fr = hc_weights(f'blk.{index}.hc_ffn_')
        hc = stack.enter_context(engine.hc_state(2560, 320))
        recurrent = headers.recurrent[index]
        pairs = [checkpoint_upload(stack, engine, store, f'blk.{index}.' + s) for s in
                 (REC_STEMS if recurrent else list(STEMS) + ['attn_output.weight'] + list(NORMS))]
        tensors, values = [p[0] for p in pairs], [p[1] for p in pairs]
        if recurrent:
            reference = rec_reference(spec, values)
            branch = stack.enter_context(engine.recurrent_layer(spec, 4, *tensors, arithmetic='reference_fp32'))
        else:
            reference = Qwen4ExpAttention(QSAConfig(2560, epsilon=spec.rms_eps),
                **dict(zip(NAMES, values[:6])),
                **dict(zip(('gamma_q', 'gamma_k', 'gamma_index_q', 'gamma_index_k'), values[6:])))
            branch = stack.enter_context(engine.qsa_layer(4, *tensors, policy='ggml_cpu',
                epsilon=spec.rms_eps, kv_policy='reference_fp32', page_size=16,
                physical_pages=1, host_budget_bytes=2**20, raw_diagnostics=True))
        ple, pr = actual_ple(stack, runtime, engine, store, actual_path(), spec.rms_eps) if index in headers.ple_layers else (None, None)
        source = CheckpointMoE(stack, engine, store, index)
        owner = source.native(stack, (), libraries(), threads=2)
        moe, executor = stack.enter_context(source.oracle(cp, (), threads=2))
        # Preserve captured full-model factory's dense upload / transpose view.
        moe.router.W_router = cp.asarray(source.router).T
        for lineage, sequence in captures.items():
            state = reference.create_state(1 if recurrent else 4)
            ps = pr.fresh_state() if pr else None
            lineage_diags[lineage] = []
            for token_index, capture in enumerate(sequence):
                prefix = f'{lineage}_{token_index}_'
                incoming = (capture[f'{index-1}_post_moe'] if index else
                            np.broadcast_to(captures['python'][token_index]['first_hc_input'], (4, 2560)).copy())
                if ple:
                    us, _ = upload(stack, target, runtime, engine, [incoming])
                    hc.initialize(us[0]); ple.step(hc, (1, 7)[token_index], token_index)
                    incoming_python, ps = pr.step((1, 7)[token_index], cp.asarray(incoming), ps)
                    incoming_native = hc_read(hc, 'streams', 2560)
                    compare(saved, checks, prefix+'ple_streams', incoming_native, incoming_python)
                    compare(saved, checks, prefix+'ple_history', ple_history(ple, ps), ps.normalized_history)
                    hc.reset()
                    incoming = incoming_native if lineage == 'native' else array(incoming_python)
                x = capture[f'{index}_hc_attention_read']
                us, _ = upload(stack, target, runtime, engine, [x, incoming])
                # Both mixers consume the same complete two-token hidden history.
                if recurrent:
                    expected = reference.step(cp.asarray(x)[None], state)[0]
                else:
                    expected, diagnostics = reference.step(cp.asarray(x), state, position=token_index, diagnostics=True)
                branch.step(us[0], token_index)
                output = np.empty(2560, np.float32); branch.readback_for_test('output', output)
                compare(saved, checks, prefix+'mixer_output', output, expected)
                states = [('history', state.conv.history), ('matrix', state.delta.memory)] if recurrent else [
                    ('raw', state.raw_index[:state.length]), ('keys', state.keys[:state.length]),
                    ('values', state.values[:state.length]), ('pooled', state.pooled[:state.length//4])]
                for name, value in states:
                    value = array(value); got = np.empty_like(value)
                    if got.size:
                        branch.readback_for_test(name, got)
                    compare(saved, checks, prefix+'mixer_'+name, got, value)
                if not recurrent:
                    ids = np.empty(state.length, np.int64); branch.readback_for_test('ids', ids)
                    compare(saved, checks, prefix+'mixer_ids', ids, diagnostics['selected_ids'], exact=True)
                # Native write's gains are copied unchanged into Python's write:
                # output, streams, gains are now IDENTICAL, not approximately equal.
                hc.initialize(us[1]); ticket = hc.read(aw)
                hidden, gains = ar.read(cp.asarray(incoming))
                ng = hc_read(hc, 'gains', 2560)
                compare(saved, checks, prefix+'attention_hidden', hc_read(hc, 'hidden', 2560), hidden)
                compare(saved, checks, prefix+'attention_gains', ng, gains)
                out, _ = upload(stack, target, runtime, engine, [output])
                hc.write(ticket, out[0])
                compare(saved, checks, prefix+'attention_write', hc_read(hc, 'streams', 2560),
                        ar.write(cp.asarray(incoming), cp.asarray(output), cp.asarray(ng)))
                compare(saved, checks, prefix+'captured_attention_write', hc_read(hc, 'streams', 2560), capture[f'{index}_post_mixer'])
                hc.reset()
                ffn_x = capture[f'{index}_hc_ffn_read']
                with owner.submit(np.ascontiguousarray(ffn_x)) as job:
                    job.wait(); native = snapshot(owner, 2560, 640)
                python = python_snapshot(moe, ffn_x)
                lineage_diags[lineage].append(python)
                for field in native:
                    compare(saved, checks, prefix+'moe_'+field, native[field], python[field],
                            exact=field in ('ids', 'cpu_rows', 'input'))
                for slot in range(10):
                    compare(saved, checks, prefix+f'cpu_row{slot}', native['cpu_rows'][slot], python['cpu_rows'][slot], exact=True)
                for impl, diag in [('native', native), ('python', python)]:
                    ordered = np.zeros(2560, np.float32)
                    for term in diag['terms']:
                        ordered += term
                    compare(saved, checks, prefix+impl+'_ordered_combine', ordered, diag['routed'], exact=True)
                    compare(saved, checks, prefix+impl+'_weight_once', diag['terms'], diag['cpu_rows']*diag['weights'][:,None], exact=True)
                streams = capture[f'{index}_post_mixer']
                us, _ = upload(stack, target, runtime, engine, [streams, native['output']])
                hc.initialize(us[0]); ticket = hc.read(fw)
                hidden, gains = fr.read(cp.asarray(streams)); ng = hc_read(hc, 'gains', 2560)
                compare(saved, checks, prefix+'ffn_hidden', hc_read(hc, 'hidden', 2560), hidden)
                compare(saved, checks, prefix+'ffn_gains', ng, gains)
                hc.write(ticket, us[1])
                compare(saved, checks, prefix+'ffn_write', hc_read(hc, 'streams', 2560),
                        fr.write(cp.asarray(streams), cp.asarray(native['output']), cp.asarray(ng)))
                compare(saved, checks, prefix+'captured_ffn_write', hc_read(hc, 'streams', 2560), capture[f'{index}_post_moe'])
                hc.reset()
                for name in ('quantize_row_q8_K', 'quantize_row_q8_0'):
                    saved[prefix+name] = python[name]
            branch.reset(); owner.reset()
            if ple: ple.reset()
        cross = []
        for t in (0, 1):
            n, p = (lineage_diags[l][t] for l in ('native', 'python'))
            cross.append(dict(input=metric(n['input'], p['input']),
                ids_changed=not np.array_equal(n['ids'], p['ids']), native_ids=n['ids'].tolist(), python_ids=p['ids'].tolist(),
                quantized_input_bytes_changed={k:int(np.count_nonzero(n[k]!=p[k])) for k in ('quantize_row_q8_K','quantize_row_q8_0')},
                output=metric(n['output'],p['output']), cpu_rows=metric(n['cpu_rows'],p['cpu_rows']),
                note='Different captured operands: amplification diagnostic, NOT same-input failure.'))
        memory = dict(native_peak_bytes=engine.stats.peak_bytes,
                      cupy_reserved_bytes=cp.get_default_memory_pool().total_bytes())
        assert memory['native_peak_bytes'] + memory['cupy_reserved_bytes'] <= TARGET
        identity = source.identity
    # Captured aggregate reconstruction is a separate lineage check, not an
    # operator gate: its inputs/history have accumulated earlier differences.
    failures = {k:v for k,v in checks.items() if 'captured_' not in k and
                (v['violations'] or (v.get('requires_exact') and not v['exact']))}
    report = dict(layer=index, recurrent=recurrent, preflight=admission, memory=memory,
                  policy=POLICY, gpu_experts=[], checks=checks, cross_lineage=cross,
                  weight_identity=identity, passed=not failures, failures=list(failures),
                  elapsed_seconds=time.monotonic()-started)
    np.savez(target/'evidence.npz', **saved)
    report['evidence_sha256'] = sha(target/'evidence.npz')
    (target/'report.json').write_text(json.dumps(report, indent=2)+'\n')
    print(json.dumps({k:report[k] for k in ('layer','passed','failures','memory','elapsed_seconds')}), flush=True)
    return report


def ple_history(ple, ps):
    out = np.empty(ps.normalized_history.shape, np.float32)
    ple.readback_for_test('history', out)
    return out


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('root', type=Path)
    parser.add_argument('--start', type=int, required=True)
    parser.add_argument('--stop', type=int, required=True)
    args = parser.parse_args()
    assert 0 <= args.start < args.stop <= 48 and args.stop-args.start <= 8
    preflight()
    keepalive = C.CDLL(os.environ['MINI_LLM_GGML_CPU_LIBRARY'])
    paths = [folder/f'{name}.npz' for folder in (SOURCE, SOURCE/'token1') for name in ('native','python')]
    captures = {name:[dict(np.load(folder/f'{name}.npz', allow_pickle=False)) for folder in (SOURCE,SOURCE/'token1')]
                for name in ('native','python')}
    provenance = dict(command=sys.argv, artifacts={str(p):sha(p) for p in paths},
        sources={p:sha(p) for p in ('tests/native_decoder_conditioned.py','tests/native_decoder_divergence.py','tests/native_moe_checkpoint_helpers.py')},
        libraries={p:sha(p) for p in (os.environ['MINI_LLM_NATIVE_LIBRARY'],os.environ['MINI_LLM_GGML_CPU_LIBRARY'])})
    results=[]
    for index in range(args.start,args.stop):
        results.append(run_layer(index,args.root,captures))
        gc.collect(); cp.get_default_memory_pool().free_all_blocks()
    path=args.root/f'batch{args.start:02d}-{args.stop:02d}.json'
    with path.open('x') as f:
        json.dump(dict(provenance=provenance,layers=[r['layer'] for r in results],passed=all(r['passed'] for r in results)),f,indent=2)
    if not all(r['passed'] for r in results):
        raise SystemExit(1)


if __name__ == '__main__':
    main()
