"""Bounded sequential placement/staging/cache ablation, full top-10 FP32 decode.

No model downloading, dequantized expert banks, speculative tokens or route skipping.
Versioned output is exclusive-created; every variant gets a separate process.
"""
import argparse
import json
import os
from pathlib import Path
import resource
import statistics
import subprocess
import sys
import time
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / 'scripts'))
from probe_0218_hc_identity import MODEL
from probe_0227_short_sequences_and_profile import digest_state

GiB = 1024**3
MODES = ('baseline', 'backbone', 'batch', 'cache', 'eager', 'fused_qsa',
         'grouped_iq', 'fused_hc', 'fused_router', 'fused_gdn')


def child(mode, output):
    from mini_llm.backend import asnumpy, xp, BACKEND_NAME
    from mini_llm.external.qwen4exp_model import Qwen4ExpGGUFModel
    from mini_llm.external import qwen4exp_weights
    from mini_llm.ops.quantized_linear import _FORMAT_ID
    if BACKEND_NAME != 'cupy':
        raise ValueError('CuPy required')
    budgets = dict(cpu_workers=4)
    if mode == 'baseline':
        # Reproduce the pre-0213 GPU capability/placement, not a changed baseline.
        qwen4exp_weights.supports_packed_gpu_type = lambda fmt: fmt in _FORMAT_ID
    else:
        budgets.update(static_device_budget_bytes=6*GiB, device_budget_bytes=14*GiB,
                       cache_bytes=6*GiB if mode not in ('backbone', 'batch') else 0)
        if mode not in ('backbone', 'batch', 'cache'):
            budgets.update(expert_promote_after=1, qsa_fast_decode=mode != 'eager',
                           hc_fast_decode=mode in ('fused_hc', 'fused_router', 'fused_gdn'),
                           router_fast_decode=mode in ('fused_router', 'fused_gdn'),
                           gdn_fast_decode=mode == 'fused_gdn')
    start = time.perf_counter()
    with Qwen4ExpGGUFModel.open(MODEL, max_length=16, **budgets) as model:
        model.expert_executor.batch_cpu_decode = mode not in ('baseline', 'backbone')
        state = model.create_generation_state(1, 16)
        xp.cuda.get_current_stream().synchronize()
        built = time.perf_counter()
        latencies, snapshots = [], []
        initial = model.expert_executor.statistics()
        for position in range(16):
            before = time.perf_counter()
            logits = model.step(position + 1, state)
            xp.cuda.get_current_stream().synchronize()
            seconds = time.perf_counter() - before
            values = asnumpy(logits)
            if not np.isfinite(values).all():
                raise ValueError(f'nonfinite logits at {position}')
            latencies.append(dict(position=position, seconds=seconds))
            if position + 1 in (1, 2, 4, 8, 16):
                np.save(output / f'{mode}_logits_{position+1}.npy', values)
                snapshots.append(dict(prefix=position+1, state_sha256=digest_state(state, asnumpy),
                                      argmax=int(values.argmax())))
        measured = [x['seconds'] for x in latencies[8:]]
        record = dict(mode=mode, input_ids=list(range(1, 17)), full_top10=True,
                      activation_dtype='float32', state_dtype='float32', speculative=False,
                      batch_cpu_decode=model.expert_executor.batch_cpu_decode,
                      build_seconds=built-start, latencies=latencies,
                      measured_positions=list(range(8, 16)),
                      mean_decode_seconds=statistics.mean(measured),
                      median_decode_seconds=statistics.median(measured),
                      measured_tokens_per_second=len(measured)/sum(measured),
                      snapshots=snapshots, plan=model.plan,
                      initial_expert_statistics=initial,
                      expert_statistics=model.expert_executor.statistics(),
                      peak_rss_bytes=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss*1024,
                      device_pool_used_bytes=xp.get_default_memory_pool().used_bytes(),
                      device_pool_reserved_bytes=xp.get_default_memory_pool().total_bytes())
    with (output / f'{mode}.json').open('x') as handle:
        json.dump(record, handle, indent=2, allow_nan=False)


def run(mode, output):
    env = dict(os.environ, MINI_LLM_BACKEND='cupy', OPENBLAS_NUM_THREADS='1', OMP_NUM_THREADS='1')
    start = time.monotonic()
    peak, reason = 0, None
    with (output / f'{mode}.log').open('x') as log:
        proc = subprocess.Popen([sys.executable, str(Path(__file__).resolve()), '--child', mode,
                                 '--output', str(output)], env=env, stdout=log, stderr=subprocess.STDOUT)
        try:
            while proc.poll() is None:
                try:
                    lines = Path(f'/proc/{proc.pid}/status').read_text().splitlines()
                    rss = next((int(s.split()[1])*1024 for s in lines if s.startswith('VmRSS:')), 0)
                    peak = max(peak, rss)
                    if rss > 20*GiB:
                        reason = '20 GiB sampled RSS guard'
                except FileNotFoundError:
                    pass
                if time.monotonic()-start > 120:
                    reason = '120-second child deadline'
                if reason:
                    proc.kill()
                    break
                time.sleep(.05)
        finally:
            if proc.poll() is None:
                proc.kill()
            proc.wait()
    result = dict(returncode=proc.returncode, seconds=time.monotonic()-start,
                  sampled_peak_rss_bytes=peak, stop_reason=reason)
    if proc.returncode or reason:
        raise RuntimeError(f'{mode}: {result}; inspect {output / (mode+".log")}')
    return result


def compare(output, mode, baseline_output=None):
    results = []
    for prefix in (1, 2, 4, 8, 16):
        a = np.load(output/f'{mode}_logits_{prefix}.npy').astype(np.float64).reshape(-1)
        b = np.load((baseline_output or output)/f'baseline_logits_{prefix}.npy').astype(np.float64).reshape(-1)
        results.append(dict(prefix=prefix, max_absolute_error=float(np.max(np.abs(a-b))),
                            relative_l2=float(np.linalg.norm(a-b)/max(np.linalg.norm(b), 1e-30)),
                            actual_argmax=int(a.argmax()), baseline_argmax=int(b.argmax()),
                            top10_overlap=len(set(np.argsort(a)[-10:]) & set(np.argsort(b)[-10:]))))
    return results


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--child', choices=MODES)
    parser.add_argument('--output', type=Path)
    parser.add_argument('--version', default='v1')
    parser.add_argument('--resume-from', help='Reuse successful variants from a preserved version')
    args = parser.parse_args()
    if args.child:
        child(args.child, args.output)
        return
    docs = ROOT/'docs/qwen4exp'
    output = docs/f'diagnostics_0229_{args.version}'
    result = docs/f'0229_decode_performance_{args.version}.json'
    if output.exists() or result.exists():
        raise FileExistsError('Preserve previous artifacts; use a new --version')
    output.mkdir()
    report = dict(status='in_progress', sequential_processes=True, runs={}, results={},
                  note='Short forced-token uninstrumented decode ablation; not natural-language greedy throughput, long-context throughput, or independent full-model numerical parity. No tail skipping or speculation. Timing positions8..15 include any cache admission/copy costs there.')
    try:
        previous = (json.loads((docs/f'0229_decode_performance_{args.resume_from}.json').read_text())
                    if args.resume_from else None)
        baseline_output = docs/f'diagnostics_0229_{args.resume_from}' if previous else output
        origin = previous
        while previous and not (baseline_output/'baseline_logits_1.npy').exists():
            source = origin['reused_successful_variants_from']
            origin = json.loads((docs/source).read_text())
            baseline_output = docs/source.replace('0229_decode_performance_', 'diagnostics_0229_').removesuffix('.json')
        if previous:
            report['reused_successful_variants_from'] = f'0229_decode_performance_{args.resume_from}.json'
        for mode in MODES:
            if previous and mode in previous['results'] and previous['runs'][mode]['returncode'] == 0:
                report['runs'][mode] = previous['runs'][mode]
                report['results'][mode] = previous['results'][mode]
                continue
            report['runs'][mode] = run(mode, output)
            report['results'][mode] = json.loads((output/f'{mode}.json').read_text())
            if mode != 'baseline':
                report['results'][mode]['same_prefix_vs_baseline'] = compare(output, mode, baseline_output)
            print(mode, report['results'][mode]['measured_tokens_per_second'], 'tokens/s', flush=True)
        report['status'] = 'completed_measurement_not_full_parity_claim'
    except Exception as exc:
        report['status'] = 'blocked'
        report['error'] = str(exc)
    with result.open('x') as handle:
        json.dump(report, handle, indent=2, allow_nan=False)
    if report['status'] == 'blocked':
        raise SystemExit(report['error'])


if __name__ == '__main__':
    main()
