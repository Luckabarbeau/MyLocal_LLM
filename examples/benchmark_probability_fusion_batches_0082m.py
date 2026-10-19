"""Benchmark end-to-end fusion TRAINING steps with different GPU batch sizes.

From repository root (run on RTX 5080):
  MINI_LLM_BACKEND=cupy python -m examples.benchmark_probability_fusion_batches_0082m \
      --batch-sizes 32,64,128,256 --warmup 2 --repeats 5

This benchmark includes full vocabulary forward, manual backward, global
clipping and AdamW. It excludes teacher inference and disk reads. Larger
batches can run out of VRAM; results are NOT equivalent to gradient
accumulation when comparing models trained with different batch sizes.
"""

import argparse
import csv
from pathlib import Path
import time
import numpy as np

from mini_llm.backend import BACKEND_NAME, xp, synchronize
from mini_llm.collective import FusionConfig, ProbabilityResidualCombiner
from mini_llm.optim.adamw import AdamW
from examples.train_probability_fusion_0082a import _clip_grad_norm_



class FusionStepTimeline:
    """Boundary profiler: event timing without synchronizing at each stage.

    GPU event intervals contain actual GPU work plus any gaps between kernels;
    host intervals measure Python/CuPy launch work, NOT separate additive
    time. Recorded boundaries are ordered on the current CUDA stream.
    Normal training never instantiates this profiler.
    """

    def __init__(self, *, gpu, memory_pool=None):
        self.gpu = bool(gpu)
        self.memory_pool = memory_pool
        self._events = []
        self._names = []
        self._host_times = []
        self._peak_bytes = 0

    def start(self):
        self._events.clear()
        self._names.clear()
        self._host_times = [time.perf_counter()]
        self._peak_bytes = 0
        self._record_event()

    def _record_event(self):
        if self.gpu:
            # Current CuPy stream, not a new stream: ordering is essential.
            event = xp.cuda.Event()
            event.record()
            self._events.append(event)
        if self.memory_pool is not None:
            self._peak_bytes = max(self._peak_bytes, int(self.memory_pool.used_bytes()))

    def mark(self, completed_stage):
        if not self._host_times:
            raise RuntimeError('call timeline.start() before marking stages')
        self._names.append(completed_stage)
        self._record_event()
        self._host_times.append(time.perf_counter())

    def snapshot(self):
        """Call after the benchmark's one end-of-update synchronize()."""
        host_ms = np.diff(np.asarray(self._host_times, dtype=np.float64)) * 1000.0
        if self.gpu:
            device_ms = np.asarray([
                xp.cuda.get_elapsed_time(self._events[i], self._events[i + 1])
                for i in range(len(self._names))
            ], dtype=np.float64)
        else:
            device_ms = host_ms.copy()
        return (dict(zip(self._names, device_ms)),
                dict(zip(self._names, host_ms)), self._peak_bytes)


def _summarize_stage_profiles(samples, *, median_update_ms, batch_rows):
    """Return deterministic order-preserving per-stage median timings."""
    if not samples:
        return []
    expected_names = list(samples[0][0])
    if any(list(dev) != expected_names or list(host) != expected_names
           for dev, host, _ in samples):
        raise ValueError('profiling stage names changed across updates')
    result = []
    for name in expected_names:
        gpu_ms = float(np.median([dev[name] for dev, _, _ in samples]))
        host_ms = float(np.median([host[name] for _, host, _ in samples]))
        result.append({
            'stage': name,
            'gpu_ms': gpu_ms,
            'host_ms': host_ms,
            'share_pct': 100.0 * gpu_ms / max(median_update_ms, 1e-12),
            'supervised_tok_s': batch_rows / (gpu_ms / 1000.0) if gpu_ms else float('inf'),
        })
    return result


def _print_profile(rows, batch_rows, *, median_update_ms, peak_bytes):
    timing_label = 'CUDA stream-event time' if BACKEND_NAME == 'cupy' else 'CPU wall time'
    print(f'  Breakdown for batch_rows={batch_rows} ({timing_label}; profiling overhead applies)')
    print(f'  {"Stage":39s} {"GPU ms":>10s} {"% update":>10s} {"host ms":>10s}')
    for row in rows:
        print(f'  {row["stage"]:39s} {row["gpu_ms"]:10.3f} '
              f'{row["share_pct"]:9.1f}% {row["host_ms"]:10.3f}')
    for category, label in (('forward.', 'FORWARD TOTAL'),
                            ('backward.', 'BACKWARD TOTAL'),
                            ('optimizer.', 'GRAD + OPTIMIZER TOTAL')):
        selection = [r for r in rows if r['stage'].startswith(category)]
        gpu = sum(r['gpu_ms'] for r in selection)
        host = sum(r['host_ms'] for r in selection)
        print(f'  {label:39s} {gpu:10.3f} '
              f'{100.0*gpu/max(median_update_ms,1e-12):9.1f}% {host:10.3f}')
    slowest = sorted(rows, key=lambda row: row['gpu_ms'], reverse=True)[:5]
    print('  Slowest stages: ' + ', '.join(
        f'{r["stage"]} ({r["gpu_ms"]:.1f} ms)' for r in slowest))
    if peak_bytes:
        print(f'  Peak allocated CuPy memory (sampled): {peak_bytes / 2**20:,.1f} MiB')
    if BACKEND_NAME == 'cupy':
        print('  Note: GPU event intervals include gaps between launches; host times overlap GPU time.')
    print('        Per-stage medians need not sum to the median whole-update time.')


def _write_profile_csv(path, reports):
    if not path:
        return
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open('w', newline='', encoding='utf-8') as f:
        w = csv.DictWriter(f, fieldnames=['batch_rows', 'stage', 'gpu_ms',
                                          'host_ms', 'share_pct', 'supervised_tok_s',
                                          'median_update_ms', 'peak_used_bytes'])
        w.writeheader()
        for report in reports:
            for row in report['stages']:
                w.writerow({'batch_rows': report['batch_rows'], **row,
                            'median_update_ms': report['median_update_ms'],
                            'peak_used_bytes': report['peak_used_bytes']})


def run_batch(batch_rows, *, members, vocab, warmup, repeats, clip, profile=False,
              context_heads=0, context_tokens=512, embed_dim=16, member_width=8,
              decoder_width=24, decoder_layers=1):
    cfg = FusionConfig(vocab_size=vocab, max_models=members,
                       context_heads=context_heads, embed_dim=embed_dim,
                       member_width=member_width, decoder_width=decoder_width,
                       decoder_layers=decoder_layers)
    model = ProbabilityResidualCombiner(cfg)
    opt = AdamW(model.parameters(), lr=2e-4, weight_decay=0.01)
    rng = np.random.default_rng(82)
    host = rng.normal(0, 1.5, size=(batch_rows, members, vocab)).astype(np.float32)
    logits = xp.asarray(host)
    targets = rng.integers(0, vocab, size=batch_rows, dtype=np.int64)
    member_ids = np.arange(members, dtype=np.int64)
    context_kwargs = {}
    if context_heads:
        context_kwargs = {
            'context_tokens': rng.integers(0, vocab,
                                           size=(batch_rows, context_tokens),
                                           dtype=np.int64),
            'context_lengths': np.full(batch_rows, context_tokens, dtype=np.int64),
        }
    times = []
    profile_samples = []
    pool = xp.get_default_memory_pool() if BACKEND_NAME == 'cupy' else None
    timeline = FusionStepTimeline(gpu=BACKEND_NAME == 'cupy', memory_pool=pool) if profile else None

    for iteration in range(warmup + repeats):
        synchronize()
        start = time.perf_counter()
        if timeline is not None:
            timeline.start()
        opt.zero_grad()
        if timeline is not None:
            timeline.mark('optimizer.zero_grad')
        model.loss_and_backward(logits, member_ids, targets,
                                return_weighted=True, defer_metrics=True,
                                check_finite=False, profile=timeline,
                                **context_kwargs)  # finite synthetic inputs
        _clip_grad_norm_(model.parameters(), clip, device_only=True)
        if timeline is not None:
            timeline.mark('optimizer.clip_gradients')
        opt.step()
        if timeline is not None:
            timeline.mark('optimizer.adamw_step')
        synchronize()
        if iteration >= warmup:
            times.append(time.perf_counter() - start)
            if timeline is not None:
                profile_samples.append(timeline.snapshot())

    elapsed = float(np.median(times))
    result = (elapsed, batch_rows / max(elapsed, 1e-12))
    if timeline is not None:
        stages = _summarize_stage_profiles(profile_samples,
                                           median_update_ms=elapsed * 1000.0,
                                           batch_rows=batch_rows)
        peak_bytes = max((sample[2] for sample in profile_samples), default=0)
        result += (stages, peak_bytes)
    del opt, model, logits, timeline
    if BACKEND_NAME == 'cupy':
        xp.get_default_memory_pool().free_all_blocks()
    return result


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--batch-sizes', default='32,64,128,256',
                        help='comma-separated fusion optimizer batch sizes')
    parser.add_argument('--members', type=int, default=2)
    parser.add_argument('--vocab', type=int, default=32768)
    parser.add_argument('--embed-dim', type=int, default=16)
    parser.add_argument('--member-width', type=int, default=8)
    parser.add_argument('--decoder-width', type=int, default=24)
    parser.add_argument('--decoder-layers', type=int, default=1)
    parser.add_argument('--context-heads', type=int, choices=[0, 2], default=0)
    parser.add_argument('--context-tokens', type=int, default=512,
                        help='synthetic causal prefix length only when context-heads=2')
    parser.add_argument('--warmup', type=int, default=2)
    parser.add_argument('--repeats', type=int, default=5)
    parser.add_argument('--grad-clip', type=float, default=1.0)
    parser.add_argument('--profile', action='store_true',
                        help='record forward/backward/optimizer CUDA-event breakdown (diagnostics only)')
    parser.add_argument('--profile-csv', default=None,
                        help='optional CSV filename for per-stage profiling results; requires --profile')
    args = parser.parse_args(argv)
    batches = [int(x.strip()) for x in args.batch_sizes.split(',') if x.strip()]
    if (not batches or min(batches) <= 0 or args.members < 1 or args.vocab < 2
            or args.warmup < 0 or args.repeats <= 0 or args.grad_clip < 0
            or min(args.embed_dim, args.member_width, args.decoder_width,
                   args.decoder_layers, args.context_tokens) < 1):
        parser.error('invalid positive sizes or repetitions')
    if args.profile_csv and not args.profile:
        parser.error('--profile-csv requires --profile')
    print(f'backend={BACKEND_NAME} models={args.members} vocab={args.vocab} '
          f'context_heads={args.context_heads} '
          f'objective=full_fusion_training (excludes teacher export/disk IO)')
    print(f'{"batch_rows":>10} {"median_ms/update":>17} {"supervised_tok/s":>18}')
    oom_errors = (MemoryError,)
    if BACKEND_NAME == 'cupy':
        oom_errors = oom_errors + (xp.cuda.memory.OutOfMemoryError,)
    reports = []
    for size in batches:
        try:
            result = run_batch(size, members=args.members,
                               vocab=args.vocab, warmup=args.warmup,
                               repeats=args.repeats, clip=args.grad_clip,
                               profile=args.profile, context_heads=args.context_heads,
                               context_tokens=args.context_tokens,
                               embed_dim=args.embed_dim, member_width=args.member_width,
                               decoder_width=args.decoder_width,
                               decoder_layers=args.decoder_layers)
            elapsed, tokens_per_sec = result[:2]
        except oom_errors:
            print(f'{size:>10} {"OUT OF MEMORY":>17}')
            if BACKEND_NAME == 'cupy':
                xp.get_default_memory_pool().free_all_blocks()
            continue
        print(f'{size:>10} {elapsed*1e3:>17.2f} {tokens_per_sec:>18,.1f}')
        if args.profile:
            stages, peak_bytes = result[2:]
            _print_profile(stages, size, median_update_ms=elapsed*1000.,
                           peak_bytes=peak_bytes)
            reports.append({'batch_rows': size, 'stages': stages,
                            'median_update_ms': elapsed*1000.,
                            'peak_used_bytes': peak_bytes})
    _write_profile_csv(args.profile_csv, reports)


if __name__ == '__main__':
    main()
