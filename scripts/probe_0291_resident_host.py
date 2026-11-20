"""Bounded actual-checkpoint CPU residency benchmark; no CUDA or full model.

Alternates identical-input registered row-pool execution from canonical mmap
and compact anonymous RAM. This measures a warm operator, not decode throughput.
"""
import argparse
import gc
import hashlib
import json
import os
from pathlib import Path
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT), str(ROOT / 'scripts')]


def process_memory():
    names = {'VmRSS', 'RssAnon', 'RssFile', 'VmLck', 'VmPin'}
    return {key: int(value.split()[0]) * 1024
            for key, value in (line.split(':', 1) for line in Path('/proc/self/status').read_text().splitlines()
                               if ':' in line) if key in names}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--source-mib', type=int, default=128)
    parser.add_argument('--workers', type=int, default=19)
    args = parser.parse_args()
    if not 32 <= args.source_mib <= 512 or not 1 <= args.workers <= 23:
        parser.error('Require source32..512MiB, workers1..23')
    from run_0290_isolated_benchmark import require_isolation
    require_isolation()
    args.output.mkdir(parents=True, exist_ok=False)
    os.environ['MINI_LLM_BACKEND'] = 'numpy'
    import numpy as np
    from mini_llm.external.qwen4exp_gguf import Qwen4ExpGGUFIndex
    from mini_llm.ops.resident_expert_source import ResidentPackedExpertSource
    from mini_llm.ops.expert_residency import ExpertKey
    from mini_llm.ops.ggml_packed_cpu import GGMLPackedCPUExecutor, GGMLPackedCPURowPool
    from probe_0218_hc_identity import MODEL
    budget = args.source_mib * 1024**2
    index = Qwen4ExpGGUFIndex.open(MODEL, model_id='0291')
    source = None
    original = copied = None
    try:
        # One real top10 group per distinct mixed-format combination, bounded by
        # complete bundle bytes. Metadata scanning never reads expert payloads.
        groups = []
        seen = set()
        used = 0
        for layer in range(index.n_layers):
            descriptors = [index.descriptors[f'blk.{layer}.ffn_{p}_exps.weight'] for p in ('gate', 'up', 'down')]
            formats = tuple(d.quant_type for d in descriptors)
            if formats in seen:
                continue
            seen.add(formats)
            size = sum(d.nbytes // index.n_experts for d in descriptors)
            stride = (size + 4095) // 4096 * 4096
            if used + 10 * stride > budget:
                continue
            keys = [ExpertKey(index.model_id, layer, int(e)) for e in np.linspace(0, index.n_experts - 1, 10, dtype=int)]
            groups.append((layer, formats, keys))
            used += 10 * stride
        if not groups:
            raise MemoryError('Budget cannot fit one complete top10 group')
        keys = [key for _, _, group in groups for key in group]
        before = process_memory()
        started = time.perf_counter()
        source = ResidentPackedExpertSource(index.provider, keys, budget_bytes=budget,
                                             reserve_bytes=256*1024**2, pin=False)
        setup_seconds = time.perf_counter() - started
        after_copy = process_memory()
        digest = hashlib.sha256()
        for key in keys:
            for a, b in zip(index.provider(key), source.weights(key)):
                np.testing.assert_array_equal(a.data, b.data)
                digest.update(b.data)
        x = np.random.default_rng(291).normal(0, .1, (1, index.d_model)).astype(np.float32)
        results = []
        cpu = GGMLPackedCPUExecutor()
        with GGMLPackedCPURowPool(cpu, index.d_model, index.d_ff, args.workers,
                                 host_help=True, scheduler='tasks', spin_us=50) as pool:
            for layer, formats, group in groups:
                original = [index.provider(key) for key in group]
                copied = [source.weights(key) for key in group]
                ids = {}
                for kind, bank in [('mmap', original), ('anonymous', copied)]:
                    ids[kind] = np.asarray([pool.register((kind, key), ws)
                                            for key, ws in zip(group, bank)], dtype=np.int32)
                for count in (1, 2, 5, 10):
                    out = {kind: np.empty((count, index.d_model), np.float32) for kind in ids}
                    def execute(kind):
                        return pool._submit_registered(x, ids[kind][:count], out[kind]).result()
                    for _ in range(3):
                        execute('mmap'); execute('anonymous')
                    np.testing.assert_array_equal(out['mmap'], out['anonymous'])
                    samples = {kind: [] for kind in ids}
                    for trial in range(8):
                        order = ('mmap', 'anonymous') if trial % 2 == 0 else ('anonymous', 'mmap')
                        for kind in order:
                            start = time.perf_counter_ns()
                            for _ in range(30):
                                execute(kind)
                            samples[kind].append((time.perf_counter_ns() - start) / 30 / 1e6)
                    np.testing.assert_array_equal(out['mmap'], out['anonymous'])
                    medians = {kind: float(np.median(values)) for kind, values in samples.items()}
                    row = dict(layer=layer, formats=formats, experts=count,
                               bit_exact=True, samples_ms=samples, median_ms=medians,
                               mmap_over_anonymous=medians['mmap']/medians['anonymous'])
                    results.append(row)
                    print(json.dumps({k: v for k, v in row.items() if k != 'samples_ms'}), flush=True)
        stats = source.statistics()
        original = copied = bank = a = b = None
        source.close(); source = None
        gc.collect()
        after_close = process_memory()
        report = dict(scope='warm registered native CPU operators only; no full-model speed claim',
                      budget_bytes=budget, workers=args.workers, host_help=True, spin_us=50,
                      setup_seconds=setup_seconds, source=stats, payload_sha256=digest.hexdigest(),
                      exact_packed_bytes=True, memory_before=before, memory_after_copy=after_copy,
                      memory_after_close=after_close, rows=results, cpu_provenance=cpu.provenance)
        (args.output / 'result.json').write_text(json.dumps(report, indent=2, allow_nan=False))
    finally:
        original = copied = None
        if source is not None:
            source.close()
        index.clear_source_cache()
        index.store.close()


if __name__ == '__main__':
    main()
