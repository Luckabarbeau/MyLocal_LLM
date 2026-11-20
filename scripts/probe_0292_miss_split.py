"""Bounded top10 CPU/GPU staging split experiment, not an integrated model path."""
import argparse
import gc
import json
import os
from pathlib import Path
import sys
import time
from types import SimpleNamespace

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT), str(ROOT / 'scripts')]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--workers', type=int, default=19)
    args = parser.parse_args()
    from run_0290_isolated_benchmark import require_isolation
    require_isolation()
    if not 1 <= args.workers <= 23:
        parser.error('workers1..23 only')
    args.output.mkdir(parents=True, exist_ok=False)
    os.environ['MINI_LLM_BACKEND'] = 'cupy'
    import numpy as np
    import cupy as cp
    from mini_llm.ops.resident_expert_source import ResidentPackedExpertSource
    from mini_llm.ops.ggml_packed_cpu import GGMLPackedCPUExecutor, GGMLPackedCPURowPool
    from mini_llm.ops.quantized_linear import PackedExpertFFNInference, packed_to_device, DevicePackedGGUFWeight
    from mini_llm.ops.mixed_q8_gpu import GroupedQ8Plan
    from mini_llm.ops.expert_residency import ExpertKey
    from mini_llm.external.qwen4exp_gguf import Qwen4ExpGGUFIndex
    from probe_0218_hc_identity import MODEL
    index = Qwen4ExpGGUFIndex.open(MODEL, model_id='0292')
    source = None
    compute = cp.cuda.Stream(non_blocking=True)
    transfer = cp.cuda.Stream(non_blocking=True)
    ready = cp.cuda.Event(disable_timing=True)
    x = np.random.default_rng(292).normal(0, .1, (1, index.d_model)).astype(np.float32)
    gx = cp.asarray(x)
    cp.cuda.get_current_stream().synchronize()
    cpu = GGMLPackedCPUExecutor()
    rows = []
    seen = set()

    def plan(experts, sources, formats):
        p = SimpleNamespace(d=index.d_model, f=index.d_ff, k=len(experts),
            device=cp.cuda.runtime.getDevice(), formats=formats,
            resident_row_bytes=tuple(w.nbytes // w.shape[0] for w in sources[0]),
            selected=cp.asarray([[w.data.data.ptr for w in (e.gate, e.up, e.down)] for e in experts], cp.uint64),
            hidden=cp.empty((len(experts), index.d_ff), cp.float32),
            terms=cp.empty((len(experts), index.d_model), cp.float32),
            routes=cp.ones(len(experts), cp.float32))
        return p, GroupedQ8Plan(p)

    try:
        with GGMLPackedCPURowPool(cpu, index.d_model, index.d_ff, args.workers,
                                 host_help=True, scheduler='tasks', spin_us=50) as pool:
            for layer in range(index.n_layers):
                formats = tuple(index.descriptors[f'blk.{layer}.ffn_{p}_exps.weight'].quant_type
                                for p in ('gate', 'up', 'down'))
                if formats in seen:
                    continue
                seen.add(formats)
                keys = [ExpertKey(index.model_id, layer, int(e))
                        for e in np.linspace(0, index.n_experts - 1, 10, dtype=int)]
                source = ResidentPackedExpertSource(index.provider, keys, budget_bytes=32*1024**2,
                    reserve_bytes=256*1024**2, pin=True, chunk_bytes=4*1024**2)
                if not all(source.dma_pointer(key) for key in keys):
                    raise RuntimeError('Complete pinned bundles required for this experiment')
                sources = [source.weights(key) for key in keys]
                locals_ = [PackedExpertFFNInference(*(packed_to_device(w) for w in ws)) for ws in sources]
                staging = cp.empty(source.nbytes, cp.uint8)
                staged = []
                for key, ws in zip(keys, sources):
                    offset = source.offsets[key][0]
                    views = []
                    for w in ws:
                        views.append(DevicePackedGGUFWeight(staging[offset:offset+w.nbytes], w.shape, w.quant_type))
                        offset += w.nbytes
                    staged.append(PackedExpertFFNInference(*views))
                ids = np.asarray([pool.register(key, ws) for key, ws in zip(keys, sources)], np.int32)
                cpu_oracle = pool.submit(x, sources).result().copy()
                oracle_p, oracle = plan(locals_, sources, formats)
                oracle.forward(gx)
                gpu_oracle = oracle_p.terms.get()
                for hits in (4, 7):
                    for count in range(min(3, 10-hits)+1):
                        resident_p, resident = plan(locals_[:hits], sources, formats)
                        if count:
                            staged_p, staged_plan = plan(staged[hits:hits+count], sources, formats)
                        else:
                            staged_p = staged_plan = None
                        cpu_ids = np.ascontiguousarray(ids[hits+count:])
                        cpu_out = np.empty((len(cpu_ids), index.d_model), np.float32)
                        cp.cuda.get_current_stream().synchronize()

                        def execute():
                            # Native workers start before Python submits GPU work.
                            batch = pool._submit_registered(x, cpu_ids, cpu_out) if len(cpu_ids) else None
                            try:
                                if count:
                                    with transfer:
                                        for key in keys[hits:hits+count]:
                                            start, end = source.offsets[key]
                                            cp.cuda.runtime.memcpyAsync(staging.data.ptr+start,
                                                source.dma_pointer(key), end-start,
                                                cp.cuda.runtime.memcpyHostToDevice, transfer.ptr)
                                        ready.record(transfer)
                                with compute:
                                    resident.forward(gx)
                                    if count:
                                        compute.wait_event(ready)
                                        staged_plan.forward(gx)
                            finally:
                                if batch is not None:
                                    batch.result()
                            compute.synchronize()

                        for _ in range(3):
                            execute()
                        np.testing.assert_array_equal(resident_p.terms.get(), gpu_oracle[:hits])
                        if count:
                            np.testing.assert_array_equal(staged_p.terms.get(), gpu_oracle[hits:hits+count])
                        if len(cpu_ids):
                            np.testing.assert_array_equal(cpu_out, cpu_oracle[hits+count:])
                        samples = []
                        for _ in range(7):
                            started = time.perf_counter_ns()
                            for _ in range(30):
                                execute()
                            samples.append((time.perf_counter_ns()-started)/30/1e6)
                        row = dict(layer=layer, formats=formats, resident=hits, staged=count,
                                   cpu=10-hits-count, complete_top10=True,
                                   same_site_oracles_bit_exact=True, samples_ms=samples,
                                   median_ms=float(np.median(samples)))
                        rows.append(row)
                        print(json.dumps({k: v for k, v in row.items() if k != 'samples_ms'}), flush=True)
                        del resident_p, resident, staged_p, staged_plan
                compute.synchronize(); transfer.synchronize()
                pool.clear_registrations()
                del sources, locals_, staged, staging, oracle, oracle_p, ws, views, w
                source.close(); source = None
                index.clear_source_cache()
                gc.collect()
        (args.output/'result.json').write_text(json.dumps(dict(
            scope='Operator scheduling envelope only; Python orchestration and completion fence included; no full-model gain',
            workers=args.workers, host_help=True, spin_us=50,
            cpu_arithmetic='GGML Q8_K/Q8_0 per format', gpu_arithmetic='native Q8_1',
            resident_staged_source='complete packed anonymous pinned bundles, <=32MiB per group',
            staging='one reusable device arena, dedicated transfer stream, event before consumption',
            rows=rows), indent=2, allow_nan=False))
    finally:
        compute.synchronize(); transfer.synchronize()
        if source is not None:
            source.close()
        index.clear_source_cache(); index.store.close()


if __name__ == '__main__':
    main()
