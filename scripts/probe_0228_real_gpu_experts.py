"""Bounded real packed CPU/GPU expert probe, NOT a model benchmark.

Reads one expert/layer and ten selected experts from one layer, never expert
banks or dense expert matrices. Fixed synthetic FP32 input is identical for
native CPU and GPU. Top10 uses real router logits without correction bias and
softmax-normalized selected logits: this is a representative dispatch workload,
not a claim of full-model router/hidden-state parity. Checkpoint is read-only.

Run with MINI_LLM_BACKEND=cupy; output is a versioned JSON report. No checkpoint
paths or runtime modules are modified. RSS includes touched mmap pages, CUDA
context, compilation and library overhead; enforce <2 GiB additional peak RSS.
"""
import argparse
from collections import Counter
import json
from pathlib import Path
import resource
import sys
import time
from unittest.mock import patch

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from mini_llm.backend import BACKEND_NAME
from mini_llm.external.qwen4exp_gguf import Qwen4ExpGGUFIndex
from mini_llm.ops.expert_residency import ExpertKey, HeterogeneousExpertExecutor
from mini_llm.ops.experts_inference import ExpertsInference
from mini_llm.ops.native_packed_cpu import packed_cpu_expert, packed_cpu_linear
from mini_llm.ops.quantized_linear import PackedExpertFFNInference, packed_to_device
from mini_llm.ops.quantized_weight import BLOCK_FORMATS, PackedGGUFWeight


LIMIT = 2 * 1024**3


def peak_rss():
    return resource.getrusage(resource.RUSAGE_SELF).ru_maxrss * 1024


def rss():
    for line in Path("/proc/self/status").read_text().splitlines():
        if line.startswith("VmRSS:"):
            return int(line.split()[1]) * 1024
    raise RuntimeError("Linux RSS accounting unavailable")


def error_metrics(got, expected):
    if not np.isfinite(got).all() or not np.isfinite(expected).all():
        raise AssertionError("Nonfinite real expert output")
    delta = got.astype(np.float64) - expected.astype(np.float64)
    norm = float(np.linalg.norm(expected.astype(np.float64)))
    metrics = dict(max_absolute_error=float(np.abs(delta).max()),
                   relative_l2=float(np.linalg.norm(delta) / max(norm, 1e-30)),
                   reference_max_abs=float(np.abs(expected).max()))
    # Normwise criteria avoid meaningless relative errors near zero outputs.
    metrics["passed"] = metrics["relative_l2"] <= 2e-4 and metrics["max_absolute_error"] <= (
        2e-7 + 2e-4 * metrics["reference_max_abs"])
    if not metrics["passed"]:
        raise AssertionError(json.dumps(metrics))
    return metrics


def timings(cpu, gpu, cp, repeats):
    cpu()  # Warm native code and mmap pages.
    gpu()
    cp.cuda.get_current_stream().synchronize()  # Exclude JIT and upload.
    cpu_ms, gpu_event_ms, gpu_wall_ms = [], [], []
    for _ in range(repeats):
        start = time.perf_counter()
        cpu()
        cpu_ms.append((time.perf_counter() - start) * 1000)
        first, last = cp.cuda.Event(), cp.cuda.Event()
        start = time.perf_counter()
        first.record()
        gpu()
        last.record()
        last.synchronize()
        gpu_wall_ms.append((time.perf_counter() - start) * 1000)
        gpu_event_ms.append(cp.cuda.get_elapsed_time(first, last))
    return dict(warm_repeats=repeats, cpu_wall_ms=float(np.median(cpu_ms)),
                gpu_event_ms=float(np.median(gpu_event_ms)),
                gpu_synchronized_wall_ms=float(np.median(gpu_wall_ms)))


def router_top10(index, layer, x):
    name = f"blk.{layer}.ffn_gate_inp.weight"
    desc = index.descriptors[name]
    raw = index.store.rows(name, 0, index.n_experts)
    if desc.quant_type in BLOCK_FORMATS:
        scores = packed_cpu_linear(x, PackedGGUFWeight(raw, desc.shape, desc.quant_type))[0]
    elif desc.quant_type in ("F32", "F16"):
        table = raw.view("<f4" if desc.quant_type == "F32" else "<f2").reshape(desc.shape)
        scores = (x @ table.astype(np.float32, copy=False).T)[0]
    elif desc.quant_type == "BF16":
        table = (raw.view("<u2").astype(np.uint32) << 16).view(np.float32).reshape(desc.shape)
        scores = (x @ table.T)[0]
    else:
        raise NotImplementedError(f"Unsupported bounded router type {desc.quant_type}")
    selected = np.argsort(-scores, kind="stable")[:index.top_k]
    route = np.exp(scores[selected] - np.max(scores[selected])).astype(np.float32)
    route /= route.sum(dtype=np.float32)
    return selected, route, desc.quant_type


def probe(model, repeats):
    if BACKEND_NAME != "cupy":
        raise RuntimeError("Run with MINI_LLM_BACKEND=cupy")
    import cupy as cp
    baseline_rss = rss()
    baseline_peak = peak_rss()
    started = time.perf_counter()
    assert cp.cuda.runtime.getDeviceCount() > 0
    def guard():
        extra = peak_rss() - baseline_rss
        if extra >= LIMIT:
            raise MemoryError(f"Additional peak RSS {extra} reaches 2 GiB limit")
    index = Qwen4ExpGGUFIndex.open(model, model_id="0228-real-iq3-s")
    original_rows = index.store.rows
    row_bytes = 0
    max_read = 0
    reads = 0
    def bounded_rows(name, first, last, *, outer=()):
        nonlocal row_bytes, max_read, reads
        desc = index.store.tensors[name]
        size = (last - first) * desc.row_bytes
        if size > 16 * 1024**2 or ("_exps." in name and len(outer) != 1):
            raise AssertionError("Unbounded checkpoint tensor access")
        result = original_rows(name, first, last, outer=outer)
        row_bytes += size
        max_read = max(max_read, size)
        reads += 1
        guard()
        return result
    index.store.rows = bounded_rows
    before = [(p.stat().st_size, p.stat().st_mtime_ns) for p in index.store.paths]
    x = np.random.default_rng(228).normal(0, .1, (1, index.d_model)).astype(np.float32)
    xd = cp.asarray(x)
    records = []
    def no_decode(*args, **kwargs):
        raise AssertionError("Full expert decoding is forbidden")
    executor = None
    try:
        with patch.object(PackedGGUFWeight, "dequantize_reference", no_decode), \
                patch.object(index.store, "packed_weight", no_decode):
            guard()
            for layer in range(index.n_layers):
                key = ExpertKey(index.model_id, layer, index.n_experts - 1)
                bundle = index.source(key)
                weights = bundle.weights
                device = tuple(packed_to_device(w) for w in weights)
                handle = PackedExpertFFNInference(*device)
                cpu = lambda: packed_cpu_expert(x, weights, backend="native")
                gpu = lambda: handle.forward(xd)
                expected = cpu()
                got = cp.asnumpy(gpu())
                record = dict(layer=layer, expert=key.expert_id,
                              formats=[w.quant_type for w in weights], packed_bytes=bundle.nbytes,
                              **error_metrics(got, expected), **timings(cpu, gpu, cp, repeats))
                records.append(record)
                guard()
                print(f"layer {layer:02d}: {record['formats']} relL2={record['relative_l2']:.3g} "
                      f"CPU={record['cpu_wall_ms']:.3f}ms GPU={record['gpu_event_ms']:.3f}ms", flush=True)
                del handle, device, bundle, weights, cpu, gpu
            signatures = Counter(tuple(row["formats"]) for row in records)
            if len(signatures) != 7:
                raise AssertionError(f"Expected seven checkpoint expert triples, found {signatures}")
            layer = index.n_layers - 1
            ids, routes, router_type = router_top10(index, layer, x)
            bank = {int(e): index.provider(ExpertKey(index.model_id, layer, int(e))) for e in ids}
            budget = sum(w.nbytes for ws in bank.values() for w in ws)
            executor = HeterogeneousExpertExecutor(lambda key: bank[key.expert_id],
                model_id=index.model_id, d_model=index.d_model, d_ff=index.d_ff,
                device_budget_bytes=budget, execution_mode="gpu", compute_dtype="float32",
                cpu_packed_backend="native", host_dequant_budget_bytes=0)
            managed = ExpertsInference(index.d_model, index.d_ff, index.n_experts,
                                       layer_id=layer, expert_executor=executor)
            rd = cp.asarray(routes.reshape(1, 1, -1))
            ed = cp.asarray(ids.astype(np.int64).reshape(1, 1, -1))
            def cpu_top10():
                out = np.zeros_like(x)
                for e in sorted(bank):
                    slot = int(np.flatnonzero(ids == e)[0])
                    out += routes[slot] * packed_cpu_expert(x, bank[e], backend="native")
                return out
            def gpu_top10():
                return managed.forward(xd.reshape(1, 1, -1), rd, ed).reshape(x.shape)
            expected = cpu_top10()
            got = cp.asnumpy(gpu_top10())
            top10 = dict(layer=layer, ids=ids.tolist(), routes=routes.tolist(), router_type=router_type,
                         selection="real uncorrected router logits on fixed synthetic input; selected softmax",
                         combine="ascending expert ID, sequential FP32 multiply then add",
                         packed_bytes=budget, **error_metrics(got, expected),
                         **timings(cpu_top10, gpu_top10, cp, repeats), execution=executor.statistics())
            guard()
            assert before == [(p.stat().st_size, p.stat().st_mtime_ns) for p in index.store.paths]
            return dict(scope="one packed expert/layer plus representative real-router top10; not model inference",
                        model_path=str(model), dtype="float32", input_seed=228, input_std=.1,
                        input_shape=list(x.shape), oracle="native packed CPU; no dense expert decode",
                        gpu=cp.cuda.runtime.getDeviceProperties(cp.cuda.Device().id)["name"].decode(),
                        cupy_version=cp.__version__, cuda_runtime=cp.cuda.runtime.runtimeGetVersion(),
                        n_layers=index.n_layers, records=records,
                        format_triples=[dict(formats=list(sig), layers=count) for sig, count in signatures.items()],
                        top10=top10, logical_mmap_read_bytes=row_bytes, mmap_row_requests=reads,
                        max_single_mmap_read_bytes=max_read, baseline_rss_bytes=baseline_rss,
                        baseline_peak_rss_bytes=baseline_peak, peak_rss_bytes=peak_rss(),
                        additional_peak_rss_bytes=peak_rss() - baseline_rss,
                        additional_rss_limit_bytes=LIMIT, checkpoint_size_mtime_unchanged=True,
                        elapsed_seconds=time.perf_counter() - started, passed=True,
                        timing_notes="warm median; excludes expert upload and JIT; CPU compute-only wall, "
                                     "GPU CUDA events and synchronized public-dispatch wall; not end-to-end model latency")
    finally:
        if executor is not None:
            executor.close()
            executor.cache.clear()
        index.store.close()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("model", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--repeats", type=int, default=5)
    args = parser.parse_args()
    if not 1 <= args.repeats <= 20:
        parser.error("--repeats must be in [1, 20]")
    if args.output.exists():
        parser.error("Refusing to overwrite a versioned report")
    report = probe(args.model, args.repeats)
    args.output.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps({k: report[k] for k in ("passed", "additional_peak_rss_bytes", "elapsed_seconds")}, indent=2))
    print("report:", args.output)


if __name__ == "__main__":
    main()
