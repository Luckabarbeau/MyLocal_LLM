#!/usr/bin/env python3
"""Isolated synthetic QSA operator A/B only: no weights, integrations, or model.

Must be launched through run_0290_isolated_benchmark.py. CUDA event timings
include repeated operator calls (including their allocations/launch gaps), not
model-step latency. Compilation, setup, host copies and FP64 checks are outside
timed regions. JSON contains every sample and old/new/FP64 numerical errors.
"""

import argparse
import json
import os
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT), str(ROOT / "scripts")]


def oracle_fp64(np, arrays, n, scale):
    q, qg, k, v, keys, values = [a.astype(np.float64) for a in arrays]
    heads, dim = q.shape
    result = []
    for h in range(heads):
        kh = h // (heads // k.shape[0])
        all_k = np.concatenate((keys[:n-1, kh], k[kh:kh+1]))
        all_v = np.concatenate((values[:n-1, kh], v[kh:kh+1]))
        logits = np.sum(all_k * q[h], axis=1) * float(np.float32(scale))
        weight = np.exp(logits - logits.max())
        context = np.sum(weight[:, None] * all_v, axis=0) / weight.sum()
        gate = qg[h, dim:]
        z = np.exp(-np.abs(gate))
        result.append(context * np.where(gate >= 0, 1/(1+z), z/(1+z)))
    return np.asarray(result).ravel()


def errors(np, actual, reference):
    reference = reference.astype(np.float64)
    delta = actual.astype(np.float64) - reference
    return dict(max_abs=float(np.abs(delta).max()),
                relative_l2=float(np.linalg.norm(delta) / max(np.linalg.norm(reference), 1e-30)))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True, help="New JSON file; parent must exist")
    parser.add_argument("--warmups", type=int, default=10)
    parser.add_argument("--repetitions", type=int, default=100)
    parser.add_argument("--trials", type=int, default=7)
    args = parser.parse_args()
    if not (1 <= args.warmups <= 100 and 1 <= args.repetitions <= 1000
            and 1 <= args.trials <= 25):
        parser.error("require warmups 1..100, repetitions 1..1000, trials 1..25")
    if not args.output.parent.is_dir() or args.output.exists():
        parser.error("output parent must exist and output must be a new file")
    from run_0290_isolated_benchmark import require_isolation
    require_isolation()  # Before any CUDA import/allocation; no direct fallback.

    os.environ["MINI_LLM_BACKEND"] = "cupy"
    import numpy as np
    import cupy as cp
    import triton
    from mini_llm.ops.qsa_decode_gpu import qsa_decode_gpu
    from mini_llm.ops.qsa_cooperative_gpu import qsa_cooperative_gpu

    rng = np.random.default_rng(314)
    heads, kv, dim, scale = 24, 2, 256, 1/16
    operators = dict(serial=qsa_decode_gpu, cooperative=qsa_cooperative_gpu)
    device = cp.cuda.runtime.getDevice()
    name = cp.cuda.runtime.getDeviceProperties(device)["name"]
    if isinstance(name, bytes):
        name = name.decode()
    result = dict(scope="synthetic standalone operators, NOT model latency or throughput",
                  precision="FP32 only; no tensor-core dot, TF32, FP16 or BF16",
                  rounding="FP32 tree reductions and split-context log-sum-exp merge differ from serial",
                  tolerance=dict(new_fp64_atol=4e-6, new_fp64_rtol=2e-5,
                                 old_fp64_atol=8e-6, old_fp64_rtol=4e-5),
                  device=device, device_name=name, cupy=cp.__version__, triton=triton.__version__,
                  heads=heads, kv_heads=kv, dim=dim, scale=scale, seed=314,
                  warmups=args.warmups, repetitions=args.repetitions, trials=args.trials,
                  timing="CuPy events on a nonblocking current stream; alternating A/B trial order",
                  cases=[])
    with cp.cuda.Stream(non_blocking=True) as stream:
        for n in (141, 435, 1024):
            host = [rng.normal(size=s).astype(np.float32) for s in (
                (heads, dim), (heads, 2*dim), (kv, dim), (kv, dim),
                (n+1, kv, dim), (n+1, kv, dim))]
            host[1][:, :dim] = np.nan
            host[-2][n-1:] = np.nan
            host[-1][n-1:] = np.nan
            inputs = [cp.asarray(a) for a in host]
            before = [cp.asnumpy(a).view(np.uint32).copy() for a in inputs]
            reference = oracle_fp64(np, host, n, scale)
            outputs = {key: cp.asnumpy(op(*inputs, n, scale)) for key, op in operators.items()}
            unchanged = all(np.array_equal(cp.asnumpy(a).view(np.uint32), b)
                            for a, b in zip(inputs, before))
            accuracy = dict(new_fp64=errors(np, outputs["cooperative"], reference),
                            old_fp64=errors(np, outputs["serial"], reference),
                            new_old=errors(np, outputs["cooperative"], outputs["serial"]))
            accepted = bool(unchanged and np.allclose(outputs["cooperative"], reference,
                                                    atol=4e-6, rtol=2e-5)
                            and np.allclose(outputs["serial"], reference, atol=8e-6, rtol=4e-5))
            for _ in range(args.warmups):
                for op in operators.values():
                    op(*inputs, n, scale)
            stream.synchronize()
            samples = {key: [] for key in operators}
            start, end = cp.cuda.Event(), cp.cuda.Event()
            for trial in range(args.trials):
                order = ("serial", "cooperative") if trial % 2 == 0 else ("cooperative", "serial")
                for key in order:
                    start.record(stream)
                    for _ in range(args.repetitions):
                        output = operators[key](*inputs, n, scale)
                    end.record(stream)
                    end.synchronize()
                    samples[key].append(float(cp.cuda.get_elapsed_time(start, end) / args.repetitions))
            medians = {key: float(np.median(value)) for key, value in samples.items()}
            result["cases"].append(dict(n=n, accepted=accepted, inputs_bitwise_unchanged=unchanged,
                                        errors=accuracy, samples_ms=samples, median_ms=medians,
                                        speedup=medians["serial"] / medians["cooperative"]))
    result["accepted"] = all(case["accepted"] for case in result["cases"])
    with args.output.open("x") as file:
        json.dump(result, file, indent=2, allow_nan=False)
        file.write("\n")
    print(json.dumps(result, allow_nan=False), flush=True)
    if not result["accepted"]:
        raise RuntimeError("operator parity failed; see recorded JSON, do not treat timing as acceptance")


if __name__ == "__main__":
    main()
