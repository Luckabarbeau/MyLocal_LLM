"""0106L packed Q6_K and explicit BF16 access, synthetic or one local expert.

No whole-model decode, model download, or automatic VRAM sizing. Native CPU
and GPU paths keep weights packed. Only the small reference expert is expanded.
"""

import argparse
import json
from pathlib import Path
import sys
import time

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np

from benchmarks.benchmark_0106d_heterogeneous_moe import packed_weight
from mini_llm.backend import BACKEND_NAME, asnumpy, xp
from mini_llm.diagnostics.quantized_numerics import packed_swiglu_reference
from mini_llm.external.qwen3next_gguf import Qwen3NextGGUFIndex
from mini_llm.ops.native_packed_cpu import packed_cpu_expert
from mini_llm.ops.quantized_linear import PackedExpertFFNInference, packed_to_device
from mini_llm.ops.quantized_weight import BLOCK_FORMATS


def errors(actual, expected):
    actual, expected = np.asarray(actual), np.asarray(expected)
    error = np.abs(actual.astype(np.float64) - expected.astype(np.float64))
    tolerance = 2e-5 + 2e-4 * np.abs(expected.astype(np.float64))
    return dict(passed=bool(actual.shape == expected.shape and actual.dtype == expected.dtype
                            and np.isfinite(actual).all() and np.all(error <= tolerance)),
                max_abs_error=float(error.max()),
                max_tolerance_ratio=float((error / tolerance).max()), rtol=2e-4, atol=2e-5)


def timed(fn, iterations):
    start = time.perf_counter()
    for _ in range(iterations):
        fn()
    return (time.perf_counter() - start) * 1000 / iterations


def bf16_sample(index):
    candidates = [t for t in index.tensors.values() if t.tensor_type.name == "BF16"
                  and not t.name.endswith("_exps.weight") and "ngram" not in t.name]
    if not candidates:
        return dict(status="absent")
    tensor = min(candidates, key=lambda t: t.data.nbytes)
    if tensor.data.nbytes > 32 * 1024**2:
        return dict(status="skipped", reason="Smallest BF16 tensor exceeds 32 MiB source cap")
    values = index.non_expert_weight(tensor.name)
    expected_bits = tensor.data.reshape(-1).view("<u2").astype(np.uint32) << 16
    exact = bool(np.array_equal(values.reshape(-1).view(np.uint32), expected_bits))
    return dict(status="measured", name=tensor.name, shape=list(values.shape),
                dtype=str(values.dtype), source_bytes=int(tensor.data.nbytes),
                decoded_bytes=int(values.nbytes), exact_bit_expansion=exact,
                finite=bool(np.isfinite(values).all()))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", type=Path)
    parser.add_argument("--layer", type=int, default=46)
    parser.add_argument("--expert", type=int, default=0)
    parser.add_argument("--d-model", type=int, default=512)
    parser.add_argument("--d-ff", type=int, default=256)
    parser.add_argument("--mixed", action="store_true", help="Synthetic Q4_K/Q5_K/Q6_K expert")
    parser.add_argument("--iterations", type=int, default=30)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.iterations < 1 or min(args.d_model, args.d_ff) < 1 or args.d_model % 256 or args.d_ff % 256:
        parser.error("Positive iterations and block-aligned positive synthetic widths required")
    index = None
    if args.model is not None:
        index = Qwen3NextGGUFIndex.open(args.model)
        weights = index.expert_weights(args.layer, args.expert)
        source = dict(path=str(args.model), layer=args.layer, expert=args.expert,
                      inventory=index.report(), bf16=bf16_sample(index))
    else:
        formats = ("Q4_K", "Q5_K", "Q6_K") if args.mixed else ("Q6_K",) * 3
        shapes = ((args.d_ff, args.d_model), (args.d_ff, args.d_model),
                  (args.d_model, args.d_ff))
        weights = tuple(packed_weight(fmt, *shape, 106120 + i)
                        for i, (fmt, shape) in enumerate(zip(formats, shapes)))
        # Keep synthetic values numerically modest; byte layout/traffic is
        # unchanged. Never alter real checkpoint payloads.
        for weight in weights:
            raw = weight.data.reshape(-1, BLOCK_FORMATS[weight.quant_type][1])
            offset = 208 if weight.quant_type == "Q6_K" else 0
            raw[:, offset:offset + 2] = np.frombuffer(np.float16(1 / 1024).tobytes(), dtype=np.uint8)
            if weight.quant_type in ("Q4_K", "Q5_K"):
                raw[:, 2:4] = np.frombuffer(np.float16(1 / 2048).tobytes(), dtype=np.uint8)
        source = dict(kind="synthetic", seed=106120, mixed=args.mixed)
    packed_bytes = sum(w.nbytes for w in weights)
    if packed_bytes > 32 * 1024**2:
        raise ValueError("Single-expert smoke exceeds explicit 32 MiB packed-byte cap")
    d_model = weights[0].shape[1]
    x = np.random.default_rng(10612).normal(0, .02, (1, d_model)).astype(np.float32)
    expected = packed_swiglu_reference(x, *weights)["output"]
    # First native compilation is excluded from measured iterations.
    native = packed_cpu_expert(x, weights, backend="native")
    native_error = errors(native, expected)
    native_ms = timed(lambda: packed_cpu_expert(x, weights, backend="native"), args.iterations)
    dense = tuple(w.dequantize_reference() for w in weights)

    def reference(matrices):
        g, u = x @ matrices[0].T, x @ matrices[1].T
        z = np.exp(-np.abs(g))
        sigmoid = np.where(g >= 0, 1 / (1 + z), z / (1 + z))
        return ((g * sigmoid) * u) @ matrices[2].T

    cached_ms = timed(lambda: reference(dense), args.iterations)
    cold_ms = timed(lambda: reference(tuple(w.dequantize_reference() for w in weights)), args.iterations)
    result = dict(patch="0106L", backend=BACKEND_NAME, source=source,
                  scope="One packed expert, not full-model generation",
                  formats=[w.quant_type for w in weights], shapes=[list(w.shape) for w in weights],
                  dtype="float32", input_seed=10612, iterations=args.iterations,
                  packed_bytes=packed_bytes, dense_reference_bytes=sum(w.nbytes for w in dense),
                  native_cpu=dict(mean_ms=native_ms, error=native_error),
                  dense_fp32_cached_blas_ms=cached_ms, dense_fp32_dequant_each_call_ms=cold_ms,
                  gpu=dict(status="skipped", reason="CuPy backend not selected"))
    if BACKEND_NAME == "cupy":
        gpu_weights = tuple(packed_to_device(w) for w in weights)
        gpu_expert = PackedExpertFFNInference(*gpu_weights)
        device_x = xp.asarray(x)
        actual = asnumpy(gpu_expert.forward(device_x))
        gpu_error = errors(actual, expected)
        start, end = xp.cuda.Event(), xp.cuda.Event()
        wall_start = time.perf_counter()
        start.record()
        for _ in range(args.iterations):
            gpu_expert.forward(device_x)
        end.record()
        end.synchronize()
        wall_ms = (time.perf_counter() - wall_start) * 1000 / args.iterations
        event_ms = float(xp.cuda.get_elapsed_time(start, end)) / args.iterations
        result["gpu"] = dict(status="measured", mean_host_wall_ms=wall_ms,
                             mean_stream_interval_ms=event_ms, error=gpu_error,
                             uploaded_packed_bytes=sum(w.nbytes for w in gpu_weights))
    result["passed"] = bool(native_error["passed"] and
        (result["gpu"]["status"] == "skipped" or result["gpu"]["error"]["passed"]) and
        (index is None or source["bf16"].get("exact_bit_expansion", True)))
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    print(json.dumps({k: v for k, v in result.items() if k != "source"}, indent=2))
    if index is not None:
        print("Actual inventory unsupported routed formats:", source["inventory"]["unsupported_routed_formats"])
        print("Actual BF16 sample:", source["bf16"])
    if not result["passed"]:
        raise AssertionError(f"Expert/BF16 parity failed; see {args.output}")


if __name__ == "__main__":
    main()
