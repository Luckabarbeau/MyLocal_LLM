"""CPU-only 0106M sequence/chunk/step checks; never requests GPU resources.

Synthetic mode reuses the independent regression fixture/oracle (pytest dev
extra required). --model loads only one mixer, not experts or a whole decoder.
Actual-model mode checks streaming parity, not independent model/logit parity.
"""

import argparse
from dataclasses import asdict
import json
from pathlib import Path
import runpy
import sys
import time

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np

from mini_llm.backend import BACKEND_NAME
from mini_llm.external.qwen3next_gguf import Qwen3NextGGUFIndex
from mini_llm.external.qwen3next_mixer import Qwen3NextRecurrentMixerInference
from mini_llm.ops.quantized_weight import PackedGGUFWeight


def compare(actual, expected):
    error = np.abs(actual.astype(np.float64) - expected.astype(np.float64))
    tolerance = 5e-6 + 3e-4 * np.abs(expected.astype(np.float64))
    return dict(passed=bool(np.isfinite(actual).all() and np.all(error <= tolerance)),
                max_abs_error=float(error.max()) if error.size else 0.0,
                max_tolerance_ratio=float((error / tolerance).max()) if error.size else 0.0,
                rtol=3e-4, atol=5e-6)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", type=Path)
    parser.add_argument("--layer", type=int, default=0)
    parser.add_argument("--tokens", type=int, default=16)
    parser.add_argument("--cpu-backend", choices=("reference", "native"), default="native")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if BACKEND_NAME != "numpy":
        parser.error("This smoke check is CPU-only; use MINI_LLM_BACKEND=numpy")
    if args.tokens < 1:
        parser.error("--tokens must be positive")
    functions = weights = None
    if args.model is None:
        functions = runpy.run_path(str(Path(__file__).resolve().parents[1] /
                                      "tests/test_0106m_qwen3next_mixer.py"))
        index, weights, _ = functions["_fixture"]()
        layer = 0
    else:
        index = Qwen3NextGGUFIndex.open(args.model)
        layer = args.layer
    mixer = Qwen3NextRecurrentMixerInference.from_index(index, layer, args.cpu_backend)
    batch = 2 if args.model is None else 1
    rng = np.random.default_rng(10613)
    prefix = rng.normal(0, .2, (batch, 3, mixer.spec.d_model)).astype(np.float32)
    hidden = rng.normal(0, .2, (batch, args.tokens, mixer.spec.d_model)).astype(np.float32)
    initial = mixer.create_state(batch)
    # Warm projections/compiled CPU kernels and both raw/recurrent histories.
    mixer.forward(prefix, initial)
    full, chunk, step = (initial.copy() for _ in range(3))
    begin = time.perf_counter()
    output = mixer.forward(hidden, full)
    sequence_ms = (time.perf_counter() - begin) * 1000
    midpoint = max(1, args.tokens // 2)
    chunked = np.concatenate((mixer.forward(hidden[:, :midpoint], chunk),
                              mixer.forward(hidden[:, midpoint:], chunk)), axis=1)
    begin = time.perf_counter()
    stepped = np.stack([mixer.step(hidden[:, t], step) for t in range(args.tokens)], axis=1)
    step_ms = (time.perf_counter() - begin) * 1000
    comparisons = dict(output_chunk=compare(chunked, output), output_step=compare(stepped, output),
                       history_chunk=compare(chunk.conv.history, full.conv.history),
                       history_step=compare(step.conv.history, full.conv.history),
                       memory_chunk=compare(chunk.delta.memory, full.delta.memory),
                       memory_step=compare(step.delta.memory, full.delta.memory))
    if functions is not None:
        reference, ref_history, ref_memory = functions["_reference"](
            hidden, weights, mixer.spec, initial.conv.history, initial.delta.memory)
        comparisons["independent_fp64_output"] = compare(output, reference)
        comparisons["independent_fp64_history"] = compare(full.conv.history, ref_history)
        comparisons["independent_fp64_memory"] = compare(full.delta.memory, ref_memory)
    expected_length = 3 + args.tokens
    lengths = dict(full=full.length, chunk=chunk.length, step=step.length)
    passed = bool(all(c["passed"] for c in comparisons.values()) and
                  all(n == expected_length for n in lengths.values()))
    result = dict(patch="0106M", backend=BACKEND_NAME,
        scope="One FP32 recurrent token mixer, not a decoder or generation throughput",
        model=str(args.model) if args.model is not None else "synthetic-ratio3-dk3-dv2",
        layer=layer, spec=asdict(mixer.spec), batch=batch, tokens=args.tokens,
        prefix_tokens=3, dtype="float32", input_seed=10613,
        initial_state_nonzero=bool(np.any(initial.conv.history) and np.any(initial.delta.memory)),
        state_bytes=int(full.conv.history.nbytes + full.delta.memory.nbytes),
        packed_projection_bytes=sum(w.nbytes for w in mixer.projection_weights.values()
                                    if isinstance(w, PackedGGUFWeight)),
        lengths=lengths, comparisons=comparisons, passed=passed,
        observed_cpu_wall_ms=dict(sequence=sequence_ms, token_steps=step_ms),
        timing_warning="CPU correctness smoke under concurrent Strata workload; not a controlled performance comparison",
        independent_model_reference=("Synthetic FP64 graph with scalar convolution and affine recurrence"
                                     if functions is not None else "Not run for actual checkpoint"),
        gpu_validation="pending; existing Strata inference left undisturbed",
        full_model_decode_available=False)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    print(json.dumps(result, indent=2))
    if not passed:
        raise AssertionError(f"Recurrent mixer parity failed; see {args.output}")


if __name__ == "__main__":
    main()
