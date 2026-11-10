"""0106K: matched-storage arithmetic audit of the original 0106B/C regressions.

Requires the dev dependency pytest: runpy loads the ACTUAL test fixture factories,
without running tests. No model, expert cache, throughput claim, or GPU decoder
reference is involved. The FP64 arithmetic oracle still uses dequantize_reference.

CPU-only (GPU measurements explicitly skipped):
  MINI_LLM_BACKEND=numpy OPENBLAS_NUM_THREADS=1 python \
      benchmarks/benchmark_0106k_matched_precision.py --output /tmp/0106k-cpu.json
CUDA audit:
  MINI_LLM_BACKEND=cupy OPENBLAS_NUM_THREADS=1 python \
      benchmarks/benchmark_0106k_matched_precision.py --output /tmp/0106k-gpu.json

Legacy failures are observations, not exit causes. Matched GPU failures (including
shape/dtype/nonfinite failures) are saved before raising. Samples, not matrices,
are written to JSON; undefined relative errors/cancellation ratios are null.
"""

import argparse
import json
from pathlib import Path
import runpy
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np

from mini_llm.backend import BACKEND_NAME
from mini_llm.diagnostics.quantized_numerics import packed_swiglu_reference
from mini_llm.ops.quantized_linear import packed_to_device, quantized_linear, quantized_swiglu

ROOT = Path(__file__).resolve().parents[1]
SAMPLE_LIMIT = 12
TINY = np.finfo(np.float64).tiny


def finite_number(value):
    value = float(value)
    return value if np.isfinite(value) else None


def ratio(numerator, denominator):
    if not np.isfinite(denominator) or abs(denominator) <= TINY:
        return None
    return finite_number(numerator / denominator)


def error_fields(actual, reference, rtol, atol):
    a, r = float(actual), float(reference)
    absolute = abs(a - r)
    tolerance = atol + rtol * abs(r)
    return dict(actual=finite_number(a), reference=finite_number(r),
                abs_error=finite_number(absolute), relative_error=ratio(absolute, abs(r)),
                requested_tolerance=finite_number(tolerance), rtol=rtol, atol=atol,
                tolerance_ratio=ratio(absolute, tolerance))


def error_arrays(actual, reference, rtol, atol):
    a = np.asarray(actual, dtype=np.float64)
    r = np.asarray(reference, dtype=np.float64)
    with np.errstate(over="ignore", invalid="ignore", divide="ignore"):
        absolute = np.abs(a - r)
        tolerance = atol + rtol * np.abs(r)
        relative = np.full(r.shape, np.nan, dtype=np.float64)
        np.divide(absolute, np.abs(r), out=relative, where=np.abs(r) > TINY)
        tolerance_ratio = absolute / tolerance
    finite = np.isfinite(a) & np.isfinite(r) & np.isfinite(absolute)
    failed = ~finite | (absolute > tolerance)
    return absolute, relative, tolerance_ratio, failed


def comparison(actual, reference, rtol, atol):
    """All arithmetic for error measurement is FP64, never FP16 subtraction."""
    actual, reference = np.asarray(actual), np.asarray(reference)
    result = dict(shape=list(actual.shape), dtype=str(actual.dtype),
                  reference_shape=list(reference.shape), reference_dtype=str(reference.dtype),
                  rtol=rtol, atol=atol, passed=False, violation_count=None,
                  max_abs_error=None, max_relative_error=None, max_tolerance_ratio=None,
                  samples=[])
    if actual.shape != reference.shape:
        result["reason"] = "shape mismatch"
        return result
    absolute, relative, tolerance_ratio, failed = error_arrays(actual, reference, rtol, atol)
    for name, values in (("max_abs_error", absolute), ("max_relative_error", relative),
                         ("max_tolerance_ratio", tolerance_ratio)):
        finite = values[np.isfinite(values)]
        result[name] = float(finite.max()) if finite.size else None
    result.update(passed=not bool(failed.any()), violation_count=int(failed.sum()),
                  nonfinite_count=int((~np.isfinite(actual) | ~np.isfinite(reference)).sum()),
                  undefined_relative_count=int((~np.isfinite(relative)).sum()))
    # Include the worst coordinates even when every comparison passes.
    order = np.argsort(np.where(np.isfinite(tolerance_ratio), tolerance_ratio, np.inf),
                       axis=None)[::-1][:SAMPLE_LIMIT]
    for flat in order:
        index = tuple(int(i) for i in np.unravel_index(flat, actual.shape))
        result["samples"].append(dict(index=list(index),
            **error_fields(actual[index], reference[index], rtol, atol)))
    return result


def legacy_failure_examples(actual, legacy, matched, rtol, atol):
    _, _, ratios, failed = error_arrays(actual, legacy, rtol, atol)
    indices = np.flatnonzero(failed)
    order = np.argsort(np.where(np.isfinite(ratios.flat[indices]),
                               ratios.flat[indices], np.inf))[::-1]
    examples = []
    for flat in indices[order[:SAMPLE_LIMIT]]:
        index = tuple(int(i) for i in np.unravel_index(flat, actual.shape))
        examples.append(dict(index=list(index), actual=finite_number(actual[index]),
            legacy_reference=finite_number(legacy[index]),
            matched_reference=finite_number(matched[index]),
            legacy=error_fields(actual[index], legacy[index], rtol, atol),
            matched=error_fields(actual[index], matched[index], rtol, atol)))
    return examples, failed


def fixtures():
    """Do not replace these with lookalike random fixtures or copied factories."""
    try:
        b = runpy.run_path(str(ROOT / "tests/test_0106b_quantized_linear.py"))
        c = runpy.run_path(str(ROOT / "tests/test_0106c_fused_packed_expert.py"))
    except ModuleNotFoundError as exc:
        if exc.name == "pytest":
            raise RuntimeError("0106K explicitly requires the dev dependency pytest") from exc
        raise
    packed_weight, make_quant = b["_packed_weight"], c["make_quant"]
    for fmt in ("Q3_K", "Q4_K"):
        yield dict(name=f"resident_full_fp32_{fmt}", kind="full_expert", dtype="float32",
                   source="tests/test_0106b_quantized_linear.py",
                   test="test_resident_packed_expert_keeps_quantized_bytes_on_gpu",
                   factory="_packed_weight", weight_seeds=[91, 92, 93], input_seed=111,
                   input_std=.01, rtol=2e-5, atol=2e-3,
                   x=np.random.default_rng(111).normal(0, .01, (2, 256)).astype(np.float32),
                   gate=packed_weight(fmt, 256, 256, seed=91),
                   up=packed_weight(fmt, 256, 256, seed=92),
                   down=packed_weight(fmt, 256, 256, seed=93))
    for gate_fmt, up_fmt in (("Q4_K", "Q4_K"), ("Q3_K", "Q4_K")):
        yield dict(name=f"fused_fp16_{gate_fmt}_{up_fmt}", kind="fused_hidden", dtype="float16",
                   source="tests/test_0106c_fused_packed_expert.py",
                   test="test_triton_fused_gate_up_matches_cpu", factory="make_quant",
                   weight_seeds=[171, 172], input_seed=173, input_std=.07,
                   rtol=2e-2, atol=6e-3,
                   x=np.random.default_rng(173).normal(0, .07, (2, 3, 512)).astype(np.float32),
                   gate=make_quant(gate_fmt, 256, 512, 171),
                   up=make_quant(up_fmt, 256, 512, 172), down=None)


def host_references(case):
    x, gate, up, down = (case[k] for k in ("x", "gate", "up", "down"))
    dtype, rtol, atol = case["dtype"], case["rtol"], case["atol"]
    # Preserve the exact OLD test formula/FP32 BLAS arithmetic, not a stable-SiLU
    # transcription or a reference recomputed after rounding the FP16 input.
    if down is not None:
        g = x @ gate.dequantize_reference().T
        u = x @ up.dequantize_reference().T
        legacy_hidden = (g / (1 + np.exp(-g))) * u
        legacy = legacy_hidden @ down.dequantize_reference().T
    else:
        legacy_hidden = legacy = quantized_swiglu(x, gate, up)
    matched = packed_swiglu_reference(x, gate, up, down, storage_dtype=dtype)
    original64 = packed_swiglu_reference(x.astype(np.float64), gate, up, down,
                                         storage_dtype=np.float64)
    rounded_input64 = packed_swiglu_reference(x.astype(dtype).astype(np.float64),
                                              gate, up, down, storage_dtype=np.float64)
    rounding = dict(
        semantics="Each effect is isolated; effect magnitudes are not additive error budgets.",
        input_storage=comparison(matched["input"], x.astype(np.float64), rtol, atol),
        input_rounding_effect={key: comparison(rounded_input64[key], original64[key], rtol, atol)
                               for key in ("gate", "up", "hidden_unrounded", "output_unrounded")},
        hidden_storage=comparison(matched["hidden"], matched["hidden_unrounded"], rtol, atol),
        output_storage=comparison(matched["output"], matched["output_unrounded"], rtol, atol),
        separate_output_store=down is not None,
        output_aliases_hidden=down is None,
    )
    if down is not None:
        rounding["hidden_rounding_effect_on_down"] = comparison(
            matched["output_unrounded"], rounded_input64["output_unrounded"], rtol, atol)
    # This is the original FP32 CPU path, not a fabricated GPU measurement.
    cpu_hidden = quantized_swiglu(x, gate, up)
    cpu_output = quantized_linear(cpu_hidden, down) if down is not None else cpu_hidden
    cpu = dict(status="measured_cpu_only", input_dtype="float32",
               note="Original unrounded FP32 input; observational, not a matched GPU contract check.",
               legacy=comparison(cpu_output, legacy, rtol, atol),
               matched=comparison(cpu_output, matched["output"], rtol, atol),
               hidden_matched=comparison(cpu_hidden, matched["hidden"], rtol, atol))
    return legacy, legacy_hidden, matched, rounding, cpu


def cancellation_at_failures(failed, legacy_hidden, matched_hidden, actual_hidden, down,
                             legacy, matched):
    """All old-failing output coordinates, but no full hidden/weight matrices."""
    weights = down.dequantize_reference().astype(np.float64)
    hs = dict(legacy_fp32_hidden=legacy_hidden, matched_stored_hidden=matched_hidden,
              actual_gpu_stored_hidden=actual_hidden)
    entries = []
    for coord in np.argwhere(failed):
        index = tuple(int(i) for i in coord)
        row, column = index[:-1], index[-1]
        entry = dict(index=list(index))
        for name, hidden in hs.items():
            terms = np.asarray(hidden[row], dtype=np.float64) * weights[column]
            sum_abs = float(np.sum(np.abs(terms), dtype=np.float64))
            reference = float(np.sum(terms, dtype=np.float64))
            entry[name] = dict(sum_abs=finite_number(sum_abs),
                               fp64_dot=finite_number(reference),
                               sum_abs_over_abs_fp64_dot=ratio(sum_abs, abs(reference)))
        entry["sum_abs_over_abs_legacy_reference"] = ratio(
            entry["legacy_fp32_hidden"]["sum_abs"], abs(float(legacy[index])))
        entry["sum_abs_over_abs_matched_reference"] = ratio(
            entry["matched_stored_hidden"]["sum_abs"], abs(float(matched[index])))
        entries.append(entry)
    return dict(definition="sum(abs(hidden[j] * down[col,j])) / abs(reference)",
                coordinate_count=len(entries), entries=entries)


def gpu_measurement(cp, case, legacy, legacy_hidden, matched):
    x, gate, up, down = (case[k] for k in ("x", "gate", "up", "down"))
    dtype, rtol, atol = case["dtype"], case["rtol"], case["atol"]
    device_x = cp.asarray(x, dtype=dtype)
    device_gate, device_up = packed_to_device(gate), packed_to_device(up)
    device_hidden = quantized_swiglu(device_x, device_gate, device_up)
    device_down = packed_to_device(down) if down is not None else None
    device_output = (quantized_linear(device_hidden, device_down)
                     if down is not None else device_hidden)
    actual_hidden, actual = cp.asnumpy(device_hidden), cp.asnumpy(device_output)
    effective_input = cp.asnumpy(device_x)
    contract = (actual.shape == matched["output"].shape
                and actual_hidden.shape == matched["hidden"].shape
                and actual.dtype == np.dtype(dtype) and actual_hidden.dtype == np.dtype(dtype))
    result = dict(status="measured", contract_passed=contract,
                  actual_shape=list(actual.shape), actual_dtype=str(actual.dtype),
                  legacy=comparison(actual, legacy, rtol, atol),
                  matched=comparison(actual, matched["output"], rtol, atol),
                  hidden_matched=comparison(actual_hidden, matched["hidden"], rtol, atol),
                  device_input_matches_oracle=bool(np.array_equal(
                      effective_input.astype(np.float64), matched["input"])))
    result["passed"] = (contract and result["matched"]["passed"]
                        and result["hidden_matched"]["passed"]
                        and result["device_input_matches_oracle"])
    if actual.shape != legacy.shape or actual_hidden.shape != matched["hidden"].shape:
        return result
    examples, old_failed = legacy_failure_examples(actual, legacy, matched["output"], rtol, atol)
    result["legacy_failing_examples"] = examples
    result["legacy_failing_examples_total"] = int(old_failed.sum())
    # Probe separate packed GEMVs with FP32 output, using values actually stored
    # on device (FP16-rounded where applicable), not the original host input.
    probe_x = cp.ascontiguousarray(device_x.astype(cp.float32))
    probes = dict(note="Independent quantized_linear FP32 dot probes, NOT direct internal "
                       "fused-kernel instrumentation; fused g/up are never stored or rounded.")
    for name, weight in (("gate", device_gate), ("up", device_up)):
        probe = cp.asnumpy(quantized_linear(probe_x, weight))
        probes[name] = comparison(probe, matched[name], rtol, atol)
    result["independent_projection_probes"] = probes
    if down is not None:
        # Hold the actual GPU hidden fixed: upstream fused error cannot enter
        # this local down-reference comparison a second time.
        local64 = (actual_hidden.astype(np.float64).reshape(-1, down.shape[1])
                   @ down.dequantize_reference().astype(np.float64).T).reshape(actual.shape)
        local_stored = local64.astype(dtype)
        result["local_down"] = dict(
            note="FP64 down dot with ACTUAL GPU stored hidden; isolates down arithmetic/store.",
            against_unrounded=comparison(actual, local64, rtol, atol),
            against_storage_rounded=comparison(actual, local_stored, rtol, atol),
            output_rounding_effect=comparison(local_stored, local64, rtol, atol))
        result["down_cancellation_at_legacy_failures"] = cancellation_at_failures(
            old_failed, legacy_hidden, matched["hidden"], actual_hidden, down,
            legacy, matched["output"])
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    cp, skip_reason = None, f"backend={BACKEND_NAME}; GPU measurement requires MINI_LLM_BACKEND=cupy"
    if BACKEND_NAME == "cupy":
        import cupy as cp
        if cp.cuda.runtime.getDeviceCount() == 0:
            cp, skip_reason = None, "No CUDA device; GPU measurements explicitly skipped"
        else:
            import triton  # Required by the production packed operators.
    payload = dict(
        benchmark="0106K", backend=BACKEND_NAME, numpy_version=np.__version__,
        scope="Isolated original-fixture packed expert arithmetic, not model/cache/throughput.",
        oracle="packed_swiglu_reference: input rounded to storage then FP64; FP64 fused g/up "
               "and stable SiLU; hidden storage rounding; optional FP64 down then output store. "
               "Decoding remains dequantize_reference; this is not an independent decoder audit.",
        relative_error_policy="abs(actual-reference)/abs(reference); null if denominator <= "
                              "FP64 tiny or result nonfinite; absolute errors remain reported.",
        legacy_failure_policy="Observational only; original tolerances unchanged for BOTH oracles.",
        projection_probe_policy="Independent FP32 quantized_linear probes, not fused instrumentation.",
        dev_dependency="pytest (actual test files loaded with runpy.run_path, no tests executed)",
        sample_limit=SAMPLE_LIMIT, cases=[], failures=[],
    )
    for case in fixtures():
        result = {key: value for key, value in case.items() if key not in ("x", "gate", "up", "down")}
        result.update(input_shape=list(case["x"].shape), original_input_dtype=str(case["x"].dtype),
                      formats={key: case[key].quant_type if case[key] is not None else None
                               for key in ("gate", "up", "down")},
                      weight_shapes={key: list(case[key].shape) if case[key] is not None else None
                                     for key in ("gate", "up", "down")})
        try:
            legacy, legacy_hidden, matched, rounding, cpu = host_references(case)
            result.update(rounding_effects=rounding, original_cpu_fp32=cpu,
                          legacy_vs_matched=comparison(legacy, matched["output"],
                                                       case["rtol"], case["atol"]))
            if cp is None:
                result["gpu"] = dict(status="skipped", reason=skip_reason, passed=None)
            else:
                result["gpu"] = gpu_measurement(cp, case, legacy, legacy_hidden, matched)
                if not result["gpu"]["passed"]:
                    payload["failures"].append(dict(case=case["name"], reason="matched GPU failure"))
        except Exception as exc:
            result["error"] = f"{type(exc).__name__}: {exc}"
            payload["failures"].append(dict(case=case["name"], reason=result["error"]))
        payload["cases"].append(result)
        gpu = result.get("gpu", {})
        print(f"{case['name']}: GPU={gpu.get('status', 'error')}; "
              f"legacy violations={gpu.get('legacy', {}).get('violation_count')}; "
              f"matched violations={gpu.get('matched', {}).get('violation_count')}")
    measured = sum(case.get("gpu", {}).get("status") == "measured" for case in payload["cases"])
    payload.update(gpu_cases_measured=measured, gpu_cases_skipped=len(payload["cases"]) - measured,
                   matched_gpu_passed=(not payload["failures"] if measured else None),
                   status="failed" if payload["failures"] else ("completed" if measured else "gpu_skipped"))
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    print(f"Saved {args.output}; matched GPU cases measured={measured}; failures={len(payload['failures'])}")
    if payload["failures"]:
        raise AssertionError(f"0106K audit failed; diagnostics saved to {args.output}")


if __name__ == "__main__":
    main()
