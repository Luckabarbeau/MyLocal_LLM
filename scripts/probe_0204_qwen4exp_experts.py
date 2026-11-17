"""Bounded real-checkpoint sparse expert diagnostics; never loads complete banks.

Run with MINI_LLM_BACKEND=numpy and the repository interpreter. This is expert
validation only, not a qwen4exp forward pass or independent model-logit oracle.
"""
import argparse
import json
from pathlib import Path
import resource
import sys
import time

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from mini_llm.external.qwen4exp_gguf import Qwen4ExpGGUFIndex
from mini_llm.ops.expert_residency import ExpertKey, HeterogeneousExpertExecutor


def probe(path):
    start = time.perf_counter()
    index = Qwen4ExpGGUFIndex.open(path, model_id="local-iq3-s")
    executor = HeterogeneousExpertExecutor(
        index.provider, model_id=index.model_id, d_model=index.d_model,
        d_ff=index.d_ff, compute_dtype="float32", execution_mode="cpu",
        cpu_packed_backend="native", host_dequant_budget_bytes=0,
    )
    before = [(p.stat().st_size, p.stat().st_mtime_ns) for p in index.store.paths]
    x = np.random.default_rng(204).normal(0, .01, (1, index.d_model)).astype(np.float32)
    seen, records = set(), []
    try:
        for layer in range(index.n_layers):
            signature = tuple(index.descriptors[f"blk.{layer}.ffn_{p}_exps.weight"].quant_type
                              for p in ("gate", "up", "down"))
            if signature in seen:
                continue
            seen.add(signature)
            expert = 511 if index.n_experts == 512 else index.n_experts - 1
            bundle = index.source(ExpertKey(index.model_id, layer, expert))
            gate, up, down = bundle.weights
            t0 = time.perf_counter()
            # Bounded independent Python decoder path versus native lane readers.
            g = gate.matvec_reference(x[0], tile_rows=8)
            u = up.matvec_reference(x[0], tile_rows=8)
            sigmoid = np.exp(-np.logaddexp(np.float32(0), -g))
            expected = down.matvec_reference((g * sigmoid) * u, tile_rows=8)
            result = executor.forward(layer, expert, x)[0]
            if not np.isfinite(expected).all() or not np.isfinite(result).all():
                raise AssertionError(f"nonfinite expert result at layer {layer}")
            error = np.abs(result.astype(np.float64) - expected.astype(np.float64))
            scale = np.maximum(np.abs(expected.astype(np.float64)), 1e-12)
            record = dict(layer=layer, expert=expert, formats=signature,
                          packed_bytes=bundle.nbytes, max_absolute_error=float(error.max()),
                          max_relative_error=float((error / scale).max()),
                          reference_max_abs=float(np.abs(expected).max()),
                          seconds=time.perf_counter() - t0, atol=2e-5, rtol=5e-5)
            records.append(record)
            np.testing.assert_allclose(result, expected, atol=record["atol"], rtol=record["rtol"],
                                       err_msg=f"layer={layer}, expert={expert}, formats={signature}")
        assert before == [(p.stat().st_size, p.stat().st_mtime_ns) for p in index.store.paths]
        return dict(scope="selected real experts, not complete model inference",
                    tensor_count=len(index.store), layers=index.n_layers,
                    experts_per_layer=index.n_experts, dtype="float32",
                    oracle="bounded Python block decoding; independent compiled GGML block parity tested separately",
                    records=records, execution=executor.statistics(),
                    peak_rss_bytes=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss * 1024,
                    elapsed_seconds=time.perf_counter() - start,
                    gpu_allocated_bytes=0, checkpoint_size_mtime_unchanged=True)
    finally:
        executor.close()
        index.store.close()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("model", type=Path)
    args = parser.parse_args()
    print(json.dumps(probe(args.model), indent=2))


if __name__ == "__main__":
    main()
