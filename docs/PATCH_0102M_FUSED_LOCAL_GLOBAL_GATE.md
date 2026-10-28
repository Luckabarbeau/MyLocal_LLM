# 0102M: Experimental fused local/global gate

Optimization-only patch after 0102L. The gating formula is unchanged:

`g = sigmoid(x @ W_gate)`, `effective = g * available`,
`context = local + effective * (global - local)`.

The new `mini_llm/ops/triton_dual_resolution_gate.py` replaces separate CuPy
sigmoid, availability masking, attention-context interpolation, gradient split,
and gate-logit derivative with **two Triton launches** (one forward, one backward).
No change to gate projection GEMMs, global visibility conditions, the compressor,
or trainable parameters. Both kernels accumulate in FP32. NumPy reference remains
untouched. The default remains **off** until tested on a CUDA GPU.

Enable via `MINI_LLM_DUAL_RESOLUTION_FUSED_GATE=1`; disable with `=0`.
Requires the CuPy backend and Triton. The short <=128-token local-only bypass
remains active and never launches this gate. The optional 0102L fused compressor
RMSNorm setting is separate and is not changed by this patch.

## Validate on RTX 5080

```bash
MINI_LLM_BACKEND=cupy python -m pytest -v tests/test_dual_resolution_fused_gate_0102m.py
```

Standalone test (small overhead, same forward+backward work both modes):

```bash
MINI_LLM_BACKEND=cupy python benchmarks/benchmark_dual_resolution_gate_0102m.py \
  --contexts 4096,16384,65536 --warmup 3 --repeats 10
```

Then benchmark full model (warmed). All attention kernels remain Triton and
packed BF16 projections and fused pooling remain enabled:

```bash
for mode in 0 1; do
  MINI_LLM_BACKEND=cupy \
  MINI_LLM_DUAL_RESOLUTION_FUSED_POOLING=1 \
  MINI_LLM_DUAL_RESOLUTION_FUSED_RMSNORM=1 \
  MINI_LLM_DUAL_RESOLUTION_FUSED_GATE=$mode \
  python benchmarks/benchmark_dual_resolution_full_model_0102d.py \
    --context 4096 --layers 20 --batch-size 1 --compression-dim 512 \
    --score-last-k 128 --head-chunk-tokens 64 --activation-checkpoint block \
    --local-kernel triton --global-kernel triton --compression-kernel triton \
    --query-chunk 512 --warmup-iterations 2 --repeat-iterations 5 \
    --output "0102m_gate_${mode}.json"
done
```

These full-model benchmarks exclude optimizer work and supervise 128 rows.
They are appropriate for a same-workload A/B test, **not** a production
supervised-token throughput measurement. The effect at 32K/65K could differ.

GPU parity and timing are not validated in the CPU-only patch build environment.
Keep `MINI_LLM_DUAL_RESOLUTION_FUSED_GATE=0` in production until tests pass.
