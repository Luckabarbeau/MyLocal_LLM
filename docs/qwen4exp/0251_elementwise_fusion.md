# 0251 — Integrated bit-exact Qwen4Exp elementwise fusion

## Objective and implementation

Reduce launch-heavy shared-expert and DeltaNet coefficient expressions identified in the warm timeline. New `ops/qwen4exp_elementwise_gpu.py` provides three cached, stride-aware FP32 ElementwiseKernels: shared SwiGLU intermediate, shared expert admission/combine, and the two-output beta/log-decay computation. Stable sigmoid, installed CuPy logaddexp arithmetic, all FP32 storage boundaries and `--fmad=false` are retained. No Q8 arithmetic or route omissions in this patch.

The previously recorded test blocker is resolved by intercepting ElementwiseKernel constructor options rather than accessing the unavailable `.options` property. The historical blocker report is preserved; it is superseded by this report.

Integration is opt-in via `Qwen4ExpGGUFModel.open(..., elementwise_fast_decode=True)`. `Qwen4ExpMoE.fast_elementwise` and recurrent mixer `fast_coefficients` control their respective stages. Defaults retain existing operations; NumPy falls back unchanged. Modified files: `mini_llm/external/qwen4exp_model.py`, `mini_llm/external/qwen4exp_mixer.py`, `mini_llm/ops/qwen4exp_elementwise_gpu.py`, `tests/test_0251_qwen4exp_elementwise_gpu.py`. New flags are checked as bool.

## Tests

- Candidate GPU suite: **43 passed,1 skipped**.
- GPU elementwise/mixer/sigmoid-decode/native-HC regression group: **112 passed,4 skipped**.
- CPU model/Qwen3-Next mixer/attention regression group: **132 passed,1 skipped**.
- Exact operator checks include signed zero, saturation, negative inputs, nondefault streams, strides, dtype/shape rejection, and large bounded arrays. No tolerance changed.

## Controlled checkpoint A/B

`diagnostics_0251_ab_v1/`: one model process with `hc_native_read=True`, original model weights, fixed IDs1..16, full top10 FP32, no speculation. After a cold warmup, passes1/3 use prior elementwise expressions; passes2/4 fuse them. The benchmark switches the corresponding operator attributes at fresh request boundaries. No state is shared across requests except immutable model weights and expert residency.

| Arm | Warm forced tokens/s |
|---|---:|
| Prior |18.02873 /17.97284|
| Fused |20.53097 /20.23955|

Mean of the two rates is approximately18.00 versus20.39 (~13.25% higher), in this short repeated-prefix cache-resident workload only. Final logits and active-state hashes are bit-exact across all five passes. Sampled peakRSS10,301,206,528bytes (~9.59GiB), below direct-child20GiB RSS guard;120s child deadline retained. Device budget unchanged. This is not prolonged natural-language emitted-token throughput and not a50t/s result.

## Interpretation and next dependency

The launch-heavy path masked most of the0249 expert-kernel improvement. Removing safe launch work improves whole-model latency without changed precision. This does not prove every gap is removable or that kernel speedups must translate directly to model throughput.

The user's requested Strata-aligned Q8 path is a separate0252 candidate. It must reproduce the exact Q8_1 producer and integer dot contracts, not merely quantize/dequantize with a generic8-bit scheme. Its independent CUDA oracle is currently blocked; no production Q8 switch is enabled. See `0252_q8_1_contract.md`.
