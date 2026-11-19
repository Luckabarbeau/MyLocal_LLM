# 0249 — Opt-in compile-time specialization of packed IQ warp kernels

`MINI_LLM_IQ_GPU_GROUPED_KERNEL=warp_specialized` constant-propagates D/F, pointer-table width and gate/up/down format/block-layout parameters in the existing warp kernels. Launch ABI, decoder equations, FP32 products/reduction ordering and route combination are unchanged. `warp` and `block` remain available. Compiled warp module cache bounded to32 entries; no cached activations or full dequantized experts. Modified `ops/iq_grouped_warp_gpu.py`, `ops/iq_packed_gpu.py`, warp tests; added `scripts/probe_0249_specialized_iq.py` for bounded real seven-format A/B.

Actual GPU warp suite **54 passed,1 skipped**. Real checkpoint resident top10 operators: all seven format triples bit-exact, warm median wall-speed ratios1.40–1.97×; sampled peakRSS1,218,166,784bytes. See `diagnostics_0249_v1/operators.json` for balanced-order8-pair samples.

Full-model same-process A/B (`diagnostics_0249_model_ab_v1/`): generic warp passes1/3 =16.2477/16.3184 forced tokens/s; specialized passes2/4 =16.4705/16.2513. Final logits and active-state hashes exact across all repeats. **No material demonstrated model throughput improvement**, despite faster expert kernels. Keep specialization opt-in. The launch-heavy host path limits how much kernel improvement reaches wall time; no50t/s claim.
