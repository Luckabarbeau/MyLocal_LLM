# 0250 — Three-launch FP32 hyper-connection read

## Objective and graph

The warm Nsight trace identified launch-heavy HC projection work. `ops/hyperconnection_read_gpu.py` computes one token's HC read in three kernels: existing four-stream normalization; fused down/inject products with their SiLU/gain epilogues; fused up projection, stable sigmoid and ordered stream mixing. Products/reductions remain FP32 with `--fmad=false`. Weight encoding and activation precision are unchanged. Reduction order differs from cuBLAS, so numerical validation rather than bit equality is required against the older path.

New API `read_gpu(streams,gamma,down,up,inject,epsilon)` returns fresh mixed/gain arrays. It validates geometry, dtype, contiguity and current device without value readback. It retains no activation or weight caches; only compiled kernels are cached. Integration is separately opt-in: `Qwen4ExpGGUFModel.open(..., hc_fast_decode=True, hc_native_read=True)`. CPU/unsupported input and default HC behavior remain unchanged. HC graph mode and native read are separate mutually exclusive options.

Modified/added: `mini_llm/ops/hyperconnection_read_gpu.py`, `mini_llm/external/qwen4exp_hyperconnection.py`, `mini_llm/external/qwen4exp_model.py`, `tests/test_0250_hc_read_gpu.py`. The terminal chat default has not been switched.

## Tests and numerical measurements

- GPU new HC plus retained HC/HC-graph tests: **90 passed,1 skipped**; skip requires a second device.
- Synthetic production geometry D2560/R320 with BF16-expanded FP32 weights: worst mix relativeL2~1.28e-7 versus the retained fast path, ~9.15e-8 versus FP64; maxabs~2.13e-7. Gains maxabs~5.96e-8 versus retained path. Existing2e-6 infinity-norm tolerance unchanged.
- Nondefault streams, fresh output ownership, inject=None, layout/dtype rejection and read-only collapse covered.
- Real checkpoint fixed IDs1..16, full top10 FP32: final logits maxabs8.34465e-6, relativeL2 **6.51844e-7** versus the preceding FP32 path; argmax17 unchanged, top10 overlap10/10. Warm repeated native-path logits and active-state hashes are bit-exact across fresh requests. Cross-path states are not claimed bit-exact.

## Performance scope

Synthetic same-process six-pair operator benchmark D2560/R320: old read approximately0.059–0.061ms versus native0.020–0.022ms. Raw samples: `0250_hc_operator_v1.json`.

Real checkpoint bounded run `diagnostics_0250_v1/`: warm16-token repeated-pass rates **17.8544 /17.7630 /17.8277 forced tokens/s**, sampled peakRSS10,302,586,880bytes. Previous batched-validation path measured16.31–16.49 in a separate process; this is not a matched model A/B speed ratio. Resident expert cache and fixed-prefix workload differ from Strata's prolonged chat with suffix speculation and tail omission. No50t/s claim. This patch is opt-in pending broader real-prompt validation; model/reference equivalence is not established by matching rankings.

## Next dependency

Continue with launch-heavy elementwise/projection work, preserving independent reference paths and deterministic state tests. Candidate0251 has a remaining test-harness API failure and is **not integrated or validated**; see its blocker report. Do not enable it until the focused suite completes.
