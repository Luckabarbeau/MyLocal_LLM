# 0334 — GPU expert capacity and CPU backend audit

## Scope and conclusion

Requested pre-optimization audit. This patch adds documentation only; it does not change inference mathematics, kernels, placement or defaults. Existing captures were reused. No new model execution, checkpoint copy, full host arena allocation or throughput measurement was performed.

**GPU expert capacity is not the principal deficit. Native CPU MoE is a major exposed critical-path component, but the available measurements do not establish that our arithmetic backend is inferior to Strata's.** The next experiment must normalize expert work rather than target CPU utilization.

## Complete GPU expert bundles

Counts below mean complete gate/up/down experts, not individual projections. The model has 48×512 = 24,576 routed experts.

| Configuration | Complete GPU experts | Packed GPU arena | CPU team |
|---|---:|---:|---|
| Local latest controlled replay, 0332 combined v2 | 3,673 | 7,515,878,400 bytes (~7.000 GiB), 7 GiB budget | 19 workers + participating service thread |
| Historical Strata 75.2 emitted tokens/s session | 2,792 | reported 5.35 GiB | 15 workers + participating host |
| Controlled Strata 0310, full top10/no drafting | 3,786 | reported 7.233398 GiB | 19 workers + participating host |

Local has **881 more bundles (+31.55%)** than the historical fast session, and **113 fewer (-2.98% relative to Strata)** than the controlled comparison. The latter has ~3.33% more arena bytes than our nominal 7 GiB budget; heterogeneous bundle sizes also affect capacity.

Local arena is full: zero free/quarantined slots; 3,673 leased slots. Seven physical size classes have capacities 383/230/76/1148/1071/689/76 at strides 1,510,400/1,715,200/1,868,800/1,971,200/2,176,000/2,329,600/2,662,400 bytes. CPU complement registration is 20,903 experts. Registered canonical CPU handles do not imply a second full resident/pinned arena. Cumulative promotions/transfer bytes in final snapshots include startup and profile replacement: they are not decode-only transfers. All slots being leased is consistent with frozen graph pointer ownership, not evidence of unused cache capacity.

Sources: `diagnostics_0332_combined_v2/result.json`, `0324_strata_sessions_v1.json`, `0324_0325_strata_residency_report.md`, `0310_0317_strata_comparison.md`.

## CPU implementation: current, not historical

Active local backend is `ggml_pool`, not the old per-expert Python executor:

- Persistent native workers and participating native service thread; mapped-doorbell handoff, no Python/GIL expert arithmetic.
- Row tasks across selected experts, task factor 3 (~60 tasks per phase with 20 participants).
- Gate/up phase, hidden quantization phase, down phase; shared format-specific input quantization.
- IQ2_S, IQ3_XXS, IQ4_NL use GGML dot kernels; IQ3_S uses validated AVX2 gather preserving GGML FMA order; Q2_0 uses the custom validated AVX2/GGML-Q8_0 path; IQ4_XS uses GGML.
- CPU activations are Q8_K or Q8_0 according to weight type. GPU Q8_1 is a distinct contract.
- Workers unpinned, host affinity 0–23, bounded spin 50 microseconds. Staging limit zero in the measured configuration; misses execute on CPU.

Relevant implementation: `mini_llm/ops/ggml_packed_cpu.py`, `mini_llm/ops/ggml_packed_cpu.cpp`, `mini_llm/ops/native_expert_service.h`. The 0258 Python-executor gap is historical and must not be attributed to current runs.

### Loaded ISA capability check

Fresh isolated feature query only, no matrix/model/GPU allocation:

- Intel Core Ultra 9 285K; hybrid PMU sets report performance cores 0–7 and atom cores 8–23; affinity permits all 24.
- Loaded library: `/home/lucka/Documents/my_local_LLM/qwen4exp-references/llama-no-prefetch/build-cpu/bin/libggml-cpu.so.0.25.3`.
- SHA256 `924ac13f3e949cb1b62eddff20221600a6efcb1a2b99eb4abd390e75c23df839`.
- AVX/AVX2/FMA/F16C/AVX-VNNI exported capability queries return 1; AVX512/AVX512-VNNI return 0. Capabilities do not prove every format selects its fastest possible kernel. Binary source-commit provenance is not independently proved by the ABI target revision.

Evidence: `diagnostics_0334_cpu_capabilities_guard_v1/result.json`. There is no evidence that the current backend accidentally uses an entirely scalar build. AVX512 optimization is not applicable to this CPU.

## What Strata does differently

Source pin: `712fec6b4297c9cfd32ab43daaa2979a9aa01edf`. Reinspected `src/kernels/cpu/native_expert.cpp`, `src/kernels/cpu/pool.cpp`, `include/strata/kernels/cpu/pool.hpp` at immutable GitHub raw URLs under that commit.

- Persistent pool with participating host and roughly three tasks per participant, like the local approach; hybrid topology-aware affinity.
- Default spin before sleep is 20 ms (20,000 microseconds), versus local 50 microseconds. This is a 400× threshold difference, **not a performance ratio**. Waiting workers can appear 100% busy without doing useful expert arithmetic.
- Singleton IQ paths commonly fall through to GGML `vec_dot`. Source availability of optional specialized kernels does not establish their activation in the installed binary. IQ3_S single-token specialization is gated by `STRATA_IQ3S_MT1`; default multi-token threshold is 2.
- Specialized multi-token paths can reuse decoding across tokens; singleton decode must be benchmarked separately from speculative/grouped windows.
- AVX2 byte-identical SIMD Q8_K quantizer; local calls the GGML quantizer. Quantization is measurable but small in the current local phase budget.
- Q2_0 specialized `q2_rows_any`/`ActQ` contract is not automatically identical to local GGML-Q8_0 arithmetic.
- Hidden rows are quantized serially between Strata row phases; local has a separate parallel phase. Previous local fused-hidden experiment did not establish a gain.
- Strata's phase-seam worker parking differs from local generation-tagged completion. Neither is inherently superior without matching work.

Only the header was saved by the attempted source snapshot: `reference_0334/include__strata__kernels__cpu__pool.hpp`, 26,795 bytes, SHA256 `a9bc33482b70a28428c7d94fce7ea0d78b865d06d87eef4290744ae4a33a7f59`. The bulk fetch timed out before a manifest was written; the two C++ files were inspected remotely, not fully archived locally. Do not represent this directory as a complete source snapshot or proof of installed binary options.

## Critical path and useful CPU work

Local hot trace `0330_critical_path_v2.json`:

- 16 warmed response-2 positions; GPU span 343.892 ms.
- Non-wait GPU operation union 220.367 ms.
- Summed GPU waits on native CPU completion 82.060 ms = **5.12874 ms/token**.
- Packed dense GPU projections ~4.99245 ms/token; HC ~2.39708; resident expert kernels ~1.84263.

Native-only hot capture `diagnostics_0330_native_v1/result.json`, final 293 production decode calls per layer, private graph warmups excluded:

| Native wall phase | ms/token |
|---|---:|
| Preparation | 0.11018 |
| Gate/up | 3.60233 |
| Hidden quantization | 0.18370 |
| Down | 1.64417 |
| Completion tail | 0.24915 |
| Total | 5.78952 |

143.382 CPU expert executions/token (~2.987/layer) out of 480 full-top10 selections: ~29.87% CPU, ~70.13% GPU-resident in this no-staging measured decode. This is not a historical Strata cache-hit comparison. Gate/up contribution is approximately 1.313 ms IQ3_XXS, 1.273 ms IQ2_S, 0.941 ms IQ3_S. Timings include scheduling and overlap GPU work: **do not add native and GPU totals**. Typical native start follows publication within ~1 microsecond in exploratory alignment; ordinary handoff delay is not demonstrated as the principal deficit. Long wait-inclusive copy API times are not evidence of physical PCIe bandwidth saturation.

Controlled Strata hot profile (`diagnostics_0310_strata_profile_v3/stderr.log`): 359 windows, average T=1, 20.99 ms/window; CPU host phase 11.11 ms/window, exposed GDN+QSA CPU wait 10.52 ms/window. It processes **6.02 CPU experts/layer**, versus ~2.99 locally. Different routing, placement, threads and timing scopes make 11.11 versus 5.79 ms an invalid backend speed comparison. Profiled 47.634 windows/s is not the separate unprofiled hot 49.26/s result.

## Interpreting utilization and RAM

Latest local four-response decode CPU-time/wall-time ratios are **8.18553, 8.26100, 8.30106, 8.27717 active-core equivalents**, ~34% of 24 CPUs. This includes service/application time and spinning, not only arithmetic. Strata's reported 16 saturated CPUs are user observations, not a newly matched per-thread capture.

Strata does more CPU-routed work in the controlled capture and spins much longer. Both can explain greater utilization. Neither proves that all the difference is waiting, or that local CPU kernels are optimal. Prior local spin-2000 experiment increased active-core equivalents from 9.517 to 18.207 while throughput fell 41.04→34.88/s. Copying affinity also regressed. Repeating these knobs without a normalized experiment is not justified.

Historical Strata pins a **46.84 GiB full expert arena**. Local keeps immutable mmap-backed sources and relies on bounded working-set/page residency; it does not preload/pin that full arena. This is a genuine configuration difference, not just a display label for used RAM. It may affect cold/page-pressure behavior, but does not prove the warm CPU arithmetic gap. A prior bounded 4 GiB anonymous host-source experiment did not improve throughput. Full pinning is unsafe under current 28/32 GiB high/hard limits, 12 GiB host reserve and swap=0; do not relax these guards or pin outside containment.

The historical 75.2 emitted tokens/s Strata session also uses route-tail omission, suffix drafting, INT8 KV and adaptive asynchronous movement. Its 78.9% cache statistic has different work/window denominators. This is not equivalent to local full-top10 fixed-trajectory forwards/s. Local latest hot-three aggregate is **47.17321 forwards/s**; target 50 remains unmet.

## Recommended next experiment

Run a bounded **same-input, same-expert native CPU A/B**, not another unrestricted model benchmark:

1. Use identical real packed expert bundles/IDs and saved FP32 activations; verify formats and preserve checkpoint storage.
2. Compare local kernels with the corresponding pinned Strata row arithmetic without porting its engine. Explicitly resolve the Q2_0 activation contract first; report numerical deltas, do not assume equality.
3. Warm exactly the same bounded bytes; test singleton decode separately from genuinely grouped tokens. Use 16 and 20 participating threads, identical task ranges and documented affinity.
4. Measure gate/up, activation/hidden quantization, down, task dispatch/completion, useful packed bytes/s and latency distributions. Preserve cold versus warm distinctions.
5. Select changes based on exposed CPU wait saved per token and validate mixed CPU/GPU output. A faster isolated operator is not a full-model gain.

Do not increase spin merely to make utilization look like Strata. Avoid rerunning performance-rejected IQ2_S gather/affinity/host-copy variants. Preserve the existing no-approximation production path and guards.

## Validation, limitations and next dependency

This is an audit, not a production implementation or new numerical-validation patch. Existing capture slot totals/configuration and CPU-time ratios were checked offline; the fresh isolated CPU feature query succeeded. No regression suite was rerun because production/test code is unchanged. Prior 0332 selected CPU/GPU regression results remain historical verified evidence, not newly executed tests. No new speedup, peak-memory benchmark or mathematical-equivalence claim.

Remaining uncertainty: matched native CPU throughput, installed Strata optional-kernel selection, degree of warm memory-bandwidth limitation, and natural-conversation profile generalization. Next dependency is the normalized native CPU A/B above; utilization alone cannot select a replacement backend.
