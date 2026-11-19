# 0254 — Longer sampled workload aligned to captured Strata requests

## Contract and files

Added `scripts/probe_0254_strata_workload.py`, `tests/test_0254_strata_workload.py`. The runner reuses captured `outputs.jsonl` requests, messages and chat template. GGUF and installed Strata tokenizer assets produce identical prompt token IDs:75 for warmup,77 for short cases. Sampling temperature1, top_p0.95, top_k20, seeds1/2/3, thinkingxhigh, zero penalties,128 warmup and512 short response caps. PRNG/backend differences mean generated sequences need not match. Each request starts a fresh state; expert cache persists. Sampler is the repository's existing `generation.sample_token`.

Reports distinguish N emitted tokens, N-1 post-prefill decode forwards, prefill completion time, TTFT, sampling/text time and64-token windows. Process memory is separated into anonymous and mmap file residency, with memory/I/O journals. Default absolute anonymous cap8GiB, system-available floor8GiB,180s watchdog. Full model payloads/checkpoints remain unchanged; no complete dequantized expert bank.

Not full configuration parity: MyLocal context allocation1024 versus Strata capacity262144, FP32 versusINT8 KV, full top10 versus conditional tail skipping, no suffix drafts versus suffix active, reclaimable host mappings versus pinned complement. These distinctions are recorded in every result. Workers can now match captured15; cache budgets/slot counts still differ. Model GPU expert arithmetic remains FP32; optional CPU GGML path uses format-specific Q8_K/Q8_0.

## Results and limitations

| Run | Outcome |
|---|---|
| `diagnostics_0254_v1` | Original sort/native FP32 CPU, no host warming;180s cap before warmup completed.96 emitted at145.50s,27.00GB storage reads,16,061 evictions,11,627 CPU fallbacks. Not a throughput result for the complete workload. |
| `diagnostics_0255_long_v1` | Heap policy + host warming; native FP32 CPU,4 workers.128-token warmup completed,127 decode forwards/71.5925s =1.77393/s.180s cap before short request completed. |
| `diagnostics_0256_long_v1` | Heap + mapped host warming + GGML CPU +15 workers. **Supervisor and child completed** in120.72s overall.128 warmup followed by427-token short response endingEOS.426 post-prefill forwards/53.00188s =**8.03745 forwards/s**; emitted427/decode_seconds =**8.05632/s**. TTFT9.09466s;77-token prefill9.09446s. |

The captured Strata short-r1 response emitted406 tokens and reported47.475385 decode tokens/s. Same prompt and requested sampling, but output length/content, RNG, arithmetic, routing, cache and speculation differ. The previously quoted~20.5 MyLocal rate was a16-token repeated-prefix/cache-fitting workload and is **not** the comparable sustained rate.

0256 memory:50,292,326,400 source expert bytes; warming touched47,795,105,792bytes with8GiB reserve; estimated45,671,370,752bytes resident. Source not completely warmed. Sampled anonymous peak825,704,448bytes; fileRSS52,317,421,568bytes. CuPy pool reservation12,334,209,024bytes, not global VRAM high-water. No private complete expert copy or pinned whole source.

## Supervisor race — unresolved

Later0257 inference children completed and wrote result files, but parent telemetry could encounter `PermissionError` while reading `/proc/<pid>/io` as the child exited. A localized correction checks `poll()` before treating this as an exit race. Its next execution still failed: `/proc/io` access was denied while `poll()` momentarily returnedNone, with returncode0 observed in cleanup. Do not mark those parent runs passed.

No further correction was made after the allowed correction/retest. Next focused experiment: reproduce/simulate that exit transition and test a bounded child-reaping check, without suppressing permission failures on a live inference process. Existing unit expectations were also corrected to disable mutually exclusive HC graph/native-read modes. Focused CPU benchmark/cache/model tests53 passed3 skipped; later CPU benchmark/model30 passed1 skipped. A green unit suite did not establish race-free parent supervision.
