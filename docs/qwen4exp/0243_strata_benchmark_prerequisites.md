# 0243 — Strata benchmark prerequisites (in progress)

## Objective

Measure the installed Strata on the same checkpoint and machine, including memory tiers, request throughput and decode timing, before choosing optimizations or matching arithmetic. This is a provenance and safety audit, **not a benchmark result**.

## Inspected 2026-11-17

- MyLocal_LLM HEAD: `314746cb0ac516e27282234e4fe81943db13b46d`; worktree has pre-existing modified and untracked files, retained.
- Installed engine build reported previously: `0.1.59`, `712fec6b4297+cuda13.0`; matching source commit `712fec6b4297c9cfd32ab43daaa2979a9aa01edf`. Exact-version source consulted: [docs/DETAILS.md](https://github.com/Niko1221/Strata/blob/712fec6b4297c9cfd32ab43daaa2979a9aa01edf/docs/DETAILS.md), [src/core/layer.cpp](https://github.com/Niko1221/Strata/blob/712fec6b4297c9cfd32ab43daaa2979a9aa01edf/src/core/layer.cpp). The older local Strata checkout has a different revision.
- `strata launch --model iq3_s --context 262144 --parallel 1 --vision off --access local --no-api-key --port 8095 --plan --output json` selects DFlash2 and says a 1.2 GB drafter download is required. The planned pack path `/home/lucka/softwares/local_llm_lab_north_v7/local_llm_lab/dflash2/qwen3.8-flash-next-dflash2.q4_0.pack` was absent.
- The existing `/home/lucka/.local/share/strata/install/strata-iq3_s.json` has `--spec 4`, `--kv int8`, `--pool-workers 15`, `--expert-cache auto`, `--max-context 262144`, and `mtp: false`. `strata drafting plan --config ... --mode auto --output json` reports `choice: off`. This saved-config plan is not the same operation as the exact launch plan and neither proves what was active during the earlier conversation. `--spec 4` alone does not prove a model drafter is loaded; other speculative mechanisms may exist.
- No Strata process was observed at inspection. `nvidia-smi` reported 14,830 MiB free and 1,002 MiB used overall (the editor accounted for 113 MiB of compute-process usage). Host `free -h` reported 62 GiB total, 54 GiB available and 4 GiB of 8 GiB swap in use. These are snapshots, not a memory reservation. A full model load can still cause severe reclaim and may change available memory for the user's work.

## Source-derived hypotheses to verify at runtime

- The exact-version documentation describes `STRATA_ROUTE_TAIL_SKIP=7` as a default on eligible CUDA configurations: a nonresident expert at rank 7 or lower for all tokens in a verify window can be skipped. Its application depends on cache and execution mode, and it changes answers. Do not treat it as equivalent to full top-10 inference.
- `layer.cpp` dispatches native packed GPU projections using `native_quantize_q8_1` and `native_mmvq`; other canonical paths use Q8_K or Q8_0 according to weight representation. BF16 projection behavior also depends on the enabled native path. This is not a complete inventory of active kernels or CPU arithmetic.
- The exact-version documentation identifies `STRATA_DECODE_TIMING=1` and `STRATA_VERIFY_PROFILE=1` for decode wall-time and GPU stage diagnostics. GPU profiling perturbs throughput, so use separate unprofiled and profiled requests. `/metrics` can report cache hits, transfer tiers, and draft acceptance where available. Startup and request logs must establish actual switches and residency; documentation defaults alone do not.
- The documentation describes normally copying experts into pinned RAM, and fallback/resident/mapped modes under memory pressure. Actual tier and size must be measured on this machine. MyLocal's mmap page warming is reclaimable file cache, not the same as a pinned RAM arena.

## Unresolved and next dependency

No live Strata model was started and no operation times or sustained tokens/s were measured in this audit. The reported 70–80 tokens/s remains user-reported and its context length, prompt, drafting, cache and numerical settings are unknown. The old MyLocal 12.03 tokens/s forced-token trace is not an apples-to-apples comparison.

To measure the **current exact launch defaults**, obtain the missing 1.2 GB drafter via Strata with user authorization, then run a supervised, memory-guarded server with a fixed recorded conversation. Keep model runs sequential and preserve existing logs. Record actual startup settings, host anonymous/file-backed memory, VRAM, tier occupancy, request lengths, draft acceptance and timings; collect stage timing in a separate profiling pass. Compare with a separately labelled drafting-off/full-routing baseline and MyLocal using the same workload. Context capacity is not actual conversation length. Never apply route skipping or quantization changes silently; keep them optional and validate them against the existing numerical reference.
