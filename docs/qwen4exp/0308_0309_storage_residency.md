# 0308–0309 — Natural-workload storage and bounded RAM-tier check

## Objective and status

Investigate why naturally sampled requests remain slower than the page-cache-hot retained trajectory, without blindly loading/pinning the complete checkpoint. **50 tokens/s remains unmet.** These are diagnostic/residency experiments, not a demonstrated new performance optimization.

## 0308: read-only complete tensor page census

Files: `scripts/probe_0308_page_residency.py`, `tests/test_0308_page_residency.py`.

`resident_bytes(address, length, chunk_pages=8192)` queries Linux `mincore` on an already-owned live mapping, using at most8192 page-status bytes per syscall. It never dereferences, faults in, pins or warms payload bytes. Partial first/last pages contribute only the actual payload bytes. Mapping lifetime remains with the store/caller. Invalid arguments fail explicitly. Because Linux can redact residency for unowned/unwritable files, the CLI rejects unowned checkpoint files rather than trusting all-one bits.

The census iterates **all1224 `store.tensors`**, not the expert-only index subset. `tensor_group` recognizes `per_layer_token_embd.weight` as PLE, even though its name contains no `ple` substring. Initial expert-only/misgrouped captures are retained but superseded by `diagnostics_0308_census_v4/result.json`.

Latest snapshot after the natural/RAM-source experiments:

| Canonical tensor group | Packed bytes | Resident payload bytes | Nonresident payload bytes |
|---|---:|---:|---:|
| Routed experts | 50,292,326,400 | 35,410,182,144 | 14,882,144,256 |
| PLE table and PLE operators | 28,865,879,040 | 696,651,584 | 28,169,227,456 |
| Other tensors | 4,448,432,640 | 4,117,328,384 | 331,104,256 |

The large PLE table alone is26.8222GiB. Routed experts total46.8384GiB. This census concerns **canonical file pages**: some nonresident expert pages may belong to experts already in VRAM, and VRAM copies do not appear as file-page residency. It is not the CPU-cache-complement residency table, a per-token page-fault trace, or unique physical RAM accounting. Tensor-edge pages may be shared; byte weighting avoids rounding whole tensors up, but does not make this a unique-page RAM allocator metric.

`/proc/self/io` read_bytes remained **23,994,368 before and after the final census**: header/index setup had already read metadata; the census itself caused zero additional payload disk reads. No checkpoint writes occurred.

Validation: **10passed** (`0308_tests_v2.xml`), including partial edges, one-page chunks, two-byte cross-page spans, arbitrary argument rejection and PLE/expert grouping. Initial768MiB-high metadata-index run was safely aborted by MemoryHigh; the successful metadata-only run used1792MiBhigh/2048MiBhard with unchanged12GiBhostreserve. All model benchmarks retained32/36GiBlimits. No safety check was bypassed.

## 0309: existing 4 GiB anonymous CPU source, no pinning or staging

Modified diagnostic interface: `probe_0302_decode_configuration.py --resident-host-gib 0|1|2|4`. It forwards an approved setup-only host-source budget into the existing model API. Default remains0; trace mode rejects this override rather than silently ignoring it. No new production arithmetic or RAM-store implementation was introduced.

The existing source selects warmup-observed nonresident expert keys, copies complete original packed projections and retains immutable canonical mappings independently. The native CPU registry uses the copied bytes. This experiment uses **no pinned expert bytes and no staged GPU misses**.

Retained `diagnostics_0309_host4_v1`:

- 2,111 complete bundles,4,294,733,824private bytes (~4GiB); source setup1.9063s.
- Exactly the same natural warmup/response IDs as0307naturalv2:371emittedtokens/370decodeforwards.
- Decode **32.51596forwards/s**, versus the earlier repeated natural35.15922/s: **no demonstrated gain**. Do not increase RAM budget automatically or promote this to default.
- Anonymous RSS high-water5,160,083,456bytes. Charged service memory peak~8.56GiB; zeroOOMkills. No complete host complement allocation.
- Warmup-to-response-end disk reads2,951,426,048bytes and40,639majorfaults. **This interval includes untimed source preparation as well as prefill/decode**, so these counters cannot establish that the source increased decode I/O. No exact decode-only I/O attribution is claimed.
- CPU/native/GPU expertise remains complete; all existing source byte/ownership tests from the471-passing CPU regression remain applicable. No production code changed after that verified regression, only diagnostic wrapper/census scripts and documentation.

Safe optional reproduction, with new output directories:

```sh
.venv/bin/python scripts/run_0290_isolated_benchmark.py \
  --memory-high-mib 32768 --memory-max-mib 36864 --timeout 200 \
  --output docs/qwen4exp/my_0309_host4_guard_v1 -- \
  .venv/bin/python scripts/probe_0302_decode_configuration.py \
    --shared-grid --natural --resident-host-gib 4 \
    --output docs/qwen4exp/my_0309_host4_v1
```

This is a diagnostic, not a recommended faster configuration. Do not add bulk warming, arbitrary larger budgets or full pinning to it.

## What is known / unresolved / next experiment

Confirmed: the natural trajectory is not fully warm even on a repeated identical token sequence; a substantial fraction of canonical expert pages is absent, and PLE is deliberately mostly disk-backed. A4GiBhot-key RAM copy did not improve the model observation. Existing GPU arithmetic accelerations do not automatically reduce latency when CPU/storage joins dominate.

Unresolved: exact per-source/per-phase faults and read bytes during **timed decode**, their overlap with native CPU work, and the contribution of placement versus Linux/cgroup cache residency. The aggregate census cannot distinguish these causes. The next useful experiment is native-worker/PLE per-source fault/I/O attribution, bracketed separately from source setup, prefill and sampling. Only then select a bounded prefetch/residency policy. Do not treat an arbitrary RAM fill or larger worker count as an evidence-supported fix.

Full previous ledger: `0295_0307_critical_path_optimization.md`. Latest structured results, guards and source SHA256s, including0308/0309: `0295_0309_optimization_results_v1.json`. Original captures are preserved.
