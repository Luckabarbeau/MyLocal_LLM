# Patch 0237 — warp-per-row grouped IQ kernels

## Objective and files

Remove repeated block-wide shared-memory reduction barriers from grouped IQ gate/up and down projections.

- New `mini_llm/ops/iq_grouped_warp_gpu.py`.
- `mini_llm/ops/iq_packed_gpu.py`: independently selectable grouped schedule.
- `tests/test_0237_warp_iq_gpu.py`.
- `0237_iq_warp_scheduling_v1.json`, `0229_decode_performance_v9.json`.

## Kernel/API contract

`MINI_LLM_IQ_GPU_GROUPED_KERNEL=block|warp`; default remains `block`. Four independent output-row warps share one128-thread CTA. Warp-local shuffles replace shared-memory reduction barriers. Canonical packed lane equations/codebooks and their MIT attribution are reused, not reimplemented.

FP32 products and activation storage remain; within-row accumulation order changes. Down outputs retain a separate stored routing multiply and the existing ordered expert reduction. No Q8 activations, expert dropping, reduced-precision tensor cores, dense weight expansion or additional weight transfers are introduced. Current streams, strided/reversed/broadcast routes and explicit route slots remain supported.

## Numerical and timing evidence

Focused synthetic tests:27 passed/1 optional benchmark skipped. With the real checkpoint benchmark enabled: **28 passed in6.00s**. All7 actual triples,10 real experts per representative layer,20 warm samples/mode in balanced order. JIT/weight upload excluded; pointer-table upload included.

- Median CUDA-event speedup versus block scheduling: **1.472287×**.
- Worst real relative L2: **3.176314e-7**; worst max absolute difference3.492460e-10.
- Additional peak RSS882,761,728 bytes, max source slice921,600 bytes; checkpoint size/mtime unchanged.

Full model: **10.197347→12.033481tokens/s**, measured positions8–15 of forced IDs1–16. Worst relative L2 versus original FP32 production baseline2.014602e-6; maximum absolute difference2.717972e-5. All checked argmax/top10 sets match. RSS10,310,451,200 bytes; default pool9,829,571,072 reserved bytes.

This is **44.10×** the original0.272867tokens/s, not the70–80 target or a natural-language/long-context benchmark. Next dependency: bounded expert transfer/admission overlap and lower host dispatch cost.
