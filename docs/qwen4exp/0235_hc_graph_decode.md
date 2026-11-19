# Patch 0235 — bounded opt-in HC graph replay

## Objective and files

Reuse the existing `CapturedArrayInference` helper around static hyper-connection reads, leaving routing, cache placement and dynamic QSA positions outside capture.

- `mini_llm/external/qwen4exp_hyperconnection.py`: `graph_decode=False`, capped private pools and graph statistics.
- `mini_llm/ops/hyperconnection_decode_gpu.py`: optional projection callback.
- `mini_llm/external/qwen4exp_weights.py`, `qwen4exp_model.py`: `hc_graph_decode=False`, explicit workspace reservation/owned-byte replanning and safe unload.
- `tests/test_0235_hc_graph_decode.py`; `0229_decode_performance_v7.json`.

## Interface and ownership

One graph per HC object; contiguous singleton FP32 inputs only. A changed shape/stream falls back without replacing a live graph. Replay copies inputs and returns fresh output ownership; no token position or persistent model state is captured. Collapse on read-only final HC excludes injection. Default remains disabled.

Each pool is capped at2MiB. The model reserves all97 pools' maximum203,423,744 bytes before construction and credits only actual graph-owned bytes during replanning. Observed97 graphs/1,552 calls reserve17,153,024 private-pool bytes. Graph-enabled unload synchronizes once before releasing buffers, including consumers on nondefault streams.

## Observed failure and localized correction

The first GPU test failed because installed CuPy rejects cuBLAS calls during stream capture. The correction reuses existing graph-capturable `matmul_inference` FP32 GEMV in the opt-in graph path; normal HC projections remain unchanged. No FP16/BF16/TF32 substitution was introduced.

Corrected focused GPU run including existing HC tests: **53 passed in1.30s**. Focused NumPy/planner/model tests: **28 passed,8 skipped in0.75s**. Tests cover real dimensions2560/rank320, changing inputs, fresh outputs, pool exhaustion, changed stream/shape and physical budget checks.

## Performance decision

Full-model forced-token rate: **8.965350tokens/s**, versus9.174106 without this graph. Same-prefix argmax/top10 sets match; worst relative L2 versus original FP32 baseline1.929070e-6. Sampled process RSS10,322,583,552 bytes; global VRAM high-water is not measured, and the default pool snapshot excludes these private pools.

**No speedup demonstrated; keep disabled.** The result does not justify graph capture at this granularity. Next dependency: coarser dispatch/launch reduction with the existing bounded ownership contracts, not speculative changes to model mathematics.
