# 0089J — Matched GPU performance gate for 0089F–I

Each candidate is opt-in and must be measured in a **new process** against
exactly the same checkpoint, batch, router seed, history length, model settings
and attention backend. `compare_0089j_optimization.py` rejects changed
checkpoint/hardware/input hashes, any confounding environment flag, profiler-
perturbed samples, and scalar-loss parity failure. It does not claim parameter-
gradient parity or full optimizer-step throughput, which require separate tests.

Run these from the repository root on RTX 5080/CuPy, with the same existing
338M checkpoint and data root. Example for history-heavy workload:

```bash
MINI_LLM_BACKEND=cupy MINI_LLM_TRAIN_ATTN_STRICT=1 \
MINI_LLM_HISTORY_BF16_BACKWARD=0 \
python benchmarks/profile_0087a_training.py \
  --checkpoint ./checkpoints/0083_intermediate_20L_1024_fullmix_100k \
  --mixed-shard-root ./data/token_shards_ensemble_32k_mixed \
  --batch-size 3 --working-length 4096 --warmup 2 --repeats 20 \
  --attention-backend triton --no-optimizer \
  --out benchmarks/results/0089j_history_base.json

MINI_LLM_BACKEND=cupy MINI_LLM_TRAIN_ATTN_STRICT=1 \
MINI_LLM_HISTORY_BF16_BACKWARD=1 \
python benchmarks/profile_0087a_training.py \
  --checkpoint ./checkpoints/0083_intermediate_20L_1024_fullmix_100k \
  --mixed-shard-root ./data/token_shards_ensemble_32k_mixed \
  --batch-size 3 --working-length 4096 --warmup 2 --repeats 20 \
  --attention-backend triton --no-optimizer \
  --out benchmarks/results/0089j_history_candidate.json

python benchmarks/compare_0089j_optimization.py \
  --baseline benchmarks/results/0089j_history_base.json \
  --candidate benchmarks/results/0089j_history_candidate.json \
  --optimization history --out benchmarks/results/0089j_history_comparison.json
```

Repeat for `ffn-dx`, `ffn-wgrad`, `cross-block` using the corresponding flag
from `benchmarks/compare_0089j_optimization.py`, changing **only that flag**.

Interpretation:

- 0089F: B3/T4096 *with non-empty actual history*; verify report history
  validity and unchanged input hashes. History lengths differ by batch, so
  aggregate latency over identical sampled input hashes only.
- 0089G/H: B3/T2304 without history, separate from history-heavy test.
- 0089I: additionally run the original 0089E benchmark at depths 2/4/8,
  including 22-tensor depth-2 numerical parity before timing.
- Do not combine flags until single-patch numerical correctness and matched
  speedups have passed. Full optimizer-step supervised tokens/s is the final
  production acceptance metric, not no-update forward/backward latency.
- Record GPU high-water memory in a separate training run: 0089A reports
  no optimizer and does **not** prove full production VRAM safety.

CPU tests:

```bash
MINI_LLM_BACKEND=numpy python -m pytest -q \
  tests/test_0089j_optimization_comparison.py
```
