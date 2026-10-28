# Patch 0102C — Experimental Triton global-summary + compressor attention

This patch continues 0102B (local Triton). **No model architecture or checkpoint tensor names change.** The compressor remains full bidirectional attention over each 128-token block, followed by the existing learned pooling. Global queries access only fully completed earlier blocks with matching document IDs. Local attention remains the already tested 0102B implementation.

The new `triton_dual_resolution_dense.py` contains one tiled, online-softmax attention core for two distinct geometries:

- **Global**: `[B, T, 16, 64]` queries against `[B, M, 4, 64]` summary K/V. Query and block document IDs, validity bits, and summary-end versus query positions are checked in each tile. Availability for the local/global gate comes from the **same exact mask** as attention. An empty global key set gives zero output, zero gradients and a false availability mask.
- **Compressor**: `[N_valid_blocks, 128, 8, 64]` bidirectional self-attention (for the default compression width 512). The existing learned pooling and all linear projections are unchanged; the tiled kernel replaces only the inner full self-attention.

The forward pass retains only output and per-query log-sum-exp. Backward recomputes softmax tiles and produces FP32 `dQ`, `dK`, and `dV`. Shared GQA K/V gradients are accumulated over all query heads in a separate key-tile pass, without atomic writes. Both paths support full Q/K/V FP32 values and BF16 (head dimension 64). Full-sequence attention matrices are not materialized.

## Opt-in switches

CuPy remains the default for global attention and compression:

```bash
MINI_LLM_DUAL_RESOLUTION_GLOBAL_KERNEL=cupy
MINI_LLM_DUAL_RESOLUTION_COMPRESSION_KERNEL=cupy
```

For GPU comparisons, toggle them independently via the benchmark flags:

```bash
MINI_LLM_BACKEND=cupy python benchmarks/benchmark_dual_resolution_0102.py \
    --contexts 4096,16384 --global-query-chunk 512 --compression-block-chunk 64 \
    --local-kernel triton --global-kernel cupy --compression-kernel cupy \
    --skip-inference --warmup 2 --repeats 5

MINI_LLM_BACKEND=cupy python benchmarks/benchmark_dual_resolution_0102.py \
    --contexts 4096,16384 --global-query-chunk 512 --compression-block-chunk 64 \
    --local-kernel triton --global-kernel triton --compression-kernel cupy \
    --skip-inference --warmup 2 --repeats 5

MINI_LLM_BACKEND=cupy python benchmarks/benchmark_dual_resolution_0102.py \
    --contexts 4096,16384 --global-query-chunk 512 --compression-block-chunk 64 \
    --local-kernel triton --global-kernel triton --compression-kernel triton \
    --skip-inference --warmup 2 --repeats 5
```

**Validate before timing:**

```bash
MINI_LLM_BACKEND=cupy python -m pytest -v \
    tests/test_triton_dual_resolution_remaining_0102c.py
```

Do not run production training with either new Triton flag until the RTX 5080 has passed the GPU tests and a short full-model training step. The 0102C Triton implementations were **not GPU-compiled in the patch-authoring environment**. The key-tile backward kernel intentionally uses a deterministic reduction over query tiles; at 65K, assess whether it is fast enough. A slower result is a valid reason to retain CuPy for the affected branch. The 0102B 512-token local Triton backend and the original CuPy branches remain independent fallback paths.

The new compressor kernel currently requires a 64-dimensional compression head (e.g. compression 256/4, 512/8, 1024/16). Widths with other head dimensions should use CuPy. The CPU NumPy backend cannot run Triton.

For a first diagnostic failure, include the complete Triton compile traceback, the exact command, CUDA/Triton/CuPy versions, and which component was enabled. A standalone timing improvement does not validate 20-layer training throughput or 16 GB peak-memory feasibility.
