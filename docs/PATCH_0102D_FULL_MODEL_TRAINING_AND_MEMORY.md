# Patch 0102D — Dual-resolution full-model training and memory validation

This patch adds an **end-to-end decoder training smoke benchmark**; it does not change
attention mathematics, the 20-layer architecture, model/checkpoint schemas, or the
existing CuPy/Triton component dispatch.

## What is timed

`benchmarks/benchmark_dual_resolution_full_model_0102d.py` instantiates the
actual `DecoderLanguageModel`, executes every Transformer layer, final RMSNorm,
chunked vocabulary projection and next-token cross entropy, then runs
`chunked_lm_head_backward` and `backward_body`. Optional `--optimizer-step`
constructs the repository's real AdamW and updates all trainable parameters.

The default target is the **final 128 next-token predictions** rather than every
position. **All context tokens still traverse all Transformer layers**, including
the full block compressor, and gradients can flow back through completed block
summaries. This is a full-model *training-path* and long-context memory smoke,
**not** equivalent to supervising all 65K positions or timing a complete normal
pretraining optimizer step. Report `scored_token_rows` with any measurements.

The benchmark checks actual registered versus estimated parameter counts, finite
loss, finite gradients, and nonzero gradients in the compressor and global
attention pathways. The selected synthetic input is one document with no EOS.
Packed-document masking is covered by the previous 0102/0102B/0102C tests.

## RTX 5080: validation sequence

Apply patches 0102A, B, C, then D. Run parity first:

```bash
MINI_LLM_BACKEND=cupy python -m pytest -q \
  tests/test_triton_dual_resolution_local_0102b.py \
  tests/test_triton_dual_resolution_remaining_0102c.py

MINI_LLM_BACKEND=numpy python -m pytest -q \
  tests/test_dual_resolution_full_model_0102d.py
```

Run a **real 20-layer** trial at 1K with checkpointing and compact head:

```bash
MINI_LLM_BACKEND=cupy MINI_LLM_OPTIMIZER_OFFLOAD=full \
python benchmarks/benchmark_dual_resolution_full_model_0102d.py \
  --context 1024 --layers 20 --batch-size 1 \
  --compression-dim 512 --score-last-k 128 --head-chunk-tokens 64 \
  --activation-checkpoint block --local-kernel triton \
  --global-kernel triton --compression-kernel triton \
  --query-chunk 512 --optimizer-step \
  --output dual_fullmodel_1k_0102d.json
```

If it succeeds, test 4096, 8192, 16384 and then larger contexts **one process
at a time**, adjusting `--context` and the JSON filename. Initially omit
`--optimizer-step` at 32K–65K to isolate forward/backward memory from optimizer
state and offload traffic. Do not run a multi-context loop until the previous
configuration has been inspected for OOM and time. To test the reduced-size
version substitute `--compression-dim 256` (368.17M vs 406.20M parameters).

For a short CPU test without CUDA:

```bash
MINI_LLM_BACKEND=numpy python benchmarks/benchmark_dual_resolution_full_model_0102d.py \
  --tiny --context 24 --layers 20 --score-last-k 4 --optimizer-step
```

Optional checkpoint-ablation at short contexts: change
`--activation-checkpoint block` to `none`, keeping all other arguments and
seeds the same. The CPU test suite explicitly compares full-model outputs and
*all* parameter gradients in both modes.

## Memory reporting and caveats

Each JSON output reports **sampled** CuPy pool used/reserved bytes and device
used memory at synchronization barriers (model initialized, forward, loss,
backward, optimizer). These are **not exact transient GPU peaks**: temporary
allocations inside the operations can rise and fall between samples. CuPy pool
reserved memory also includes cached reusable allocations and differs from
live memory. For a rigorous peak VRAM measurement, independently monitor the
process with NVIDIA tools or a CUDA allocation profiler. A successful run with
sampled memory below 16GB is not proof that its instantaneous peak was below
that number; successful allocation is evidence the tested step fit.

Using the full 65K sequence still retains full-resolution block inputs for
replay. `--activation-checkpoint block` is strongly recommended. The chunked
head is critical: without it, a [B,T,V] projection can dominate memory.
All three Triton kernels remain **opt-in**.

## Actual shard-backed training (separate from synthetic smoke)

After the full-model smoke fits, run the repository's existing training CLI
with your real pretokenized dataset. Example **one-update** configuration:

```bash
MINI_LLM_BACKEND=cupy \
MINI_LLM_DUAL_RESOLUTION_LOCAL_KERNEL=triton \
MINI_LLM_DUAL_RESOLUTION_GLOBAL_KERNEL=triton \
MINI_LLM_DUAL_RESOLUTION_COMPRESSION_KERNEL=triton \
MINI_LLM_LM_HEAD_CHUNK_TOKENS=64 \
MINI_LLM_ACTIVATION_CHECKPOINT=block \
MINI_LLM_OPTIMIZER_OFFLOAD=full \
python train_model.py \
  --model dual-resolution-1024-20l --precision bf16-mixed \
  --runtime-profile reference --context-length 1024 \
  --mixed-shard-root /PATH/TO/EXISTING/MIXED_SHARDS \
  --batch-size 1 --grad-accum-steps 1 --total-steps 1000 \
  --max-run-steps 1 --warmup-steps 100 --peak-lr 1e-4 \
  --val-interval 0 --save-interval 1000 --log-interval 1 \
  --checkpoint-dir ./checkpoints/dual_resolution_0102d_smoke
```

The `reference` runtime profile avoids silently enabling unrelated experimental
fast paths, while the explicit three Triton switches select the tested attention
path. Substitute an actual mixed shard root; do not run production training
against random synthetic tokens. The real trainer may have additional
configuration/data requirements and has **not** been GPU-tested by this patch.

## Acceptance and research interpretation

1. GPU parity tests pass, including document and gradient masks.
2. The 20-layer 1K smoke completes with finite loss and compressor/global gradients.
3. One actual optimizer update completes when requested.
4. Record both total token positions processed and **scored** target rows.
5. Increase context cautiously, measuring sampled VRAM and externally monitored
   peak VRAM; stop and report the first failing size.
6. Only then compare against the original trained 338M: report training data,
   supervised-token budget, optimizer, train FLOPs/wall-clock, model sizes
   (new default 406.20M), and held-out evaluation, not just early loss.

No result from these smoke benchmarks establishes model quality or the ability
of one 128-token summary to retain arbitrary detailed facts. That requires
controlled long-range retrieval and language-modeling evaluation after training.
