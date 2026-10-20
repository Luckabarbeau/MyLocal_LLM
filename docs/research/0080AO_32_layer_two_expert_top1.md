# 0080AO — deep-500m-memory-64k, 32 layers, 2 experts, top-1

## Goal and parameter audit

This is an **alternative new-scratch main Transformer** behind the existing
0080AN **v4** direct 768-dimensional probability-embedding decoder. It does NOT
modify `wide-500m-memory-64k` or existing checkpoint weights. “32” refers to
32 Transformer blocks, not a 32K context limit. Retain the exact 64K historical
store, 4K dense targets, 2K routed prefix, 6,144 active token slots, and all
12 attention heads per layer (6 local/2 dilated/2 global/2 learned retrieval).

| Property | Old main trunk | New 0080AO trunk |
| --- | ---: | ---: |
| Layers | 12 | 32 |
| Hidden width | 768 | 768 |
| MoE experts/layer | 6 | 2 |
| MoE selected/token | 2 | 1 |
| Expert SwiGLU internal width | 2304 | 2368 |
| Main model stored parameters | 501,237,576 | 499,213,120 |
| Stored MoE expert parameters | 382,205,952 | 349,175,808 |
| Active MoE expert parameters/token | 127,401,984 | 174,587,904 |
| Attention projection parameters | 17,694,720 | 47,185,920 |
| Approx. active params/token (incl. embedding and vocab heads) | 245M | 322M |

The new **main** trunk is 499.2M; the additional learned probability embedding,
four-horizon proposal head and 2-layer 768-wide decoder add approximately 63M,
so the complete v4 system is ~562M parameters. “Same total parameter count”
does not imply same FLOPs, memory or throughput. Attention and normalization
now run 32 rather than 12 times, even though only one expert executes per layer.

## Essential router gradient fix

The existing MoE `selected_topk_softmax` is **mathematically degenerate for
k=1**, as softmax of one selected value equals 1 with zero Jacobian. Training
a fresh 2-expert top-1 MoE with the old rule would produce zero LM gradients
for its router (without an auxiliary loss).

All MoE configurations use the same mathematical rule after 0080AP:

    p = softmax([z0, ..., zE-1])  # over EVERY expert
    i = argsort(-z)[:top_k]       # sparse execution
    y = sum(p[i] * E_i(x) for i in selected)

The selected weights are **not** renormalized after selection. This ensures
that top-1 is differentiable and that all unselected router logits receive a
signal from the full-softmax denominator; the unselected experts themselves
do not run. The model has no special top-1 flag or branch.

This changes the previous selected-only-softmax *top-2* weighting in old
models too: sums of selected probabilities are generally < 1. Checkpoints
trained under old routing semantics should not be silently resumed as an
identical objective; run a new scratch experiment or explicitly migrate and
validate them. Existing 0080AO checkpoint JSON may contain the obsolete flag
and needs metadata migration before resuming under the updated dataclass.

## Memory and training considerations

Initial launcher uses a new checkpoint/log name, group size 8, pinned G/U
activation offloading in the first 24 MoE layers, per-layer router metadata
cache limit 4, in-place FP32 final CE, and periodic CuPy pool trimming. The
previous 16 GB RTX 5080 experienced OOM at group size 128 with 12 layers;
the 32-layer run is still **unmeasured on GPU** and could exceed capacity.
Never assume the increased depth is free or that the previous 12-layer cache
limit of 64 is safe here. If the smoke OOMs, reduce `BLOCK_GROUP_SIZE` further
or adjust offload/checkpoint policy after profiling. Pinned RAM offload can
increase wall time, so measure both supervised tokens/s and peak live VRAM.

The 2-expert top-1 router can experience *expert collapse* (most tokens routed
to one expert) without a balancing objective. The patch preserves the existing
learning loss `final CE + 0.5 proposal CE`, adding no balancing auxiliary loss.
Measure per-layer expert selection fractions early; consider a controlled
load-balance regularizer **only if diagnostics justify it**.

## Application & validation

Apply 0080AO after 0080AN, using `git apply --check` then `git apply`.

    MINI_LLM_BACKEND=numpy OPENBLAS_NUM_THREADS=1 python -m pytest -q \
      tests/test_0080ao_deep_top1_moe.py tests/test_moe_router.py
    MINI_LLM_BACKEND=cupy python -m pytest -q \
      tests/test_0080ao_top1_cupy.py tests/test_block_proposal_0080an_cupy.py -x -vv

The available reference regression subset (block-proposal 0080*, MoE,
config/checkpoint, and 0080AO) passed 419 tests with 58 skips. Five separate
historical audit tests fail at **collection time** because their imported audit
scripts are not present in the reconstructed repository; they are not 0080AO
execution failures. The RTX BF16 checks and integrated 32-layer smoke have
not been executed locally.

From repository root, with compiled CuPy runtime installed and mixed shards
present at the default location:

```bash
CHECKPOINT_DIR=./checkpoints/deep32_top1 MODEL=deep-500m-memory-64k \
OBJECTIVE=block-proposal bash ./run_training.sh smoke
CHECKPOINT_DIR=./checkpoints/deep32_top1 MODEL=deep-500m-memory-64k \
OBJECTIVE=block-proposal bash ./run_training.sh continue
```

Smoke starts from random initialization. Continue loads the same deep32
checkpoint. Additional block-proposal controls can be passed after `--`; check
VRAM before increasing group or batch sizes.
