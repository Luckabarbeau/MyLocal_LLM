#!/bin/sh
# Deliberate Model B parallel-attention architectural fork.
# Source checkpoint (NEVER mutated): checkpoints/model_b_batch1_8k_norecompute_long
#   step 18150, ~31.99B measured supervised tokens, hybrid-Muon, single worker
#   (no heterogeneous_state in its training_state.json).
# Every scientific/runtime value below is recovered verbatim from that
# checkpoint's training_resume_contract: batch 1 x grad-accum 1024 x seq 8192,
# supervised_tokens weighting, peak-lr 7.5e-5 (the intentional 16200 down-salt
# from 1.5e-4), min_lr = peak/10 = 7.5e-6, warmup 750, total 100000,
# Muon momentum 0.95 / ns 5 / bf16 matmul, Adam beta 0.9/0.95 eps 1e-8,
# weight decay 0.1, grad clip 1.0, unembedding-lr-ratio 1, offload full.
# The 76 MINI_LLM_* launch variables are exported verbatim from the recorded
# runtime_environment (logs/model_b_parallel_fork_migration_long.runtime.env).
# The ONLY intentional changes: MINI_LLM_MATRIX_POOL_PARALLEL_BRANCHES=1
# (the architectural fork), a NEW output directory, and the two-GPU
# heterogeneous peer (5070 Ti). The source checkpoint records no
# heterogeneous_state, so adding the peer forks the effective batch
# (~2x supervised tokens per update); that is acknowledged explicitly.
set -eu
ulimit -c 0
# Pin the owner to the RTX 5080 by UUID (index 0 today; UUID survives re-enumeration).
export CUDA_VISIBLE_DEVICES=GPU-bea264b4-9e76-3be5-5883-7bde3fe9f82d
export TRITON_DEFAULT_BACKEND=nvidia
export LD_LIBRARY_PATH=
export OPENBLAS_NUM_THREADS=1
export OMP_NUM_THREADS=1
. logs/model_b_parallel_fork_migration_long.runtime.env
export MINI_LLM_MATRIX_POOL_PARALLEL_BRANCHES=1
if [ "${MINI_LLM_MATRIX_POOL_PARALLEL_BRANCHES:-}" != "1" ]; then
    echo 'FATAL: MINI_LLM_MATRIX_POOL_PARALLEL_BRANCHES != 1; refusing to fork in legacy mode.' >&2
    exit 2
fi
exec .venv/bin/python -u train_model.py \
    --backend cupy \
    --model dual-resolution-matrix-pool-sparse-1024-20l \
    --objective ar \
    --precision bf16-mixed \
    --context-length 8192 \
    --mixed-shard-root ./token_shards_ensemble_32k_mixed \
    --batch-size 1 \
    --grad-accum-steps 1024 \
    --gradient-accumulation-weighting supervised_tokens \
    --optimizer hybrid-muon \
    --muon-momentum 0.95 \
    --muon-ns-steps 5 \
    --muon-matmul-precision bf16 \
    --adam-beta1 0.9 \
    --adam-beta2 0.95 \
    --adam-eps 1e-8 \
    --weight-decay 0.1 \
    --grad-clip 1.0 \
    --unembedding-lr-ratio 1 \
    --total-steps 100000 \
    --warmup-steps 750 \
    --peak-lr 7.5e-5 \
    --optimizer-offload full \
    --seed 42 \
    --max-run-steps 2 \
    --val-interval 100 \
    --save-interval 2 \
    --log-interval 1 \
    --checkpoint-dir ./checkpoints/model_b_parallel_fork_migration_long \
    --log-file ./logs/model_b_parallel_fork_migration_long.csv \
    --resume-from ./checkpoints/model_b_batch1_8k_norecompute_long \
    --heterogeneous-peer-config ./configs/model_b_5070ti_peer.json \
    --acknowledge-heterogeneous-fork \
    "$@"
