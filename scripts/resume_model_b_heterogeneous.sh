#!/bin/sh
# Run from repository root. Continue the verified fork, not the original parent.
# Default: 10 further updates. Appended CLI options can override the step bound.
set -eu
ulimit -c 0
if [ ! -f checkpoints/model_b_8k_heterogeneous_trial_15700/training_state.json ]; then
    echo 'Verified heterogeneous checkpoint is missing; run the trial first.' >&2
    exit 2
fi
export TRITON_DEFAULT_BACKEND=nvidia
export LD_LIBRARY_PATH=
export OPENBLAS_NUM_THREADS=1
export OMP_NUM_THREADS=1
exec .venv/bin/python -u train_model.py \
    --backend cupy \
    --model dual-resolution-matrix-pool-sparse-1024-20l \
    --objective ar \
    --precision bf16-mixed \
    --runtime-profile consumer-gpu \
    --mixed-shard-root ./token_shards_ensemble_32k_mixed \
    --context-length 8192 \
    --batch-size 1 \
    --grad-accum-steps 1024 \
    --gradient-accumulation-weighting supervised_tokens \
    --optimizer hybrid-muon \
    --muon-momentum 0.95 \
    --muon-ns-steps 5 \
    --muon-matmul-precision bf16 \
    --total-steps 100000 \
    --warmup-steps 750 \
    --peak-lr 1.5e-4 \
    --grad-clip 1.0 \
    --weight-decay 0.1 \
    --val-interval 100 \
    --val-steps 4 \
    --val-per-source-steps 1 \
    --val-fixed-mixed-steps 4 \
    --val-fixed-mixed-seed 78078 \
    --save-interval 50 \
    --log-interval 1 \
    --seed 42 \
    --checkpoint-dir ./checkpoints/model_b_8k_heterogeneous_trial_15700 \
    --log-file ./logs/model_b_8k_heterogeneous_trial.csv \
    --benchmark-suite ./benchmarks/configs/foundation_capabilities_0097_progressive.json \
    --resume-from ./checkpoints/model_b_8k_heterogeneous_trial_15700 \
    --unembedding-lr-ratio 1 \
    --component-update-monitor \
    --component-update-interval 10 \
    --component-update-sample-elements 16384 \
    --component-update-output ./logs/model_b_8k_heterogeneous_trial.components.csv \
    --heterogeneous-peer-config ./configs/model_b_heterogeneous_production_peer.json \
    --max-run-steps 10 \
    "$@"
