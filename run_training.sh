#!/usr/bin/env bash
# Single configurable launcher for MyLocal_LLM training.
#
# Required for a real run:
#   CHECKPOINT_DIR=./checkpoints/my_run bash ./run_training.sh smoke
#   CHECKPOINT_DIR=./checkpoints/my_run bash ./run_training.sh continue
#
# Override MODEL, OBJECTIVE, MIXED_SHARD_ROOT, LOG_FILE, optimization variables,
# or append train_model.py options after `--`. DRY_RUN=1 prints without running.
set -Eeuo pipefail

usage() {
    cat <<'EOF'
Usage: bash ./run_training.sh MODE [-- EXTRA_TRAIN_MODEL_ARGS...]

MODE:
  smoke     New checkpoint, stop after two optimizer steps.
  fresh     New checkpoint, run through TOTAL_STEPS.
  continue  Resume model and optimizer from CHECKPOINT_DIR.

Required environment:
  CHECKPOINT_DIR       New or existing checkpoint directory, depending on MODE.

Common overrides:
  MODEL                Model preset (default: ensemble-deep-512-dense-memory-64k)
  OBJECTIVE            ar or block-proposal (default: ar)
  MIXED_SHARD_ROOT     Packed mixture root (default: token_shards_ensemble_32k_mixed)
  LOG_FILE             CSV path (default: logs/<checkpoint-name>.csv)
  TRAIN_BATCH_SIZE     Default: 2
  GRAD_ACCUM_STEPS     Default: 32
  TOTAL_STEPS          Default: 100000
  DRY_RUN              Set to 1 to validate and print the command without execution.

Example:
  CHECKPOINT_DIR=./checkpoints/experiment MODEL=wide24-500m-memory-64k \
    OBJECTIVE=block-proposal bash ./run_training.sh smoke -- \
    --block-proposal-k 4 --block-proposal-soft-interface proposal_latents_direct
EOF
}

if [[ "${1:-}" == "--help" || "${1:-}" == "-h" ]]; then
    usage
    exit 0
fi

MODE="${1:-}"
case "$MODE" in
    smoke|fresh|continue) ;;
    *) usage >&2; exit 2 ;;
esac
shift
if [[ "${1:-}" == "--" ]]; then
    shift
fi

CHECKPOINT_DIR="${CHECKPOINT_DIR:-}"
if [[ -z "$CHECKPOINT_DIR" ]]; then
    echo "CHECKPOINT_DIR is required; refusing to select or overwrite a checkpoint implicitly." >&2
    exit 2
fi

MODEL="${MODEL:-ensemble-deep-512-dense-memory-64k}"
OBJECTIVE="${OBJECTIVE:-ar}"
MIXED_SHARD_ROOT="${MIXED_SHARD_ROOT:-./token_shards_ensemble_32k_mixed}"
CHECKPOINT_NAME="$(basename "$CHECKPOINT_DIR")"
LOG_FILE="${LOG_FILE:-./logs/${CHECKPOINT_NAME}.csv}"
PYTHON_BIN="${PYTHON_BIN:-python}"
DRY_RUN="${DRY_RUN:-0}"

case "$OBJECTIVE" in
    ar|block-proposal) ;;
    *) echo "OBJECTIVE must be 'ar' or 'block-proposal', got: $OBJECTIVE" >&2; exit 2 ;;
esac

if [[ ! -f "$MIXED_SHARD_ROOT/tokenizer.json" || ! -f "$MIXED_SHARD_ROOT/mixture_manifest.json" ]]; then
    echo "Missing tokenizer.json or mixture_manifest.json in $MIXED_SHARD_ROOT" >&2
    exit 1
fi

if [[ "$MODE" == "continue" ]]; then
    if [[ ! -f "$CHECKPOINT_DIR/training_state.json" ]]; then
        echo "Cannot continue: $CHECKPOINT_DIR/training_state.json is missing" >&2
        exit 1
    fi
elif [[ -d "$CHECKPOINT_DIR" && -n "$(ls -A "$CHECKPOINT_DIR")" ]]; then
    echo "$MODE requires a new or empty CHECKPOINT_DIR: $CHECKPOINT_DIR" >&2
    exit 1
fi

export MINI_LLM_BACKEND="${MINI_LLM_BACKEND:-cupy}"

ARGS=(
    --model "$MODEL"
    --objective "$OBJECTIVE"
    --precision "${PRECISION:-bf16-mixed}"
    --runtime-profile "${RUNTIME_PROFILE:-gpu-fast}"
    --mixed-shard-root "$MIXED_SHARD_ROOT"
    --batch-size "${TRAIN_BATCH_SIZE:-2}"
    --grad-accum-steps "${GRAD_ACCUM_STEPS:-32}"
    --gradient-accumulation-weighting supervised_tokens
    --optimizer "${OPTIMIZER:-hybrid-muon}"
    --muon-momentum "${MUON_MOMENTUM:-0.95}"
    --muon-ns-steps "${MUON_NS_STEPS:-5}"
    --muon-matmul-precision "${MUON_MATMUL_PRECISION:-bf16}"
    --total-steps "${TOTAL_STEPS:-100000}"
    --warmup-steps "${WARMUP_STEPS:-500}"
    --peak-lr "${PEAK_LR:-3e-4}"
    --grad-clip "${GRAD_CLIP:-1.0}"
    --weight-decay "${WEIGHT_DECAY:-0.1}"
    --val-steps "${VAL_STEPS:-4}"
    --val-per-source-steps "${VAL_PER_SOURCE_STEPS:-1}"
    --log-interval "${LOG_INTERVAL:-1}"
    --seed "${SEED:-42}"
    --checkpoint-dir "$CHECKPOINT_DIR"
    --log-file "$LOG_FILE"
)

if [[ "$MODE" == "smoke" ]]; then
    ARGS+=(--max-run-steps 2 --profile-steps 1 --val-interval 2 --save-interval 1)
elif [[ "$MODE" == "continue" ]]; then
    ARGS+=(--resume-from "$CHECKPOINT_DIR" --profile-steps 0
           --val-interval "${VAL_INTERVAL:-100}" --save-interval "${SAVE_INTERVAL:-100}")
else
    ARGS+=(--profile-steps "${PROFILE_STEPS:-0}"
           --val-interval "${VAL_INTERVAL:-100}" --save-interval "${SAVE_INTERVAL:-100}")
fi

ARGS+=("$@")

printf 'Training command:'
printf ' %q' "$PYTHON_BIN" train_model.py "${ARGS[@]}"
printf '\n'
printf 'Mode=%s model=%s objective=%s checkpoint=%s log=%s\n' \
    "$MODE" "$MODEL" "$OBJECTIVE" "$CHECKPOINT_DIR" "$LOG_FILE"

if [[ "$DRY_RUN" == "1" ]]; then
    exit 0
fi

mkdir -p "$CHECKPOINT_DIR" "$(dirname "$LOG_FILE")"
exec "$PYTHON_BIN" train_model.py "${ARGS[@]}"
