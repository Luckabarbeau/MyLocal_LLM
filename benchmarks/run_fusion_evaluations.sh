#!/usr/bin/env bash
# Benchmark trained Deep+Wide fusion on the standard checkpoint datasets.
# Run directly from the MyLocal_LLM repository root.
#
# BENCHMARKS='wikitext_plain ptb' MAX_TOKENS=8192 bash benchmarks/run_fusion_evaluations.sh
# BENCHMARKS='wikitext_plain ptb c4 pg19_plain lambada' bash benchmarks/run_fusion_evaluations.sh
# DRY_RUN=1 bash benchmarks/run_fusion_evaluations.sh
set -Eeuo pipefail

PYTHON_BIN="${PYTHON_BIN:-python}"
DATA_ROOT="${DATA_ROOT:-../benchmarks}"
RESULT_ROOT="${RESULT_ROOT:-./benchmarks/results/fusion_0082s}"
FUSION_CKPT="${FUSION_CKPT:-./checkpoints/fusion_0082f_deepwide_streaming_weighted_context_accum3.npz}"
DEEP_CKPT="${DEEP_CKPT:-./checkpoints/0080BX_deep512_fullmix_100k}"
WIDE_CKPT="${WIDE_CKPT:-./checkpoints/0080BX_wide768_fullmix_100k}"
CONTEXT_LENGTH="${CONTEXT_LENGTH:-512}"
STRIDE="${STRIDE:-512}"
SEGMENT_WINDOWS="${SEGMENT_WINDOWS:-128}"
LM_HEAD_ROWS="${LM_HEAD_ROWS:-64}"
BATCH_WINDOWS="${BATCH_WINDOWS:-1}"
BATCH_ROWS="${BATCH_ROWS:-8}"
MAX_TOKENS="${MAX_TOKENS:-}"
MAX_EXAMPLES="${MAX_EXAMPLES:-}"
FORCE="${FORCE:-0}"
DRY_RUN="${DRY_RUN:-0}"

if [[ -n "${BENCHMARKS:-}" ]]; then
  read -r -a SELECTED_BENCHMARKS <<< "$BENCHMARKS"
else
  SELECTED_BENCHMARKS=(wikitext_plain ptb c4 pg19_plain lambada)
fi

export MINI_LLM_BACKEND="${MINI_LLM_BACKEND:-cupy}"
export MINI_LLM_LM_HEAD_CHUNK_TOKENS="${MINI_LLM_LM_HEAD_CHUNK_TOKENS:-1024}"

WIKITEXT_DATA="${WIKITEXT_DATA:-}"
if [[ -z "$WIKITEXT_DATA" ]]; then
  for path in \
    "$DATA_ROOT/wikitext-2-raw/wiki.test.raw" \
    "./benchmarks/wikitext-2-raw/wiki.test.raw" \
    "$DATA_ROOT/wiki_test_2/wiki.test.raw" \
    "$DATA_ROOT/wikitext-2/wiki.test.raw"; do
    if [[ -f "$path" ]]; then WIKITEXT_DATA="$path"; break; fi
  done
fi
if [[ -z "$WIKITEXT_DATA" && "$DRY_RUN" == 1 ]]; then
  WIKITEXT_DATA="$DATA_ROOT/wikitext-2-raw/wiki.test.raw"
fi

[[ "$DRY_RUN" == 1 || -f ./benchmarks/evaluate_probability_fusion_checkpoint_0082s.py ]] || {
  echo 'Run from MyLocal_LLM repository root.' >&2; exit 2;
}
if [[ "$DRY_RUN" != 1 ]]; then
  [[ -f "$FUSION_CKPT" ]] || { echo "Missing fusion checkpoint: $FUSION_CKPT" >&2; exit 2; }
  [[ -f "$DEEP_CKPT/config.json" && -f "$WIDE_CKPT/config.json" ]] || {
    echo 'Missing frozen member checkpoint config.json.' >&2; exit 2;
  }
fi

for benchmark in "${SELECTED_BENCHMARKS[@]}"; do
  case "$benchmark" in
    wikitext_plain) input="$WIKITEXT_DATA"; format=text; dataset='WikiText-2 test (continuous)' ;;
    ptb) input="$DATA_ROOT/ptb/ptb.test.txt"; format=text; dataset='Penn Treebank test' ;;
    c4) input="$DATA_ROOT/c4-validation/c4_validation_first_5000.jsonl"; format=jsonl; dataset='C4 validation fixed' ;;
    pg19_plain) input="$DATA_ROOT/pg19-test/pg19_test.jsonl"; format=jsonl; dataset='PG-19 test' ;;
    lambada) input="$DATA_ROOT/lambada-openai/lambada_test_en.jsonl"; format=lambada; dataset='LAMBADA English test' ;;
    *) echo "Unknown benchmark: $benchmark (supported: wikitext_plain ptb c4 pg19_plain lambada)" >&2; exit 2 ;;
  esac
  if [[ "$DRY_RUN" != 1 && ! -f "$input" ]]; then
    echo "Missing benchmark input: $input" >&2; exit 2
  fi
  output="$RESULT_ROOT/$benchmark"
  command=("$PYTHON_BIN" benchmarks/evaluate_probability_fusion_checkpoint_0082s.py
    --fusion-checkpoint "$FUSION_CKPT"
    --member "0:$DEEP_CKPT" --member "1:$WIDE_CKPT"
    --data "$input" --format "$format" --dataset-name "$dataset"
    --context-length "$CONTEXT_LENGTH" --stride "$STRIDE"
    --segment-windows "$SEGMENT_WINDOWS" --lm-head-rows "$LM_HEAD_ROWS"
    --batch-windows "$BATCH_WINDOWS" --batch-rows "$BATCH_ROWS"
    --json-output "$output/results.json")
  # Match the parent checkpoint harness's fixed 1M-token C4/PG-19 budget.
  budget="$MAX_TOKENS"
  if [[ -z "$budget" && ( "$benchmark" == c4 || "$benchmark" == pg19_plain ) ]]; then
    budget=1000000
  fi
  if [[ -n "$budget" && "$format" != lambada ]]; then
    command+=(--max-tokens "$budget")
  fi
  if [[ -n "$MAX_EXAMPLES" && "$format" == lambada ]]; then
    command+=(--max-examples "$MAX_EXAMPLES")
  fi
  if [[ "$DRY_RUN" == 1 ]]; then
    printf '[DRY RUN] %s: ' "$benchmark"
    printf '%q ' "${command[@]}"
    printf '\n'
    continue
  fi
  if [[ "$FORCE" != 1 && -f "$output/evaluation.log.ok" ]]; then
    echo "[SKIP] $benchmark; set FORCE=1 to re-evaluate"
    continue
  fi
  mkdir -p "$output"
  rm -f "$output/evaluation.log.ok"
  printf '\n===== %s: collective fusion =====\n' "$benchmark"
  if "${command[@]}" 2>&1 | tee "$output/evaluation.log"; then
    touch "$output/evaluation.log.ok"
    echo "[PASS] $benchmark -> $output/results.json"
  else
    echo "[FAIL] $benchmark -> $output/evaluation.log" >&2
    exit 1
  fi
done
