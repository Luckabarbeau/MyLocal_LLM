#!/usr/bin/env bash
# Compare ordinary CE and adaptive 30% at 28,700, with 28,500/22,500 references.
# Run from the MyLocal_LLM repository root:
#   bash /path/to/run_0073G_checkpoint_benchmarks.sh
# Optional:
#   MODELS='control_28700 adaptive30_28700' BENCHMARKS='ptb c4 wikitext_plain' bash ...
#   DRY_RUN=1 bash ...              # preview all commands, no paths required
#   FORCE=1 bash ...                # rerun successful evaluations
#   DATA_ROOT=../benchmarks bash ...
set -Eeuo pipefail

PYTHON_BIN="${PYTHON_BIN:-python}"
DATA_ROOT="${DATA_ROOT:-../benchmarks}"
CHECKPOINT_ROOT="${CHECKPOINT_ROOT:-./checkpoints}"
RESULT_ROOT="${RESULT_ROOT:-./benchmark_results/muon_checkpoint_comparison}"
DRY_RUN="${DRY_RUN:-0}"
FORCE="${FORCE:-0}"

# Checkpoint names are exactly those supplied for this experiment.
declare -A CHECKPOINTS=(
  [control_35500]="wide-500m-memory-64k_0060_30b"
  [muon_40500]="wide-500m-memory-64k_0060_30b_muon_run"
  [baseline_28500]="wide-500m-memory-64k_0060_30b_back_up_28500_before_opt_change"
  [baseline_22500]="wide-500m-memory-64k_0060_30b_back_up22500"
)

ALL_MODELS=(control_35500 muon_40500 baseline_28500 baseline_22500)
ALL_BENCHMARKS=(ptb c4 pg19_plain pg19_routed lambada wikitext_plain wikitext_routed)
# Whitespace-delimited filters; leave unset to run all 28 combinations.
if [[ -n "${MODELS:-}" ]]; then read -r -a SELECTED_MODELS <<< "$MODELS"; else SELECTED_MODELS=("${ALL_MODELS[@]}"); fi
if [[ -n "${BENCHMARKS:-}" ]]; then read -r -a SELECTED_BENCHMARKS <<< "$BENCHMARKS"; else SELECTED_BENCHMARKS=("${ALL_BENCHMARKS[@]}"); fi

for tag in "${SELECTED_MODELS[@]}"; do
  if [[ -z "${CHECKPOINTS[$tag]+x}" ]]; then
    echo "Unknown model tag: $tag (valid: ${ALL_MODELS[*]})" >&2; exit 2
  fi
done
for bench in "${SELECTED_BENCHMARKS[@]}"; do
  case " $bench " in
    ' ptb '|' c4 '|' pg19_plain '|' pg19_routed '|' lambada '|' wikitext_plain '|' wikitext_routed ') ;;
    *) echo "Unknown benchmark: $bench (valid: ${ALL_BENCHMARKS[*]})" >&2; exit 2;;
  esac
done

export MINI_LLM_BACKEND=cupy
export MINI_LLM_LM_HEAD_CHUNK_TOKENS=1024

if [[ "$DRY_RUN" != "1" ]]; then
  [[ -f ./train_model.py ]] || { echo "Run from MyLocal_LLM repository root (train_model.py not found)." >&2; exit 2; }
fi

# Auto-detect WikiText-2 in either a shared benchmark directory or local benchmark directory.
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
# In dry-run mode use the usual path if detection is unavailable.
if [[ -z "$WIKITEXT_DATA" && "$DRY_RUN" == "1" ]]; then WIKITEXT_DATA="$DATA_ROOT/wikitext-2-raw/wiki.test.raw"; fi

check_file() {
  if [[ "$DRY_RUN" != "1" && ! -f "$1" ]]; then
    echo "Missing file: $1" >&2
    if [[ "$1" == *wiki.test.raw* ]]; then echo "Set WIKITEXT_DATA to your WikiText-2 test-file path." >&2; fi
    exit 2
  fi
}

run_one() {
  local bench="$1" tag="$2"; shift 2
  local outfile="$RESULT_ROOT/$bench/$tag.log"
  if [[ "$DRY_RUN" == "1" ]]; then
    printf '[DRY RUN] %-17s %-17s ' "$bench" "$tag"
    printf '%q ' "$@"
    printf '\n'
    return
  fi
  mkdir -p "$RESULT_ROOT/$bench"
  if [[ "$FORCE" != "1" && -f "${outfile}.ok" ]]; then
    echo "[SKIP] $bench / $tag: ${outfile}.ok exists (set FORCE=1 to rerun)"
    return
  fi
  rm -f "${outfile}.ok"
  printf '\n===== %s / %s =====\n' "$bench" "$tag"
  if {
    printf 'Model tag:       %s\n' "$tag"
    printf 'Benchmark:       %s\n' "$bench"
    printf 'UTC started:     %s\n' "$(date -u +'%Y-%m-%dT%H:%M:%SZ')"
    printf 'Command:         '; printf '%q ' "$@"; printf '\n'
    "$@"
  } 2>&1 | tee "$outfile"; then
    touch "${outfile}.ok"
    echo "[PASS] $bench / $tag -> $outfile"
  else
    echo "[FAIL] $bench / $tag -> $outfile (no .ok marker; rerunning will retry)" >&2
    exit 1
  fi
}

for bench in "${SELECTED_BENCHMARKS[@]}"; do
  case "$bench" in
    ptb) check_file ./benchmarks/evaluate_corpus.py; check_file "$DATA_ROOT/ptb/ptb.test.txt" ;;
    c4) check_file ./benchmarks/evaluate_corpus.py; check_file "$DATA_ROOT/c4-validation/c4_validation_first_5000.jsonl" ;;
    pg19_plain|pg19_routed) check_file ./benchmarks/evaluate_corpus.py; check_file "$DATA_ROOT/pg19-test/pg19_test.jsonl" ;;
    lambada) check_file ./benchmarks/evaluate_lambada.py; check_file "$DATA_ROOT/lambada-openai/lambada_test_en.jsonl" ;;
    wikitext_plain|wikitext_routed) check_file ./benchmarks/evaluate_wikitext.py; check_file "$WIKITEXT_DATA" ;;
  esac

  for tag in "${SELECTED_MODELS[@]}"; do
    checkpoint="$CHECKPOINT_ROOT/${CHECKPOINTS[$tag]}"
    if [[ "$DRY_RUN" != "1" && ! -e "$checkpoint" ]]; then
      echo "Missing checkpoint: $checkpoint" >&2; exit 2
    fi
    case "$bench" in
      ptb)
        run_one "$bench" "$tag" "$PYTHON_BIN" benchmarks/evaluate_corpus.py \
          --checkpoint "$checkpoint" \
          --data "$DATA_ROOT/ptb/ptb.test.txt" \
          --dataset-name 'Penn Treebank' --format text \
          --memory-mode plain --context-length 4096 --stride 4096
        ;;
      c4)
        run_one "$bench" "$tag" "$PYTHON_BIN" benchmarks/evaluate_corpus.py \
          --checkpoint "$checkpoint" \
          --data "$DATA_ROOT/c4-validation/c4_validation_first_5000.jsonl" \
          --dataset-name 'C4 validation fixed 1M' --format jsonl --text-field text \
          --memory-mode plain --context-length 4096 --stride 4096 --max-tokens 1000000
        ;;
      pg19_plain)
        run_one "$bench" "$tag" "$PYTHON_BIN" benchmarks/evaluate_corpus.py \
          --checkpoint "$checkpoint" \
          --data "$DATA_ROOT/pg19-test/pg19_test.jsonl" \
          --dataset-name 'PG-19 test fixed 1M' --format jsonl --text-field text \
          --memory-mode plain --context-length 4096 --stride 4096 --max-tokens 1000000
        ;;
      pg19_routed)
        run_one "$bench" "$tag" "$PYTHON_BIN" benchmarks/evaluate_corpus.py \
          --checkpoint "$checkpoint" \
          --data "$DATA_ROOT/pg19-test/pg19_test.jsonl" \
          --dataset-name 'PG-19 test fixed 1M' --format jsonl --text-field text \
          --memory-mode routed --context-length 4096 --stride 4096 --max-tokens 1000000
        ;;
      lambada)
        run_one "$bench" "$tag" env \
          MINI_LLM_FUSED_EXPERT_GEMM=0 \
          MINI_LLM_FUSED_SWIGLU=1 \
          MINI_LLM_ROUTED_PREFIX_REFRESH_TOKENS=128 \
          "$PYTHON_BIN" benchmarks/evaluate_lambada.py \
          --checkpoint "$checkpoint" \
          --data "$DATA_ROOT/lambada-openai/lambada_test_en.jsonl" \
          --context-length 4096 --progress-every 500
        ;;
      wikitext_plain)
        run_one "$bench" "$tag" "$PYTHON_BIN" benchmarks/evaluate_wikitext.py \
          --checkpoint "$checkpoint" --data "$WIKITEXT_DATA" \
          --memory-mode plain --context-length 4096 --stride 4096
        ;;
      wikitext_routed)
        run_one "$bench" "$tag" "$PYTHON_BIN" benchmarks/evaluate_wikitext.py \
          --checkpoint "$checkpoint" --data "$WIKITEXT_DATA" \
          --memory-mode routed --context-length 4096 --stride 4096
        ;;
    esac
  done
done

if [[ "$DRY_RUN" != "1" ]]; then
  echo
  echo 'Completed all requested benchmark/model combinations.'
  echo "Results: $RESULT_ROOT/<benchmark>/<model>.log"
  echo "Successful results have matching .log.ok marker files."
fi
