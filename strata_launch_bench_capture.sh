#!/usr/bin/env bash
# Strata IQ3_S: launch, capture effective runtime, benchmark, and exercise the API.
# Requires: strata, curl, python3. Keeps the Strata server running after completion.
set -Eeuo pipefail

URL="http://127.0.0.1:${PORT:-8095}"
OUT="${OUT:-$HOME/strata_iq3s_$(date +%Y%m%d_%H%M%S)}"
mkdir -p "$OUT"
printf '%s\n' "$OUT" > "$OUT/output_directory.txt"

echo "Output directory: $OUT"
command -v strata >/dev/null || { echo 'ERROR: strata CLI not found' >&2; exit 1; }
command -v curl >/dev/null || { echo 'ERROR: curl not found' >&2; exit 1; }
command -v python3 >/dev/null || { echo 'ERROR: python3 not found' >&2; exit 1; }

# Original launch settings, with --no-tui for scripting and --restart for a fresh server.
launch=(
  strata launch
  --family qwen
  --model iq3_s
  --context 262144
  --parallel 1
  --vision off
  --access local
  --no-api-key
  --port "${PORT:-8095}"
  --yes
  --no-tui
)
# Optional: DRAFTING=off (strict no-drafting), DRAFTING=auto, mtp, dflash2.
if [[ -n "${DRAFTING:-}" ]]; then
  launch+=(--drafting "$DRAFTING")
fi
# Optional: MODEL_PATH=/absolute/path/to/IQ3_S/shard-00001.gguf or directory.
if [[ -n "${MODEL_PATH:-}" ]]; then
  launch+=(--model-path "$MODEL_PATH")
fi

printf '%q ' "${launch[@]}" --restart > "$OUT/launch_command.sh"
printf '\n' >> "$OUT/launch_command.sh"
(strata --version || true) > "$OUT/strata_version.txt" 2>&1
(strata launch --help || true) > "$OUT/launch_help.txt" 2>&1
"${launch[@]}" --plan > "$OUT/launch_plan.log" 2>&1 || echo 'Launch plan unavailable; see launch_plan.log.'

echo 'Launching Strata; this may stop/restart an existing server on the same port.'
"${launch[@]}" --restart 2>&1 | tee "$OUT/launch.log"

echo 'Checking health...'
ready=0
for _ in $(seq 1 90); do
  if curl --noproxy '*' --connect-timeout 2 --max-time 5 -fsS "$URL/health" > "$OUT/health.json" 2>/dev/null; then
    ready=1
    break
  fi
  sleep 2
done
if [[ "$ready" != 1 ]]; then
  echo "ERROR: server did not become healthy; see $OUT/launch.log" >&2
  exit 1
fi

# Capture effective/runtime state before any benchmark request.
(strata status --url "$URL" --json || true) > "$OUT/strata_status_before.json" 2> "$OUT/status_stderr.log"
(strata monitor --url "$URL" --once --json || true) > "$OUT/monitor_once_before.json" 2> "$OUT/monitor_stderr.log"
for endpoint in props metrics status slots v1/models; do
  name="${endpoint//\//_}"
  curl --noproxy '*' -fsS --max-time 15 "$URL/$endpoint" > "$OUT/${name}_before.json" || true
done
(ps -eo pid,ppid,etime,args | grep -Ei '[s]trata|[l]lama-server' || true) > "$OUT/runtime_processes.txt"
(lspci -tv || true) > "$OUT/pci_topology.txt" 2>&1
(lspci -nn | grep -Ei 'VGA|3D controller|Display controller' || true) > "$OUT/gpu_pci_inventory.txt" 2>&1
if command -v nvidia-smi >/dev/null; then
  (nvidia-smi -q || true) > "$OUT/nvidia_smi_q.txt" 2>&1
fi
if command -v amd-smi >/dev/null; then
  (amd-smi list || true) > "$OUT/amd_smi_list.txt" 2>&1
fi
{
  for d in /sys/bus/pci/devices/*; do
    [[ -r "$d/class" ]] || continue
    case "$(< "$d/class")" in
      0x030000|0x030200|0x038000)
        printf '\n%s\n' "${d##*/}"
        for field in current_link_speed current_link_width max_link_speed max_link_width; do
          if [[ -r "$d/$field" ]]; then printf '  %s = %s\n' "$field" "$(< "$d/$field")"; fi
        done
        ;;
    esac
  done
} > "$OUT/pcie_links.txt"

# Official comparable benchmark: standardized prompts and per-request timings.
echo 'Running Strata standard benchmark...'
if strata bench run --url "$URL" --suite standard-v1 --out "$OUT/benchmark" \
    2>&1 | tee "$OUT/benchmark_console.log"; then
  echo 'Benchmark completed.'
else
  echo 'Benchmark failed or marked nonconforming; consult logs.' >&2
fi

# A post-benchmark API exercise, intentionally after the standardized benchmark.
for endpoint in props metrics status; do
  curl --noproxy '*' -fsS --max-time 15 "$URL/$endpoint" > "$OUT/${endpoint}_after.json" || true
done

python3 - "$OUT/interactive_request.json" <<'PY'
import json, sys
with open(sys.argv[1], 'w', encoding='utf-8') as f:
    json.dump({
        'model': 'strata', 'stream': False, 'temperature': 0,
        'max_tokens': 512,
        'messages': [
            {'role': 'system', 'content': 'You are a scientific Python programming assistant.'},
            {'role': 'user', 'content': 'Write a compact Python function implementing conjugate gradient for a sparse symmetric positive definite matrix. Explain the stopping criterion.'},
        ],
    }, f, indent=2)
PY

echo 'Sending separate interactive coding/scientific query...'
curl --noproxy '*' -fsS --max-time 300 "$URL/v1/chat/completions" \
  -H 'Content-Type: application/json' \
  -d @"$OUT/interactive_request.json" > "$OUT/interactive_response.json" || true

python3 - "$OUT/interactive_response.json" <<'PY'
import json, sys
try:
    data = json.load(open(sys.argv[1], encoding='utf-8'))
    print('\n--- Interactive test ---')
    print('Usage:', data.get('usage', {}))
    choices = data.get('choices') or []
    if choices:
        message = choices[0].get('message') or {}
        print('Finish:', choices[0].get('finish_reason'))
        print('Content:', str(message.get('content', ''))[:1800])
        if message.get('reasoning_content'):
            print('Reasoning tokens/text were returned separately.')
except Exception as exc:
    print('Could not parse interactive response:', exc)
PY

echo
echo 'DONE. Strata remains running for your agent / Zed.'
echo "API: $URL/v1"
echo "Artifacts: $OUT"
echo 'View live metrics with: strata monitor --url '"$URL"
echo 'Review benchmark: strata bench report '"$OUT/benchmark"
