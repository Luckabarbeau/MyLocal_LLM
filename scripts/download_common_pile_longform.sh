#!/usr/bin/env bash
set -euo pipefail

STAGE="${1:-priority1}"
DATA_ROOT="${2:-../pretraining_data}"
CONFIG="${3:-configs/common_pile_longform_v01.json}"

case "$STAGE" in
  priority1) PRIORITIES="1" ;;
  priority2) PRIORITIES="2" ;;
  all) PRIORITIES="1 2" ;;
  *) echo "usage: $0 {priority1|priority2|all} [data-root] [config]" >&2; exit 2 ;;
esac

python - "$STAGE" "$DATA_ROOT" "$CONFIG" <<'PY'
import json, shutil, sys
from pathlib import Path
stage, root, config = sys.argv[1], Path(sys.argv[2]), Path(sys.argv[3])
cfg = json.loads(config.read_text())
priorities = {"priority1": {1}, "priority2": {2}, "all": {1, 2}}[stage]
remaining = 0
for source in cfg["sources"].values():
    if source["priority"] not in priorities:
        continue
    target = root / source["raw_subdir"]
    for name in source["files"]:
        if not (target / name).is_file():
            remaining += source["repository_bytes"] / len(source["files"])
free = shutil.disk_usage(root).free
reserve = 20 * 1024**3
if free < remaining + reserve:
    raise SystemExit(f"insufficient free space: {free/1024**3:.1f} GiB free; "
                     f"need about {remaining/1024**3:.1f} GiB plus 20 GiB reserve")
print(f"download preflight: {free/1024**3:.1f} GiB free; "
      f"approximately {remaining/1024**3:.1f} GiB remains")
PY

for PRIORITY in $PRIORITIES; do
  while IFS=$'\t' read -r DATASET REVISION RAW_SUBDIR FILE; do
    DEST="$DATA_ROOT/$RAW_SUBDIR"
    mkdir -p "$DEST"
    if [ -s "$DEST/$FILE" ]; then
      echo "Already present: $DEST/$FILE"
      continue
    fi
    URL="https://huggingface.co/datasets/$DATASET/resolve/$REVISION/$FILE?download=true"
    echo "Downloading $DATASET/$FILE"
    curl --fail --location --continue-at - --retry 12 --retry-all-errors \
      --connect-timeout 30 --speed-time 120 --speed-limit 1024 \
      --output "$DEST/$FILE.part" "$URL"
    mv "$DEST/$FILE.part" "$DEST/$FILE"
  done < <(python - "$PRIORITY" "$CONFIG" <<'PY'
import json, sys
priority = int(sys.argv[1])
cfg = json.load(open(sys.argv[2], encoding="utf-8"))
for source in cfg["sources"].values():
    if source["priority"] == priority:
        for name in source["files"]:
            print(source["dataset"], source["revision"], source["raw_subdir"], name, sep="\t")
PY
)
done

echo "Download stage '$STAGE' complete. Files remain gzip-compressed; no duplicate HF cache was created."
