#!/bin/sh
# Run from the repository root: sh scripts/run_r9700_preflight.sh
# Process-local settings only; does not modify the NVIDIA environment.
set -u
ulimit -c 0
export ROCM_HOME="${ROCM_HOME:-/opt/rocm-7.2.4}"
export ROCM_PATH="$ROCM_HOME"
export LD_LIBRARY_PATH="$PWD/.venv-rocm/rocm-deps$ROCM_HOME/lib:$ROCM_HOME/lib${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
export MINI_LLM_BACKEND=cupy
export TRITON_DEFAULT_BACKEND=amd

if [ ! -d "$ROCM_HOME/lib" ]; then
    echo "Missing ROCm library directory: $ROCM_HOME/lib" >&2
    exit 2
fi

# Optionally run another bounded AMD diagnostic under the same environment.
if [ "$#" -gt 0 ]; then
    exec .venv-rocm/bin/python "$@"
fi

status=0
.venv-rocm/bin/python -u scripts/probe_0107a_r9700.py || status=1
.venv-rocm/bin/python -u scripts/probe_0107b_amd_bf16.py || status=1
exit "$status"
