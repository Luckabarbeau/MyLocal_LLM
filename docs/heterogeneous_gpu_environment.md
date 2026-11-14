# Native heterogeneous GPU process environments

## Scope

Use separate Python processes for the installed CUDA and HIP CuPy builds. Do not
load both builds into one interpreter or copy extensions between environments.
`scripts/preflight_heterogeneous.py` uses only Python's standard library in its
parent process. Each child imports native CuPy, checks its compiled runtime,
selects the configured device, and checks a 16×16 FP32 ones matmul (expected
entries: exactly 16). Triton's installed provider/version is inspected through
`importlib.metadata`; Triton and PyTorch are not imported. No PyTorch dependency
is required by this preflight.

This is an environment/matmul gate, **not** a gradient, BF16, Triton kernel,
Model B, collective communication, concurrent training, or throughput test.
Workers run sequentially in fresh processes, each with a 120-second subprocess
timeout. A failed worker does not prevent checking the remaining workers.

## Existing launch settings

Repository root on this machine:
`/home/lucka/Documents/my_local_LLM/MyLocal_LLM`.

The existing AMD launcher is `scripts/run_r9700_preflight.sh`. With its default
ROCm location and no inherited library path, its effective settings are:

| Setting | AMD | NVIDIA |
| --- | --- | --- |
| Python | `.venv-rocm/bin/python` | `.venv/bin/python` |
| `MINI_LLM_BACKEND` | `cupy` | `cupy` |
| `ROCM_HOME` | `/opt/rocm-7.2.4` | not required |
| `ROCM_PATH` | `/opt/rocm-7.2.4` | not required |
| `TRITON_DEFAULT_BACKEND` | `amd` | `nvidia` (explicit process-local selection) |
| `LD_LIBRARY_PATH` | see below | no AMD overlay required |

AMD library path, as one literal line:

```text
/home/lucka/Documents/my_local_LLM/MyLocal_LLM/.venv-rocm/rocm-deps/opt/rocm-7.2.4/lib:/opt/rocm-7.2.4/lib
```

The shell launcher appends an inherited `LD_LIBRARY_PATH` after these entries
and honors an inherited `ROCM_HOME`. JSON overrides are literal strings: no
`$PWD`, `$ROCM_HOME`, tilde, or other shell expansion occurs in environment
values. Supply the fully expanded library path. Interpreter paths may be
absolute or repository-relative; `~` is expanded for the interpreter only.

`docs/0107d_rocm_bf16.md` records previous successful native AMD checks with
Python 3.12, patched CuPy 14.2.0, ROCm 7.2.4, Triton 3.8.0, and target
`hip/gfx1201`. That environment includes a local MIOpen extraction and a patched
CuPy BLAS binding. Reinstalling packages can destroy that established state.
`docs/0107f_amd_nvidia_ffn_comparison.md` records successful NVIDIA commands
using `MINI_LLM_BACKEND=cupy .venv/bin/python`; those commands do not explicitly
set `TRITON_DEFAULT_BACKEND`. The explicit `nvidia` setting above was verified by the new native matmul
preflight and isolates the process from an inherited AMD selection.

Missing libraries in an unrelated ROCm PyTorch installation do not diagnose
native CuPy. Do not import PyTorch or install packages to repair this gate.

## Configuration and invocation

The coordinator/config owner should supply a JSON object containing a nonempty
`workers` list. Each worker accepts:

- `name`: unique nonempty string.
- `python`: interpreter path, resolved relative to the repository root.
- `env`: optional dictionary of string environment overrides, default `{}`.
- `device`: optional nonnegative integer, default `0`.
- `expected_runtime`: required `cuda` or `hip`.

Use the AMD/NVIDIA settings above in their respective worker environments.
The parent environment is inherited before overrides. `MINI_LLM_BACKEND` is
forced to `cupy`; a conflicting worker override is rejected. Visibility variables
such as `CUDA_VISIBLE_DEVICES`, `HIP_VISIBLE_DEVICES`, and
`ROCR_VISIBLE_DEVICES` are inherited unless overridden. Device indices refer to
**process-visible** devices after those filters, not necessarily physical GPU
indices. Start from a clean shell or explicitly override unwanted inherited
settings. No global environment or activation scripts are changed.

Run with any available standard-library Python, for example:

```sh
python3 scripts/preflight_heterogeneous.py --config /absolute/path/to/workers.json
```

Children always use the repository root as their working directory, regardless
of the caller's directory. The config path itself is relative to the caller.
Output is a JSON report with per-worker stdout/stderr, exit status, device name,
runtime identity/version, CuPy version, Triton distribution versions, and matmul
error. Exit codes: `0` all probes passed; `1` at least one worker failed;
`2` invalid configuration or command-line usage. Nothing is installed, no model
or checkpoint is loaded, and no training is started.

## Verification status for this change

Syntax compilation using Python's built-in `compile` passed without writing
bytecode. A temporary JSON config was then passed to the actual new script;
both native workers passed with exit code 0. The tested overrides were exactly
the table above, with NVIDIA additionally using `LD_LIBRARY_PATH=""` and both
workers omitting `device` (therefore selecting device 0). `MINI_LLM_BACKEND=cupy`
was injected by the script. No visibility overrides were supplied.

| Observation | NVIDIA | AMD |
| --- | --- | --- |
| Device | NVIDIA GeForce RTX 5080 | AMD Radeon Graphics |
| Architecture | not reported by this gate | `gfx1201` |
| Visible device count | 1 | 1 |
| CuPy | 14.2.0 | 14.2.0 |
| Triton distribution | 3.8.0 | 3.8.0+rocm7.2.4.git669b31ac |
| CuPy runtime | `cuda` | `hip` |
| Raw runtime version | 13020 | 70253211 |
| FP32 matmul maximum absolute error | 0.0 | 0.0 |

Both worker stderr streams were empty. These observations verify the new script
and tiny native matmuls, not Triton kernel execution or full training.

Two earlier terminal commands timed out (10 seconds and 30 seconds) with no
captured output. The successful validation used a longer outer terminal bound
(260 seconds); each worker retained its own 120-second bound. The terminal
reported `pyenv: cannot rehash: /home/lucka/.pyenv/shims/.pyenv-shim exists` before
successful execution. The cause of that shell/startup warning was not investigated
or changed, and it did not prevent either native probe from passing.
