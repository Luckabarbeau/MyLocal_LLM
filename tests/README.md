# Test suite

Run commands from the repository root after activating the project environment:

```bash
source .venv/bin/activate
export PYTHONPATH=.
```

The suite stays under one collection root so historical regressions can import
shared helpers by module name. Pytest assigns markers automatically from stable
filenames; individual files do not need marker boilerplate.

> **Current GPU restriction:** the protected external training run is user-reported
> at step 165; its checkpoint and host are not available in this workspace. Do not
> run GPU tests or the GPU Model B regression gate on any device owned by training,
> inference, or serving. Use the NumPy commands below when ownership is uncertain,
> and run `nvidia-smi` before any CuPy test run.

## Markers

| Marker | Purpose |
|---|---|
| `unit` | Focused numerical, API, and component tests |
| `integration` | Training, checkpoint, data, streaming, generation, and benchmark workflows |
| `gpu` | Files whose names explicitly target CuPy, CUDA, or GPU behavior |
| `slow` | Audits, benchmarks, production checks, and extended workflows |

Markers overlap. For example, a GPU benchmark is both `gpu` and `slow`. The
collection policy lives in `tests/conftest.py` and infers these categories from
filename hints; marker selection does not itself change the backend.

## Safe CPU workflow

Always force the NumPy backend when another process owns the GPU:

```bash
MINI_LLM_BACKEND=numpy python -m pytest -q -m "unit and not gpu and not slow"
```

Useful narrower commands:

```bash
# Fast numerical smoke tests
MINI_LLM_BACKEND=numpy python -m pytest -q \
  tests/test_embedding.py tests/test_rmsnorm.py tests/test_loss.py

# CPU integration tests, excluding long-running audits and benchmarks
MINI_LLM_BACKEND=numpy python -m pytest -q \
  -m "integration and not gpu and not slow"

# Verify collection and inferred marker assignments without executing tests
MINI_LLM_BACKEND=numpy python -m pytest --collect-only -q
```

Generated test data must use pytest's `tmp_path` fixture. Tests must not write to
repository-level `checkpoints/`, `logs/`, shard directories, benchmark results, or
feature caches.

## GPU tests

Only on a device not owned by training, inference, or serving, inspect active
compute processes and free VRAM with `nvidia-smi`, then run:

```bash
MINI_LLM_BACKEND=cupy python -m pytest -q -m gpu
```

Do not run this concurrently with training, inference, or serving. Test allocations
can reduce available VRAM, trigger out-of-memory failures in the owner process, or
disturb throughput. A small amount of reported free memory is not permission to
share a protected device.

## Model B regression gate

The bounded regression runner launches every suite in a fresh interpreter because
backend selection and several runtime toggles are process-global. It pins NumPy or
CuPy explicitly, removes unrelated inherited `MINI_LLM_*` overrides, stops at the
first failed suite, and can write a machine-readable JSON report.

The CPU gate is safe while another process owns the GPU:

```bash
python scripts/run_0103a_regressions.py \
  --mode cpu --report /tmp/model_b_cpu_regressions.json
```

Its exact, versioned selectors cover:

- public model APIs, canonical model loading, automatic inference policy, the
  production architecture contract, and the regression runner itself;
- generation sessions, remote/OpenAI-compatible serving and sampling, fixed
  generation-benchmark behavior, and the unified benchmark runner;
- matrix-pool dual-resolution numerics and metadata;
- stride-16 sparse historical attention and incremental decode;
- checkpoint compatibility and isolated runtime defaults;
- model-centered checkpoint transactions, Model B configuration round trips,
  adaptive fine-shape buckets, training-run behavior, residual specialists, and
  specialist composition/checkpoint behavior;
- production-profile, training-path parity, specialist sampling, specialist
  training-run, and specialist CLI contracts.

The architecture contract protects the current 348,261,376-parameter Model B:
20 layers, hidden size 1024, FFN size 4096, vocabulary size 32768, architectural
context 65536, matrix-pool blocks of 128 tokens with four summaries, 512/4096
local windows, and one sparse historical head at stride 16. CPU tests validate
contracts and bounded numerical fixtures; they deliberately do not allocate the
full production model or prove GPU speed.

List the GPU subprocesses without executing them:

```bash
python scripts/run_0103a_regressions.py --mode gpu --list
```

After the device is genuinely free, run the GPU gate:

```bash
python scripts/run_0103a_regressions.py \
  --mode gpu --report /tmp/model_b_gpu_regressions.json
```

GPU coverage includes sparse stride-history Triton parity and gradients,
production-width matrix-pool Triton parity, and native Model B CUDA Graph
activation/fallback parity. The runner requires working CuPy, Triton, and a visible
CUDA device. It checks free memory before each subprocess and blocks below 2 GiB,
but that threshold is only an out-of-memory guard—not authorization to share a
GPU.

The production-profile contract is opt-in because it is meaningful only under the
exact profile environment:

```bash
python scripts/run_0103a_regressions.py \
  --mode gpu --model-b-profile \
  --report /tmp/model_b_gpu_profile_regressions.json
```

`--mode all` runs CPU and GPU tiers sequentially and therefore must also wait for a
free GPU. A passing gate establishes the listed correctness contracts, not
end-to-end throughput. In particular, the CUDA Graph test asserts capture and
parity; it is not a performance benchmark.

## Naming and optional historical regressions

- General tests: `test_<component>.py`
- Historical regressions: `test_<experiment-id>_<topic>.py`
- GPU-specific files include `cupy`, `cuda`, or `gpu` in the filename.
- Reusable fixtures and collection policy belong in `tests/conftest.py`.
- Measurement-only workloads belong in `benchmarks/`; tests should assert a stable
  correctness or API contract.

Some historical regressions target research utilities that are not part of the
current checkout. Load those modules with `pytest.importorskip` so an absent
optional artifact produces a documented skip rather than breaking collection.
`tests/test_source_audit_0078d.py` currently follows this policy for the absent
`benchmarks.audit_source_sampling_0078d` utility.

For historical context and superseded validation decisions, see
[`docs/DEVELOPMENT_LOG.md`](../docs/DEVELOPMENT_LOG.md).
