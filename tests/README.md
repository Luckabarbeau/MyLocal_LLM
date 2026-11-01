# Test suite

The test suite is intentionally kept under one collection root so historical
regression tests can continue importing shared helpers by module name. Pytest
classifies tests automatically from their stable filenames; individual files do
not need marker boilerplate.

## Categories

| Marker | Purpose |
|---|---|
| `unit` | Focused numerical, API, and component tests |
| `integration` | Training, checkpoint, data, streaming, generation, and benchmark workflows |
| `gpu` | Files whose names explicitly target CuPy, CUDA, or GPU behavior |
| `slow` | Audits, benchmarks, production checks, and extended workflows |

Markers overlap. For example, a GPU benchmark is both `gpu` and `slow`.

## Safe CPU workflow

Always select the NumPy backend when a training process already owns the GPU:

```bash
MINI_LLM_BACKEND=numpy pytest -q -m "unit and not gpu and not slow"
```

Useful narrower commands:

```bash
# Fast numerical smoke tests
MINI_LLM_BACKEND=numpy pytest -q \
  tests/test_embedding.py tests/test_rmsnorm.py tests/test_loss.py

# CPU integration tests, excluding long-running audits/benchmarks
MINI_LLM_BACKEND=numpy pytest -q -m "integration and not gpu and not slow"

# Verify collection and marker assignments without executing tests
MINI_LLM_BACKEND=numpy pytest --collect-only -q
```

## GPU tests

Run GPU tests only when no training or inference job is using the device:

```bash
MINI_LLM_BACKEND=cupy pytest -q -m gpu
```

Before doing so, inspect active compute processes with `nvidia-smi`. Do not run
this command concurrently with a significant training job: test allocations can
reduce available VRAM or disturb throughput.

## Naming conventions

- General tests: `test_<component>.py`
- Historical regressions: `test_<experiment-id>_<topic>.py`
- GPU-specific files include `cupy`, `cuda`, or `gpu` in the filename.
- Reusable fixtures and collection policy belong in `tests/conftest.py`.
- Benchmarks that primarily report measurements belong in `benchmarks/`; tests
  should assert correctness or a stable contract.

Generated test data must use pytest's `tmp_path` fixture. Tests must not write to
repository-level `checkpoints/`, `logs/`, shard directories, or feature caches.

## Optional historical tests

Some historical regressions target research utilities that are not part of the
current checkout. Those modules must be loaded with `pytest.importorskip` so an
absent optional artifact produces a documented skip rather than preventing the
entire suite from collecting. `test_source_audit_0078d.py` currently follows
this policy for the absent `benchmarks.audit_source_sampling_0078d` utility.

## Model B 0103A regression gate

Run the recommended bounded baseline correctness suites in independent
processes instead of mixing runtime-profile tests with numerical tests:

```bash
python scripts/run_0103a_regressions.py --mode cpu --report /tmp/0103a_cpu.json
```

When the CUDA device is available and **not** occupied by training, run:

```bash
python scripts/run_0103a_regressions.py --mode gpu --report /tmp/0103a_gpu.json
```

These tests protect the current **348M Model B + stride-16 sparse historical
head**. See `docs/0103A_BASELINE_REGRESSION.md` for the precise coverage and
limitations. Passing correctness checks does not establish GPU throughput.
