"""0084C: host NumPy context under the CuPy runtime must not call .get()."""
from types import SimpleNamespace

import numpy as np

from mini_llm import backend
from mini_llm.collective import probability_fusion
from mini_llm.collective.probability_fusion import FusionConfig, ProbabilityResidualCombiner


class _FakeDeviceArray:
    """Device value stub: models the existing CuPy ndarray .get() interface."""

    def __init__(self, data):
        self._data = np.asarray(data)
        self.calls = 0

    def get(self):
        self.calls += 1
        return self._data.copy()


def test_asnumpy_accepts_host_arrays_even_with_cupy_backend(monkeypatch):
    monkeypatch.setattr(backend, 'BACKEND_NAME', 'cupy')
    monkeypatch.setattr(backend, 'xp', SimpleNamespace(ndarray=_FakeDeviceArray))
    host_tokens = np.arange(12, dtype=np.int64).reshape(3, 4)
    host_lengths = np.asarray([1, 2, 4], dtype=np.int64)
    np.testing.assert_array_equal(backend.asnumpy(host_tokens), host_tokens)
    np.testing.assert_array_equal(backend.asnumpy(host_lengths), host_lengths)
    np.testing.assert_array_equal(backend.asnumpy([1, 2, 3]), [1, 2, 3])


def test_asnumpy_still_downloads_device_arrays(monkeypatch):
    monkeypatch.setattr(backend, 'BACKEND_NAME', 'cupy')
    monkeypatch.setattr(backend, 'xp', SimpleNamespace(ndarray=_FakeDeviceArray))
    device = _FakeDeviceArray([2, 4, 8])
    np.testing.assert_array_equal(backend.asnumpy(device), [2, 4, 8])
    assert device.calls == 1


def test_context_aware_fusion_with_host_context_in_cupy_runtime(monkeypatch):
    """Exercise the call chain that failed in 0084B's first GPU update.

    NumPy runs the reference fusion math, while the backend conversion helper
    is switched into the CuPy branch.  This isolates the host/device boundary
    without requiring a GPU in CI.
    """
    model = ProbabilityResidualCombiner(FusionConfig(
        vocab_size=13, context_heads=2, context_head_dim=4,
        context_hidden=8, seed=21,
    ))
    logits = np.linspace(-1.0, 1.0, 2 * 2 * 13, dtype=np.float32).reshape(2, 2, 13)
    tokens = np.asarray([[1, 2, 0, 0], [3, 4, 5, 6]], dtype=np.int64)
    lengths = np.asarray([2, 4], dtype=np.int64)
    targets = np.asarray([3, 7], dtype=np.int64)

    monkeypatch.setattr(backend, 'BACKEND_NAME', 'cupy')
    monkeypatch.setattr(backend, 'xp', SimpleNamespace(ndarray=_FakeDeviceArray))
    # Keep reference NumPy math while simulating the actual GPU conversion branch.
    monkeypatch.setattr(probability_fusion, 'scalar', lambda value: float(value))
    ce, arithmetic, weighted = model.loss_and_backward(
        logits, [0, 1], targets, return_weighted=True,
        context_tokens=tokens, context_lengths=lengths,
    )
    assert np.isfinite([ce, arithmetic, weighted]).all()
    assert np.isfinite(np.asarray(model.reference_logits.grad)).all()
