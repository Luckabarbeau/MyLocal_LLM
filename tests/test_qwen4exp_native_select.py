"""Native QSA selection must retain exact value-only reference heap ties."""
import numpy as np
import pytest

from mini_llm.backend import asnumpy, xp
from mini_llm.external.qwen4exp_attention import select_cells
from mini_llm.ops.native_packed_cpu import qsa_select_native


@pytest.mark.parametrize('count,width', [(1, 1), (2, 1), (2, 2), (3, 2), (4, 3),
    (127, 1), (127, 16), (128, 127), (128, 128), (4096, 2051), (262144, 2051)])
@pytest.mark.parametrize('kind', ['random', 'ties', 'infinities', 'increasing', 'decreasing'])
def test_membership_matches_reference_heap(monkeypatch, count, width, kind):
    rng = np.random.default_rng(5080)
    scores = rng.normal(size=count).astype(np.float32)
    if kind == 'ties':
        scores = rng.integers(-2, 3, count).astype(np.float32)
    elif kind == 'infinities':
        scores[::3] = np.inf
        scores[1::5] = -np.inf
    elif kind == 'increasing':
        scores = np.arange(count, dtype=np.float32)
    elif kind == 'decreasing':
        scores = -np.arange(count, dtype=np.float32)
    monkeypatch.setenv('MINI_LLM_QSA_NATIVE_SELECT', '0')
    expected = asnumpy(select_cells(xp.asarray(scores), width))
    np.testing.assert_array_equal(qsa_select_native(scores, width), expected)
    monkeypatch.setenv('MINI_LLM_QSA_NATIVE_SELECT', '1')
    np.testing.assert_array_equal(asnumpy(select_cells(xp.asarray(scores), width)), expected)


def test_noncontiguous_and_validation():
    scores = np.arange(12, dtype=np.float32)[::2]
    np.testing.assert_array_equal(qsa_select_native(scores, 2), [4, 5])
    for width in (0, 7, True, 1.5):
        with pytest.raises(ValueError):
            qsa_select_native(scores, width)
    with pytest.raises(ValueError):
        qsa_select_native(np.array([np.nan], np.float32), 1)
    with pytest.raises(TypeError):
        qsa_select_native(scores.astype(np.float64), 1)
