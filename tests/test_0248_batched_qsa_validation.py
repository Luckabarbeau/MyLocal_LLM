"""Batched validation retains nonfinite rejection before QSA state commit."""
import numpy as np
import pytest

from mini_llm.backend import BACKEND_NAME, asnumpy, xp
from mini_llm.ops.finite_checks_gpu import all_finite_gpu

pytestmark = pytest.mark.skipif(BACKEND_NAME != 'cupy', reason='actual CuPy required')


@pytest.mark.parametrize('count', [1, 6, 7, 8])
def test_finite_arrays_strides_empty_and_nondefault_stream(count):
    arrays = [xp.arange(257 + i, dtype=xp.float32)[::2] for i in range(count)]
    arrays[0] = xp.empty((0,), dtype=xp.float32)
    xp.cuda.get_current_stream().synchronize()
    with xp.cuda.Stream(non_blocking=True):
        assert all_finite_gpu(*arrays)
        if count > 1:
            for bad in (xp.nan, xp.inf, -xp.inf):
                arrays[-1][-1] = bad
                assert not all_finite_gpu(*arrays)
            arrays[-1][-1] = 0
            assert all_finite_gpu(*arrays)
    with pytest.raises(TypeError, match='FP32'):
        all_finite_gpu(xp.ones(1, dtype=xp.float16))
    with pytest.raises(ValueError, match='1..8'):
        all_finite_gpu()


@pytest.mark.parametrize('name', ['query_gate', 'key', 'value', 'index_key', 'output'])
@pytest.mark.parametrize('bad', [np.nan, np.inf, -np.inf])
def test_bad_projection_cannot_be_masked_or_committed(name, bad):
    from test_0230_qsa_gpu_decode import fixture
    from mini_llm.external.qwen4exp_attention import Qwen4ExpAttention
    config, weights, rng = fixture()
    model = Qwen4ExpAttention(config, **weights, fast_decode=True)
    state = model.create_state(4)
    hidden = xp.asarray(rng.normal(size=7), dtype=xp.float32)
    for position in range(3):
        model.step(hidden, state, position=position)
    before = {k: asnumpy(getattr(state, k)).copy()
              for k in ('keys', 'values', 'raw_index', 'pooled')}
    _, size = model.projections[name]
    model.projections[name] = (lambda x: xp.full(size, bad, dtype=xp.float32), size)
    if name != 'output':
        # Even a callback that suppresses all upstream NaNs must not hide them.
        _, output_size = model.projections['output']
        model.projections['output'] = (lambda x: xp.zeros(output_size, dtype=xp.float32), output_size)
    with pytest.raises(ValueError, match='finite'):
        model.step(hidden, state, position=3)
    assert state.length == 3
    for key, value in before.items():
        np.testing.assert_array_equal(asnumpy(getattr(state, key)), value)
