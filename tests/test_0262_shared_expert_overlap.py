"""Shared GPU work is independent of routed CPU work; event-order and parity."""
import time
from types import SimpleNamespace

import numpy as np
import pytest

from mini_llm.backend import BACKEND_NAME, xp
from mini_llm.external.qwen4exp_model import Qwen4ExpMoE

pytestmark = pytest.mark.skipif(BACKEND_NAME != 'cupy', reason='CuPy stream scheduling')


@pytest.mark.parametrize('fast', [False, True])
@pytest.mark.parametrize('nondefault', [False, True])
def test_exact_output_producer_order_and_overlap(fast, nondefault):
    rng = np.random.default_rng(262)
    d, f = 64, 96
    wg, wu, wd = (xp.asarray(rng.normal(0, .1, shape).astype(np.float32))
                  for shape in ((d, f), (d, f), (f, d)))
    score = xp.asarray(rng.normal(0, .1, d).astype(np.float32))
    events, streams = [], []
    def gate(x):
        streams.append(xp.cuda.get_current_stream().ptr)
        return x @ wg
    def down(x):
        result = x @ wd
        event = xp.cuda.Event(disable_timing=True)
        event.record()
        events.append(event)
        return result
    router = SimpleNamespace(forward=lambda x: (xp.ones((1, 1, 1), xp.float32),
                                               xp.zeros((1, 1, 1), xp.int32)))
    check_overlap = [False]
    def routed(x, weights, ids):
        # Simulate the native CPU wait. Independent shared work must already be
        # enqueued, not delayed until after the routed expert returns.
        if check_overlap[0]:
            assert events
            deadline = time.monotonic()+3
            while not events[-1].done and time.monotonic() < deadline:
                time.sleep(.001)
            assert events[-1].done
        return x * xp.float32(.5)
    kwargs = dict(router=router, experts=SimpleNamespace(forward=routed), gate=gate,
                  up=lambda x: x @ wu, down=down, shared_gate=score, fast_elementwise=fast)
    baseline = Qwen4ExpMoE(**kwargs)
    candidate = Qwen4ExpMoE(**kwargs, overlap_shared=True)
    stream = xp.cuda.Stream(non_blocking=True) if nondefault else xp.cuda.get_current_stream()
    with stream:
        x = xp.zeros(d, xp.float32)
        snapshots = []
        for i in range(5):
            x.fill(float(i+1))  # Producer is on the caller stream.
            check_overlap[0] = False
            expected = baseline(x)
            events.clear()
            check_overlap[0] = True
            got = candidate(x)
            snapshots.append((got, expected))
        stream.synchronize()
        for got, expected in snapshots:
            np.testing.assert_array_equal(got.get(), expected.get())
    assert candidate._shared_stream.ptr in streams
    assert stream.ptr in streams


def test_failure_fences_shared_work():
    done = []
    def down(x):
        result = x.copy()
        event = xp.cuda.Event(disable_timing=True)
        event.record()
        done.append(event)
        return result
    def fail(*args):
        raise RuntimeError('routed failure')
    moe = Qwen4ExpMoE(SimpleNamespace(forward=fail), None, lambda x: x,
                     lambda x: x, down, xp.ones(64, xp.float32), overlap_shared=True)
    with pytest.raises(RuntimeError, match='routed failure'):
        moe(xp.ones(64, xp.float32))
    xp.cuda.get_current_stream().synchronize()
    assert done[-1].done
