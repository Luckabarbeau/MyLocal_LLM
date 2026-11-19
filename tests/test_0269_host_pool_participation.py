"""Same packed arithmetic with a participating host and explicit worker affinity."""
import ctypes as C
import os
import threading

import numpy as np
import pytest

from mini_llm.ops.ggml_packed_cpu import GGMLPackedCPURowPool
from test_0256_ggml_packed_cpu import executor, selected_library, DOT
from test_0259_ggml_row_pool import bundle


@pytest.mark.parametrize('workers', [1, 4, 15])
@pytest.mark.parametrize('spin', [0, 500])
def test_host_phase_barriers_exact(executor, workers, spin):
    ws = [bundle(i) for i in range(10)]
    x = np.random.default_rng(269).normal(0,.1,(1,256)).astype(np.float32)
    expected = np.concatenate([executor.expert(x,w) for w in ws])
    with GGMLPackedCPURowPool(executor,256,320,workers,host_help=True,spin_us=spin) as pool:
        for repeat in range(120):
            count = 1+repeat%10
            np.testing.assert_array_equal(pool.submit(x,ws[:count]).result(),expected[:count])


@pytest.mark.parametrize('help_host', [False, True])
def test_host_really_computes_rows(executor, help_host):
    functions = executor._functions.copy()
    expected_ws = bundle(1)
    x = np.ones((1,256), np.float32)
    expected = executor.expert(x, expected_ws)
    owner = threading.get_native_id()
    participants = set()
    activation, quant, pointer = functions['IQ3_S']
    original = DOT(pointer)
    def counted(*args):
        participants.add(threading.get_native_id())
        original(*args)
    callback = DOT(counted)
    try:
        executor._functions['IQ3_S'] = (activation,quant,C.cast(callback,C.c_void_p).value)
        with GGMLPackedCPURowPool(executor,256,320,1,host_help=help_host,spin_us=0) as pool:
            actual = pool.submit(x,[expected_ws]).result()
            np.testing.assert_array_equal(actual,expected)
        assert (owner in participants) == help_host
    finally:
        executor._functions = functions


def test_linux_worker_affinity_does_not_pin_caller(executor):
    if not hasattr(os,'sched_getaffinity'):
        pytest.skip('Linux affinity test')
    allowed = sorted(os.sched_getaffinity(0))
    if len(allowed) < 4:
        pytest.skip('Need four available CPUs')
    before = set(os.listdir('/proc/self/task'))
    with GGMLPackedCPURowPool(executor,256,320,4,cpu_ids=allowed[:4],host_help=True) as pool:
        after = set(os.listdir('/proc/self/task'))
        observed = sorted(tuple(os.sched_getaffinity(int(t))) for t in after-before)
        assert observed == [(c,) for c in allowed[:4]]
        assert os.sched_getaffinity(0) == set(allowed)
        pool.submit(np.ones((1,256),np.float32),[bundle()]).result()


@pytest.mark.parametrize('cpus', [(0,), (0,0), (-1,0), (True,1)])
def test_affinity_validation(executor,cpus):
    with pytest.raises(ValueError,match='cpu_ids'):
        GGMLPackedCPURowPool(executor,256,320,2,cpu_ids=cpus)
