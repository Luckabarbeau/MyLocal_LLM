"""Bounded-spin policy cannot alter row arithmetic or batch lifetime."""
import os
import time
from concurrent.futures import ThreadPoolExecutor

import numpy as np
import pytest

from mini_llm.ops.ggml_packed_cpu import GGMLPackedCPURowPool
from test_0256_ggml_packed_cpu import selected_library, executor
from test_0259_ggml_row_pool import bundle


@pytest.mark.parametrize('spin_us', [0, 50, 500, 2000])
@pytest.mark.parametrize('workers', [1, 4, 15])
def test_phase_publication_epochs_and_drain(executor, spin_us, workers):
    experts = [bundle(i) for i in range(10)]
    x = np.random.default_rng(267).normal(0, .1, (1, 256)).astype(np.float32)
    reference = np.concatenate([executor.expert(x, w) for w in experts])
    with GGMLPackedCPURowPool(executor, 256, 320, workers, spin_us=spin_us) as pool:
        for repeat in range(60):
            count = 1+repeat % 10
            job = pool.submit(x, experts[:count])
            if repeat % 5 == 0:
                with ThreadPoolExecutor(1) as waiter:
                    got = waiter.submit(job.result).result(timeout=5)
            else:
                got = job.result()
            np.testing.assert_array_equal(got, reference[:count])
            if repeat % 11 == 0:
                time.sleep(.003)  # Also exercise the spin->sleep->wake seam.


@pytest.mark.parametrize('value', [-1, True, .5, 2001])
def test_invalid_policy(executor, value):
    with pytest.raises(ValueError, match='spin_us'):
        GGMLPackedCPURowPool(executor, 256, 320, 4, spin_us=value)


def test_optional_operator_timing(executor):
    if os.environ.get('MINI_LLM_POOL_SPIN_TIMING') != '1':
        pytest.skip('Explicit warmed batch timing only')
    ws = [bundle(i, 2560, 640) for i in range(5)]
    x = np.random.default_rng(268).normal(0, .1, (1, 2560)).astype(np.float32)
    results = {}
    for spin in (0, 50, 500):
        with GGMLPackedCPURowPool(executor, 2560, 640, 15, spin_us=spin) as pool:
            for _ in range(10):
                pool.submit(x, ws).result()
            samples = []
            for _ in range(100):
                start = time.perf_counter()
                pool.submit(x, ws).result()
                samples.append(time.perf_counter()-start)
                time.sleep(.0003)  # Approximate intervening layer work, not throughput.
            results[spin] = float(np.median(samples))
    print('0267 batch5 D2560/F640 median seconds by spin_us:', results)
