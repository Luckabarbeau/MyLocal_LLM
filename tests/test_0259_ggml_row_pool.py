"""Persistent row pool versus retained per-expert GGML arithmetic, no model load."""
import ctypes as C
import os
import threading
import time
from concurrent.futures import ThreadPoolExecutor

import numpy as np
import pytest

from mini_llm.ops.ggml_packed_cpu import GGMLPackedCPURowPool
from test_0256_ggml_packed_cpu import (executor, selected_library, packed, TRIPLES,
                                       QUANT, DOT)


def bundle(i=0, d=256, f=320):
    formats = TRIPLES[i % len(TRIPLES)]
    return (packed(formats[0], f, d, 100+i), packed(formats[1], f, d, 200+i),
            packed(formats[2], d, f, 300+i))


@pytest.mark.parametrize('workers', [1, 4, 15])
@pytest.mark.parametrize('count', [1, 2, 7, 10])
def test_exact_unweighted_rows(executor, workers, count):
    experts = [bundle(i) for i in range(count)]
    x = np.random.default_rng(259).normal(0, .1, (1, 256)).astype(np.float32)
    expected = np.concatenate([executor.expert(x, ws) for ws in experts])
    out = np.empty_like(expected)
    with GGMLPackedCPURowPool(executor, 256, 320, workers) as pool:
        job = pool.submit(x, experts, out=out)
        assert job.result() is out
        np.testing.assert_array_equal(out, expected)
        assert job.result() is out


def test_back_to_back_epochs_and_output_ownership(executor):
    experts = [bundle(i) for i in range(10)]
    inputs = np.random.default_rng(260).normal(0, .1, (4, 256)).astype(np.float32)
    expected = [[executor.expert(x[None], ws)[0] for ws in experts] for x in inputs]
    with GGMLPackedCPURowPool(executor, 256, 320, 8) as pool:
        old = []
        for epoch in range(120):
            count = 1 + epoch % 10
            job = pool.submit(inputs[epoch % 4:epoch % 4+1], experts[:count])
            with pytest.raises(RuntimeError, match='busy'):
                pool.submit(inputs[:1], experts[:1])
            np.testing.assert_array_equal(job.result(), expected[epoch % 4][:count])
            if epoch < 4:
                old.append((job, job.result().copy()))
        for job, value in old:
            np.testing.assert_array_equal(job.result(), value)


def test_shared_quantization_and_close_drain(executor):
    functions = executor._functions.copy()
    calls, callbacks = [], []
    try:
        # All formats sharing a type must use the same function pointer.
        by_type = {}
        for fmt, (activation, quant, dot) in functions.items():
            if activation not in by_type:
                original = QUANT(quant)
                def counted(x, y, k, original=original, activation=activation):
                    calls.append((activation, k))
                    original(x, y, k)
                callback = QUANT(counted)
                callbacks.append(callback)
                by_type[activation] = C.cast(callback, C.c_void_p).value
            executor._functions[fmt] = (activation, by_type[activation], dot)
        experts = [bundle(i) for i in range(10)]
        pool = GGMLPackedCPURowPool(executor, 256, 320, 4)
        job = pool.submit(np.ones((1, 256), np.float32), experts)
        pool.close()
        pool.close()
        assert job.result().shape == (10, 256)
        assert calls.count((15, 256)) == 1
        assert calls.count((8, 256)) == 1
        assert calls.count((8, 320)) == 10
        with pytest.raises(RuntimeError, match='closed'):
            pool.submit(np.ones((1, 256), np.float32), experts)
    finally:
        executor._functions = functions


def test_delayed_dot_retains_sources_and_drains(executor):
    functions = executor._functions.copy()
    entered, release = threading.Event(), threading.Event()
    errors = []
    activation, quant, pointer = functions['IQ3_S']
    original = DOT(pointer)
    def delayed(k, out, bs, w, bx, x, by, rows):
        entered.set()
        if not release.wait(5):
            errors.append('worker release timed out')
        original(k, out, bs, w, bx, x, by, rows)
    callback = DOT(delayed)
    try:
        executor._functions['IQ3_S'] = (activation, quant, C.cast(callback, C.c_void_p).value)
        with GGMLPackedCPURowPool(executor, 256, 320, 4) as pool:
            experts = [bundle(1)]
            job = pool.submit(np.ones((1, 256), np.float32), experts)
            try:
                assert entered.wait(3)
                assert job.weights is not None
                with ThreadPoolExecutor(1) as waiter:
                    future = waiter.submit(job.result)
                    release.set()
                    assert future.result(timeout=5).shape == (1, 256)
            finally:
                release.set()
        assert not errors
    finally:
        release.set()
        executor._functions = functions


def test_bounds_and_validation(executor):
    for args in ((256, 320, 0), (256, 320, True), (255, 320, 4)):
        with pytest.raises(ValueError):
            GGMLPackedCPURowPool(executor, *args)
    with GGMLPackedCPURowPool(executor, 256, 320, 2, capacity=2) as pool:
        x, ws = np.ones((1, 256), np.float32), bundle()
        for experts in ([], [ws]*3):
            with pytest.raises(ValueError, match='capacity'):
                pool.submit(x, experts)
        with pytest.raises(ValueError, match='one shared'):
            pool.submit(np.ones((2, 256), np.float32), [ws])
        with pytest.raises(TypeError, match='FP32'):
            pool.submit(x.astype(np.float16), [ws])
        with pytest.raises(ValueError, match='projection shapes'):
            pool.submit(x, [(ws[0], ws[1], ws[0])])
        with pytest.raises(ValueError, match='Output'):
            pool.submit(x, [ws], out=np.empty((256, 1), np.float32))
        with pytest.raises(ValueError, match='alias'):
            pool.submit(x, [ws], out=x)
        assert pool.submit(x, [ws]).result().shape == (1, 256)


def test_heterogeneous_decode_batch(executor):
    from mini_llm.backend import BACKEND_NAME, xp
    from mini_llm.ops.expert_residency import HeterogeneousExpertExecutor
    experts = [bundle(i) for i in range(10)]
    x_np = np.random.default_rng(261).normal(0, .1, (1, 256)).astype(np.float32)
    weights_np = np.linspace(.01, .1, 10, dtype=np.float32)
    expected = np.zeros_like(x_np)
    for i, ws in enumerate(experts):
        expected += weights_np[i] * executor.expert(x_np, ws)
    managed = HeterogeneousExpertExecutor(lambda key: experts[key.expert_id], model_id='0259',
        d_model=256, d_ff=320, cpu_workers=4, cpu_packed_backend='ggml_pool',
        compute_dtype='float32', execution_mode='cpu')
    try:
        with managed.dispatch_scope(0, tuple(range(10))):
            output = managed.try_forward_decode_batch(0, tuple(range(10)), xp.asarray(x_np),
                                                     xp.asarray(weights_np), tuple(range(10)))
        if BACKEND_NAME == 'cupy':
            output = output.get()
        np.testing.assert_array_equal(output, expected)
        assert managed.counters['ggml_row_batches'] == 1
        assert managed.counters['ggml_cpu_experts'] == 10
    finally:
        managed.close()


def test_optional_timing(executor):
    if os.environ.get('MINI_LLM_ROW_POOL_TIMING') != '1':
        pytest.skip('Set MINI_LLM_ROW_POOL_TIMING=1 for warmed operator timing')
    # Model-like D/F; identical inputs, formats and worker count for both paths.
    d, f, workers = 2048, 768, 15
    all_experts = [bundle(i, d, f) for i in range(10)]
    x = np.random.default_rng(262).normal(0, .1, (1, d)).astype(np.float32)
    with GGMLPackedCPURowPool(executor, d, f, workers) as row_pool, ThreadPoolExecutor(workers) as old:
        for count in (1, 3, 10):
            experts = all_experts[:count]
            timings = {}
            for label, run in (
                    ('whole_expert_futures', lambda: list(old.map(lambda ws: executor.expert(x, ws), experts))),
                    ('row_pool', lambda: row_pool.submit(x, experts).result())):
                for _ in range(5):
                    run()
                samples = []
                for _ in range(15):
                    start = time.perf_counter()
                    run()
                    samples.append(time.perf_counter()-start)
                timings[label] = float(np.median(samples))
            print('0259 warmed batch', count, 'D/F=', (d, f), 'seconds=', timings,
                  'speedup=', timings['whole_expert_futures']/timings['row_pool'])
