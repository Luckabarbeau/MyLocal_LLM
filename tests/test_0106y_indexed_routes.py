"""Route-slot indirection preserves packed expert arithmetic and avoids gathers."""

import numpy as np
import pytest

from mini_llm.backend import BACKEND_NAME, asnumpy, xp
from mini_llm.ops import grouped_quantized_experts as grouped
from test_0106t_grouped_quantized_experts import (
    FORMATS, _hosts, _dense_fp64, _upload, simulated_gpu,
)


@pytest.mark.parametrize('slots', [(2, 0, 1), (1, 1, 0)])
def test_cpu_indexed_launch_graph_and_fp64(simulated_gpu, slots):
    sim = simulated_gpu
    hosts = _hosts(('Q4_K', 'Q5_K', 'Q6_K'))
    handles = sim.device(hosts)
    x_host = np.random.default_rng(10625).normal(0, .02, (1, 256)).astype(np.float32)
    routes_host = np.array([-.25, .125, .75], np.float32)
    x, routes = sim.cp.asarray(x_host), sim.cp.asarray(routes_host)
    sim.calls.clear()
    sim.transfers.clear()
    assert grouped.supports_grouped_quantized_experts(x, handles, routes, routing_slots=slots)
    assert not sim.calls and not sim.transfers
    actual = grouped.grouped_quantized_experts_forward(x, handles, routes, routing_slots=slots)
    np.testing.assert_allclose(actual, _dense_fp64(x_host, hosts, routes_host[list(slots)]),
                               rtol=4e-4, atol=8e-6)
    assert sim.transfers == [((3, 4), np.dtype('uint64'))]
    assert len(sim.calls) == 3
    np.testing.assert_array_equal(routes, routes_host)


@pytest.mark.parametrize('slots', [(), (0, 1), (0, 1, 3), (-1, 0, 1),
                                  (True, 0, 1), (.5, 0, 1), np.array([0, 1, 2])])
def test_invalid_host_slots_rejected_without_gpu_work(simulated_gpu, slots):
    sim = simulated_gpu
    handles = sim.device(_hosts(('Q8_0', 'Q8_0', 'Q8_0')))
    x, routes = sim.cp.asarray(np.ones((1, 256), np.float32)), sim.cp.asarray(np.ones(3, np.float32))
    sim.calls.clear()
    sim.transfers.clear()
    assert not grouped.supports_grouped_quantized_experts(x, handles, routes, routing_slots=slots)
    with pytest.raises(ValueError, match='routing_slots'):
        grouped.grouped_quantized_experts_forward(x, handles, routes, routing_slots=slots)
    assert not sim.calls and not sim.transfers


@pytest.mark.skipif(BACKEND_NAME != 'cupy', reason='Requires CUDA packed kernels')
@pytest.mark.parametrize('formats', FORMATS)
@pytest.mark.parametrize('layout', ['normal', 'reverse', 'broadcast'])
def test_cuda_indexed_routes_bitwise_equal_to_materialized(formats, layout):
    with xp.cuda.Stream(non_blocking=True) as stream:
        handles = _upload(_hosts(formats))
        handles = [handles[0], handles[1], handles[0], handles[2]]
        x = xp.asarray(np.random.default_rng(10625).normal(0, .03, (1, 256)).astype(np.float32))
        routes = xp.asarray([.5, -.25, .125, .75], dtype=xp.float32)
        if layout == 'reverse':
            routes = routes[::-1]
        elif layout == 'broadcast':
            routes = xp.broadcast_to(routes[:1], (4,))
        slots = (3, 0, 0, 2)
        before = routes.copy()
        materialized = xp.concatenate([routes[s:s + 1] for s in slots])
        expected = grouped.grouped_quantized_experts_forward(x, handles, materialized)
        actual = grouped.grouped_quantized_experts_forward(x, handles, routes, routing_slots=slots)
        stream.synchronize()
        np.testing.assert_array_equal(asnumpy(actual), asnumpy(expected))
        np.testing.assert_array_equal(asnumpy(routes), asnumpy(before))
        # Subsequent calls return independent storage, even with a new slot map.
        saved = actual.copy()
        other = grouped.grouped_quantized_experts_forward(x, handles, routes, routing_slots=(0, 1, 2, 3))
        stream.synchronize()
        np.testing.assert_array_equal(asnumpy(actual), asnumpy(saved))
        assert actual.data.ptr != other.data.ptr
