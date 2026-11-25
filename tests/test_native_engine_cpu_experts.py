"""WP5 native caller gate: synthetic packed banks only, no model allocations."""
import ctypes as C
import gc
import os
from pathlib import Path
import subprocess
import sys
import textwrap

import numpy as np
import pytest

from mini_llm.native import Runtime, NativeError
from mini_llm.ops.ggml_packed_cpu import GGMLPackedCPUExecutor, _load_bridge, _FORMATS
from test_0256_ggml_packed_cpu import selected_library, packed
from test_0201_gguf_store import write_gguf

ACTUAL_TRIPLES = (
    ('IQ3_XXS', 'IQ3_XXS', 'IQ4_NL'),
    ('IQ2_S', 'IQ2_S', 'Q2_0'),
    ('IQ2_S', 'IQ2_S', 'IQ4_NL'),
    ('IQ3_XXS', 'IQ3_XXS', 'Q2_0'),
    ('IQ3_S', 'IQ3_S', 'IQ4_NL'),
    ('IQ3_S', 'IQ3_S', 'Q2_0'),
    ('IQ4_XS', 'IQ4_XS', 'IQ4_NL'),
)
D, F = 256, 320


@pytest.fixture
def runtime():
    return Runtime()


@pytest.fixture
def libraries(selected_library):
    return dict(bridge_library=_load_bridge()._name, ggml_library=selected_library)


@pytest.mark.parametrize('next_owner', ['ctypes', 'native', 'python_pool', 'moe'])
def test_ggml_process_residency_without_retainer(libraries, tmp_path, next_owner):
    # A new interpreter is essential: an oracle/fixture CDLL in this pytest
    # process can hide the native last-owner unload/reload bug.
    code = textwrap.dedent('''
        import ctypes as C
        import gc
        import resource
        import sys
        from contextlib import ExitStack
        from pathlib import Path
        resource.setrlimit(resource.RLIMIT_CORE, (0, 0))
        sys.path.insert(0, str(Path.cwd() / 'tests'))
        from mini_llm.native import Runtime
        bridge, ggml, next_owner, directory = sys.argv[1:]
        libraries = dict(bridge_library=bridge, ggml_library=ggml)
        r = Runtime()
        def cycle():
            with r.engine() as e:
                with e.cpu_experts(d_model=256, d_ff=320, threads=1, **libraries) as p:
                    stale = p._handle
            gc.collect()
            assert r._lib.ml_cpu_experts_destroy(stale) == 1
        cycle()
        print('native last owner closed; next=' + next_owner, flush=True)
        if next_owner == 'ctypes':
            import _ctypes
            for _ in range(3):
                lib = C.CDLL(ggml)
                lib.ggml_cpu_init()
                _ctypes.dlclose(lib._handle)
                del lib
                cycle()
        elif next_owner == 'native':
            for _ in range(3):
                cycle()
        elif next_owner == 'python_pool':
            from mini_llm.ops.ggml_packed_cpu import GGMLPackedCPUExecutor, GGMLPackedCPURowPool
            executor = GGMLPackedCPUExecutor(ggml)
            for _ in range(3):
                pool = GGMLPackedCPURowPool(executor, 256, 320, 1)
                pool.close()
                cycle()
        elif next_owner == 'moe':
            from test_native_engine_moe import fixture, native_setup
            path = Path(directory) / 'moe.gguf'
            data = fixture(path)
            for _ in range(2):
                with ExitStack() as stack:
                    native_setup(stack, path, data, libraries, ())
                cycle()
            assert str(path) not in Path('/proc/self/maps').read_text()
        assert ggml in Path('/proc/self/maps').read_text()
        print('residency and owner transitions passed', flush=True)
    ''')
    result = subprocess.run(
        [sys.executable, '-u', '-c', code, libraries['bridge_library'],
         libraries['ggml_library'], next_owner, str(tmp_path)],
        cwd=Path(__file__).resolve().parents[1], env=os.environ.copy(),
        capture_output=True, text=True, timeout=30)
    assert result.returncode == 0, (result.returncode, result.stdout, result.stderr[:4000])
    assert 'residency and owner transitions passed' in result.stdout


def bank(tmp_path, formats=ACTUAL_TRIPLES[0]):
    experts = [tuple(packed(fmt, rows, cols, 500+10*e+p)
        for p, (fmt, rows, cols) in enumerate(zip(formats, (F,F,D), (D,D,F))))
        for e in range(10)]
    tensors, offset, sizes = [], 0, []
    for p, (name, fmt, rows, cols) in enumerate(zip(('gate','up','down'), formats, (F,F,D), (D,D,F))):
        payload = b''.join(ws[p].data.tobytes() for ws in experts)
        offset = (offset+31)//32*32
        tensors.append((name, (cols, rows, 10), _FORMATS[fmt][0], offset, payload))
        sizes.append(experts[0][p].nbytes)
        offset += len(payload)
    path = tmp_path/'experts.gguf'
    write_gguf(path, tensors)
    return path, experts, sizes


def setup(runtime, libraries, path, sizes):
    engine = runtime.engine()
    pool = engine.cpu_experts(d_model=D, d_ff=F, threads=4, capacity=10, **libraries)
    store = runtime.gguf(path)
    leases = [store.lease(name) for name in ('gate','up','down')]
    indices = [pool.register(*leases, offsets=tuple(e*s for s in sizes)) for e in range(10)]
    return engine, pool, store, leases, indices


@pytest.mark.parametrize('formats', ACTUAL_TRIPLES)
def test_actual_triples_ten_exact_native_rows(runtime, libraries, tmp_path, formats, monkeypatch):
    path, experts, sizes = bank(tmp_path, formats)
    oracle = GGMLPackedCPUExecutor(libraries['ggml_library'])
    x = np.random.default_rng(55).normal(0,.02,(1,D)).astype(np.float32)
    expected = np.concatenate([oracle.expert(x, ws) for ws in experts])
    engine, pool, store, leases, ids = setup(runtime,libraries,path,sizes)
    try:
        assert pool.arithmetic == 'ggml_q8'
        # The caller is not allowed to dispatch experts through the Python oracle.
        monkeypatch.setattr(GGMLPackedCPUExecutor, 'expert', lambda *a: pytest.fail('Python expert dispatch'))
        for count in (1,10,3,10):
            with pool.submit(x, ids[:count]) as job:
                job.wait()
                np.testing.assert_array_equal(job.readback_for_test(), expected[:count])
                np.testing.assert_array_equal(job.readback_for_test(), expected[:count])
    finally:
        pool.close()
        for lease in leases: lease.close()
        store.close(); engine.close()


def test_native_ownership_busy_close_pending_and_detached_results(runtime, libraries, tmp_path):
    path, experts, sizes = bank(tmp_path)
    x = np.random.default_rng(56).normal(0,.02,(1,D)).astype(np.float32)
    oracle = GGMLPackedCPUExecutor(libraries['ggml_library'])
    expected = np.concatenate([oracle.expert(x,w) for w in experts])
    engine, pool, store, leases, ids = setup(runtime,libraries,path,sizes)
    job = pool.submit(x,ids)
    x.fill(99)  # Native input owns its snapshot.
    with pytest.raises(NativeError, match='busy'): pool.submit(x,ids)
    with pytest.raises(NativeError, match='busy'): pool.register(*leases)
    for lease in leases: lease.close()
    store.close(); engine.close()
    assert any(str(path) in line for line in Path('/proc/self/maps').read_text().splitlines())
    pool.close()  # Explicitly drains even though the job is still public.
    pool.close()
    with pytest.raises(NativeError, match='closed'): pool.submit(x,ids)
    np.testing.assert_array_equal(job.readback_for_test(),expected)
    stale = job._handle
    detached = job.readback_for_test()
    job.close(); job.close(); gc.collect()
    assert not any(str(path) in line for line in Path('/proc/self/maps').read_text().splitlines())
    assert runtime._lib.ml_cpu_job_wait(stale) == 1
    assert runtime._lib.ml_cpu_job_destroy(stale) == 1
    np.testing.assert_array_equal(detached,expected)


def test_invalid_subtensor_jobs_and_stale_handles(runtime,libraries,tmp_path):
    path, _, sizes = bank(tmp_path)
    engine, pool, store, leases, ids = setup(runtime,libraries,path,sizes)
    x = np.zeros((1,D),np.float32)
    try:
        for offsets in ((1,0,0), (sizes[0]*10,0,0), (2**64-1,0,0)):
            with pytest.raises(NativeError,match='offset'): pool.register(*leases,offsets=offsets)
        with pytest.raises(NativeError,match='shape'): pool.register(leases[2],leases[1],leases[0])
        with pytest.raises(NativeError,match='index'): pool.submit(x,[10])
        for indices in ([], list(range(11)), [-1], [True]):
            with pytest.raises(ValueError): pool.submit(x,indices)
        for invalid in (x.astype(np.float64), np.zeros((2,D),np.float32)):
            with pytest.raises(ValueError): pool.submit(invalid,ids)
        with pytest.raises(NativeError,match='nonfinite'): pool.submit(np.full((1,D),np.nan,np.float32),ids)
        with pool.submit(x,ids) as job:
            output = np.full(10*D,123,np.float32)
            assert runtime._lib.ml_cpu_job_readback_for_test(job._handle,
                output.ctypes.data_as(C.POINTER(C.c_float)),1) == 1
            assert np.all(output==123)
            job.wait()
        # Closing an unconsumed job also drains and permits the next epoch.
        pool.submit(x,ids).close()
        with pool.submit(x,ids) as job:
            np.testing.assert_array_equal(job.readback_for_test(),np.zeros((10,D),np.float32))
        out=C.c_uint64(123)
        assert runtime._lib.ml_cpu_experts_submit(pool._handle,None,1,None,D,C.byref(out))==1
        assert out.value==0
        stale=pool._handle
        pool.close()
        assert runtime._lib.ml_cpu_experts_destroy(stale)==1
        assert runtime._lib.ml_cpu_experts_destroy(0)==0
        assert runtime._lib.ml_cpu_job_destroy(0)==0
    finally:
        pool.close()
        for lease in leases: lease.close()
        store.close(); engine.close()


def test_explicit_libraries_and_arithmetic(runtime,libraries):
    with runtime.engine() as engine:
        base=dict(d_model=D,d_ff=F,threads=2,**libraries)
        for key in ('bridge_library','ggml_library'):
            with pytest.raises(NativeError): engine.cpu_experts(**dict(base,**{key:'/nonexistent/explicit.so'}))
        with pytest.raises(NativeError):
            engine.cpu_experts(**dict(base,ggml_library=libraries['bridge_library']))
        with pytest.raises(ValueError,match='ggml_q8'):
            engine.cpu_experts(**base,arithmetic='reference_fp32')
        for key, value in (('d_model',255),('d_ff',0),('threads',0),('capacity',0)):
            with pytest.raises((ValueError,NativeError)):
                engine.cpu_experts(**dict(base,**{key:value}))
