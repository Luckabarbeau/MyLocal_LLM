"""Real CUDA tests. Missing library/device is a failure, never a CPU fallback."""
import ctypes as C
import gc
import weakref

import pytest

from mini_llm.native import NativeError, Runtime, Stats

pytestmark = pytest.mark.gpu


@pytest.fixture
def runtime():
    return Runtime()


def test_synthetic_step_capacity_reset(runtime):
    with runtime.engine() as engine, engine.session(capacity=2) as session:
        assert session.step(42) == (42, 0)
        assert session.step(2147483647) == (2147483647, 1)
        with pytest.raises(NativeError) as error:
            session.step(3)
        assert error.value.status == 2
        session.reset()
        assert session.step(0) == (0, 0)
        assert engine.stats.device_bytes == 4
        assert engine.stats.pinned_bytes == 8


def test_accounting_budget_and_release(runtime):
    with runtime.engine(budget_bytes=24, max_sessions=3) as engine:
        assert engine.stats.sessions == engine.stats.peak_bytes == 0
        a, b = engine.session(), engine.session()
        assert engine.stats.sessions == 2
        assert engine.stats.device_bytes + engine.stats.pinned_bytes == 24
        with pytest.raises(NativeError, match="budget"):
            engine.session()
        assert engine.stats.sessions == 2
        a.close()
        a.close()
        assert engine.stats.sessions == 1
        with engine.session() as c:
            for _ in range(10):
                c.reset()
                assert c.step(9) == (9, 0)
            assert engine.stats.peak_bytes == 24
        b.close()
        assert engine.stats.sessions == 0
        assert engine.stats.device_bytes == engine.stats.pinned_bytes == 0
        assert engine.stats.peak_bytes == 24


def test_session_limit(runtime):
    with runtime.engine(budget_bytes=120, max_sessions=1) as engine:
        with engine.session():
            with pytest.raises(NativeError):
                engine.session()


def test_python_strong_ownership_and_gc(runtime):
    engine = runtime.engine()
    reference = weakref.ref(engine)
    session = engine.session()
    del engine
    gc.collect()
    assert reference() is not None
    assert session.step(17) == (17, 0)
    del session
    gc.collect()
    assert reference() is None


def test_native_ownership_survives_engine_close(runtime):
    engine = runtime.engine()
    session = engine.session()
    engine.close()
    assert session.step(8) == (8, 0)
    session.reset()
    assert session.step(7) == (7, 0)
    session.close()
    session.close()
    with pytest.raises(NativeError, match="closed"):
        session.reset()
    with pytest.raises(NativeError):
        engine.session()


def test_validation(runtime):
    for kwargs in ({"budget_bytes": 0}, {"max_sessions": 0}):
        with pytest.raises(NativeError):
            runtime.engine(**kwargs)
    with pytest.raises(ValueError):
        runtime.engine(device=-1)
    with pytest.raises(NativeError) as error:
        runtime.engine(device=2147483647)
    assert error.value.status == 3
    with runtime.engine() as engine:
        with pytest.raises(NativeError):
            engine.session(0)
        for capacity in (-1, 2**64, 1.5, True):
            with pytest.raises(ValueError):
                engine.session(capacity)
        with engine.session() as session:
            for token in (-1, 2**31, 1.5, True):
                with pytest.raises(ValueError):
                    session.step(token)
            assert session.step(1) == (1, 0)


def test_c_abi_errors_stale_and_null(runtime):
    lib = runtime._lib
    assert lib.ml_engine_create(0, 12, 1, None) == 1
    assert lib.ml_last_error()
    handle = C.c_uint64(999)
    assert lib.ml_engine_create(-1, 12, 1, C.byref(handle)) == 1
    assert handle.value == 0
    assert lib.ml_session_create(0, 1, C.byref(handle)) == 1
    assert lib.ml_session_reset(0) == 1
    assert lib.ml_engine_destroy(0) == 0
    assert lib.ml_session_destroy(0) == 0
    with runtime.engine() as engine:
        assert lib.ml_engine_stats(engine._handle, None) == 1
        session = engine.session()
        stale = session._handle
        out, pos = C.c_int32(99), C.c_uint64(99)
        assert lib.ml_session_step(stale, -1, C.byref(out), C.byref(pos)) == 1
        assert (out.value, pos.value) == (99, 99)
        assert lib.ml_session_step(stale, 1, None, C.byref(pos)) == 1
        assert session.step(6) == (6, 0)
        session.close()
        assert lib.ml_session_destroy(stale) == 1
        assert lib.ml_session_reset(stale) == 1
        assert engine.stats.sessions == 0
    assert lib.ml_engine_stats(engine._handle, C.byref(Stats())) == 1


def test_missing_library_is_explicit(monkeypatch):
    monkeypatch.delenv("MINI_LLM_NATIVE_LIBRARY", raising=False)
    with pytest.raises(RuntimeError, match="MINI_LLM_NATIVE_LIBRARY"):
        Runtime()
    with pytest.raises(OSError):
        Runtime("/nonexistent/mini_llm_native.so")
