"""Exact stored-byte CUDA uploads; isolated from production model execution."""
import ctypes as C
import gc
import weakref

import pytest

from mini_llm.native import Runtime, NativeError, UploadedTensor, _GGUFTensor
from test_0201_gguf_store import write_gguf
from test_native_engine_payload import payload_file, mapped

pytestmark = pytest.mark.gpu


@pytest.fixture
def runtime():
    return Runtime()


# Every storage geometry currently accepted by the native parser. Arbitrary raw
# bits deliberately include invalid floating encodings/packed values: no decode.
FORMATS = [(0, 1, 4), (1, 1, 2), (2, 32, 18), (3, 32, 20),
           (6, 32, 22), (7, 32, 24), (8, 32, 34), (10, 256, 84),
           (11, 256, 110), (12, 256, 144), (13, 256, 176),
           (14, 256, 210), (15, 256, 292), (18, 256, 98),
           (20, 32, 18), (21, 256, 110), (22, 256, 82),
           (23, 256, 136), (30, 1, 2), (42, 64, 18)]


def readback(upload, offset, size):
    buffer = bytearray(size)
    assert upload.readback_for_test(offset, size, buffer) == size
    return bytes(buffer)


@pytest.mark.parametrize("kind,elements,block_bytes", FORMATS)
def test_all_formats_exact(runtime, tmp_path, kind, elements, block_bytes):
    path = tmp_path / "raw.gguf"
    raw = bytes((i * 37 + 11) % 256 for i in range(block_bytes * 6))
    write_gguf(path, [("raw", (elements * 2, 3), kind, 32, raw)])
    with runtime.gguf(path) as store, store.lease("raw") as lease:
        descriptor = lease.descriptor
        with runtime.engine(budget_bytes=max(12, len(raw))) as engine:
            with engine.upload(lease) as upload:
                assert isinstance(upload, UploadedTensor)
                assert upload.descriptor == descriptor
                assert isinstance(upload.query(), bool)
                upload.wait()
                upload.wait()
                assert upload.query()
                assert readback(upload, 0, len(raw)) == raw
                assert readback(upload, 1, 7) == raw[1:8]
                assert readback(upload, len(raw), 0) == b""
                assert engine.stats.device_bytes == len(raw)
                assert engine.stats.pinned_bytes == engine.stats.sessions == 0
            assert engine.stats.device_bytes == 0
            assert engine.stats.peak_bytes == len(raw)


def test_native_ownership_public_close_and_retirement(runtime, payload_file):
    path, payloads = payload_file
    store = runtime.gguf(path)
    lease = store.lease("dense")
    engine = runtime.engine()
    upload = engine.upload(lease)
    refs = [weakref.ref(x) for x in (store, lease, engine)]
    stale = [x._handle for x in (store, lease, engine)]
    store.close()
    lease.close()
    engine.close()
    del store, lease, engine
    gc.collect()
    assert all(ref() is None for ref in refs)  # Ownership must be native.
    assert mapped(path)  # No query/wait has retired the consumer's lease yet.
    assert runtime._lib.ml_gguf_destroy(stale[0]) == 1
    assert runtime._lib.ml_payload_lease_destroy(stale[1]) == 1
    assert runtime._lib.ml_engine_destroy(stale[2]) == 1
    upload.wait()
    assert not mapped(path)
    assert readback(upload, 0, 48) == payloads["dense"]
    upload.close()
    upload.close()
    for operation in (upload.query, upload.wait, lambda: upload.descriptor,
                      lambda: upload.readback_for_test(0, 0, bytearray())):
        with pytest.raises(NativeError, match="closed"):
            operation()


@pytest.mark.parametrize("upload_first", [True, False])
def test_shared_budget_both_directions(runtime, payload_file, upload_first):
    path, _ = payload_file
    with runtime.gguf(path) as store, store.lease("dense") as lease:
        with runtime.engine(budget_bytes=59) as engine:
            first = engine.upload(lease) if upload_first else engine.session()
            before = engine.stats
            with pytest.raises(NativeError, match="budget") as error:
                engine.session() if upload_first else engine.upload(lease)
            assert error.value.status == 2
            assert bytes(engine.stats) == bytes(before)
            first.close()
            assert engine.stats.device_bytes == engine.stats.pinned_bytes == 0
        with runtime.engine(budget_bytes=60) as engine:
            with engine.session(), engine.upload(lease):
                assert engine.stats.device_bytes == 52
                assert engine.stats.pinned_bytes == 8
                assert engine.stats.sessions == 1
                assert engine.stats.peak_bytes == 60
            assert engine.stats.device_bytes == engine.stats.pinned_bytes == 0
            assert engine.stats.peak_bytes == 60


def test_upload_gc_and_destructor_drains(runtime, payload_file):
    path, _ = payload_file
    with runtime.engine() as engine:
        for _ in range(10):
            store = runtime.gguf(path)
            lease = store.lease("packed")
            upload = engine.upload(lease)
            stale = upload._handle
            store.close()
            lease.close()
            del upload  # No query or wait: destructor must drain itself.
            gc.collect()
            assert runtime._lib.ml_uploaded_destroy(stale) == 1
            assert not mapped(path)
            assert engine.stats.device_bytes == engine.stats.pinned_bytes == 0
        assert engine.stats.peak_bytes == 204


@pytest.mark.parametrize("offset,size", [(-1, 1), (True, 1), (0, False),
    (1.5, 1), (0, "1"), (2**64, 0), (0, 2**64), (49, 0), (47, 2),
    (2**64 - 1, 2), (1, 2**64 - 1)])
def test_python_ranges(runtime, payload_file, offset, size):
    path, _ = payload_file
    with runtime.gguf(path) as store, store.lease("dense") as lease:
        with runtime.engine() as engine, engine.upload(lease) as upload:
            dest = bytearray(b"sentinel")
            with pytest.raises(ValueError):
                upload.readback_for_test(offset, size, dest)
            assert dest == b"sentinel"


def test_python_buffers_and_inputs(runtime, payload_file):
    path, _ = payload_file
    with runtime.gguf(path) as store, store.lease("dense") as lease:
        with runtime.engine() as engine, engine.upload(lease) as upload:
            for dest in (b"readonly", bytearray(1), memoryview(bytearray(10))[::2]):
                with pytest.raises(ValueError):
                    upload.readback_for_test(0, 2, dest)
            for invalid in (None, store, 1):
                with pytest.raises(ValueError):
                    engine.upload(invalid)
            other = Runtime()
            with other.gguf(path) as other_store, other_store.lease("dense") as other_lease:
                with pytest.raises(ValueError):
                    engine.upload(other_lease)
            lease.close()
            with pytest.raises(NativeError, match="closed"):
                engine.upload(lease)


def test_c_boundary_outputs_bounds_stale(runtime, payload_file):
    path, payloads = payload_file
    lib = runtime._lib
    assert lib.ml_uploaded_destroy(0) == 0
    with runtime.gguf(path) as store, store.lease("dense") as lease:
        with runtime.engine() as engine:
            assert lib.ml_engine_upload(engine._handle, lease._handle, None) == 1
            out = C.c_uint64(123)
            for a, b in [(0, lease._handle), (engine._handle, 0),
                         (lease._handle, engine._handle)]:
                out.value = 123
                assert lib.ml_engine_upload(a, b, C.byref(out)) == 1
                assert out.value == 0
            with engine.upload(lease) as upload:
                handle = upload._handle
                assert lib.ml_uploaded_descriptor(handle, None) == 1
                assert lib.ml_uploaded_query(handle, None) == 1
                dest = (C.c_ubyte * 50)(*([0xAB] * 50))
                copied = C.c_uint64(123)
                for offset, size, capacity, target in [
                    (49, 0, 50, dest), (48, 1, 50, dest),
                    (2**64 - 1, 2, 50, dest), (1, 2**64 - 1, 50, dest),
                    (0, 2, 1, dest), (0, 1, 50, None)]:
                    copied.value = 123
                    assert lib.ml_uploaded_readback_for_test(handle, offset, size,
                        target, capacity, C.byref(copied)) == 1
                    assert copied.value == 0
                    assert bytes(dest) == bytes([0xAB] * 50)
                assert lib.ml_uploaded_readback_for_test(handle, 0, 1, dest, 50, None) == 1
                assert lib.ml_uploaded_readback_for_test(handle, 48, 0, None, 0, C.byref(copied)) == 0
                assert lib.ml_uploaded_readback_for_test(handle, 0, 48, dest, 50, C.byref(copied)) == 0
                assert bytes(dest) == payloads["dense"] + bytes([0xAB] * 2)
            for invalid in (0, handle, lease._handle, engine._handle):
                d = _GGUFTensor()
                d.nbytes = 123
                ready = C.c_int(123)
                assert lib.ml_uploaded_descriptor(invalid, C.byref(d)) == 1
                assert d.nbytes == 123
                assert lib.ml_uploaded_query(invalid, C.byref(ready)) == 1
                assert ready.value == 0
                assert lib.ml_uploaded_wait(invalid) == 1
                copied.value = 123
                assert lib.ml_uploaded_readback_for_test(invalid, 0, 0, None, 0, C.byref(copied)) == 1
                assert copied.value == 0
            assert lib.ml_uploaded_destroy(handle) == 1
            assert engine.stats.device_bytes == 0


def test_capacity_failure_zero_output_and_no_accounting(runtime, payload_file):
    path, _ = payload_file
    with runtime.gguf(path) as store, store.lease("dense") as lease:
        with runtime.engine(budget_bytes=47) as engine:
            out = C.c_uint64(123)
            assert runtime._lib.ml_engine_upload(engine._handle, lease._handle, C.byref(out)) == 2
            assert out.value == 0
            assert engine.stats.device_bytes == engine.stats.pinned_bytes == engine.stats.peak_bytes == 0
            assert lease.read(0, 4)
        with runtime.engine(budget_bytes=2**64 - 1) as engine:
            with engine.upload(lease), engine.session():
                assert engine.stats.device_bytes + engine.stats.pinned_bytes == 60


def test_nonzero_device_and_restore(runtime, payload_file):
    path, payloads = payload_file
    cudart = C.CDLL("/usr/local/cuda/lib64/libcudart.so")
    count, previous, current = C.c_int(), C.c_int(), C.c_int()
    assert cudart.cudaGetDeviceCount(C.byref(count)) == 0
    if count.value < 2:
        pytest.skip("nonzero-device test requires two CUDA devices")
    assert cudart.cudaGetDevice(C.byref(previous)) == 0
    assert cudart.cudaSetDevice(0) == 0
    current = C.c_int()
    try:
        with runtime.gguf(path) as store, store.lease("dense") as lease:
            with runtime.engine(device=1) as engine, engine.upload(lease) as upload:
                upload.query()
                upload.wait()
                assert readback(upload, 0, 48) == payloads["dense"]
                assert cudart.cudaGetDevice(C.byref(current)) == 0
                assert current.value == 0
        assert cudart.cudaGetDevice(C.byref(current)) == 0
        assert current.value == 0
    finally:
        assert cudart.cudaSetDevice(previous.value) == 0
