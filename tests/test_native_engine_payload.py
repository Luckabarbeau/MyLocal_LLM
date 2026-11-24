"""Host-only payload consumers: exact source bytes and shared native mmap lifetime."""
import ctypes as C
import gc
import os
from pathlib import Path
import struct
import subprocess
import sys
import threading
import weakref

import pytest

from mini_llm.native import Runtime, NativeError, PayloadLease, _GGUFTensor
from test_0201_gguf_store import write_gguf, pair


@pytest.fixture
def runtime():
    return Runtime()


@pytest.fixture
def payload_file(tmp_path):
    path = tmp_path / "payload.gguf"
    dense = struct.pack("<12f", *[i * 0.25 - 1 for i in range(12)])
    packed = bytes((i * 37 + 11) % 256 for i in range(204))
    write_gguf(path, [("dense", (4, 3), 0, 32, dense),
                      ("packed", (32, 2, 3), 8, 128, packed)])
    return path, {"dense": dense, "packed": packed}


def mapped(path):
    return any(str(path) in line for line in Path("/proc/self/maps").read_text().splitlines())


@pytest.mark.parametrize("name", ["dense", "packed"])
def test_source_bytes_descriptor_and_detached_copy(runtime, payload_file, name):
    path, payloads = payload_file
    with runtime.gguf(path) as store, store.lease(name) as lease:
        assert isinstance(lease, PayloadLease)
        assert lease.descriptor == store.tensors[name]
        d = lease.descriptor
        with path.open("rb") as source:
            source.seek(d.data_offset)
            expected = source.read(d.nbytes)
        assert expected == payloads[name] == lease.read(0, d.nbytes)
        for offset, size in [(0, 1), (3, 7), (d.nbytes - 1, 1), (d.nbytes, 0), (0, 0)]:
            assert lease.read(offset, size) == expected[offset:offset + size]
        detached = lease.read(1, 9)
    assert detached == expected[1:10]
    assert not mapped(path)


@pytest.mark.parametrize("offset,size", [
    (-1, 1), (0, -1), (True, 1), (0, False), (1.5, 1), (0, "1"),
    (2**64, 0), (0, 2**64), (2**64 - 1, 2), (1, 2**64 - 1),
    (49, 0), (48, 1), (47, 2),
])
def test_bounds_before_allocation(runtime, payload_file, monkeypatch, offset, size):
    path, _ = payload_file
    with runtime.gguf(path) as store, store.lease("dense") as lease:
        # Any ctypes payload allocation would consult c_ubyte; invalid ranges must not.
        monkeypatch.setattr(C, "c_ubyte", None)
        with pytest.raises(ValueError):
            lease.read(offset, size)


def test_explicit_size_and_name_validation(runtime, payload_file):
    path, _ = payload_file
    with runtime.gguf(path) as store, store.lease("dense") as lease:
        with pytest.raises(TypeError):
            lease.read(0)
        for name in (None, 1, b"dense", "dense\0suffix"):
            with pytest.raises(ValueError):
                store.lease(name)
        with pytest.raises(NativeError, match="unknown"):
            store.lease("missing")
        # Detached public snapshots are not the lease identity authority.
        store.tensors.clear()
        with store.lease("dense") as second:
            assert second.descriptor.nbytes == 48


def test_independent_multishard_leases_and_last_release(runtime, tmp_path):
    a, b = pair(tmp_path)
    store = runtime.gguf(b)
    first = store.lease("experts")
    sibling = store.lease("experts")
    other = store.lease("ple")
    assert first.descriptor.shard_index == 0
    assert other.descriptor.shard_index == 1
    store.close()
    with pytest.raises(NativeError, match="closed"):
        store.lease("ple")
    first.close()
    assert sibling.read(135, 69) == bytes(range(135, 204))
    assert other.read(1, 53) == bytes(range(1, 54))
    assert mapped(a) and mapped(b)
    sibling.close()
    # Each lease retains the whole GGUF, including both shards.
    assert mapped(a) and mapped(b)
    other.close()
    assert not mapped(a) and not mapped(b)
    for lease in (first, sibling, other):
        lease.close()
        with pytest.raises(NativeError, match="closed"):
            lease.read(0, 0)
        with pytest.raises(NativeError, match="closed"):
            _ = lease.descriptor


def test_store_gc_and_lease_gc(runtime, payload_file):
    path, payloads = payload_file
    store = runtime.gguf(path)
    lease = store.lease("packed")
    reference = weakref.ref(store)
    stale_store = store._handle
    del store
    gc.collect()
    assert reference() is None  # Ownership is native, not a hidden Python store ref.
    assert runtime._lib.ml_gguf_destroy(stale_store) == 1
    assert mapped(path)
    assert lease.read(0, 204) == payloads["packed"]
    stale_lease = lease._handle
    del lease
    gc.collect()
    assert runtime._lib.ml_payload_lease_destroy(stale_lease) == 1
    assert not mapped(path)


def test_c_boundary_null_bounds_outputs_stale_and_wrong_handles(runtime, payload_file):
    path, payloads = payload_file
    lib = runtime._lib
    out = C.c_uint64(123)
    assert lib.ml_gguf_lease_create(0, b"dense", C.byref(out)) == 1
    assert out.value == 0
    assert lib.ml_payload_lease_destroy(0) == 0
    with runtime.gguf(path) as store:
        assert lib.ml_gguf_lease_create(store._handle, b"dense", None) == 1
        for name in (None, b"missing"):
            out.value = 123
            assert lib.ml_gguf_lease_create(store._handle, name, C.byref(out)) == 1
            assert out.value == 0
            assert lib.ml_last_error()
        with store.lease("dense") as lease:
            handle = lease._handle
            dest = (C.c_ubyte * 50)(*([0xAB] * 50))
            copied = C.c_uint64(123)
            assert lib.ml_payload_lease_descriptor(handle, None) == 1
            for offset, size, capacity, destination in [
                (49, 0, 50, dest), (48, 1, 50, dest),
                (2**64 - 1, 2, 50, dest), (1, 2**64 - 1, 50, dest),
                (0, 2, 1, dest), (0, 1, 50, None),
            ]:
                copied.value = 123
                assert lib.ml_payload_lease_copy(handle, offset, size, destination,
                                                 capacity, C.byref(copied)) == 1
                assert copied.value == 0
                assert bytes(dest) == bytes([0xAB] * 50)
                assert lib.ml_last_error()
            assert lib.ml_payload_lease_copy(handle, 0, 1, dest, 50, None) == 1
            assert bytes(dest) == bytes([0xAB] * 50)
            assert lib.ml_payload_lease_copy(handle, 48, 0, None, 0, C.byref(copied)) == 0
            assert copied.value == 0
            assert lib.ml_payload_lease_copy(handle, 0, 48, dest, 50, C.byref(copied)) == 0
            assert copied.value == 48
            assert bytes(dest) == payloads["dense"] + bytes([0xAB] * 2)
        with store.lease("dense") as replacement:
            assert replacement._handle != handle
            for invalid in (0, handle, store._handle):
                d = _GGUFTensor()
                d.nbytes = 123
                copied.value = 123
                assert lib.ml_payload_lease_descriptor(invalid, C.byref(d)) == 1
                assert d.nbytes == 123
                assert lib.ml_payload_lease_copy(invalid, 0, 0, None, 0, C.byref(copied)) == 1
                assert copied.value == 0
            assert lib.ml_payload_lease_destroy(handle) == 1
            assert replacement.read(0, 1) == payloads["dense"][:1]


def test_large_tensor_relative_offset(runtime, tmp_path):
    path = tmp_path / "sparse.gguf"
    size = 2**32 + 4
    base = write_gguf(path, [("large", (size // 4,), 0, 0, None)])
    with path.open("r+b") as source:
        source.truncate(base + size)
        source.seek(base + 2**32)
        source.write(b"tail")
    with runtime.gguf(path) as store, store.lease("large") as lease:
        assert lease.read(2**32, 4) == b"tail"
        assert lease.read(size, 0) == b""
        with pytest.raises(ValueError):
            lease.read(size - 1, 2)


@pytest.mark.parametrize("target", ["store", "lease"])
def test_read_serializes_close(runtime, payload_file, monkeypatch, target):
    path, payloads = payload_file
    store = runtime.gguf(path)
    lease = store.lease("dense")
    entered, release, closing, closed = (threading.Event() for _ in range(4))
    results, failures = [], []
    original = runtime._call

    def blocked(name, *args):
        if name == "ml_payload_lease_copy":
            entered.set()
            assert release.wait(5)
        return original(name, *args)

    def read():
        try:
            results.append(lease.read(0, 48))
        except BaseException as error:
            failures.append(error)

    def close():
        closing.set()
        try:
            (store if target == "store" else lease).close()
            closed.set()
        except BaseException as error:
            failures.append(error)

    monkeypatch.setattr(runtime, "_call", blocked)
    reader, closer = threading.Thread(target=read), threading.Thread(target=close)
    try:
        reader.start()
        assert entered.wait(5)
        closer.start()
        assert closing.wait(5)
        assert not closed.wait(0.05)
        release.set()
        reader.join(5)
        closer.join(5)
        assert not reader.is_alive() and not closer.is_alive()
        assert not failures
        assert closed.is_set() and results == [payloads["dense"]]
    finally:
        release.set()
        reader.join(5)
        if closer.ident is not None:
            closer.join(5)
        lease.close()
        store.close()
    assert not mapped(path)


def test_python_checks_native_error_outputs(runtime, payload_file, monkeypatch):
    path, _ = payload_file
    original = runtime._call
    with runtime.gguf(path) as store, store.lease("dense") as lease:
        def bad_output(name, *args):
            if name in ("ml_gguf_lease_create", "ml_payload_lease_copy"):
                return  # Simulate success without filling required outputs.
            return original(name, *args)
        monkeypatch.setattr(runtime, "_call", bad_output)
        with pytest.raises(NativeError, match="null handle"):
            store.lease("dense")
        with pytest.raises(NativeError, match="byte count"):
            lease.read(0, 1)


def test_repeated_cleanup_and_process_teardown(runtime, payload_file):
    path, _ = payload_file
    baseline = len(os.listdir("/proc/self/fd"))
    for _ in range(20):
        with runtime.gguf(path) as store:
            with store.lease("dense") as lease:
                store.close()
                assert lease.read(0, 4) == struct.pack("<f", -1)
        assert not mapped(path)
    assert len(os.listdir("/proc/self/fd")) == baseline
    # Exercise native registry teardown with live store and lease handles, not GC cleanup.
    code = """
import ctypes as C
import sys
from mini_llm.native import Runtime
r = Runtime()
paths = (C.c_char_p * 1)(sys.argv[1].encode())
s, lease = C.c_uint64(), C.c_uint64()
r._call('ml_gguf_open', paths, 1, C.byref(s))
r._call('ml_gguf_lease_create', s.value, b'dense', C.byref(lease))
"""
    result = subprocess.run([sys.executable, "-c", code, str(path)],
                            capture_output=True, text=True, timeout=30)
    assert result.returncode == 0, result.stdout + result.stderr
