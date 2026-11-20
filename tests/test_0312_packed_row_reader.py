"""Synthetic CPU-only packed-row tests; no model/backend imports or workloads."""
import builtins
from concurrent.futures import ThreadPoolExecutor, TimeoutError
import fcntl
import importlib.util
from itertools import repeat
import os
from pathlib import Path
import sys
import threading

import numpy as np
import pytest


SOURCE = Path(__file__).resolve().parents[1] / 'mini_llm/ops/packed_row_reader.py'


def load_reader():
    # Avoid mini_llm.__init__, which imports model classes and backend selection.
    spec = importlib.util.spec_from_file_location('packed_row_reader_cpu_test', SOURCE)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


packed = load_reader()
PackedRowReader = packed.PackedRowReader


@pytest.fixture
def table(tmp_path):
    rows = np.arange(8 * 90, dtype=np.uint16).astype(np.uint8).reshape(8, 90)
    path = tmp_path / 'packed.bin'
    prefix = b'header:' + bytes(30)
    path.write_bytes(prefix + rows.tobytes() + b'trailer')
    return path, len(prefix), rows


def reader_for(table, **kwargs):
    path, offset, rows = table
    return PackedRowReader(path, offset=offset, row_count=len(rows),
                           row_bytes=rows.shape[1], **kwargs)


def test_import_does_not_load_model_or_gpu(monkeypatch):
    original_import = builtins.__import__

    def guarded(name, *args, **kwargs):
        assert not name.startswith(('mini_llm', 'cupy', 'torch'))
        return original_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, '__import__', guarded)
    assert load_reader().PackedRowReader.MAX_BATCH_ROWS == 16


@pytest.mark.parametrize('name,value', [
    ('offset', -1), ('offset', True), ('offset', 1.0),
    ('row_count', 0), ('row_count', -1), ('row_count', False),
    ('row_count', 1.5), ('row_count', '8'),
    ('row_bytes', 0), ('row_bytes', -1), ('row_bytes', True),
    ('row_bytes', 90.0), ('row_bytes', None),
    ('cache_rows', -1), ('cache_rows', True), ('cache_rows', 1.0),
    ('workers', 0), ('workers', -1), ('workers', True),
    ('workers', 9), ('workers', 2.0),
    ('offset', np.bool_(True)), ('row_count', np.bool_(False)),
    ('offset', sys.maxsize + 1), ('row_bytes', sys.maxsize),
    ('row_count', sys.maxsize),
])
def test_invalid_arguments_before_open(table, monkeypatch, name, value):
    path, offset, rows = table
    options = dict(offset=offset, row_count=len(rows), row_bytes=90)
    options[name] = value
    monkeypatch.setattr(packed.os, 'open',
                        lambda *args: pytest.fail('invalid arguments opened a file'))
    with pytest.raises(ValueError):
        PackedRowReader(path, **options)


@pytest.mark.parametrize('offset,count,width', [(0, 100, 90), (10**6, 1, 1),
                                               (0, 1, 1000), (759, 1, 6)])
def test_entire_table_must_fit_and_constructor_closes_fd(table, monkeypatch,
                                                        offset, count, width):
    path, _, _ = table
    opened, closed = [], []
    real_open, real_close = os.open, os.close

    def opening(*args):
        fd = real_open(*args)
        opened.append(fd)
        return fd

    def closing(fd):
        closed.append(fd)
        real_close(fd)

    monkeypatch.setattr(packed.os, 'open', opening)
    monkeypatch.setattr(packed.os, 'close', closing)
    with pytest.raises(ValueError, match='file size'):
        PackedRowReader(path, offset=offset, row_count=count, row_bytes=width)
    assert len(opened) == 1 and closed == opened


def test_read_only_separate_fd_advice_and_unchanged_file(table, monkeypatch):
    path, offset, rows = table
    path.chmod(0o444)
    original = path.read_bytes()
    before = path.stat()
    opened, advice = [], []
    real_open = os.open

    def opening(path, flags):
        opened.append(flags)
        return real_open(path, flags)

    monkeypatch.setattr(packed.os, 'open', opening)
    monkeypatch.setattr(packed.os, 'POSIX_FADV_RANDOM', 1, raising=False)
    monkeypatch.setattr(packed.os, 'posix_fadvise',
                        lambda *args: advice.append(args), raising=False)
    with path.open('rb') as external, reader_for(table) as reader:
        fd = reader._fd
        assert fd != external.fileno()
        assert opened == [os.O_RDONLY | os.O_CLOEXEC]
        assert fcntl.fcntl(fd, fcntl.F_GETFL) & os.O_ACCMODE == os.O_RDONLY
        assert not os.get_inheritable(fd)
        assert not (opened[0] & getattr(os, 'O_DIRECT', 0))
        assert advice == [(fd, offset, rows.nbytes, os.POSIX_FADV_RANDOM)]
        np.testing.assert_array_equal(reader.read_batch([7, 0]), rows[[7, 0]])
        assert external.tell() == 0
    after = path.stat()
    assert path.read_bytes() == original
    assert (after.st_size, after.st_blocks, after.st_mtime_ns, after.st_ctime_ns) == (
        before.st_size, before.st_blocks, before.st_mtime_ns, before.st_ctime_ns)


def test_missing_advice_api_is_supported(table, monkeypatch):
    monkeypatch.delattr(packed.os, 'posix_fadvise', raising=False)
    with reader_for(table) as reader:
        np.testing.assert_array_equal(reader.read_batch([0]), table[2][[0]])


def test_advice_failure_closes_constructor_fd(table, monkeypatch):
    closed = []
    real_close = os.close

    def closing(fd):
        closed.append(fd)
        real_close(fd)

    def refuse(*args):
        raise OSError('advice refused')

    monkeypatch.setattr(packed.os, 'close', closing)
    monkeypatch.setattr(packed.os, 'POSIX_FADV_RANDOM', 1, raising=False)
    monkeypatch.setattr(packed.os, 'posix_fadvise', refuse, raising=False)
    with pytest.raises(OSError, match='advice refused'):
        reader_for(table)
    assert len(closed) == 1


def test_unique_duplicates_owned_copy_and_stats(table, monkeypatch):
    requests = []
    real_pread = os.pread

    def reading(fd, size, offset):
        requests.append((size, offset))
        return real_pread(fd, size, offset)

    monkeypatch.setattr(packed.os, 'pread', reading)
    with reader_for(table) as reader:
        output = reader.read_batch([2, 1, 2, 0])
        np.testing.assert_array_equal(output, table[2][[2, 1, 2, 0]])
        assert output.dtype == np.uint8 and output.flags.owndata
        assert output.flags.c_contiguous and output.base is None
        assert sorted(requests) == [(90, table[1] + i * 90) for i in range(3)]
        assert all(isinstance(value, bytes) for value in reader._cache.values())
        output[:] = 0
        again = reader.read_batch(np.array([2, 1, 2], dtype=np.int64))
        np.testing.assert_array_equal(again, table[2][[2, 1, 2]])
        assert len(requests) == 3 and not np.shares_memory(output, again)
        assert reader.stats() == dict(calls=2, read_rows=7, cache_hits=3,
            cache_misses=4, disk_logical_read_bytes=270, cached_bytes=270)
        snapshot = reader.stats()
        snapshot['calls'] = -1
        assert reader.stats()['calls'] == 2
    assert reader.stats()['cached_bytes'] == 0


def test_lru_bound_and_recent_duplicate_order(table):
    with reader_for(table, cache_rows=2) as reader:
        reader.read_batch([0, 1, 0])
        assert list(reader._cache) == [1, 0]
        reader.read_batch([2])
        assert list(reader._cache) == [0, 2]
        reader.read_batch([0, 1])
        assert list(reader._cache) == [0, 1]
        reader.read_batch([3, 4, 5, 3])
        assert list(reader._cache) == [5, 3]
        assert reader.stats()['cached_bytes'] == 180
        assert reader.stats()['disk_logical_read_bytes'] == 7 * 90


def test_zero_cache_reference_mode_and_numpy_integer_dimensions(table):
    path, offset, rows = table
    with PackedRowReader(path, offset=np.int64(offset), row_count=np.int64(8),
                         row_bytes=np.int64(90), cache_rows=0, workers=1) as reader:
        for _ in range(2):
            np.testing.assert_array_equal(reader.read_batch([1, 1]), rows[[1, 1]])
        assert not reader._cache
        assert reader.stats() == dict(calls=2, read_rows=4, cache_hits=0,
            cache_misses=4, disk_logical_read_bytes=180, cached_bytes=0)
        empty = reader.read_batch([])
        assert empty.shape == (0, 90) and empty.flags.owndata
        assert reader.stats()['calls'] == 3
        assert reader.stats()['disk_logical_read_bytes'] == 180


@pytest.mark.parametrize('indices,error', [
    ([0, -1], IndexError), ([0, 8], IndexError),
    ([0, True], ValueError), ([0, np.bool_(True)], ValueError),
    ([0, 1.0], ValueError), ([0, '1'], ValueError),
    ([0, None], ValueError), (np.array([[0], [1]]), ValueError),
    ([0] * 17, ValueError), (None, TypeError),
])
def test_bad_batch_fails_before_any_io_or_cache_change(table, monkeypatch, indices, error):
    with reader_for(table) as reader:
        before = reader.stats()
        monkeypatch.setattr(packed.os, 'pread',
                            lambda *args: pytest.fail('bad indices caused I/O'))
        with pytest.raises(error):
            reader.read_batch(indices)
        assert reader.stats() == before and not reader._cache


def test_infinite_request_is_bounded(table):
    consumed = []

    def indices():
        for value in repeat(0):
            consumed.append(value)
            yield value

    with reader_for(table) as reader:
        with pytest.raises(ValueError, match='at most 16'):
            reader.read_batch(indices())
        assert len(consumed) == 17 and reader.stats()['calls'] == 0


def test_short_reads_and_interruption_retry_exact_remaining_bytes(table, monkeypatch):
    real_pread = os.pread
    requests = []

    def short(fd, size, offset):
        requests.append((size, offset))
        if len(requests) == 1:
            raise InterruptedError()
        return real_pread(fd, min(size, 13), offset)

    monkeypatch.setattr(packed.os, 'pread', short)
    with reader_for(table, workers=1) as reader:
        np.testing.assert_array_equal(reader.read_batch([3]), table[2][[3]])
        start = table[1] + 3 * 90
        assert requests == [(90, start)] + [(90 - n, start + n) for n in range(0, 90, 13)]
        assert reader.stats()['disk_logical_read_bytes'] == 90
        assert reader._cache[3] == table[2][3].tobytes()


def test_eof_does_not_cache_partial_or_other_rows_in_failed_batch(table, monkeypatch):
    real_pread = os.pread
    start = table[1] + 90

    def truncated(fd, size, offset):
        if offset == start:
            return real_pread(fd, 7, offset)
        if offset == start + 7:
            return b''
        return real_pread(fd, size, offset)

    with reader_for(table) as reader:
        reader.read_batch([2])
        monkeypatch.setattr(packed.os, 'pread', truncated)
        with pytest.raises(EOFError, match='row 1: expected 90 bytes, got 7'):
            reader.read_batch([0, 1])
        assert list(reader._cache) == [2]
        assert reader.stats()['cached_bytes'] == 90
        assert reader.stats()['disk_logical_read_bytes'] == 270
        monkeypatch.setattr(packed.os, 'pread', real_pread)
        np.testing.assert_array_equal(reader.read_batch([1]), table[2][[1]])
        assert reader.stats()['disk_logical_read_bytes'] == 360


def test_failed_batch_drains_other_workers_before_return(table, monkeypatch):
    real_pread = os.pread
    blocked, release = threading.Event(), threading.Event()

    def reading(fd, size, offset):
        if offset == table[1]:
            assert blocked.wait(3)
            raise OSError('synthetic read failure')
        blocked.set()
        assert release.wait(3)
        return real_pread(fd, size, offset)

    monkeypatch.setattr(packed.os, 'pread', reading)
    with reader_for(table, workers=2) as reader, ThreadPoolExecutor(1) as callers:
        future = callers.submit(reader.read_batch, [0, 1])
        try:
            assert blocked.wait(3)
            with pytest.raises(TimeoutError):
                future.result(timeout=0.05)
        finally:
            release.set()
        with pytest.raises(OSError, match='synthetic read failure'):
            future.result(timeout=3)
        assert not reader._cache


def test_large_sparse_table_offset_over_4gib_stays_sparse(tmp_path, monkeypatch):
    path = tmp_path / 'large-sparse.bin'
    offset, count, width = (1 << 32) + 4099, 320_000_000, 90
    expected = {0: bytes(range(90)), 12345: bytes([71]) * 90,
                count - 1: bytes([203]) * 90}
    with path.open('wb') as file:
        file.truncate(offset + count * width)
        for index, row in expected.items():
            file.seek(offset + index * width)
            file.write(row)
    before = path.stat()
    assert before.st_blocks * 512 < 128 * 1024
    real_pread = os.pread
    requests = []

    def reading(fd, size, position):
        requests.append((size, position))
        return real_pread(fd, size, position)

    monkeypatch.setattr(packed.os, 'pread', reading)
    with PackedRowReader(path, offset=offset, row_count=count, row_bytes=width,
                         cache_rows=2) as reader:
        indices = [count - 1, 0, 12345, count - 1]
        result = reader.read_batch(indices)
        assert result.tobytes() == b''.join(expected[index] for index in indices)
        assert sorted(requests) == sorted((width, offset + index * width)
                                          for index in expected)
        assert reader.stats()['cached_bytes'] == 180
    after = path.stat()
    assert (after.st_size, after.st_blocks, after.st_mtime_ns) == (
        before.st_size, before.st_blocks, before.st_mtime_ns)


def test_at_most_16_pending_and_8_workers_across_concurrent_batches(tmp_path, monkeypatch):
    path = tmp_path / 'sixteen.bin'
    path.write_bytes(bytes(range(16)))
    release, submitted, started = threading.Event(), threading.Event(), threading.Event()
    guard = threading.Lock()
    worker_ids = set()
    pools = []
    real_pread = os.pread

    class TrackingExecutor(ThreadPoolExecutor):
        def __init__(self, *args, **kwargs):
            super().__init__(*args, **kwargs)
            self.pending = self.peak = self.total = 0
            pools.append(self)

        def submit(self, *args, **kwargs):
            with guard:
                self.pending += 1
                self.total += 1
                self.peak = max(self.peak, self.pending)
            future = super().submit(*args, **kwargs)

            def done(_):
                with guard:
                    self.pending -= 1

            future.add_done_callback(done)
            if self.total == 16:
                submitted.set()
            return future

    def blocked(fd, size, position):
        with guard:
            worker_ids.add(threading.get_ident())
            if len(worker_ids) == 8:
                started.set()
        assert release.wait(3)
        return real_pread(fd, size, position)

    monkeypatch.setattr(packed, 'ThreadPoolExecutor', TrackingExecutor)
    monkeypatch.setattr(packed.os, 'pread', blocked)
    with PackedRowReader(path, offset=0, row_count=16, row_bytes=1) as reader:
        with ThreadPoolExecutor(2) as callers:
            first = callers.submit(reader.read_batch, range(16))
            second = None
            try:
                assert submitted.wait(3) and started.wait(3)
                second = callers.submit(reader.read_batch, [0])
                with pytest.raises(TimeoutError):
                    second.result(timeout=0.05)
                assert pools[0].total == pools[0].peak == 16
                assert len(worker_ids) == 8
            finally:
                release.set()
            assert first.result(timeout=3).tobytes() == bytes(range(16))
            assert second.result(timeout=3).tobytes() == b'\0'
        assert pools[0].total == 16 and pools[0].pending == 0
        assert reader.stats()['cache_hits'] == 1
    assert all(not thread.is_alive() for thread in pools[0]._threads)


def test_multithread_read_close_waits_and_closes_once(table, monkeypatch):
    real_pread, real_close = os.pread, os.close
    entered, release, closing = threading.Event(), threading.Event(), threading.Event()
    closed = []
    reader = reader_for(table)
    fd = reader._fd

    def blocked(*args):
        entered.set()
        assert release.wait(3)
        return real_pread(*args)

    def close_fd(value):
        if value == fd:
            closed.append(value)
        real_close(value)

    def close_reader():
        closing.set()
        reader.close()

    def racing_read():
        try:
            return reader.read_batch([1, 0, 1])
        except ValueError as error:
            assert 'closed' in str(error)
            return None

    monkeypatch.setattr(packed.os, 'pread', blocked)
    monkeypatch.setattr(packed.os, 'close', close_fd)
    try:
        with ThreadPoolExecutor(6) as callers:
            first = callers.submit(reader.read_batch, [0, 1, 0])
            try:
                assert entered.wait(3)
                others = [callers.submit(racing_read) for _ in range(3)]
                closer = callers.submit(close_reader)
                assert closing.wait(3)
                with pytest.raises(TimeoutError):
                    closer.result(timeout=0.05)
                assert not closed
                os.fstat(fd)
            finally:
                release.set()
            np.testing.assert_array_equal(first.result(timeout=3), table[2][[0, 1, 0]])
            for future in others:
                result = future.result(timeout=3)
                if result is not None:
                    np.testing.assert_array_equal(result, table[2][[1, 0, 1]])
            closer.result(timeout=3)
        reader.close()
        reader.close()
        assert reader.closed and closed == [fd]
        assert reader.stats()['cached_bytes'] == 0
        with pytest.raises(OSError):
            os.fstat(fd)
        with pytest.raises(ValueError, match='closed'):
            reader.read_batch([])
        with pytest.raises(ValueError, match='closed'):
            reader.__enter__()
    finally:
        release.set()
        reader.close()


def test_context_exception_and_best_effort_destructor(table):
    reader = reader_for(table)
    with pytest.raises(RuntimeError, match='caller error'):
        with reader:
            raise RuntimeError('caller error')
    assert reader.closed
    abandoned = reader_for(table)
    fd = abandoned._fd
    abandoned.__del__()
    abandoned.__del__()
    assert abandoned.closed
    with pytest.raises(OSError):
        os.fstat(fd)
