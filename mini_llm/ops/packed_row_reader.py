"""Bounded CPU reads of opaque packed rows, without mapping/loading the table.

A separate read-only descriptor receives POSIX_FADV_RANDOM where available to
suppress read-ahead. This is NOT O_DIRECT: the OS page cache remains in use.
The caller owns the backing store's lifetime and must explicitly close readers
before retiring model/store handles; the destructor is only a best-effort guard.
"""
from collections import OrderedDict
from concurrent.futures import ThreadPoolExecutor, wait
from itertools import islice
from numbers import Integral
import os
import sys
import threading

import numpy as np


class PackedRowReader:
    """Read at most 16 row indices per synchronous :meth:`read_batch` call.

    Indices must be non-bool integers. Results are owned uint8 arrays in request
    order, including duplicates. Calls (including close) serialize on one lock;
    only distinct missing rows run concurrently, with at most 8 workers and 16
    submitted futures. The LRU holds complete immutable bytes, never views.

    ``stats()`` returns a snapshot: calls counts validated attempts (also empty
    or failed batches); read_rows counts requested positions, including repeats.
    cache_hits/cache_misses classify those positions against the cache at batch
    entry. disk_logical_read_bytes counts row_bytes per submitted unique miss,
    including failed reads, NOT physical OS disk traffic or short-read retries.
    cached_bytes is current retained payload, and becomes zero on close.
    """

    MAX_BATCH_ROWS = 16
    MAX_WORKERS = 8

    def __init__(self, path, *, offset, row_count, row_bytes,
                 cache_rows=65536, workers=8):
        self._lock = threading.Lock()
        self._fd = None
        self._executor = None
        self._cache = OrderedDict()
        self._counts = dict(calls=0, read_rows=0, cache_hits=0,
                            cache_misses=0, disk_logical_read_bytes=0)
        values = dict(offset=offset, row_count=row_count, row_bytes=row_bytes,
                      cache_rows=cache_rows, workers=workers)
        for name, value in values.items():
            minimum = 0 if name in ('offset', 'cache_rows') else 1
            if (isinstance(value, bool) or not isinstance(value, Integral)
                    or not minimum <= value <= sys.maxsize):
                raise ValueError(f'{name} must be a non-bool integer >= {minimum} '
                                 'and <= sys.maxsize')
        self.offset = int(offset)
        self.row_count = int(row_count)
        self.row_bytes = int(row_bytes)
        self.cache_rows = int(cache_rows)
        self.workers = int(workers)
        if self.workers > self.MAX_WORKERS:
            raise ValueError(f'workers must be <= {self.MAX_WORKERS}')
        table_bytes = self.row_count * self.row_bytes
        # Bound both signed file offsets and the largest possible NumPy result.
        if (self.offset + table_bytes > min(sys.maxsize, (1 << 63) - 1)
                or self.row_bytes * self.MAX_BATCH_ROWS > sys.maxsize):
            raise ValueError('table layout or batch dimensions exceed safe limits')
        try:
            self._fd = os.open(path, os.O_RDONLY | os.O_CLOEXEC)
            if self.offset + table_bytes > os.fstat(self._fd).st_size:
                raise ValueError('entire table must fit within the file size')
            if hasattr(os, 'posix_fadvise') and hasattr(os, 'POSIX_FADV_RANDOM'):
                os.posix_fadvise(self._fd, self.offset, table_bytes,
                                os.POSIX_FADV_RANDOM)
            self._executor = ThreadPoolExecutor(max_workers=self.workers,
                                                thread_name_prefix='packed-row')
        except BaseException:
            self.close()
            raise

    @property
    def closed(self):
        with self._lock:
            return self._fd is None

    def _read_row(self, index):
        position = self.offset + index * self.row_bytes
        remaining = self.row_bytes
        chunks = []
        while remaining:
            try:
                chunk = os.pread(self._fd, remaining, position)
            except InterruptedError:
                continue
            if not chunk:
                received = self.row_bytes - remaining
                raise EOFError(f'row {index}: expected {self.row_bytes} bytes, '
                               f'got {received} before EOF')
            chunks.append(chunk)
            position += len(chunk)
            remaining -= len(chunk)
        return b''.join(chunks)

    def read_batch(self, indices):
        """Validate the whole bounded batch before I/O; cache only on success.

        Iterables are consumed only up to 17 entries, so an oversized (even
        infinite) request cannot create an unbounded list or executor queue.
        A failed batch drains all submitted work before releasing the lock.
        """
        with self._lock:
            if self._fd is None:
                raise ValueError('PackedRowReader is closed')
            requested = list(islice(iter(indices), self.MAX_BATCH_ROWS + 1))
            if len(requested) > self.MAX_BATCH_ROWS:
                raise ValueError(f'batch must contain at most {self.MAX_BATCH_ROWS} rows')
            for index in requested:
                if isinstance(index, bool) or not isinstance(index, Integral):
                    raise ValueError('row indices must be non-bool integers')
                if not 0 <= index < self.row_count:
                    raise IndexError(f'row index {index} outside [0, {self.row_count})')
            requested = [int(index) for index in requested]
            result = np.empty((len(requested), self.row_bytes), dtype=np.uint8)
            rows = {index: self._cache[index] for index in dict.fromkeys(requested)
                    if index in self._cache}
            missing = [index for index in dict.fromkeys(requested) if index not in rows]
            hits = sum(index in rows for index in requested)
            self._counts['calls'] += 1
            self._counts['read_rows'] += len(requested)
            self._counts['cache_hits'] += hits
            self._counts['cache_misses'] += len(requested) - hits
            futures = []
            try:
                for index in missing:
                    future = self._executor.submit(self._read_row, index)
                    futures.append((index, future))
                    self._counts['disk_logical_read_bytes'] += self.row_bytes
                for index, future in futures:
                    rows[index] = future.result()
            finally:
                wait([future for _, future in futures])
            for position, index in enumerate(requested):
                row = rows[index]
                result[position] = np.frombuffer(row, dtype=np.uint8)
                if self.cache_rows:
                    self._cache[index] = row
                    self._cache.move_to_end(index)
                    if len(self._cache) > self.cache_rows:
                        self._cache.popitem(last=False)
            return result

    def stats(self):
        """Return an independent, lock-consistent counter/payload snapshot."""
        with self._lock:
            return dict(self._counts,
                        cached_bytes=len(self._cache) * self.row_bytes)

    def close(self):
        """Wait for readers/workers, discard cached bytes, and close exactly once."""
        with self._lock:
            try:
                if self._executor is not None:
                    self._executor.shutdown(wait=True)
                    self._executor = None
            finally:
                fd, self._fd = self._fd, None
                try:
                    if fd is not None:
                        os.close(fd)
                finally:
                    self._cache.clear()

    def __enter__(self):
        with self._lock:
            if self._fd is None:
                raise ValueError('PackedRowReader is closed')
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        self.close()

    def __del__(self):
        try:
            self.close()
        except Exception:
            pass
