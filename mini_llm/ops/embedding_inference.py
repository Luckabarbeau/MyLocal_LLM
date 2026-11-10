"""Generic inference-only row lookup with an optional bounded host cache."""

from collections import OrderedDict
from numbers import Integral

import numpy as np

from mini_llm.backend import asnumpy, dtype_name, xp


def _integer(value, name, minimum):
    if isinstance(value, bool) or not isinstance(value, Integral):
        raise TypeError(f"{name} must be an integer")
    if value < minimum:
        raise ValueError(f"{name} must be >= {minimum}")
    return int(value)


class RowEmbeddingInference:
    """Look up CPU-provider rows without retaining a device-side table.

    ``row_provider(int)`` must return a CPU array-like of shape
    ``(embedding_dim,)`` convertible to the configured float16/32/64 dtype.
    Provider data and flags are never modified. Cached rows are owned host
    copies; the byte budget counts their array data, not Python metadata.
    A budget smaller than one row disables persistent caching.

    Each forward loads a distinct row at most once, even without caching.
    Distinct rows access the LRU in first-occurrence order. Only requested
    rows are transferred to the backend; no backend arrays persist here.
    Providers should supply stable rows while they remain cached. Instances
    are not thread-safe.
    """

    def __init__(self, n_rows, embedding_dim, row_provider, *, cache_bytes=0,
                 dtype="float32"):
        self.n_rows = _integer(n_rows, "n_rows", 1)
        self.embedding_dim = _integer(embedding_dim, "embedding_dim", 1)
        self.cache_bytes = _integer(cache_bytes, "cache_bytes", 0)
        if not callable(row_provider):
            raise TypeError("row_provider must be callable")
        try:
            dtype = np.dtype(dtype)
        except (TypeError, ValueError) as exc:
            raise ValueError("dtype must be float16, float32 or float64") from exc
        if dtype_name(dtype) not in {"float16", "float32", "float64"}:
            raise ValueError("dtype must be float16, float32 or float64")
        self.dtype = dtype
        self.row_provider = row_provider
        self._row_bytes = self.embedding_dim * self.dtype.itemsize
        self._capacity = self.cache_bytes // self._row_bytes
        self._cache = OrderedDict()
        self._provider_loads = 0
        self._cache_hits = 0

    def statistics(self):
        """Return a fresh snapshot of cache occupancy and lifetime counters.

        ``resident_rows``: number of owned host rows in the persistent LRU.
        ``used_bytes``: resident row data bytes (always <= cache_bytes).
        ``provider_loads``: provider-call attempts, including calls that fail
        or return malformed rows.
        ``cache_hits``: distinct-row accesses served by the persistent LRU;
        repeated IDs within one forward do not add hits.
        """
        return {
            "resident_rows": len(self._cache),
            "used_bytes": len(self._cache) * self._row_bytes,
            "provider_loads": self._provider_loads,
            "cache_hits": self._cache_hits,
        }

    def _row(self, index):
        if index in self._cache:
            self._cache_hits += 1
            self._cache.move_to_end(index)
            return self._cache[index]
        self._provider_loads += 1
        supplied = self.row_provider(index)
        try:
            row = np.asarray(supplied)
        except (TypeError, ValueError) as exc:
            raise ValueError(f"row_provider({index}) must return a CPU row array") from exc
        if row.shape != (self.embedding_dim,):
            raise ValueError(
                f"row_provider({index}) must have shape ({self.embedding_dim},), "
                f"got {row.shape}"
            )
        try:
            row = np.array(row, dtype=self.dtype, copy=True, order="C")
        except (TypeError, ValueError) as exc:
            raise ValueError(
                f"row_provider({index}) cannot be converted to {self.dtype}"
            ) from exc
        if self._capacity:
            if len(self._cache) == self._capacity:
                self._cache.popitem(last=False)
            self._cache[index] = row
        return row

    def forward(self, token_ids):
        """Return a backend array of shape ``token_ids.shape + (D,)``.

        IDs must be a backend integer ndarray (not bool), all in [0, n_rows).
        All ID validation precedes any provider calls or cache changes. Empty
        integer inputs return an empty array with no provider calls/readback.
        Nonempty GPU inputs use ONE explicit ``asnumpy`` ID readback, which
        synchronizes device work; this is not a synchronization-free API.
        Provider failures propagate; previously loaded rows may remain cached.
        """
        if not isinstance(token_ids, xp.ndarray):
            raise TypeError("token_ids must be a backend xp.ndarray")
        if token_ids.dtype.kind not in "iu":
            raise TypeError("token_ids must have an integer dtype (not bool)")
        shape = token_ids.shape + (self.embedding_dim,)
        if token_ids.size == 0:
            return xp.empty(shape, dtype=self.dtype)
        ids = asnumpy(token_ids).reshape(-1)
        if np.any(ids < 0) or np.any(ids >= self.n_rows):
            raise ValueError(f"token_ids must be in [0, {self.n_rows})")
        unique, first, inverse = np.unique(
            ids, return_index=True, return_inverse=True,
        )
        rows = np.empty((unique.size, self.embedding_dim), dtype=self.dtype)
        for position in np.argsort(first):
            rows[position] = self._row(int(unique[position]))
        return xp.asarray(rows)[xp.asarray(inverse.reshape(-1))].reshape(shape)
