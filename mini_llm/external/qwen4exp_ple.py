"""Patch0206: bounded-table, inference-only Qwen4Exp PLE (one sequence).

Canonical output-first projections; converted gammas already include any +1.
Only caller-owned PLEState carries token or convolution history.
"""

from dataclasses import dataclass
import math
import operator
import os
from typing import Any

import numpy as np

from ..backend import xp
from ..ops.quantized_weight import PackedGGUFWeight
from .qwen4exp_hyperconnection import _fp32, _sigmoid


_MASK64 = (1 << 64) - 1


def _integer(value, name, low=0, high=_MASK64):
    if isinstance(value, (bool, np.bool_)):
        raise ValueError(f"{name} must be an integer")
    try:
        value = operator.index(value)
    except TypeError as exc:
        raise ValueError(f"{name} must be an integer") from exc
    if not low <= value <= high:
        raise ValueError(f"{name} out of range")
    return int(value)


@dataclass(frozen=True)
class PLEConfig:
    multipliers: tuple[int, ...]
    offsets: tuple[int, ...]
    vocabs: tuple[int, ...]
    eos: int
    table_rows: int

    def __post_init__(self):
        for name, size in (("multipliers", 3), ("offsets", 16), ("vocabs", 16)):
            values = tuple(_integer(v, name, low=1 if name == "vocabs" else 0)
                           for v in getattr(self, name))
            if len(values) != size:
                raise ValueError(f"{name} requires {size} integers")
            object.__setattr__(self, name, values)
        object.__setattr__(self, "eos", _integer(self.eos, "PLE eos", high=2**31 - 1))
        object.__setattr__(self, "table_rows", _integer(self.table_rows, "table rows", 1, 2**31))
        if any(o + v > self.table_rows for o, v in zip(self.offsets, self.vocabs)):
            raise ValueError("PLE head range exceeds table rows")

    @classmethod
    def from_metadata(cls, metadata, table_rows):
        required = {"qwen4exp.ple.ngram_size": 3,
                    "qwen4exp.ple.heads_per_ngram": 8,
                    "qwen4exp.ple.conv_kernel": 4,
                    "qwen4exp.embedding_length_per_layer_input": 160,
                    "qwen4exp.embedding_length": 2560}
        for name, expected in required.items():
            if _integer(metadata[name], name) != expected:
                raise ValueError(f"{name} must equal {expected}")
        return cls(metadata["qwen4exp.ple.layer_multipliers"],
                   metadata["qwen4exp.ple.head_offsets"],
                   metadata["qwen4exp.ple.head_vocab_sizes"],
                   metadata["qwen4exp.ple.eos_token_id"], table_rows)

    def rows(self, token, predecessors):
        """Sixteen exact uint64 hashes; predecessors are oldest-first, length two."""
        token = _integer(token, "token", high=2**31 - 1)
        if len(predecessors) != 2:
            raise ValueError("two oldest-first predecessors required")
        previous = tuple(_integer(t, "predecessor", -1, 2**31 - 1) for t in predecessors)
        context = [token]
        cut = False
        for t in reversed(previous):
            cut = cut or t < 0 or t == self.eos
            context.append(self.eos if cut else t)
        products = [(t * m) & _MASK64 for t, m in zip(context, self.multipliers)]
        bigram = products[0] ^ products[1]
        trigram = bigram ^ products[2]
        return tuple((bigram if h < 8 else trigram) % self.vocabs[h] + self.offsets[h]
                     for h in range(16))


@dataclass(frozen=True)
class PLEState:
    """One sequence: oldest-first tokens and normalized z history [9,4,2560].

    Arrays remain caller-owned; step never mutates an input state. Do not edit
    a state while it is in use on an asynchronous backend stream.
    """
    predecessors: tuple[int, int]
    normalized_history: Any


class Qwen4ExpPLE:
    """Single-token/single-sequence FP32 reference; no implicit request state.

    Projections are dense xp FP32 [4D,D]/[D,D], or pure callables mapping an
    xp FP32 [D] vector to [4D]/[D]. Callables must implement the exact linear
    projection, not a training layer or an approximation. D is fixed at 2560.
    """
    width = 2560

    def __init__(self, store, table_name, config, key, value, gamma_key,
                 gamma_query, gamma_conv, conv, *, epsilon):
        if not isinstance(config, PLEConfig):
            raise TypeError("config must be PLEConfig")
        desc = store.tensors[table_name]
        if desc.shape != (config.table_rows, 160) or desc.quant_type != "IQ4_NL":
            raise ValueError("PLE table must be IQ4_NL [table_rows,160]")
        self.store, self.table_name, self.config = store, table_name, config
        self.key = self._projection(key, "key", (4 * self.width, self.width))
        self.value = self._projection(value, "value", (self.width, self.width))
        for name, array, shape in (("gamma_key", gamma_key, (4, self.width)),
                                   ("gamma_query", gamma_query, (4, self.width)),
                                   ("gamma_conv", gamma_conv, (4, self.width)),
                                   ("conv", conv, (4 * self.width, 4))):
            _fp32(name, array)
            if array.shape != shape:
                raise ValueError(f"{name} must have shape {shape}")
            if not bool(xp.all(xp.isfinite(array))):
                raise ValueError(f"{name} must be finite")
            setattr(self, name, array.copy())
        epsilon = float(epsilon)
        if (not math.isfinite(epsilon) or epsilon < float(np.nextafter(np.float32(0), np.float32(1)))
                or epsilon > np.finfo(np.float32).max):
            raise ValueError("epsilon must be finite positive float32")
        self.epsilon = xp.float32(epsilon)
        self._batched_rows = os.environ.get("MINI_LLM_PLE_BATCHED_ROWS", "0") == "1"
        self._row_reader = None
        self._closed = False

    @staticmethod
    def _projection(weight, name, shape):
        if callable(weight):
            return weight
        _fp32(name, weight)
        if weight.shape != shape:
            raise ValueError(f"{name} must have shape {shape}")
        if not bool(xp.all(xp.isfinite(weight))):
            raise ValueError(f"{name} must be finite")
        return weight.copy()

    @classmethod
    def from_store(cls, store, *, key, value, gamma_key, gamma_query,
                   gamma_conv, conv, epsilon=None,
                   table_name="per_layer_token_embd.weight"):
        """Validate headers/metadata only; caller supplies decoded small weights.

        Store must remain open during use. No graph loader or residency manager
        is involved. Tokenizer EOS is deliberately not used as PLE EOS.
        """
        config = PLEConfig.from_metadata(store.metadata, store.tensors[table_name].shape[0])
        if epsilon is None:
            epsilon = store.metadata["qwen4exp.attention.layer_norm_rms_epsilon"]
        return cls(store, table_name, config, key, value, gamma_key, gamma_query,
                   gamma_conv, conv, epsilon=epsilon)

    def fresh_state(self):
        """Explicit sequence reset: missing predecessors and zero conv history."""
        return PLEState((-1, -1), xp.zeros((9, 4, self.width), dtype=xp.float32))

    def embedding(self, token, predecessors):
        """Opt-in bounded packed-row I/O; identical IQ4_NL decoding and layout."""
        if self._closed or self.store.closed:
            raise RuntimeError("PLE or GGUF store is closed")
        if not self._batched_rows:
            return self.embedding_reference(token, predecessors)
        indices = self.config.rows(token, predecessors)
        if self._row_reader is None:
            from ..ops.packed_row_reader import PackedRowReader
            desc = self.store.tensors[self.table_name]
            self._row_reader = PackedRowReader(self.store.paths[desc.shard_index],
                offset=desc.data_offset, row_count=desc.shape[0], row_bytes=desc.row_bytes)
        raw = self._row_reader.read_batch(indices)
        rows = PackedGGUFWeight(raw, (16, 160), "IQ4_NL").dequantize_reference()
        return xp.asarray(rows.reshape(self.width), dtype=xp.float32)

    def embedding_reference(self, token, predecessors):
        """Original sixteen independent mapped row decodes, retained as oracle."""
        if self._closed or self.store.closed:
            raise RuntimeError("PLE or GGUF store is closed")
        indices = self.config.rows(token, predecessors)
        rows = np.empty((16, 160), dtype=np.float32)
        for h, index in enumerate(indices):
            raw = self.store.rows(self.table_name, index, index + 1)
            rows[h] = PackedGGUFWeight(raw, (1, 160), "IQ4_NL").dequantize_rows_reference(
                np.array([0], dtype=np.int64), tile_rows=1)[0]
        return xp.asarray(rows.reshape(self.width), dtype=xp.float32)

    def close(self):
        """Retire bounded I/O workers before the model releases the GGUF store."""
        if self._row_reader is not None:
            self._row_reader.close()
        self._closed = True

    def _norm(self, array, gamma):
        return array / xp.sqrt(xp.mean(array * array, axis=-1, keepdims=True,
                                       dtype=xp.float32) + self.epsilon) * gamma

    def _project(self, weight, embedding, size):
        result = weight(embedding) if callable(weight) else weight @ embedding
        _fp32("projection output", result)
        if result.shape != (size,):
            raise ValueError(f"projection output must have shape {(size,)}")
        return result

    def step(self, token, streams, state):
        """Return (updated [4,D], new state); batching/shared tokens unsupported.

        EOS changes future hash context only. It does not clear convolution
        history; use fresh_state() for an actual new sequence.
        """
        _fp32("streams", streams)
        if streams.shape != (4, self.width):
            raise ValueError("step supports only one token/sequence shaped [4,2560]")
        if not isinstance(state, PLEState):
            raise TypeError("state must be explicit PLEState")
        history = _fp32("normalized_history", state.normalized_history)
        if history.shape != (9, 4, self.width):
            raise ValueError("normalized_history must have shape [9,4,2560]")
        emb = self.embedding(token, state.predecessors)
        result, new_history = self.apply_embedding(streams, emb, history)
        new_state = PLEState((int(state.predecessors[1]), int(token)), new_history)
        return result, new_state

    def apply_embedding(self, streams, emb, history):
        """GPU-only PLE math; row lookup and immutable token history stay outside capture."""
        if self._closed:
            raise RuntimeError("PLE is closed")
        for name, array, shape in (("streams", streams, (4, self.width)),
                ("embedding", emb, (self.width,)),
                ("normalized_history", history, (9, 4, self.width))):
            _fp32(name, array)
            if array.shape != shape:
                raise ValueError(f"{name} must have shape {shape}")
        key = self._norm(self._project(self.key, emb, 4 * self.width).reshape(4, self.width),
                         self.gamma_key)
        query = self._norm(streams, self.gamma_query)
        value = self._project(self.value, emb, self.width)
        score = xp.sum(key * query, axis=-1, dtype=xp.float32) * xp.float32(1 / math.sqrt(self.width))
        gate = _sigmoid(xp.sign(score) * xp.sqrt(xp.maximum(xp.abs(score), xp.float32(1e-6))))
        u = gate[:, None] * value[None, :]
        z = self._norm(u, self.gamma_conv)
        weights = self.conv.reshape(4, self.width, 4)
        conv = history[0] * weights[..., 0]
        conv = conv + history[3] * weights[..., 1]
        conv = conv + history[6] * weights[..., 2]
        conv = conv + z * weights[..., 3]
        result = streams + (u + conv * _sigmoid(conv))
        # Store normalized z exactly once; the four residual streams are not time.
        new_history = xp.concatenate((history[1:], z[None, ...]), axis=0)
        return result, new_history
