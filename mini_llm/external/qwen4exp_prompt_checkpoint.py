"""One prompt rollback point: small host copies, never a copy of QSA history.

QSA rows below the prompt length and completed pooled blocks are immutable.
Decode may complete a partial pooled block, but that slot lies beyond the
checkpoint's completed blocks and is ignored/recomputed after rollback.
The caller serializes access and discards the checkpoint whenever state resets.
"""
import weakref

from ..backend import BACKEND_NAME, asnumpy, xp
from .qwen4exp_ple import PLEState


def _fence():
    if BACKEND_NAME == 'cupy':
        xp.cuda.get_current_stream().synchronize()


def _bindings(state):
    """Identities only: do not keep the aggregate state or huge QSA arrays alive."""
    return (id(state.owner), state.max_length,
            tuple((i, id(s), id(s.conv), id(s.delta), id(s.conv.history), id(s.delta.memory))
                  for i, s in state.recurrent.items()),
            tuple((i, id(s), id(s.keys), id(s.values), id(s.raw_index), id(s.pooled))
                  for i, s in state.qsa.items()), tuple(state.ple))


class PromptCheckpoint:
    @classmethod
    def capture(cls, state, tokens, logits):
        """Return None for unsupported fake/legacy states; never copy QSA arrays."""
        if not all(hasattr(state, name) for name in
                   ('owner', 'max_length', 'length', 'failed', 'recurrent', 'qsa', 'ple')):
            return None
        if not all(isinstance(getattr(state, name), dict) for name in ('recurrent', 'qsa', 'ple')):
            return None
        try:
            reference = weakref.ref(state)
            bindings = _bindings(state)
        except (TypeError, AttributeError):
            return None
        if (state.failed or state.length != len(tokens) or
                any(s.conv.length != state.length or s.delta.length != state.length
                    for s in state.recurrent.values()) or
                any(s.length != state.length for s in state.qsa.values()) or
                any(not isinstance(s, PLEState) for s in state.ple.values())):
            return None
        _fence()
        checkpoint = cls()
        checkpoint._state = reference
        checkpoint._bindings = bindings
        checkpoint.tokens = tuple(tokens)
        checkpoint.recurrent = {
            i: (asnumpy(s.conv.history).copy(), s.conv.length,
                asnumpy(s.delta.memory).copy(), s.delta.length)
            for i, s in state.recurrent.items()}
        checkpoint.ple = {i: (tuple(s.predecessors), asnumpy(s.normalized_history).copy())
                          for i, s in state.ple.items()}
        checkpoint.qsa_lengths = {i: s.length for i, s in state.qsa.items()}
        checkpoint.length = state.length
        checkpoint.logits = asnumpy(logits).copy()
        return checkpoint

    @property
    def host_array_bytes(self):
        return (self.logits.nbytes + sum(h.nbytes + m.nbytes for h, _, m, _ in self.recurrent.values())
                + sum(h.nbytes for _, h in self.ple.values()))

    def restore(self, state, incoming, consumed):
        """Return prompt logits, or None without mutation if provenance does not match."""
        if (self._state() is not state or state is None or state.failed or
                state.length != len(consumed) or state.length < self.length or
                tuple(incoming[:self.length]) != self.tokens or
                tuple(consumed[:self.length]) != self.tokens or
                _bindings(state) != self._bindings or
                any(s.length != state.length for s in state.qsa.values()) or
                any(s.conv.length != state.length or s.delta.length != state.length
                    for s in state.recurrent.values())):
            return None
        _fence()
        for i, (history, conv_length, memory, delta_length) in self.recurrent.items():
            s = state.recurrent[i]
            s.conv.history[...] = xp.asarray(history)
            s.delta.memory[...] = xp.asarray(memory)
            s.conv.length, s.delta.length = conv_length, delta_length
        for i, (predecessors, history) in self.ple.items():
            state.ple[i] = PLEState(predecessors, xp.asarray(history).copy())
        for i, length in self.qsa_lengths.items():
            state.qsa[i].length = length
            # Host gather currently overwrites staging on every call. Incremental
            # staging must honor this invalidation marker when that optimization lands.
            state.qsa[i].staged_prefix_length = 0
        state.length = self.length
        logits = xp.asarray(self.logits).copy()
        _fence()  # Host copies may be released/replaced as soon as we return.
        return logits
