"""Experimental WP8 native worker. WP7 OPEN: not a production/HTTP backend."""
import ctypes as C
import threading

from . import _Owned, _integer, NativeError, SampleResult, Sampler
from .decoder import NativeDecoder


class GenerationEvent(C.Structure):
    _fields_ = [('kind', C.c_int32), ('reason', C.c_int32),
                ('status', C.c_int32), ('pending_token', C.c_int32),
                ('position', C.c_uint64), ('sampled', C.c_uint64),
                ('pending', C.c_int32), ('failed', C.c_int32),
                                ('in_flight', C.c_int32), ('reserved', C.c_int32),
                ('sample', SampleResult), ('error', C.c_char * 512)]

    @property
    def termination(self):
        return ('none', 'max_tokens', 'eos', 'capacity', 'cancelled', 'error')[self.reason]


class NativeGeneration(_Owned):
    """Retains exclusive decoder/sampler owners until close returns them.

    poll() returns a token or terminal event, or None on timeout/end. Errors are
    terminal events after queued tokens, not hidden fallback or poll exceptions.
    close() discards unread events and returns final sampled/consumed metadata.
    """
    @classmethod
    def start(cls, decoder, sampler, input_token, max_tokens, *, eos_ids=(),
              temperature=0., top_p=1., top_k=None, queue_capacity=1):
        if not isinstance(decoder, NativeDecoder) or not isinstance(sampler, Sampler):
            raise TypeError('native decoder and sampler required')
        if decoder._runtime is not sampler._runtime:
            raise TypeError('same-runtime native owners required')
        _integer(input_token, 'input_token', 2**31-1)
        _integer(max_tokens, 'max_tokens', 2**64-1)
        _integer(queue_capacity, 'queue_capacity', 4096)
        if top_k is not None:
            _integer(top_k, 'top_k', 2**64-1)
            if not top_k and temperature != 0:
                raise ValueError('top_k must be >= 1')
        eos_ids = tuple(eos_ids)
        for token in eos_ids:
            _integer(token, 'eos_id', 2**31-1)
        runtime = decoder._runtime
        out = C.c_uint64()
        with runtime._lock:
            runtime._call('ml_generation_start', decoder._require_open(), sampler._require_open(),
                          input_token, max_tokens, (C.c_int32*len(eos_ids))(*eos_ids),
                          len(eos_ids), temperature, top_p, top_k or 0, queue_capacity, C.byref(out))
        result = cls(runtime, out.value)
        result._owners = (decoder, sampler)
        result._close_lock = threading.Lock()
        result.final = None
        return result

    def _control(self, name, *args):
        # No Runtime._lock: cancellation must not wait for a concurrent poll,
        # close/join, or a Python caller blocked inside a native GPU step.
        status = getattr(self._runtime._lib, name)(*args)
        if status:
            raise NativeError(status, self._runtime._lib.ml_last_error().decode('utf-8', 'replace'))

    def poll(self, timeout_ms=0):
        _integer(timeout_ms, 'timeout_ms', 120000)
        event = GenerationEvent()
        self._control('ml_generation_poll', self._require_open(), timeout_ms, C.byref(event))
        return event if event.kind else None

    def wait(self, timeout_ms=1000):
        return self.poll(timeout_ms)

    @property
    def state(self):
        event, queued, done = GenerationEvent(), C.c_uint64(), C.c_int32()
        self._control('ml_generation_state', self._require_open(), C.byref(event),
                      C.byref(queued), C.byref(done))
        return event, queued.value, bool(done.value)

    def cancel(self):
        self._control('ml_generation_cancel', self._require_open())

    def close(self):
        if not getattr(self, '_handle', 0):
            return getattr(self, 'final', None)
        with self._close_lock:
            if self._handle:
                event = GenerationEvent()
                self._control('ml_generation_close', self._handle, C.byref(event))
                self.final = event
                self._handle = 0
                self._owners = ()
        return self.final
