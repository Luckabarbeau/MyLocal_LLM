"""Bounded current-stream replay of an array-only inference computation.

The enqueue callback receives private input/state buffers. Warmup and capture
never mutate caller state. A private pool owns all capture intermediates, even
when their Python array wrappers disappear; it is never used for replay-time
external allocations. Architecture adapters select replay for eligible calls.
"""

import numpy as np

from mini_llm.backend import BACKEND_NAME, xp


class CapturedArrayInference:
    def __init__(self, enqueue, x, mutable_arrays, *, pool_limit_bytes=32 * 1024**2):
        if BACKEND_NAME != 'cupy':
            raise RuntimeError('CUDA graph replay requires CuPy')
        arrays = (x, *mutable_arrays)
        if not arrays or any(not isinstance(a, xp.ndarray) or not a.flags.c_contiguous
                             for a in arrays):
            raise ValueError('Graph operands must be contiguous backend arrays')
        device = xp.cuda.runtime.getDevice()
        if any(a.device.id != device for a in arrays):
            raise ValueError('Graph operands must be on the current device')
        self.device = device
        self.signature = tuple((a.shape, a.dtype.str) for a in arrays)
        self.pool = xp.cuda.MemoryPool()
        self.pool.set_limit(size=int(pool_limit_bytes))
        self.capture_stream = xp.cuda.Stream(non_blocking=True)
        ready = xp.cuda.Event(disable_timing=True)
        ready.record(xp.cuda.get_current_stream())
        self.capture_stream.wait_event(ready)
        with self.capture_stream, xp.cuda.using_allocator(self.pool.malloc):
            self.input = x.copy()
            self.mutable = tuple(a.copy() for a in mutable_arrays)
            # Warm every high-level operation/toolchain on private state first.
            warm_output = enqueue(self.input, self.mutable)
            self.capture_stream.synchronize()
            # Capture starts from caller values, not the warmed private state.
            for target, source in zip(self.mutable, mutable_arrays):
                xp.copyto(target, source)
            self.capture_stream.synchronize()
            self.capture_stream.begin_capture()
            try:
                self.output = enqueue(self.input, self.mutable)
                self.graph = self.capture_stream.end_capture()
            except Exception:
                if self.capture_stream.is_capturing():
                    self.capture_stream.end_capture()
                raise
            self.graph.upload(stream=self.capture_stream)
            self.capture_stream.synchronize()
            # Keep warm arrays live until capture/upload is complete.
            del warm_output
        self.calls = 0

    def matches(self, x, mutable_arrays):
        arrays = (x, *mutable_arrays)
        return (xp.cuda.runtime.getDevice() == self.device and
                len(arrays) == len(self.signature) and
                all(isinstance(a, xp.ndarray) and a.flags.c_contiguous and
                    a.device.id == self.device and (a.shape, a.dtype.str) == spec
                    for a, spec in zip(arrays, self.signature)))

    def accepts_current_stream(self):
        """Metadata-only eligibility check; never migrate live graph scratch."""
        return not self.calls or xp.cuda.get_current_stream().ptr == self._replay_stream

    def forward(self, x, mutable_arrays):
        if not self.matches(x, mutable_arrays):
            raise ValueError('Graph operand shape/dtype/device/layout changed')
        stream = xp.cuda.get_current_stream()
        # The adapter's serial execution contract orders graph replay on one
        # stream. Reject cross-stream reuse rather than overwrite live scratch.
        if not self.accepts_current_stream():
            raise ValueError('A captured inference call must reuse its replay stream')
        self._replay_stream = stream.ptr
        xp.copyto(self.input, x)
        for target, source in zip(self.mutable, mutable_arrays):
            xp.copyto(target, source)
        self.graph.launch(stream=stream)
        for target, source in zip(mutable_arrays, self.mutable):
            xp.copyto(target, source)
        # Return fresh ownership: subsequent replays cannot overwrite outputs.
        out = self.output.copy()
        self.calls += 1
        return out

    def statistics(self):
        return dict(calls=self.calls, pool_reserved_bytes=int(self.pool.total_bytes()),
                    pool_used_bytes=int(self.pool.used_bytes()),
                    pool_limit_bytes=int(self.pool.get_limit()))
