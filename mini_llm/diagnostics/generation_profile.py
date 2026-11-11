"""Inclusive host-wall observations of existing generation components.

Use only with serial, exclusive access to the model and its component objects;
concurrent calls, overlapping profilers and shared-object use are unsupported.
No backend imports, array inspection, device events or synchronization occur.
Host times include Python/dispatch work and any waits already inside forward;
they are not isolated GPU kernel times. Nested stages overlap and must not be
summed as model latency.

This file can be loaded directly with importlib without initializing a backend.
Normal dotted imports still execute the repository's existing package initializers.
"""

from time import perf_counter_ns


class HostStageProfiler:
    """Temporarily time component instance ``forward`` methods in ``model.blocks``.

    Set ``phase`` (normally ``'prefill'`` or ``'decode'``) before model calls.
    ``clear()`` resets counters between requests, not the phase or wrappers.
    Call it between forwards, never from within an in-flight forward. Reusing
    the context preserves counters unless explicitly cleared. Missing/noncallable
    forwards and objects without writable instance attributes are skipped.

    Aliased components are wrapped once; their first encountered stage owns all
    calls. Only instance attributes are changed, never classes or model weights.
    """

    _STAGES = ('norm1', 'mixer', 'norm2', 'moe')
    _NESTED_STAGES = ('router', 'experts')

    def __init__(self, model):
        self.model = model
        self.phase = 'unassigned'
        self._phases = {}
        self._patches = []
        self._active = False

    def clear(self):
        """Discard all request counters while retaining phase and instrumentation."""
        self._phases.clear()

    def _wrap(self, forward, stage):
        def timed_forward(*args, **kwargs):
            phase = str(self.phase)
            start = perf_counter_ns()
            try:
                return forward(*args, **kwargs)
            finally:
                elapsed = perf_counter_ns() - start
                stats = self._phases.setdefault(phase, {}).setdefault(stage, [0, 0, 0])
                stats[0] += 1
                stats[1] += elapsed
                stats[2] = max(stats[2], elapsed)
        return timed_forward

    def _instrument(self, component, stage, seen):
        if id(component) in seen:
            return
        forward = getattr(component, 'forward', None)
        attributes = getattr(component, '__dict__', None)
        if not callable(forward) or attributes is None:
            return
        had_forward = 'forward' in attributes
        original = attributes.get('forward')
        try:
            setattr(component, 'forward', self._wrap(forward, stage))
        except (AttributeError, TypeError):
            return
        self._patches.append((component, had_forward, original))
        seen.add(id(component))

    def __enter__(self):
        if self._active:
            raise RuntimeError('HostStageProfiler requires serial, exclusive use')
        self._active = True
        seen = set()
        try:
            blocks = getattr(self.model, 'blocks', ())
            for block in (() if blocks is None else blocks):
                for stage in self._STAGES:
                    self._instrument(getattr(block, stage, None), stage, seen)
                moe = getattr(block, 'moe', None)
                for stage in self._NESTED_STAGES:
                    self._instrument(getattr(moe, stage, None), stage, seen)
        except BaseException:
            self.__exit__(None, None, None)
            raise
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        try:
            while self._patches:
                component, had_forward, original = self._patches.pop()
                if had_forward:
                    setattr(component, 'forward', original)
                else:
                    delattr(component, 'forward')
        finally:
            self._active = False
        return False

    def snapshot(self):
        """Return detached JSON-safe aggregates, including failed forward calls."""
        return {
            'phases': {
                phase: {
                    stage: {
                        'calls': calls,
                        'total_ms': total_ns / 1_000_000,
                        'mean_ms': total_ns / calls / 1_000_000,
                        'max_ms': max_ns / 1_000_000,
                    }
                    for stage, (calls, total_ns, max_ns) in stages.items()
                }
                for phase, stages in self._phases.items()
            },
            'timing_semantics': (
                'Inclusive host wall time measured with perf_counter_ns; nested '
                'stages overlap and are not additive. No device synchronization '
                'or device events; not isolated GPU kernel time.'
            ),
        }
