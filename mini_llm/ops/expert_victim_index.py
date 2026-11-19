"""Bounded, handle-free victim ranking for opt-in expert placement.

At request t, log2(decayed_score / bytes) equals
log2(score) - log2(bytes) + last_step / half_life - t / half_life.
The common final term can be omitted without exponentiating (or overflowing).
Floating log rounding, very large timestamps, and exact-score underflow can
change near-tie ordering versus the sort policy: this is not bit-exact policy
parity. Admission must still use the cache's exact current effective score.
"""

from contextlib import contextmanager
import heapq
import math


class ExpertVictimIndex:
    """Lazy generations; only keys and scalars, never cache/weight handles.

    Updates/removals are O(log N) amortized. Compaction alone scans live index
    entries, only when heap storage exceeds 2 * live entries + 64. Iteration
    temporarily removes entries and restores them even on early exit/error.
    Ties use last request then resident insertion order, like stable sorting.
    The cache serializes updates and victim iteration (not thread safe).
    """

    def __init__(self, half_life):
        self.half_life = half_life
        self._heap = []
        self._live = {}
        self._generation = 0
        self._order = 0

    def update(self, key, score, size_bytes, last_step):
        previous = self._live.get(key)
        if previous is None:
            self._order += 1
            order = self._order
        else:
            order = previous[2]
        self._generation += 1
        priority = math.log2(score) - math.log2(size_bytes) + last_step / self.half_life
        entry = (priority, last_step, order, self._generation, key)
        self._live[key] = entry
        heapq.heappush(self._heap, entry)
        self._compact()

    def remove(self, key):
        self._live.pop(key, None)
        self._compact()

    def _compact(self):
        if len(self._heap) > 2 * len(self._live) + 64:
            self._heap = list(self._live.values())
            heapq.heapify(self._heap)

    @contextmanager
    def victims(self):
        held = []

        def ordered():
            while self._heap:
                entry = heapq.heappop(self._heap)
                if self._live.get(entry[-1]) is not entry:
                    continue
                held.append(entry)
                yield entry[-1]

        try:
            yield ordered()
        finally:
            for entry in held:
                if self._live.get(entry[-1]) is entry:
                    heapq.heappush(self._heap, entry)
            self._compact()

    def clear(self):
        self._heap.clear()
        self._live.clear()
        self._generation = 0
        self._order = 0
