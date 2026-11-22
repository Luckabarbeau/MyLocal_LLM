from collections import deque
from types import SimpleNamespace
import weakref

from mini_llm.ops import gated_delta_decode


def test_completed_chunk_owners_retire_before_new_state_admission(monkeypatch):
    class Owner:
        pass
    owner = Owner()
    reference = weakref.ref(owner)
    queued = deque([(SimpleNamespace(done=True),(owner,))])
    blocked = deque([(SimpleNamespace(done=False),(object(),)),
                     (SimpleNamespace(done=True),(object(),))])
    monkeypatch.setattr(gated_delta_decode,'_PENDING',{(0,1):queued,(0,2):blocked})
    del owner
    assert gated_delta_decode.retire_completed_scratch()==1
    assert not queued
    assert reference() is None
    assert len(blocked)==2  # Never release a producer still in flight.
    blocked[0][0].done=True
    assert gated_delta_decode.retire_completed_scratch()==2
    assert not blocked
    assert gated_delta_decode.retire_completed_scratch()==0
