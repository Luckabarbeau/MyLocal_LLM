"""Active-state hashing and opt-in instrumentation contracts, no model loading."""
from pathlib import Path
from types import SimpleNamespace as NS
import sys
import numpy as np
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'scripts'))
from probe_0227_short_sequences_and_profile import digest_state,Profile


def test_digest_compares_all_active_state_and_ignores_unwritten_kv():
    s=NS(length=2,recurrent={0:NS(delta=NS(length=2,memory=np.zeros((1,2,3,3),np.float32)),
          conv=NS(length=2,history=np.zeros((1,3,5),np.float32)))},
         qsa={3:NS(length=2,keys=np.zeros((16,1,4),np.float32),values=np.zeros((16,1,4),np.float32),
          raw_index=np.zeros((16,4),np.float32),pooled=np.zeros((5,4),np.float32))},
         ple={1:NS(normalized_history=np.zeros((9,4,2),np.float32),predecessors=(1,2))})
    baseline=digest_state(s,np.asarray)
    s.qsa[3].keys[15]=100
    assert digest_state(s,np.asarray)==baseline
    s.recurrent[0].delta.memory[0,0,1,1]=1
    assert digest_state(s,np.asarray)!=baseline
    s.recurrent[0].delta.memory.fill(0)
    s.ple[1].predecessors=(2,3)
    assert digest_state(s,np.asarray)!=baseline


def test_profile_does_not_touch_unprofiled_warmup():
    p=Profile(None);calls=[]
    wrapped=p.wrap(lambda x:calls.append(x) or x*2,0,'unit')
    assert wrapped(3)==6 and calls==[3] and p.records==[]
