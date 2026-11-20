"""mincore census counts payload edge bytes, not whole overlapping pages."""
import mmap
import os
import numpy as np
import pytest
from scripts.probe_0308_page_residency import resident_bytes,tensor_group


def test_resident_edges_and_chunks():
    page=os.sysconf('SC_PAGE_SIZE')
    owner=mmap.mmap(-1,4*page,flags=mmap.MAP_PRIVATE|mmap.MAP_ANONYMOUS)
    raw=np.frombuffer(owner,np.uint8)
    try:
        raw[:]=123
        address=int(raw.ctypes.data)
        assert resident_bytes(address,4*page,chunk_pages=1)==4*page
        assert resident_bytes(address+13,3*page+29,chunk_pages=2)==3*page+29
        assert resident_bytes(address+page-1,2,chunk_pages=1)==2
        assert resident_bytes(address,0)==0
        assert (raw==123).all()
    finally:
        del raw; owner.close()


@pytest.mark.parametrize('name,expected',[('per_layer_token_embd.weight','ple'),
    ('blk.0.ple_key.weight','ple'),('blk.0.ffn_down_exps.weight','routed_experts'),
    ('token_embd.weight','other')])
def test_tensor_groups(name,expected): assert tensor_group(name)==expected


@pytest.mark.parametrize('address,length,chunk',[(0,1,1),(1,-1,1),(True,1,1),(1,1,8193),(1,1,True)])
def test_invalid(address,length,chunk):
    with pytest.raises(ValueError): resident_bytes(address,length,chunk_pages=chunk)
