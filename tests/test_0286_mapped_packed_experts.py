"""Bounded mapped host storage: exact bytes, GPU arithmetic, lease lifetime."""
import gc
import numpy as np
import pytest
from mini_llm.backend import BACKEND_NAME,xp,asnumpy
from mini_llm.ops import mapped_packed_experts as mapped
from mini_llm.ops import mixed_q8_gpu as gpu
from mini_llm.ops.quantized_linear import packed_to_device
from test_0238_packed_expert_slots import weights

pytestmark=pytest.mark.skipif(BACKEND_NAME!='cupy',reason='Actual CUDA mapped memory')


@pytest.mark.parametrize('formats',[
    ('IQ2_S','IQ2_S','IQ4_NL'),('IQ3_XXS','IQ3_XXS','Q2_0'),
    ('IQ3_S','IQ3_S','IQ4_NL'),('IQ4_XS','IQ4_XS','IQ4_NL')])
def test_remote_bytes_dots_and_lifetime(formats):
    source=weights(formats)
    tier=mapped.MappedPackedExpertTier(1024**2,reserve_bytes=0)
    expert=tier.add(0,source)
    x=xp.asarray(np.random.default_rng(286).normal(0,.1,(1,256)).astype(np.float32))
    for host,remote in zip(source,(expert.gate,expert.up,expert.down)):
        np.testing.assert_array_equal(asnumpy(remote.data),host.data.reshape(-1))
        np.testing.assert_array_equal(asnumpy(gpu.linear(x,remote)),asnumpy(gpu.linear(x,packed_to_device(host))))
    stats=tier.statistics()
    assert stats['used_host_bytes']==sum(w.nbytes for w in source)
    assert stats['reserved_host_bytes']<=1024**2 and stats['gpu_weight_bytes']==0
    # An exported projection retains the underlying pinned allocation.
    retained=expert.gate
    xp.cuda.get_current_stream().synchronize()
    tier.close(); del expert,tier; gc.collect()
    np.testing.assert_array_equal(asnumpy(retained.data),source[0].data.reshape(-1))
    np.testing.assert_array_equal(asnumpy(gpu.linear(x,retained)),asnumpy(gpu.linear(x,packed_to_device(source[0]))))


def test_budget_duplicates_and_ram_floor(monkeypatch):
    source=weights()
    with mapped_context(1024**2) as tier:
        tier.add(0,source)
        with pytest.raises(ValueError,match='already'): tier.add(0,source)
        with pytest.raises(TypeError,match='packed'): tier.add(1,source[:2])
        with pytest.raises(MemoryError,match='budget'): 
            for i in range(1,100): tier.add(i,source)
        assert tier.used_bytes<=tier.nbytes
    monkeypatch.setattr(mapped,'available_bytes',lambda:100)
    with pytest.raises(MemoryError,match='RAM'): mapped.MappedPackedExpertTier(1024,reserve_bytes=0)


from contextlib import contextmanager
@contextmanager
def mapped_context(size):
    tier=mapped.MappedPackedExpertTier(size,reserve_bytes=0)
    try: yield tier
    finally:
        xp.cuda.get_current_stream().synchronize()
        tier.close()
