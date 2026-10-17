"""RTX 5080 smoke/gradient checks, opt in with MINI_LLM_BACKEND=cupy.

These tests MUST run on the actual CUDA workstation.  The NumPy CI host cannot
certify BF16 kernel compatibility, peak VRAM or physical training throughput.
"""
import numpy as np
import pytest
from mini_llm.backend import BACKEND_NAME, xp
from mini_llm.parameter import Parameter
from mini_llm.block_proposal.soft_vocab_read import SoftVocabularyRead

pytestmark = pytest.mark.skipif(BACKEND_NAME != 'cupy', reason='requires CUDA CuPy')


def _reference(emb, wq, wo, bias, states, logits, beta=1.3, temperature=1.1, heads=2):
    g,k,v=logits.shape;dh=emb.shape[-1]//heads
    q=(states@wq).reshape(g,k,heads,dh)
    result=np.empty((g,k,emb.shape[-1]),dtype=np.float64)
    for m in range(heads):
        section=slice(m*dh,(m+1)*dh)
        e=emb[:,section]
        l=logits/temperature
        high=np.max(l,axis=-1,keepdims=True)
        log_prior=l-high-np.log(np.sum(np.exp(l-high),axis=-1,keepdims=True))
        score=(log_prior[:,None,:,:]+bias[m][None,:,:,None]+(beta/np.sqrt(dh))*(q[:,:,m,:]@e.T)[:,:,None,:])
        score=score.reshape(g,k,-1)
        p=np.exp(score-score.max(axis=-1,keepdims=True));p/=p.sum(axis=-1,keepdims=True)
        result[:,:,section]=p.reshape(g,k,k,v).sum(axis=2)@e
    return result@wo


def test_cupy_fp32_all_source_against_dense_numpy_and_selected_fd():
    rg=np.random.default_rng(91)
    emb=rg.normal(size=(17,8)).astype('float32')
    states=rg.normal(size=(2,4,6)).astype('float32')
    logits=rg.normal(size=(2,4,17)).astype('float32')
    obj=SoftVocabularyRead(Parameter(xp.asarray(emb),name='E'),6,source_scope='all',
                           block_size=4,n_heads=2,execution='tiled',vocab_tile=5,block_tile=1,
                           temperature=1.1,read_scale=1.3)
    obj.output_projection.W.data[:]=.04
    obj.source_bias.data[:]=xp.asarray(rg.normal(scale=.1,size=(2,4,4)).astype('float32'))
    out,cache=obj.forward(xp.asarray(states),xp.asarray(logits))
    ref=_reference(emb,xp.asnumpy(obj.query_projection.W.data),
        xp.asnumpy(obj.output_projection.W.data),xp.asnumpy(obj.source_bias.data),
        states.astype('float64'),logits.astype('float64'))
    np.testing.assert_allclose(xp.asnumpy(out),ref,rtol=1e-4,atol=1e-5)
    upstream=xp.asarray(rg.normal(size=states.shape).astype('float32'))
    obj.zero_grad();obj.embedding.zero_grad()
    dstate,dlogit=obj.backward(upstream,cache)
    assert dstate.shape==states.shape and dlogit.shape==logits.shape
    assert xp.asnumpy(xp.all(xp.isfinite(dlogit)))
    assert xp.asnumpy(xp.any(xp.abs(dlogit[:,3,:])>1e-8))
    np.testing.assert_allclose(xp.asnumpy(xp.sum(dlogit,axis=-1)),0,atol=2e-6)
    # Spot-check dL for a distinct source of DEC_1's only loss.
    one=xp.zeros_like(upstream);one[:,0,:]=upstream[:,0,:]
    obj.zero_grad();obj.embedding.zero_grad()
    dstate,cross=obj.backward(one,cache)
    assert xp.asnumpy(xp.any(xp.abs(cross[:,3,:])>1e-8))


@pytest.mark.parametrize('source_scope', ['own','all'])
def test_cupy_bf16_full_vocabulary_memory_bounded(source_scope):
    import ml_dtypes
    rg=np.random.default_rng(36)
    v,k,g,d=65280,4,4,8
    emb=Parameter(xp.asarray(rg.normal(size=(v,d)).astype('float32')).astype(ml_dtypes.bfloat16),name='E')
    obj=SoftVocabularyRead(emb,6,n_heads=2,source_scope=source_scope,block_size=k,
        execution='tiled',vocab_tile=2048,block_tile=2)
    obj.output_projection.W.data[:]=xp.asarray(np.full(obj.output_projection.W.shape,.03,dtype='float32')).astype(ml_dtypes.bfloat16)
    states=xp.asarray(rg.normal(size=(g,k,6)).astype('float32'))
    logits=xp.asarray(rg.normal(size=(g,k,v)).astype('float32')).astype(ml_dtypes.bfloat16)
    pool=xp.get_default_memory_pool()
    before=pool.used_bytes()
    correction,cache=obj.forward(states,logits)
    dx,dl=obj.backward(xp.ones_like(correction),cache)
    assert correction.shape==(g,k,6)
    assert dx.shape==(g,k,6)
    assert dl.shape==(g,k,v)
    assert str(dl.dtype)=='float32'
    assert bool(xp.asnumpy(xp.all(xp.isfinite(dl))))
    assert bool(xp.asnumpy(xp.all(xp.isfinite(emb.grad))))
    # With d=8 and g=4, a 512 MiB peak would indicate bad broadcasting.
    # This is a *module-only* scratch bound, not full 500M model headroom.
    assert pool.used_bytes()-before < 512*1024*1024


@pytest.mark.parametrize('source_scope', ['own','all'])
def test_cupy_bf16_teacher_forced_decoder_single_pass(source_scope):
    """Exercises actual BF16 decoder layers and residual VJP on CUDA."""
    import ml_dtypes
    from mini_llm.block_proposal.decoder import BlockPrefixDecoder
    from mini_llm.block_proposal.soft_interface_config import SoftInterfaceConfig
    from mini_llm.model.decoder_lm import DecoderLanguageModel
    from test_block_proposal_0080l_baselines import _configuration
    import numpy as np
    model=DecoderLanguageModel(_configuration(),rng_seed=8080,dtype='bfloat16')
    dec=BlockPrefixDecoder(model,4,width=12,n_layers=2,n_heads=3,rng_seed=8082,
        soft_interface_config=SoftInterfaceConfig(mode='full_vocab',source_scope=source_scope),
        experimental_gpu_full_vocab=True)
    dec.vocab_read.output_projection.W.data[:]=xp.asarray(.02,dtype=ml_dtypes.bfloat16)
    rg=np.random.default_rng(5080)
    logits=xp.asarray(rg.normal(size=(2,4,101)).astype('float32')).astype(ml_dtypes.bfloat16)
    targets=xp.asarray(rg.integers(0,101,size=(2,4)))
    valid=xp.ones((2,4),dtype=xp.bool_)
    dec.zero_grad();model.embedding.W.zero_grad()
    final,cache=dec.forward(logits,targets,valid)
    d_prior=dec.backward(xp.ones(final.shape,dtype=xp.float32)*.01,cache)
    assert d_prior.shape==logits.shape
    assert bool(xp.asnumpy(xp.all(xp.isfinite(d_prior.astype(xp.float32)))))
    assert bool(xp.asnumpy(xp.any(xp.abs(dec.vocab_read.output_projection.W.grad)>0)))
    assert bool(xp.asnumpy(xp.any(xp.abs(model.embedding.W.grad)>0)))
