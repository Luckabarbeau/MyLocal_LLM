"""0080AH+AI: NumPy oracle, CPU execution of GPU tiling, manual VJP/causality.

CuPy numerical smoke tests live in the separate GPU-only module.  These tests
run without CUDA but execute the identical streamed math used by the GPU.
"""
import numpy as np
import pytest

from mini_llm.backend import BACKEND_NAME
from mini_llm.block_proposal.soft_vocab_read import SoftVocabularyRead
from mini_llm.parameter import Parameter

pytestmark = pytest.mark.skipif(BACKEND_NAME != 'numpy', reason='NumPy oracle')


def _case(scope='own', k=4, vocab=13, beta=1.4, temperature=1.3, seed=40):
    rg = np.random.default_rng(seed)
    emb = rg.normal(scale=.7, size=(vocab, 8)).astype(np.float64)
    states = rg.normal(size=(2, k, 6)).astype(np.float64)
    logits = rg.normal(size=(2, k, vocab)).astype(np.float64)
    upstream = rg.normal(size=(2, k, 6)).astype(np.float64)
    reads = []
    for execution in ('reference', 'tiled'):
        param = Parameter(emb.copy(), name='main.embedding.W')
        obj = SoftVocabularyRead(param, 6, n_heads=2, source_scope=scope,
                                 block_size=k, temperature=temperature,
                                 read_scale=beta, execution=execution,
                                 vocab_tile=4, block_tile=1)
        obj.output_projection.W.data[:] = .037
        # exact same projections in both copies
        reads.append(obj)
    reads[1].query_projection.W.data[:] = reads[0].query_projection.W.data
    reads[1].output_projection.W.data[:] = reads[0].output_projection.W.data
    if scope == 'all':
        reads[0].source_bias.data[:] = rg.normal(size=(2,k,k))*.1
        reads[1].source_bias.data[:] = reads[0].source_bias.data
    return reads, states, logits, upstream


@pytest.mark.parametrize('scope', ['own','all'])
@pytest.mark.parametrize('k', [1,4])
@pytest.mark.parametrize('beta', [0., 1.4])
def test_tiled_same_as_dense_on_every_gradient(scope, k, beta):
    reads, states, logits, grad = _case(scope,k,beta=beta)
    results = []
    for obj in reads:
        out,cache=obj.forward(states,logits)
        ds,dl=obj.backward(grad,cache)
        results.append((out,cache.read,ds,dl,obj.embedding.grad.copy(),
                        [p.grad.copy() for p in obj.parameters()]))
    for a,b in zip(results[0][:5],results[1][:5]):
        np.testing.assert_allclose(a,b,rtol=2e-12,atol=5e-13)
    for a,b in zip(results[0][5],results[1][5]):
        np.testing.assert_allclose(a,b,rtol=2e-12,atol=5e-13)


def _fd(array, coord, f, eps=1e-5):
    orig = array[coord]
    try:
        array[coord] = orig+eps
        plus=f()
        array[coord] = orig-eps
        minus=f()
    finally:
        array[coord] = orig
    return (plus-minus)/(2*eps)


@pytest.mark.parametrize('execution', ['reference','tiled'])
def test_all_source_cross_horizon_fd_and_source_bias(execution):
    reads, states, logits, grad = _case('all',k=4,vocab=7,beta=1.1)
    obj=reads[0] if execution=='reference' else reads[1]
    obj.zero_grad()
    obj.embedding.zero_grad()
    output,cache=obj.forward(states,logits)
    ds,dl=obj.backward(grad,cache)
    def loss():
        return float(np.sum(obj.forward(states,logits)[0]*grad))
    for index in ((0,0,0),(1,3,4),(0,2,6)):
        np.testing.assert_allclose(dl[index], _fd(logits,index,loss),atol=2e-10,rtol=1e-5)
    for index in ((0,1,0),(1,2,4)):
        np.testing.assert_allclose(ds[index], _fd(states,index,loss),atol=2e-10,rtol=1e-5)
    for index in ((2,0),(5,7)):
        np.testing.assert_allclose(obj.embedding.grad[index],
                                   _fd(obj.embedding.data,index,loss),atol=2e-10,rtol=1e-5)
    for index in ((0,0),(5,7)):
        np.testing.assert_allclose(obj.query_projection.W.grad[index],
                                   _fd(obj.query_projection.W.data,index,loss),atol=2e-10,rtol=1e-5)
    for index in ((0,0),(7,3)):
        np.testing.assert_allclose(obj.output_projection.W.grad[index],
                                   _fd(obj.output_projection.W.data,index,loss),atol=2e-10,rtol=1e-5)
    for index in ((0,0,2),(1,3,0)):
        np.testing.assert_allclose(obj.source_bias.grad[index],
                                   _fd(obj.source_bias.data,index,loss),atol=2e-10,rtol=1e-5)
    # Contextual cross-horizon gradient: DEC1 queries all 4 P sources.
    only_first=np.zeros_like(grad); only_first[:,0,:]=grad[:,0,:]
    obj.zero_grad(); obj.embedding.zero_grad()
    _,c=obj.forward(states,logits)
    _,cross=obj.backward(only_first,c)
    assert np.any(np.abs(cross[:,3,:])>1e-10)


@pytest.mark.parametrize('scope', ['own','all'])
def test_stable_large_logits_and_source_shift_invariance(scope):
    reads,states,logits,grad=_case(scope,k=4,vocab=17)
    logits=logits*30
    logits[:,0,:] += 10000.
    logits[:,3,:] -= 20000.
    obj=reads[1]
    out,cache=obj.forward(states,logits)
    _,dlogits=obj.backward(grad,cache)
    assert np.all(np.isfinite(out))
    assert np.all(np.isfinite(dlogits))
    if scope=='all':
        changed=logits.copy();changed[:,2,:]+=5432.0
        np.testing.assert_allclose(obj.forward(states,changed)[0],out,rtol=1e-10,atol=1e-10)
        np.testing.assert_allclose(np.sum(dlogits,axis=-1),0,rtol=0,atol=1e-12)


def test_beta_zero_own_matches_expected_embedding():
    reads,states,logits,_=_case('own',beta=0.)
    obj=reads[1]
    _,cache=obj.forward(states,logits)
    score=logits/obj.temperature
    p=np.exp(score-score.max(axis=-1,keepdims=True));p/=p.sum(axis=-1,keepdims=True)
    np.testing.assert_allclose(cache.read,p@obj.embedding.data,rtol=2e-13,atol=2e-13)


def test_all_source_decoder_causality_and_parameter_uniqueness():
    from mini_llm.block_proposal.decoder import BlockPrefixDecoder
    from mini_llm.block_proposal.soft_interface_config import SoftInterfaceConfig
    from mini_llm.model.decoder_lm import DecoderLanguageModel
    from test_block_proposal_0080l_baselines import _configuration
    main=DecoderLanguageModel(_configuration(),rng_seed=8080,dtype='float64')
    dec=BlockPrefixDecoder(main,4,width=12,n_layers=2,n_heads=3,rng_seed=8082,
        soft_interface_config=SoftInterfaceConfig(mode='full_vocab',source_scope='all'),
        experimental_cpu_full_vocab=True)
    dec.vocab_read.output_projection.W.data[:]=.01
    rg=np.random.default_rng(31)
    logits=rg.normal(size=(1,4,101))
    target=rg.integers(0,101,size=(1,4))
    valid=np.ones_like(target,dtype=bool)
    out,_=dec.forward(logits,target,valid)
    for pos in range(4):
        changed=target.copy();changed[:,pos]=(changed[:,pos]+9)%101
        test,_=dec.forward(logits,changed,valid)
        np.testing.assert_array_equal(out[:,:pos+1],test[:,:pos+1])
    params=dec.parameters()
    assert len({id(p) for p in params})==len(params)
    assert len({p.name for p in params})==len(params)
    assert any(p.name.endswith('source_bias') for p in params)


def test_full_65280_vocabulary_all_source_streams_without_dense_attention_cache():
    rg=np.random.default_rng(920)
    vocab=65280
    param=Parameter(rg.normal(size=(vocab,8)).astype('float32'),name='E')
    read=SoftVocabularyRead(param,6,n_heads=2,source_scope='all',block_size=4,
                            execution='tiled',vocab_tile=1024,block_tile=1)
    read.output_projection.W.data[:]=.02
    x=rg.normal(size=(1,4,6)).astype('float32')
    logits=rg.normal(size=(1,4,vocab)).astype('float32')
    y,cache=read.forward(x,logits)
    assert y.shape==x.shape
    assert cache.attention_norm.shape==(1,4,2)
    assert cache.source_norm.shape==(1,4)
    assert cache.read.shape==(1,4,8)
    assert not any(getattr(item,'shape',())[-2:]==(vocab,4)
                   for item in vars(cache).values())
    dx,dl=read.backward(np.ones_like(x),cache)
    assert dx.shape==x.shape
    assert dl.shape==logits.shape
    assert np.all(np.isfinite(dl))
    assert np.any(abs(dl[0,3,:])>1e-10)


def test_all_source_training_one_main_backward_and_cross_horizon_gradients():
    from mini_llm.block_proposal.decoder import BlockPrefixDecoder, TwoStageBlockModel
    from mini_llm.block_proposal.soft_interface_config import SoftInterfaceConfig
    from mini_llm.block_proposal.proposals import HorizonProposalHead
    from mini_llm.block_proposal.training import SinglePassBlockTrainer
    from mini_llm.model.decoder_lm import DecoderLanguageModel
    from test_block_proposal_0080l_baselines import _configuration, _window
    model=DecoderLanguageModel(_configuration(),rng_seed=8080,dtype='float64')
    head=HorizonProposalHead(model,8,hidden_width=12,rng_seed=8081)
    decoder=BlockPrefixDecoder(model,8,width=12,n_layers=2,n_heads=3,rng_seed=8082,
        soft_interface_config=SoftInterfaceConfig(mode='full_vocab',source_scope='all'),
        experimental_cpu_full_vocab=True)
    decoder.vocab_read.output_projection.W.data[:]=.03
    net=TwoStageBlockModel(head,decoder)
    source,targets,valid,meta=_window()
    result=SinglePassBlockTrainer(net).train_window(
        source,targets,valid_target_mask=valid,memory_metadata=meta,block_group_size=2)
    assert result['main_forward_passes']==1
    assert result['main_backward_passes']==1
    assert result['unique_supervised_tokens']==13
    assert np.any(np.abs(decoder.vocab_read.source_bias.grad)>0)
    assert np.any(np.abs(decoder.vocab_read.query_projection.W.grad)>0)
