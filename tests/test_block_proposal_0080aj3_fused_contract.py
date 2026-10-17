"""0080AJ3 CPU contract: fused CUDA is opt-in, NumPy uses batched oracle."""
import numpy as np
import pytest

from mini_llm.backend import BACKEND_NAME, xp
from mini_llm.parameter import Parameter
from mini_llm.block_proposal.soft_vocab_read import SoftVocabularyRead

pytestmark = pytest.mark.skipif(BACKEND_NAME != 'numpy', reason='NumPy contract')


@pytest.mark.parametrize('k', [1, 4])
@pytest.mark.parametrize('vocab_tile', [1, 7, 23])
@pytest.mark.parametrize('beta', [0.0, 1.25])
def test_fused_selector_matches_batched_numpy_and_all_gradients(k, vocab_tile, beta):
    rng = np.random.default_rng(823)
    g, v, h, dh, decoder_width = 3, 23, 4, 3, 7
    embedding = Parameter(rng.normal(size=(v, h*dh)), name='shared.E')
    read = SoftVocabularyRead(embedding, decoder_width, n_heads=h,
                              block_size=k, source_scope='all', execution='tiled',
                              vocab_tile=vocab_tile, block_tile=2,
                              temperature=1.13, read_scale=beta,
                              all_source_algorithm='batched_mixture')
    read.output_projection.W.data[:] = .02
    read.source_bias.data[:] = rng.normal(scale=.4, size=(h,k,k))
    states = rng.normal(size=(g,k,decoder_width))
    logits = rng.normal(size=(g,k,v))
    dy = rng.normal(size=states.shape)
    results=[]
    for algorithm in ('batched_mixture','fused_mixture'):
        read.all_source_algorithm=algorithm
        read.zero_grad(); embedding.zero_grad()
        y, cache = read.forward(states, logits)
        dx, dl = read.backward(dy, cache)
        results.append([a.copy() for a in (y, dx, dl, embedding.grad)] +
                       [p.grad.copy() for p in read.parameters()])
    for a,b in zip(*results):
        np.testing.assert_allclose(a,b,rtol=0,atol=0)


def test_algorithm_cannot_change_between_forward_and_backward():
    rng = np.random.default_rng(823)
    emb = Parameter(rng.normal(size=(13,12)), name='E')
    read = SoftVocabularyRead(emb, 5, n_heads=4, source_scope='all', block_size=4,
                              execution='tiled', all_source_algorithm='fused_mixture')
    out,cache=read.forward(np.ones((2,4,5)), rng.normal(size=(2,4,13)))
    read.all_source_algorithm='batched_mixture'
    with pytest.raises(ValueError, match='changed'):
        read.backward(np.ones_like(out),cache)


def test_fused_requires_tiled_all_scope():
    e = Parameter(np.zeros((11,12)), name='E')
    with pytest.raises(ValueError, match='all-source'):
        SoftVocabularyRead(e, 5, n_heads=4, execution='tiled',
                           all_source_algorithm='fused_mixture')
    with pytest.raises(ValueError, match='tiled'):
        SoftVocabularyRead(e, 5, n_heads=4, block_size=4,
                           source_scope='all', execution='reference',
                           all_source_algorithm='fused_mixture')
