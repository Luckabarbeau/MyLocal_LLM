from mini_llm.backend import xp, RandomStream
from mini_llm.ops.embedding import Embedding


def test_embedding_repeated_id_accumulates_gradient():
    rng = RandomStream(30)
    emb = Embedding(10, 4, 0.02, rng, dtype="float64")
    ids = xp.asarray([[2,2,5]], dtype="int64")
    y, cache = emb.forward(ids)
    emb.backward(xp.ones_like(y), cache)

    assert float(xp.max(xp.abs(emb.W.grad[2]-2.0))) == 0.0
    assert float(xp.max(xp.abs(emb.W.grad[5]-1.0))) == 0.0
