from mini_llm.backend import xp
from mini_llm.parameter import Parameter
from mini_llm.optim.adamw import AdamW


def test_adamw_first_step_matches_formula():
    p = Parameter(xp.asarray([1.0,-2.0], dtype="float64"), decay=True)
    p.grad[...] = xp.asarray([0.5,-0.25], dtype="float64")
    opt = AdamW([p], lr=1e-3, beta1=0.9, beta2=0.95, eps=1e-8, weight_decay=0.1)

    old, g = p.data.copy(), p.grad.copy()
    expected = old*(1.0-1e-3*0.1) - 1e-3*g/(xp.sqrt(g*g)+1e-8)
    opt.step()
    assert float(xp.max(xp.abs(p.data-expected))) < 1e-12
