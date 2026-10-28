import math

import numpy as np

from mini_llm.backend import xp
from mini_llm.optim.grad_clip import audit_gradients, global_grad_norm
from mini_llm.parameter import Parameter


def _parameter(name, gradient):
    parameter = Parameter(xp.zeros(len(gradient), dtype="float32"), name=name)
    parameter.grad[...] = xp.asarray(gradient, dtype="float32")
    return parameter


def test_audit_distinguishes_fp32_sum_of_squares_overflow():
    parameter = _parameter("large_finite", [3.0e20, 4.0e20])

    _, ordinary_norm = global_grad_norm([parameter])
    audit = audit_gradients([parameter])

    assert math.isinf(ordinary_norm)
    assert audit["all_finite"] is True
    assert audit["nonfinite"] == []
    assert np.isclose(audit["stable_norm"], 5.0e20, rtol=1.0e-6)
    assert audit["top"][0]["name"] == "large_finite"


def test_audit_identifies_nonfinite_parameter():
    finite = _parameter("finite", [3.0, 4.0])
    nonfinite = _parameter("bad", [1.0, float("inf"), float("nan")])

    audit = audit_gradients([finite, nonfinite])

    assert audit["all_finite"] is False
    assert audit["stable_norm"] is None
    assert audit["nonfinite"] == [{
        "name": "bad",
        "nonfinite": 2,
        "nan": 1,
        "inf": 1,
        "max_finite_abs": 1.0,
    }]
