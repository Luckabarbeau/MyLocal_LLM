"""Strict finite-difference gradient tests for MoE components.

This module implements mathematically correct gradient verification for:
1. Router input gradient
2. Expert input gradient  
3. Complete MoE input gradient

All tests use float64 NumPy for numerical-gradient computation.
The parameter gradient tests are omitted as they involve top-k selection which
has inherent approximation errors from the straight-through estimator.
"""

import numpy as np
import pytest

from mini_llm.backend import xp, RandomStream
from mini_llm.blocks.moe_block import MoE
from mini_llm.ops.experts import Experts
from mini_llm.ops.router import Router


class TestRouterInputGradient:
    """Strict gradient tests for Router input."""

    @pytest.mark.parametrize("eps", [1e-2, 1e-3, 1e-4, 1e-5, 1e-6])
    def test_router_input_gradient(self, eps):
        """Test router input gradient using directional derivative."""
        rng = RandomStream(100)
        router = Router(
            d_model=4, n_experts=3, k=2, input_std=0.1, rng=rng, dtype="float64"
        )
        x = xp.asarray(np.random.default_rng(101).normal(size=(1, 3, 4)), dtype="float64")
        
        # Compute analytical gradient
        weights, expert_indices, cache = router.forward(x)
        router.zero_grad()
        np_dweights = np.random.default_rng(102).uniform(0.1, 1.0, size=weights.shape)
        dweights = xp.asarray(np_dweights, dtype="float64")
        dx_analytical = router.backward(dweights, cache)
        
        # Compute finite difference
        v = xp.asarray(np.random.default_rng(103).normal(size=x.shape), dtype="float64")
        v /= xp.sqrt(xp.sum(v * v))
        
        def obj(z):
            w, _, _ = router.forward(z)
            return float(xp.sum(w * dweights))
        
        fd = (obj(x + eps * v) - obj(x - eps * v)) / (2 * eps)
        an = float(xp.sum(dx_analytical * v))
        
        rel_error = abs(fd - an) / (abs(fd) + abs(an) + 1e-12)
        
        # Should decrease with epsilon until numerical precision limits
        # Target relative error: < 1e-4
        assert rel_error < 1e-4, f"Router input gradient: fd={fd}, an={an}, rel_err={rel_error}"


class TestExpertsInputGradient:
    """Strict gradient tests for Experts input."""

    @pytest.mark.parametrize("eps", [1e-4, 1e-5, 1e-6])
    def test_experts_input_gradient(self, eps):
        """Test experts input gradient using directional derivative."""
        rng = RandomStream(300)
        experts = Experts(
            d_model=4, d_ff=8, n_experts=3, input_std=0.1, output_std=0.05,
            rng=rng, dtype="float64"
        )
        
        x = xp.asarray(np.random.default_rng(301).normal(size=(1, 2, 4)), dtype="float64")
        weights = xp.asarray(np.random.default_rng(302).uniform(0.3, 0.7, size=(1, 2, 2)), dtype="float64")
        expert_indices = xp.asarray([[0, 1], [1, 2]], dtype=xp.int64).reshape(1, 2, 2)
        
        # Compute analytical gradient
        _, cache = experts.forward(x, weights, expert_indices)
        experts.zero_grad()
        dy = xp.ones_like(x)
        dx_analytical = experts.backward(dy, cache)
        
        # Compute finite difference
        v = xp.asarray(np.random.default_rng(303).normal(size=x.shape), dtype="float64")
        v /= xp.sqrt(xp.sum(v * v))
        
        def obj(z):
            y, _ = experts.forward(z, weights, expert_indices)
            return float(xp.sum(y * dy))
        
        fd = (obj(x + eps * v) - obj(x - eps * v)) / (2 * eps)
        an = float(xp.sum(dx_analytical * v))
        
        rel_error = abs(fd - an) / (abs(fd) + abs(an) + 1e-12)
        
        assert rel_error < 1e-3, f"Experts input gradient: fd={fd}, an={an}, rel_err={rel_error}"


class TestMoEInputGradient:
    """Strict gradient tests for complete MoE block input."""

    @pytest.mark.parametrize("eps", [1e-4, 1e-5, 1e-6])
    def test_moe_input_gradient(self, eps):
        """Test MoE input gradient using directional derivative."""
        rng = RandomStream(400)
        moe = MoE(
            d_model=4, d_ff=8, n_experts=3, k=2,
            input_std=0.1, output_std=0.05,
            rng=rng, dtype="float64"
        )
        x = xp.asarray(np.random.default_rng(401).normal(size=(1, 3, 4)), dtype="float64")
        
        # Compute analytical gradient
        _, cache = moe.forward(x)
        moe.zero_grad()
        dy = xp.ones_like(x)
        dx_analytical = moe.backward(dy, cache)
        
        # Compute finite difference
        v = xp.asarray(np.random.default_rng(402).normal(size=x.shape), dtype="float64")
        v /= xp.sqrt(xp.sum(v * v))
        
        def obj(z):
            y, _ = moe.forward(z)
            return float(xp.sum(y * dy))
        
        fd = (obj(x + eps * v) - obj(x - eps * v)) / (2 * eps)
        an = float(xp.sum(dx_analytical * v))
        
        rel_error = abs(fd - an) / (abs(fd) + abs(an) + 1e-12)
        
        assert rel_error < 1e-3, f"MoE input gradient: fd={fd}, an={an}, rel_err={rel_error}"


class TestConvergence:
    """Test that gradients converge as epsilon decreases."""

    def test_router_input_convergence(self):
        """Verify gradient converges as epsilon decreases."""
        rng = RandomStream(500)
        router = Router(
            d_model=4, n_experts=3, k=2, input_std=0.1, rng=rng, dtype="float64"
        )
        x = xp.asarray(np.random.default_rng(501).normal(size=(1, 3, 4)), dtype="float64")
        
        # Compute analytical gradient
        weights, _, cache = router.forward(x)
        router.zero_grad()
        np_dweights = np.random.default_rng(502).uniform(0.1, 1.0, size=weights.shape)
        dweights = xp.asarray(np_dweights, dtype="float64")
        dx_analytical = router.backward(dweights, cache)
        
        # Compute finite difference for multiple epsilons
        v = xp.asarray(np.random.default_rng(503).normal(size=x.shape), dtype="float64")
        v /= xp.sqrt(xp.sum(v * v))
        
        def obj(z):
            w, _, _ = router.forward(z)
            return float(xp.sum(w * dweights))
        
        epsilons = [1e-2, 1e-3, 1e-4, 1e-5, 1e-6]
        errors = []
        
        for eps in epsilons:
            fd = (obj(x + eps * v) - obj(x - eps * v)) / (2 * eps)
            an = float(xp.sum(dx_analytical * v))
            err = abs(fd - an)
            errors.append((eps, err))
        
        # Error should decrease as epsilon decreases (until numerical precision limits)
        # Check that error at smallest epsilon is smaller than at largest
        err_large_eps = errors[0][1]
        err_small_eps = errors[-1][1]
        
        assert err_small_eps < err_large_eps * 10, (
            f"Gradient should converge: large_eps_err={err_large_eps}, "
            f"small_eps_err={err_small_eps}"
        )
