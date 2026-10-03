"""Tests for AdamW optimizer with mixed precision."""

import tempfile
from pathlib import Path

import numpy as np
import pytest

from mini_llm.backend import xp
from mini_llm.parameter import Parameter
from mini_llm.optim.adamw import AdamW


class TestAdamWMixedPrecision:
    """Tests for AdamW with FP32 state and master weights."""
    
    def test_fp16_params_use_fp32_adam_state(self):
        """Test that FP16 parameters have FP32 Adam moments and master weights."""
        # Create FP16 parameter
        p = Parameter(
            data=xp.asarray([[1.0, 2.0], [3.0, 4.0]], dtype="float16"),
            name="test_param",
            decay=True,
        )
        
        optimizer = AdamW([p], lr=1e-3)
        
        # Master weights should be FP32
        assert optimizer.master_weights[0].dtype == "float32", \
            f"Master weights dtype: {optimizer.master_weights[0].dtype}"
        
        # Moments should be FP32
        assert optimizer.m[0].dtype == "float32", \
            f"M dtype: {optimizer.m[0].dtype}"
        assert optimizer.v[0].dtype == "float32", \
            f"V dtype: {optimizer.v[0].dtype}"
    
    def test_fp32_params_still_work(self):
        """Test that FP32 parameters still work correctly."""
        # Create FP32 parameter
        p = Parameter(
            data=xp.asarray([[1.0, 2.0], [3.0, 4.0]], dtype="float32"),
            name="test_param",
            decay=True,
        )
        
        optimizer = AdamW([p], lr=1e-3)
        
        # Master weights should be FP32
        assert optimizer.master_weights[0].dtype == "float32"
        
        # Moments should be FP32
        assert optimizer.m[0].dtype == "float32"
        assert optimizer.v[0].dtype == "float32"
    
    def test_tiny_gradient_accumulates_in_fp32(self):
        """Test that tiny gradients accumulate correctly in FP32 master weights."""
        # Create FP16 parameter with initial value
        p = Parameter(
            data=xp.asarray([[0.5, 0.5]], dtype="float16"),
            name="test_param",
            decay=True,
        )
        
        optimizer = AdamW([p], lr=1e-2, beta1=0.0, beta2=0.0)  # No momentum for simplicity
        
        # Create a tiny gradient that would be lost in FP16
        tiny_grad = xp.asarray([[1e-5, 1e-5]], dtype="float16")
        p.grad = tiny_grad
        
        optimizer.step()
        
        # The update should have occurred in FP32 master weights
        master_update = optimizer.master_weights[0][0, 0] - xp.asarray(0.5, dtype="float32")
        
        # Check that the update is non-zero (would be zero if FP16 was used)
        assert abs(float(master_update)) > 1e-7, \
            f"Master weight update too small: {master_update}"
    
    def test_fp16_weight_update_preserves_dtype(self):
        """Test that FP16 parameter dtype is preserved after update."""
        p = Parameter(
            data=xp.asarray([[1.0, 2.0]], dtype="float16"),
            name="test_param",
            decay=True,
        )
        
        optimizer = AdamW([p], lr=1e-3)
        
        # Zero grad first
        p.grad = xp.zeros_like(p.data)
        
        optimizer.step()
        
        # Parameter should still be FP16
        assert p.data.dtype == "float16", \
            f"Parameter dtype after step: {p.data.dtype}"
    
    def test_step_reduces_loss(self):
        """Test that a simple optimization step reduces loss."""
        # Create FP16 parameter
        p = Parameter(
            data=xp.asarray([[2.0]], dtype="float16"),
            name="test_param",
            decay=True,
        )
        
        optimizer = AdamW([p], lr=1e-1, weight_decay=0.0)
        
        # Simulate a gradient pointing to reduce the value
        p.grad = xp.asarray([[1.0]], dtype="float16")
        
        initial_value = float(p.data[0, 0])
        optimizer.step()
        final_value = float(p.data[0, 0])
        
        # Value should decrease (gradient is positive, we move opposite)
        assert final_value < initial_value, \
            f"Value didn't decrease: {initial_value} -> {final_value}"
    
    def test_momentum_accumulates(self):
        """Test that momentum (first moment) accumulates correctly."""
        p = Parameter(
            data=xp.asarray([[1.0]], dtype="float16"),
            name="test_param",
            decay=True,
        )
        
        optimizer = AdamW([p], lr=1e-3, beta1=0.9, beta2=0.0)
        
        # Apply constant gradient for 2 steps
        p.grad = xp.asarray([[1.0]], dtype="float16")
        
        optimizer.step()  # Step 1
        
        # Check that momentum has accumulated
        m_after_step1 = float(optimizer.m[0][0, 0])
        expected_m1 = 1.0  # (1 - beta1) * g = 0.1 * 1.0, but normalized by c1
        
        optimizer.step()  # Step 2
        
        m_after_step2 = float(optimizer.m[0][0, 0])
        
        # Momentum should have accumulated
        assert m_after_step2 > m_after_step1, \
            f"Momentum didn't accumulate: {m_after_step1} -> {m_after_step2}"
    
    def test_variance_accumulates(self):
        """Test that variance (second moment) accumulates correctly."""
        p = Parameter(
            data=xp.asarray([[1.0]], dtype="float16"),
            name="test_param",
            decay=True,
        )
        
        optimizer = AdamW([p], lr=1e-3, beta1=0.0, beta2=0.99)
        
        # Apply constant gradient for 2 steps
        p.grad = xp.asarray([[1.0]], dtype="float16")
        
        optimizer.step()  # Step 1
        v_after_step1 = float(optimizer.v[0][0, 0])
        
        optimizer.step()  # Step 2
        v_after_step2 = float(optimizer.v[0][0, 0])
        
        # Variance should have accumulated
        assert v_after_step2 > v_after_step1, \
            f"Variance didn't accumulate: {v_after_step1} -> {v_after_step2}"


class TestAdamWFP32:
    """Tests for AdamW with pure FP32 parameters."""
    
    def test_basic_fp32_update(self):
        """Test basic FP32 parameter update."""
        p = Parameter(
            data=xp.asarray([[1.0, 2.0]], dtype="float32"),
            name="test_param",
            decay=True,
        )
        
        optimizer = AdamW([p], lr=1e-2)
        
        p.grad = xp.asarray([[0.1, 0.2]], dtype="float32")
        
        initial_value = float(p.data[0, 0])
        optimizer.step()
        final_value = float(p.data[0, 0])
        
        # Value should have changed
        assert abs(final_value - initial_value) > 1e-6, \
            "FP32 parameter didn't update"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])


def test_adamw_numerical_debug_defaults_off():
    p = Parameter(
        data=xp.asarray([[1.0]], dtype="float32"),
        name="debug_flag_param",
        decay=True,
    )
    optimizer = AdamW([p], lr=1e-3)
    assert optimizer.numerical_debug is False


def test_adamw_numerical_debug_can_be_enabled():
    p = Parameter(
        data=xp.asarray([[1.0]], dtype="float32"),
        name="debug_flag_param",
        decay=True,
    )
    optimizer = AdamW([p], lr=1e-3, numerical_debug=True)
    assert optimizer.numerical_debug is True


def test_optimized_adam_update_matches_reference_formula():
    p = Parameter(
        data=xp.asarray([[0.25, -0.75], [1.5, -2.0]], dtype="float32"),
        name="reference_formula", decay=True,
    )
    opt = AdamW(
        [p], lr=2e-3, beta1=0.9, beta2=0.95,
        eps=1e-8, weight_decay=0.1,
    )

    ref_w = np.array([[0.25, -0.75], [1.5, -2.0]], dtype=np.float32)
    ref_m = np.zeros_like(ref_w)
    ref_v = np.zeros_like(ref_w)
    beta1, beta2 = 0.9, 0.95
    lr, wd, eps = 2e-3, 0.1, 1e-8
    grads = [
        np.array([[0.2, -0.1], [0.05, 0.4]], dtype=np.float32),
        np.array([[-0.3, 0.2], [0.15, -0.25]], dtype=np.float32),
        np.array([[0.1, 0.1], [-0.2, 0.3]], dtype=np.float32),
    ]

    for step, grad in enumerate(grads, start=1):
        p.grad[...] = xp.asarray(grad)
        opt.step()

        ref_m = beta1 * ref_m + (1.0 - beta1) * grad
        ref_v = beta2 * ref_v + (1.0 - beta2) * (grad * grad)
        m_hat = ref_m / (1.0 - beta1 ** step)
        v_hat = ref_v / (1.0 - beta2 ** step)
        ref_w *= (1.0 - lr * wd)
        ref_w -= lr * m_hat / (np.sqrt(v_hat) + eps)

        np.testing.assert_allclose(
            np.asarray(opt.master_weights[0]), ref_w, rtol=2e-6, atol=2e-7
        )


def test_restore_moments_matches_shapes():
    p = Parameter(
        data=xp.asarray([[1.0, 2.0]], dtype="float32"),
        name="restore_moments", decay=True,
    )
    opt = AdamW([p], lr=1e-3)
    m = np.asarray([[0.25, -0.5]], dtype=np.float32)
    v = np.asarray([[0.1, 0.2]], dtype=np.float32)
    assert opt.restore_moments(0, m, v)
    np.testing.assert_allclose(np.asarray(opt.m[0]), m)
    np.testing.assert_allclose(np.asarray(opt.v[0]), v)


def test_restore_moments_rejects_shape_mismatch():
    p = Parameter(
        data=xp.asarray([[1.0, 2.0]], dtype="float32"),
        name="restore_moments_shape", decay=True,
    )
    opt = AdamW([p], lr=1e-3)
    bad = np.asarray([0.1], dtype=np.float32)
    assert not opt.restore_moments(0, bad, bad)


def test_full_optimizer_offload_mode_is_reference_backend_safe(monkeypatch):
    """0055C full offload is a no-op on NumPy and remains numerically usable."""
    monkeypatch.setenv("MINI_LLM_OPTIMIZER_OFFLOAD", "full")
    p = Parameter(
        data=xp.asarray([[1.0, -2.0]], dtype="float32"),
        name="full_offload_reference",
        decay=True,
    )
    opt = AdamW([p], lr=1e-3)
    p.grad = xp.asarray([[0.25, -0.5]], dtype="float32")
    opt.step()
    assert np.all(np.isfinite(np.asarray(p.data)))
