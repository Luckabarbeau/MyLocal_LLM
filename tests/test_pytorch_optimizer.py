"""Focused tests for the optional PyTorch hybrid Muon/AdamW optimizer."""

import math

import numpy as np
import pytest


torch = pytest.importorskip("torch")

from mini_llm.backends.pytorch.optimizer import HybridMuonAdamW


def _ns_reference(gradient, steps=5):
    x = np.asarray(gradient, dtype=np.float32)
    norm = np.sqrt(np.sum(x * x, dtype=np.float32), dtype=np.float32)
    x = (x / max(float(norm), 1e-7)).astype(np.float32)
    transposed = x.shape[0] > x.shape[1]
    if transposed:
        x = x.T
    a, b, c = 3.4445, -4.7750, 2.0315
    for _ in range(steps):
        gram = (x @ x.T).astype(np.float32)
        polynomial = (b * gram + c * (gram @ gram)).astype(np.float32)
        x = (a * x + polynomial @ x).astype(np.float32)
    return (x.T if transposed else x).astype(np.float32)


def test_named_grouping_preserves_names_decay_and_unique_membership():
    parameters = [
        ("blocks.0.attention.Wq", torch.nn.Parameter(torch.ones(3, 4))),
        ("blocks.0.moe.router.W_router", torch.nn.Parameter(torch.ones(4, 2))),
        ("embedding.W", torch.nn.Parameter(torch.ones(5, 4))),
        ("blocks.0.norm1.gamma", torch.nn.Parameter(torch.ones(4))),
    ]
    decay = {
        "blocks.0.attention.Wq": True,
        "blocks.0.moe.router.W_router": True,
        "embedding.W": False,
        "blocks.0.norm1.gamma": False,
    }

    optimizer = HybridMuonAdamW(parameters, decay_policy=decay)

    grouped = [
        (name, parameter)
        for group in optimizer.param_groups
        for name, parameter in zip(group["param_names"], group["params"])
    ]
    assert sorted(name for name, _ in grouped) == sorted(name for name, _ in parameters)
    assert len({id(parameter) for _, parameter in grouped}) == len(parameters)
    assert optimizer.muon_names == {"blocks.0.attention.Wq"}
    assert optimizer.adam_names == {
        "blocks.0.moe.router.W_router",
        "embedding.W",
        "blocks.0.norm1.gamma",
    }
    assert optimizer.decay_by_name == decay


def test_fp32_step_matches_independent_numpy_native_reference():
    muon_initial = np.array(
        [[0.3, -0.2, 0.1], [0.4, 0.6, -0.5]], dtype=np.float32
    )
    adam_initial = np.array([0.2, -0.4, 0.7], dtype=np.float32)
    muon_gradient = np.array(
        [[0.07, -0.03, 0.11], [-0.05, 0.02, 0.09]], dtype=np.float32
    )
    adam_gradient = np.array([0.04, -0.08, 0.03], dtype=np.float32)
    muon = torch.nn.Parameter(torch.from_numpy(muon_initial.copy()))
    adam = torch.nn.Parameter(torch.from_numpy(adam_initial.copy()))
    muon.grad = torch.from_numpy(muon_gradient.copy())
    adam.grad = torch.from_numpy(adam_gradient.copy())
    lr, beta1, beta2, eps, weight_decay = 2e-3, 0.9, 0.95, 1e-8, 0.1
    optimizer = HybridMuonAdamW(
        [("blocks.0.attention.Wq", muon), ("blocks.0.norm1.gamma", adam)],
        lr=lr,
        beta1=beta1,
        beta2=beta2,
        eps=eps,
        weight_decay=weight_decay,
        decay_policy={
            "blocks.0.attention.Wq": True,
            "blocks.0.norm1.gamma": False,
        },
        muon_matmul_precision="fp32",
    )

    optimizer.step()

    muon_m = np.float32(0.05) * muon_gradient
    nesterov = np.float32(0.95) * muon_m + np.float32(0.05) * muon_gradient
    direction = _ns_reference(nesterov)
    direction *= np.float32(0.2 * math.sqrt(3))
    expected_muon = (
        muon_initial * np.float32(1.0 - lr * weight_decay)
        - np.float32(lr) * direction
    )

    adam_m = np.float32(1.0 - beta1) * adam_gradient
    adam_v = np.float32(1.0 - beta2) * (adam_gradient * adam_gradient)
    correction1 = 1.0 - beta1
    correction2 = 1.0 - beta2
    step_size = lr * math.sqrt(correction2) / correction1
    eps_scaled = eps * math.sqrt(correction2)
    expected_adam = adam_initial - np.float32(step_size) * adam_m / (
        np.sqrt(adam_v).astype(np.float32) + np.float32(eps_scaled)
    )

    np.testing.assert_allclose(muon.detach().numpy(), expected_muon, rtol=2e-6, atol=2e-7)
    np.testing.assert_allclose(adam.detach().numpy(), expected_adam, rtol=2e-6, atol=2e-7)
    np.testing.assert_allclose(
        optimizer.state[muon]["muon_m"].numpy(), muon_m, rtol=1e-6, atol=1e-7
    )
    np.testing.assert_allclose(
        optimizer.state[adam]["m"].numpy(), adam_m, rtol=1e-6, atol=1e-7
    )
    np.testing.assert_allclose(
        optimizer.state[adam]["v"].numpy(), adam_v, rtol=1e-6, atol=1e-7
    )


def test_checkpoint_schema_and_exact_continuation():
    first = torch.nn.Parameter(torch.arange(6, dtype=torch.float32).reshape(2, 3) / 10)
    second = torch.nn.Parameter(torch.tensor([0.2, -0.3], dtype=torch.float32))
    names = ["blocks.1.attention.Wo", "blocks.1.norm2.gamma"]
    optimizer = HybridMuonAdamW(
        list(zip(names, (first, second))),
        lr=1e-3,
        decay_policy={names[0]: True, names[1]: False},
        muon_matmul_precision="fp32",
    )
    first.grad = torch.full_like(first, 0.03)
    second.grad = torch.tensor([0.04, -0.02])
    optimizer.step()
    checkpoint = optimizer.state_dict()

    assert set(checkpoint) == {"master_weights", "m", "v", "muon_m", "step"}
    assert set(checkpoint["master_weights"]) == set(names)
    assert set(checkpoint["muon_m"]) == {names[0]}
    assert set(checkpoint["m"]) == set(checkpoint["v"]) == {names[1]}
    assert checkpoint["step"] == 1

    restored_parameters = [
        torch.nn.Parameter(first.detach().clone()),
        torch.nn.Parameter(second.detach().clone()),
    ]
    restored = HybridMuonAdamW(
        list(zip(names, restored_parameters)),
        lr=1e-3,
        decay_policy={names[0]: True, names[1]: False},
        muon_matmul_precision="fp32",
    )
    restored.load_state_dict(checkpoint)
    for original, copy in zip((first, second), restored_parameters):
        gradient = torch.full_like(original, -0.015)
        original.grad = gradient.clone()
        copy.grad = gradient.clone()
    optimizer.step()
    restored.step()

    assert restored.step_index == optimizer.step_index == 2
    for original, copy in zip((first, second), restored_parameters):
        torch.testing.assert_close(copy, original, rtol=0.0, atol=0.0)


@pytest.mark.parametrize("offload", ["moments", "full"])
def test_cpu_offload_state_placement(offload):
    muon = torch.nn.Parameter(torch.ones(2, 3))
    adam = torch.nn.Parameter(torch.ones(3))
    optimizer = HybridMuonAdamW(
        [("blocks.0.attention.Wq", muon), ("blocks.0.norm.gamma", adam)],
        offload=offload,
        muon_matmul_precision="fp32",
    )

    assert optimizer.state[muon]["muon_m"].device.type == "cpu"
    assert optimizer.state[adam]["m"].device.type == "cpu"
    assert optimizer.state[adam]["v"].device.type == "cpu"
    if offload == "full":
        assert optimizer.state[muon]["master_weights"].device.type == "cpu"
        assert optimizer.state[adam]["master_weights"].device.type == "cpu"
    if torch.cuda.is_available():
        assert optimizer.state[muon]["muon_m"].is_pinned()
        assert optimizer.state[adam]["m"].is_pinned()
        assert optimizer.state[adam]["v"].is_pinned()
        if offload == "full":
            assert optimizer.state[muon]["master_weights"].is_pinned()
            assert optimizer.state[adam]["master_weights"].is_pinned()
