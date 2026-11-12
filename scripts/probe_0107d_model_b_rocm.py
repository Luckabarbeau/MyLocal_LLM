#!/usr/bin/env python3
"""Full registered Model B ROCm diagnostic, not the pinned training profile.

Run: sh scripts/run_r9700_preflight.sh scripts/probe_0107d_model_b_rocm.py
Uses random weights and synthetic tokens; never reads or writes checkpoints.
"""
import argparse
import os
from dataclasses import replace
from pathlib import Path
import sys
import time

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dtype", choices=("float32", "bfloat16"), default="float32")
    parser.add_argument("--optimizer-step", action="store_true",
                        help="check one Hybrid Muon/AdamW update and post-update loss")
    args = parser.parse_args()
    if args.optimizer_step and args.dtype != "bfloat16":
        parser.error("--optimizer-step requires --dtype bfloat16")
    if args.optimizer_step:
        os.environ.setdefault("MINI_LLM_OPTIMIZER_OFFLOAD", "full")
        os.environ.setdefault("MINI_LLM_OPTIMIZER_OFFLOAD_CHUNK_MB", "16")
    import numpy as np
    from mini_llm.backend import BACKEND_NAME, xp, synchronize, validate_bfloat16_backend
    from mini_llm.model_registry import registry
    from mini_llm.model.decoder_lm import DecoderLanguageModel
    from mini_llm.ops.dual_resolution_attention import build_dual_resolution_metadata

    if BACKEND_NAME != "cupy" or not xp.cuda.runtime.is_hip:
        raise RuntimeError("This diagnostic requires CuPy HIP, not CUDA or CPU")
    props = xp.cuda.runtime.getDeviceProperties(0)
    arch = props.get("gcnArchName", b"")
    if isinstance(arch, bytes):
        arch = arch.decode()
    if not str(arch).startswith("gfx1201"):
        raise RuntimeError(f"Refusing non-R9700 device: {props}")
    print("AMD target:", arch, flush=True)
    if args.dtype == "bfloat16":
        validate_bfloat16_backend()
    config = replace(registry.config("model_b_348m_sparse16"), dtype=args.dtype)
    start = time.perf_counter()
    model = DecoderLanguageModel(config, rng_seed=107, dtype=args.dtype)
    count = sum(int(p.size) for p in model.parameters())
    assert count == config.estimated_parameter_count()
    synchronize()
    print(f"Initialized Model B: {count:,} parameters, {args.dtype} diagnostic", flush=True)
    raw = np.random.default_rng(107).integers(
        0, config.vocab_size - 1, size=(1, 257), dtype=np.int64)
    tokens = xp.asarray(raw[:, :-1])
    metadata = build_dual_resolution_metadata(
        raw[:, :-1], eos_token_id=config.vocab_size - 1,
        block_size=config.dual_resolution.block_size)
    head, cache = model.forward_body(
        tokens, activation_checkpoint=True, dual_resolution_metadata=metadata)
    synchronize()
    print("Forward body completed", flush=True)
    loss, loss_cache = model.chunked_lm_head_loss_forward(
        head, xp.asarray(raw[:, -8:]), chunk_tokens=8,
        target_slice=(248, 256), return_device_loss=True)
    value = float(loss.item())
    assert np.isfinite(value), value
    print(f"Loss: {value:.6f}", flush=True)
    dx = model.chunked_lm_head_backward(loss_cache)
    model.backward_body(dx, cache)
    synchronize()
    for p in model.parameters():
        assert bool(xp.all(xp.isfinite(p.grad)).item()), p.name
    print(f"PASS: Model B {args.dtype} forward/loss/backward; {time.perf_counter()-start:.2f}s cold",
          flush=True)
    if not args.optimizer_step:
        return

    from mini_llm.optim.hybrid_muon import HybridMuonAdamW
    from mini_llm.optim.grad_clip import clip_grad_global_norm
    parameters = list(model.optimization_parameters())
    assert all(str(p.grad.dtype) == "float32" for p in parameters)
    norm, scale, finite = clip_grad_global_norm(parameters, max_norm=1.0)
    assert finite and np.isfinite(float(norm.item())), "Invalid gradient norm"
    print(f"Gradient norm: {float(norm.item()):.6f}; clip scale: {scale:.6f}", flush=True)
    optimizer = HybridMuonAdamW(
        parameters, lr=1.5e-4, beta1=0.9, beta2=0.95, eps=1e-8,
        weight_decay=0.1, muon_momentum=0.95, muon_ns_steps=5,
        muon_matmul_precision="bf16", numerical_debug=True,
    )
    assert optimizer.muon.parameters and optimizer.adam.parameters
    print(f"Optimizer: {len(optimizer.muon.parameters)} Muon tensors, "
          f"{len(optimizer.adam.parameters)} AdamW tensors; "
          f"offload={os.environ['MINI_LLM_OPTIMIZER_OFFLOAD']}", flush=True)
    # Bounded snapshots prove both master and model weights actually change.
    def host_sample(array):
        sample = array.reshape(-1)[:4096]
        return (sample.get() if hasattr(sample, "get") else np.asarray(sample)).copy()

    groups = (optimizer.muon, optimizer.adam)
    before = [[(host_sample(w), host_sample(p.data))
               for p, w in zip(group.parameters, group.master_weights)]
              for group in groups]
    synchronize()
    update_start = time.perf_counter()
    optimizer.step()
    synchronize()
    update_seconds = time.perf_counter() - update_start
    assert optimizer.step_index == optimizer.muon.step_index == optimizer.adam.step_index == 1
    for name, group, snapshots in zip(("Muon", "AdamW"), groups, before):
        master_changed = data_changed = 0
        for index, (p, w) in enumerate(zip(group.parameters, group.master_weights)):
            assert bool(xp.all(xp.isfinite(p.data)).item()), p.name
            for state_list in (group.master_weights, group.m):
                state = state_list[index]
                finite_state = (bool(xp.all(xp.isfinite(state)).item())
                                if hasattr(state, "get") else bool(np.isfinite(state).all()))
                assert finite_state, f"Nonfinite {name} state: {p.name}"
            if hasattr(group, "v"):
                state = group.v[index]
                assert (bool(xp.all(xp.isfinite(state)).item()) if hasattr(state, "get")
                        else bool(np.isfinite(state).all())), p.name
            master_changed += not np.array_equal(snapshots[index][0], host_sample(w))
            data_changed += not np.array_equal(snapshots[index][1], host_sample(p.data))
        assert master_changed > 0 and data_changed > 0, f"No observed {name} update"
        print(f"{name}: finite state; changed samples in {master_changed} master / "
              f"{data_changed} model tensors", flush=True)
    model.refresh_compute_buffers()
    model.zero_grad()
    del head, cache, loss_cache, dx
    updated_head = model.forward_body(
        tokens, return_cache=False, dual_resolution_metadata=metadata)
    updated_loss, _ = model.chunked_lm_head_loss_forward(
        updated_head, xp.asarray(raw[:, -8:]), chunk_tokens=8,
        target_slice=(248, 256), return_device_loss=True)
    after = float(updated_loss.item())
    assert np.isfinite(after), after
    print(f"Post-update loss: {after:.6f} (before {value:.6f})", flush=True)
    print(f"PASS: Hybrid Muon/AdamW BF16-mixed optimizer step; "
          f"update {update_seconds:.2f}s; total {time.perf_counter()-start:.2f}s", flush=True)


if __name__ == "__main__":
    main()
