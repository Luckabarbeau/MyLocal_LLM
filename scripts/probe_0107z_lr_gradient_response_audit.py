"""0107Z: read-only audit of the reported 'learning rate changes the gradient norm' effect.

The experiment reproduces the *mechanism* on deterministic synthetic inputs without
touching the active training run, its checkpoints or its logs.

Part A - shape-faithful optimizer measurement. The real HybridMuonAdamW is applied
to the real Model B parameter names/shapes (a representative subset) with BF16
model tensors + FP32 masters, gradients calibrated to the observed production global
gradient norm, at the two production learning rates from the experiment CSV.

Part B - end-to-end trainer measurement. The real ExtendedTrainer gradient-accumulation
path (supervised-token weighting + global clipping + Hybrid Muon) is run from identical
initial state at two learning rates, and every branch is then evaluated on the SAME
fixed probe batch.

Tests performed:
  1. pre-update gradients are learning-rate independent
  2. realized parameter displacement scales as eta with an eta-independent direction
  3. fixed-probe loss/gradient response after one update
  4. optimizer-vs-reference update equations (delegated to the existing tests)
  5. fixed update direction swept over several magnitudes (curvature isolation)
"""

import argparse
import copy
import json
import math
from pathlib import Path

import numpy as np

from mini_llm.backend import BACKEND_NAME, xp, resolve_dtype, asnumpy
from mini_llm.config import ModelConfig
from mini_llm.model.decoder_lm import DecoderLanguageModel
from mini_llm.parameter import Parameter
from mini_llm.optim.grad_clip import clip_grad_global_norm, global_grad_norm
from mini_llm.optim.hybrid_muon import HybridMuonAdamW, muon_eligible

PRODUCTION_LRS = (0.0, 7.1e-5, 1.42e-4, 2.84e-4)
PRODUCTION_GRAD_NORM = 0.303665  # observed global gradient norm at step 16201
INIT_STD = 0.02          # ModelConfig.init_std for Model B
RESIDUAL_INIT_STD = 0.0031622776601683794  # ModelConfig.residual_init_std


def host(value):
    return np.asarray(asnumpy(value), dtype=np.float64)


def _scalar(value):
    return float(value.get() if hasattr(value, "get") else value)


def rms(values):
    array = np.asarray(values, dtype=np.float64).reshape(-1)
    if array.size == 0:
        return 0.0
    return float(np.sqrt(np.mean(array * array)))


def norm(values):
    array = np.asarray(values, dtype=np.float64).reshape(-1)
    return float(np.linalg.norm(array))


def cosine(left, right):
    a = np.asarray(left, dtype=np.float64).reshape(-1)
    b = np.asarray(right, dtype=np.float64).reshape(-1)
    denom = float(np.linalg.norm(a) * np.linalg.norm(b))
    return float(np.dot(a, b) / denom) if denom > 0 else 0.0


def model_b_group_shapes():
    """Representative real Model B parameter names and shapes (one block worth)."""
    groups = [
        ("embedding.W", (32768, 1024), INIT_STD),
        ("blocks.0.attention.Wq", (1024, 1024), INIT_STD),
        ("blocks.0.attention.Wk", (1024, 256), INIT_STD),
        ("blocks.0.attention.Wv", (1024, 256), INIT_STD),
        ("blocks.0.attention.Wo", (1024, 1024), INIT_STD),
        ("blocks.0.attention.global.Wq", (1024, 128), RESIDUAL_INIT_STD),
        ("blocks.0.attention.global.Wk", (1024, 128), RESIDUAL_INIT_STD),
        ("blocks.0.attention.global.Wv", (1024, 128), RESIDUAL_INIT_STD),
        ("blocks.0.attention.global.Wo", (128, 1024), RESIDUAL_INIT_STD),
        ("blocks.0.dense_ffn.expert.W_gate", (1024, 4096), INIT_STD),
        ("blocks.0.dense_ffn.expert.W_up", (1024, 4096), INIT_STD),
        ("blocks.0.dense_ffn.expert.W_down", (4096, 1024), INIT_STD),
        ("blocks.0.norm1.gamma", (1024,), 1.0),
    ]
    return groups


def build_part_a_parameters(seed=20261115, dtype=None):
    """Real Model B names/shapes at the requested parameter storage dtype."""
    from types import SimpleNamespace
    if dtype is None:
        dtype = "bfloat16" if BACKEND_NAME == "cupy" else "float32"
    storage = resolve_dtype(dtype)
    rng = np.random.default_rng(seed)
    parameters = []
    for name, shape, init_std in model_b_group_shapes():
        values = (rng.normal(size=shape) * init_std).astype(np.float32)
        eligible = muon_eligible(SimpleNamespace(name=name,
                                                data=SimpleNamespace(shape=shape)))
        parameters.append(Parameter(
            xp.asarray(values, dtype=storage), name=name, decay=bool(eligible)))
    return parameters


def calibrate_gradients(parameters, target_global_norm, seed=777):
    """Synthetic gradient field rescaled to the observed production global norm."""
    rng = np.random.default_rng(seed)
    total = 0.0
    for parameter in parameters:
        gradient = rng.normal(size=parameter.data.shape).astype(np.float32)
        # Heavier matrices carry relatively smaller per-element gradients, which
        # matches how a trained transformer's gradient RMS falls with width.
        gradient *= 1.0 / math.sqrt(max(parameter.data.shape))
        parameter.grad[...] = xp.asarray(gradient)
        total += float(np.sum(asnumpy(parameter.grad).astype(np.float64) ** 2))
    current = math.sqrt(total)
    scale = target_global_norm / current if current > 0 else 0.0
    for parameter in parameters:
        parameter.grad[...] = xp.asarray(asnumpy(parameter.grad) * scale)
    return scale


def measure_branch(learning_rate, dtype=None):
    """One optimizer update; report pre-update gradients and realized displacement.

    Fresh identical weights and identical gradients are rebuilt for every branch,
    so the only cross-branch difference is the learning rate."""
    parameters = build_part_a_parameters(dtype=dtype)
    calibrate_gradients(parameters, PRODUCTION_GRAD_NORM)
    initial = {p.name: host(p.data) for p in parameters}
    norm_backend, clip_scale, finite = clip_grad_global_norm(
        parameters, max_norm=1.0,
    )
    optimizer = HybridMuonAdamW(
        parameters, lr=learning_rate, beta1=0.9, beta2=0.95, eps=1e-8,
        weight_decay=0.1, muon_momentum=0.95, muon_ns_steps=5,
        muon_matmul_precision="fp32",
    )
    ordered = [(group, index, parameter)
               for group in (optimizer.adam, optimizer.muon)
               for index, parameter in enumerate(group.parameters)]
    masters_before = {}
    for group, index, parameter in ordered:
        masters_before[parameter.name] = host(group.master_weights[index]).copy()
    optimizer.step(lr=learning_rate)
    rows = []
    sample_deltas = []
    master_sumsq = device_sumsq = 0.0
    for group, index, parameter in ordered:
        before = masters_before[parameter.name]
        after = host(group.master_weights[index])
        master_delta = before - after
        device_delta = initial[parameter.name] - host(parameter.data)
        master_sumsq += float(np.sum(master_delta * master_delta))
        device_sumsq += float(np.sum(device_delta * device_delta))
        if group is optimizer.muon:
            sample_deltas.append(master_delta.reshape(-1)[:65536])
        rows.append({
            "parameter": parameter.name,
            "optimizer": ("muon" if group is optimizer.muon else "adamw"),
            "shape": list(parameter.data.shape),
            "parameter_rms": rms(before),
            "master_update_rms": rms(master_delta),
            "device_update_rms": rms(device_delta),
            "master_update_norm": norm(master_delta),
            "device_update_norm": norm(device_delta),
            "update_to_parameter_rms": (
                rms(master_delta) / rms(before) if rms(before) > 0 else 0.0),
            "realized_device_to_intended_rms": (
                rms(device_delta) / rms(master_delta)
                if rms(master_delta) > 0 else 0.0),
        })
    delta_sample = np.concatenate(sample_deltas) if sample_deltas else np.zeros(0)
    return {
        "learning_rate": learning_rate,
        "pre_update_global_grad_norm": float(_scalar(norm_backend)),
        "clip_scale": clip_scale,
        "all_finite": finite,
        "total_master_update_norm": math.sqrt(master_sumsq),
        "total_device_update_norm": math.sqrt(device_sumsq),
        "total_master_update_rms": math.sqrt(
            master_sumsq / sum(int(g.parameters[i].data.size)
                               for g, i, _ in ordered)),
        "rows": rows,
        "muon_delta_sample": delta_sample,
    }


def part_a(report, dtype="float32"):
    key = f"part_a_{dtype}"
    branches = {}
    for learning_rate in PRODUCTION_LRS:
        branch = measure_branch(learning_rate, dtype=dtype)
        branches[learning_rate] = branch
        report[key][str(learning_rate)] = {
            k: v for k, v in branch.items() if k != "muon_delta_sample"}
        branch["muon_delta_sample"] = branch["muon_delta_sample"]
        print(f"[{dtype}] lr={learning_rate:.6e} pre-update global grad norm="
              f"{branch['pre_update_global_grad_norm']:.6f} clip_scale="
              f"{branch['clip_scale']:.6f} master|dW|={branch['total_master_update_norm']:.6f} "
              f"device|dW|={branch['total_device_update_norm']:.6f}")
    reference = branches[PRODUCTION_LRS[1]]
    print("\n-- displacement proportionality / direction consistency --")
    for learning_rate in PRODUCTION_LRS[2:]:
        branch = branches[learning_rate]
        ratio_norm = (branch["total_master_update_norm"] /
                      max(reference["total_master_update_norm"], 1e-300))
        direction = cosine(branch["muon_delta_sample"],
                           reference["muon_delta_sample"])
        print(f"lr ratio={learning_rate / PRODUCTION_LRS[1]:.3f}  "
              f"|dW| ratio={ratio_norm:.3f}  "
              f"cos(direction_low, direction_high)={direction:.6f}")
    print("\n-- per-component displacement (high LR branch) --")
    high = branches[2.84e-4]
    for row in high["rows"]:
        print(f"{row['parameter']:<34} RMS(W)={row['parameter_rms']:.4e} "
              f"RMS(dW_master)={row['master_update_rms']:.4e} "
              f"RMS(dW)/RMS(W)={row['update_to_parameter_rms']:.4e}")
    print("\n-- Muon update RMS versus the 0.2*eta design value --")
    for learning_rate in PRODUCTION_LRS[1:]:
        muon_rows = [r for r in branches[learning_rate]["rows"]
                     if r["optimizer"] == "muon"]
        values = [r["master_update_rms"] for r in muon_rows]
        print(f"lr={learning_rate:.6e} expected=0.2*lr={0.2 * learning_rate:.4e} "
              f"observed RMS(dW) range=[{min(values):.4e}, {max(values):.4e}]")
    return branches


def part_b(report, work_dir):
    from mini_llm.train_extended import ExtendedTrainer

    print("\n=== PART B: real trainer accumulation + fixed probe batch ===")
    shard = work_dir / "part_b_train.bin"
    if not shard.exists():
        data = (np.arange(65536, dtype=np.uint16) % 250).astype(np.uint16)
        data.tofile(shard)

    results = {}
    for learning_rate in (7.1e-5, 2.84e-4):
        model = DecoderLanguageModel(ModelConfig.tiny_inspection(),
                                    rng_seed=1011, dtype="float32")
        trainer = ExtendedTrainer(
            model, [str(shard)], [str(shard)],
            batch_size=4, seq_length=32,
            grad_accum_steps=8,
            gradient_accumulation_weighting="supervised_tokens",
            warmup_steps=0, total_steps=1000, peak_lr=learning_rate,
            grad_clip=1.0, weight_decay=0.1,
            optimizer_kind="hybrid-muon", rng_seed=99,
            component_update_monitor=True,
            component_update_interval=1,
            component_update_output=str(work_dir /
                                        f"components_{learning_rate}.csv"),
        )
        masters_before = {}
        optimizer = trainer.optimizer
        for group in (optimizer.adam, optimizer.muon):
            for index, parameter in enumerate(group.parameters):
                masters_before[parameter.name] = host(group.master_weights[index]).copy()
        loss, grad_norm = trainer.train_step()

        probe = probe_gradients(model, work_dir)
        displacement = {}
        for group in (optimizer.adam, optimizer.muon):
            for index, parameter in enumerate(group.parameters):
                after = host(group.master_weights[index])
                displacement[parameter.name] = masters_before[parameter.name] - after
        results[learning_rate] = {
            "trainer_loss": loss,
            "trainer_grad_norm": grad_norm,
            "probe": probe,
            "displacement_flat": np.concatenate(
                [d.reshape(-1) for d in displacement.values()]),
            "displacement": displacement,
            "masters_before": masters_before,
        }
        print(f"lr={learning_rate:.6e} trainer loss={loss:.6f} "
              f"trainer grad norm (pre-update)={grad_norm:.6f}")

    low, high = results[7.1e-5], results[2.84e-4]
    print("\n-- Test 1: pre-update gradients across branches --")
    print(f"trainer-reported pre-update grad norm: low={low['trainer_grad_norm']:.6f} "
          f"high={high['trainer_grad_norm']:.6f} "
          f"ratio={high['trainer_grad_norm'] / max(low['trainer_grad_norm'], 1e-300):.4f}")
    print(f"trainer loss: low={low['trainer_loss']:.6f} high={high['trainer_loss']:.6f}")
    print("-- Test 2: realized displacement --")
    print(f"|dW| low={norm(low['displacement_flat']):.6e} "
          f"high={norm(high['displacement_flat']):.6e} "
          f"ratio={norm(high['displacement_flat']) / max(norm(low['displacement_flat']), 1e-300):.4f} "
          f"cos={cosine(low['displacement_flat'], high['displacement_flat']):.6f}")
    print("-- Test 3: fixed probe batch after one update --")
    for learning_rate in (7.1e-5, 2.84e-4):
        probe = results[learning_rate]["probe"]
        print(f"lr={learning_rate:.6e} probe loss={probe['loss']:.6f} "
              f"probe global grad norm={probe['global_grad_norm']:.6f} "
              f"muon group={probe['muon_grad_norm']:.6f} "
              f"adamw group={probe['adamw_grad_norm']:.6f}")

    report["part_b"] = {
        str(k): {"trainer_loss": v["trainer_loss"],
                 "trainer_grad_norm": v["trainer_grad_norm"],
                 "probe": v["probe"],
                 "displacement_norm": norm(v["displacement_flat"])}
        for k, v in results.items()
    }
    return results


def probe_gradients(model, work_dir, direction=None):
    """Fixed synthetic probe batch: loss + gradient statistics on identical inputs.

    When `direction` (unit vectors per parameter name) is supplied, the projection
    of the probe gradient onto that direction is reported. That projection is the
    finite directional derivative of the loss along the update direction.
    """
    rng = np.random.default_rng(4242)
    tokens = xp.asarray(rng.integers(1, 250, size=(4, 32)), dtype=xp.int64)
    targets = xp.asarray(rng.integers(1, 250, size=(4, 32)), dtype=xp.int64)
    logits, cache = model.forward(tokens)
    loss, loss_cache = model.compute_loss(logits, targets)
    model.zero_grad()
    model.backward(model.backward_loss(loss_cache), cache)
    muon_parameters, adam_parameters = [], []
    for parameter in model.optimization_parameters():
        if muon_eligible(parameter):
            muon_parameters.append(parameter)
        else:
            adam_parameters.append(parameter)
    def group_norm(parameters):
        total = 0.0
        for parameter in parameters:
            gradient = asnumpy(parameter.grad).astype(np.float64)
            total += float(np.sum(gradient * gradient))
        return math.sqrt(total)
    statistics = {
        "loss": float(asnumpy(loss)),
        "global_grad_norm": group_norm(model.optimization_parameters()),
        "muon_grad_norm": group_norm(muon_parameters),
        "adamw_grad_norm": group_norm(adam_parameters),
    }
    if direction is not None:
        projection = projection_muon = 0.0
        for parameter in model.optimization_parameters():
            gradient = asnumpy(parameter.grad).astype(np.float64).reshape(-1)
            unit = np.asarray(direction[parameter.name], dtype=np.float64).reshape(-1)
            dot = float(np.dot(gradient, unit))
            projection += dot
            if muon_eligible(parameter):
                projection_muon += dot
        statistics["directional_gradient_projection"] = projection
        statistics["directional_gradient_projection_muon"] = projection_muon
    model.zero_grad()
    return statistics


def part_c_curvature_sweep(report, work_dir):
    """Test 5: one fixed update direction, several magnitudes, same probe batch."""
    from mini_llm.train_extended import ExtendedTrainer

    print("\n=== PART C: fixed update direction, swept magnitude (curvature isolation) ===")
    shard = work_dir / "part_b_train.bin"
    model = DecoderLanguageModel(ModelConfig.tiny_inspection(),
                                rng_seed=1011, dtype="float32")
    trainer = ExtendedTrainer(
        model, [str(shard)], [str(shard)],
        batch_size=4, seq_length=32,
        grad_accum_steps=8, gradient_accumulation_weighting="supervised_tokens",
        warmup_steps=0, total_steps=1000, peak_lr=7.1e-5,
        grad_clip=1.0, weight_decay=0.1, optimizer_kind="hybrid-muon", rng_seed=99,
    )
    optimizer = trainer.optimizer
    groups = (optimizer.adam, optimizer.muon)
    masters0 = {}
    for group in groups:
        for index, parameter in enumerate(group.parameters):
            masters0[(group, index)] = host(group.master_weights[index]).copy()
    trainer.train_step()
    delta = {}
    total_sumsq = 0.0
    for group in groups:
        for index, parameter in enumerate(group.parameters):
            step = masters0[(group, index)] - host(group.master_weights[index])
            delta[(group, index)] = step
            total_sumsq += float(np.sum(step * step))
    global_step_norm = math.sqrt(total_sumsq)
    unit = {}
    for key, step in delta.items():
        unit[key] = step / global_step_norm if global_step_norm > 0 else step
    direction = {group.parameters[index].name: unit[(group, index)]
                 for group in groups
                 for index in range(len(group.parameters))}
    rows = []
    for magnitude in (0.0, 1.0, 2.0, 4.0, 8.0, 16.0, 64.0):
        for group in groups:
            for index, parameter in enumerate(group.parameters):
                target = masters0[(group, index)] - magnitude * delta[(group, index)]
                group.master_weights[index][...] = xp.asarray(
                    target, dtype="float32")
                owner = group.parameters[index]
                owner.data[...] = xp.asarray(target, dtype=owner.data.dtype)
        probe = probe_gradients(model, work_dir, direction=direction)
        probe["magnitude"] = magnitude
        rows.append(probe)
    baseline = rows[0]
    for row in rows:
        print(f"c={row['magnitude']:.1f} probe loss={row['loss']:.6f} "
              f"global grad norm={row['global_grad_norm']:.6f} "
              f"muon={row['muon_grad_norm']:.6f} adamw={row['adamw_grad_norm']:.6f} "
              f"dPhi/dc(along u)={-row['directional_gradient_projection']:.6e}")
    print("-- loss/gradient response along one fixed direction --")
    for row in rows[1:]:
        print(f"c={row['magnitude']:.1f}: loss delta vs c=0 = "
              f"{row['loss'] - baseline['loss']:+.6f}; "
              f"grad-norm ratio vs c=0 = "
              f"{row['global_grad_norm'] / max(baseline['global_grad_norm'], 1e-300):.4f}; "
              f"ratio vs c=4 over c=1 = "
              f"{row['global_grad_norm'] / max(baseline['global_grad_norm'], 1e-300):.4f}")
    report["part_c_sweep"] = rows
    return rows


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", default="./logs/lr_gradient_audit_0107z/report.json")
    parser.add_argument("--work-dir", default="./logs/lr_gradient_audit_0107z")
    args = parser.parse_args()
    work_dir = Path(args.output).parent
    work_dir.mkdir(parents=True, exist_ok=True)
    report = {"part_a_float32": {}, "part_a_bfloat16": {},
              "part_b": {}, "part_c_sweep": []}
    print(f"backend={BACKEND_NAME}")
    print("\n=== PART A: Model B shapes, real HybridMuonAdamW, production LRs ===")
    part_a(report, "float32")
    part_a(report, "bfloat16")
    part_b(report, work_dir)
    part_c_curvature_sweep(report, work_dir)
    Path(args.output).write_text(json.dumps(report, indent=2, sort_keys=True))
    print(f"\nreport written: {args.output}")


if __name__ == "__main__":
    main()
