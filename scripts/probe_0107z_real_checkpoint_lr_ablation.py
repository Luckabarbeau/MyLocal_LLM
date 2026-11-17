"""Phase-2 controlled LR ablation using the real Model B checkpoint (step 16200).

Runs entirely on CPU (``MINI_LLM_BACKEND=numpy``) and touches no production path:
fresh log/JSON outputs, read-only checkpoint access.

For every branch the checkpoint's weights, FP32 masters, Muon momentum,
AdamW moments, birth steps, optimizer step counter and sampler RNG state are
restored from ``checkpoints/model_b_batch1_8k_norecompute_long_16200``.  Only the
batch size is reduced for CPU feasibility; the model, the accumulator/weighting
path, the clip path, the hybrid Muon/AdamW optimizers and the checkpoint state
are the production ones.

Measured per branch:
  * pre-update global gradient norm (trainer-reported) and independently
    recomputed group/component norms,
  * the realised parameter displacement ``W_after - W_checkpoint``,
  * the *fixed probe batch* loss and gradient norms after the update
    (identical tokens for every branch, because the sampler state is restored),
  * cosine similarity between probe-gradient directions and between update
    directions across branches.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
# Production launches train_model.py from the repository root, so make the
# root module importable here too (mini_llm itself resolves to this same tree
# through the editable install).
import sys

if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
CKPT = ROOT / "checkpoints" / "model_b_batch1_8k_norecompute_long_16200"
OUT = ROOT / "logs" / "lr_gradient_audit_0107z"

SHARD = (ROOT / "token_shards_ensemble_32k_mixed" / "cosmopedia_v2" / "file_00000"
         / "train_shard_00000.bin")

ETA_HIGH = 2.841753146565134e-04    # high branch lr at step 16201 in the CSV
ETA_LOW = 7.104382866412835e-05     # low  branch lr at step 16201 in the CSV
SUBSAMPLE = 4096


def component_of(name: str) -> str:
    if name == "embedding.W":
        return "embedding_tied"
    if "matrix_pool" in name:
        return "matrix_pool"
    if ".global." in name:
        return "global_attn"
    if "attention" in name:
        return "local_attn"
    if "ffn" in name:
        return "ffn"
    if "norm" in name:
        return "norm"
    return "other"


def layer_of(name: str):
    parts = name.split(".")
    if parts and parts[0] == "blocks":
        return int(parts[1])
    return None


def norm2(values) -> float:
    flat = np.asarray(values, dtype=np.float64).reshape(-1)
    return float(np.dot(flat, flat))


def rms(values) -> float:
    flat = np.asarray(values, dtype=np.float64).reshape(-1)
    if flat.size == 0:
        return 0.0
    return math.sqrt(np.dot(flat, flat) / flat.size)


def subsample(values, count: int = SUBSAMPLE) -> np.ndarray:
    flat = np.asarray(values, dtype=np.float64).reshape(-1)
    if flat.size <= count:
        return flat
    step = flat.size // count
    return flat[::step][:count]


def cosine_of(a: np.ndarray, b: np.ndarray) -> float:
    left = math.sqrt(norm2(a))
    right = math.sqrt(norm2(b))
    if left == 0.0 or right == 0.0:
        return 0.0
    return float(np.dot(a.reshape(-1), b.reshape(-1)) / (left * right))


def load_state() -> dict:
    """Checkpoint optimizer state as host arrays (never mutated in place)."""
    def arrays(sub: str) -> dict:
        return {p.stem: np.load(p)
                for p in (CKPT / "optimizer_state" / sub).glob("*.npy")}
    scalars = json.loads((CKPT / "optimizer_state" / "scalars.json").read_text())
    return {"step": int(scalars["step"]),
            "master_weights": arrays("master_weights"),
            "m": arrays("m"),
            "v": arrays("v"),
            "muon_m": arrays("muon_m")}


def build(seq_length: int, accum: int, branch_lr: float, work_log: Path):
    from mini_llm.config import ModelConfig
    from mini_llm.model.decoder_lm import DecoderLanguageModel
    from mini_llm.train_extended import ExtendedTrainer
    from train_model import load_tokenizer_file, tokenizer_eos_id

    config = ModelConfig.dual_resolution_matrix_pool_sparse_1024_20l()
    model = DecoderLanguageModel(config, rng_seed=42, dtype="float32")

    names = [p.name for p in model.parameters()]
    disk = {p.stem for p in (CKPT / "model_params").glob("*.npy")}
    if set(names) != disk:
        missing = sorted(set(names) - disk)[:6]
        extra = sorted(disk - set(names))[:6]
        raise SystemExit(f"architecture/checkpoint mismatch missing={missing} extra={extra}")
    for p in model.parameters():
        values = np.load(CKPT / "model_params" / f"{p.name}.npy")
        if values.dtype.kind == "V" and values.dtype.itemsize == 2:
            from ml_dtypes import bfloat16
            values = np.frombuffer(np.ascontiguousarray(values).tobytes(), dtype=bfloat16
                                   ).reshape(values.shape)
        p.data[...] = np.asarray(values, dtype=np.float32)

    tokenizer = load_tokenizer_file(CKPT / "tokenizer.json")
    trainer = ExtendedTrainer(
        model, [str(SHARD)], [str(SHARD)],
        batch_size=1, seq_length=seq_length, grad_accum_steps=accum,
        gradient_accumulation_weighting="supervised_tokens",
        warmup_steps=0, total_steps=1, peak_lr=branch_lr,
        grad_clip=1000.0,          # probe norms exceed 1.0 at small batches; keep clip inactive
        weight_decay=0.1, optimizer_kind="hybrid-muon",
        muon_momentum=0.95, muon_ns_steps=5, muon_matmul_precision="bf16",
        rng_seed=42, eos_token_id=tokenizer_eos_id(tokenizer),
        val_interval=10 ** 9, save_interval=10 ** 9,
        log_file=str(work_log),
    )
    return trainer, model


def restore(trainer, state) -> None:
    opt = trainer.optimizer
    counts = opt.restore_checkpoint(state, from_hybrid=True)
    opt.restore_parameter_birth_steps(
        json.loads((CKPT / "training_state.json").read_text())
        .get("optimizer_parameter_birth_steps", {}))
    opt.step_index = int(state["step"])
    return counts


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--stage", default="full")
    parser.add_argument("--seq", type=int, default=2048)
    parser.add_argument("--accum", type=int, default=1)
    parser.add_argument("--pilot", action="store_true")
    args = parser.parse_args()

    OUT.mkdir(parents=True, exist_ok=True)
    t0 = time.time()
    state = load_state()
    trainer, model = build(args.seq, args.accum, ETA_HIGH,
                           OUT / f"ablation_seq{args.seq}_accum{args.accum}.csv")
    opt = trainer.optimizer
    adam_names = {p.name for p in opt.adam.parameters}
    muon_names = set(opt.muon_names)
    print(f"[setup] params={sum(p.data.size for p in model.parameters())} "
          f"adam={len(adam_names)} muon={len(muon_names)} "
          f"clip={trainer.grad_clip if hasattr(trainer,'grad_clip') else '?'} "
          f"({time.time() - t0:.1f}s)", flush=True)

    if args.pilot:
        restore(trainer, state)
        loss, grad_norm = trainer.train_step()
        print(f"[pilot] loss={loss:.6f} grad_norm={grad_norm:.6f} "
              f"({time.time() - t0:.1f}s)", flush=True)
        return

    branches = [
        ("base_no_update",   0.0, 0.0, frozenset()),
        ("both_low",         ETA_LOW, ETA_LOW, frozenset()),
        ("both_high",        ETA_HIGH, ETA_HIGH, frozenset()),
        ("muon_high_adam_low", ETA_HIGH, ETA_LOW, frozenset()),
        ("muon_low_adam_high", ETA_LOW, ETA_HIGH, frozenset()),
        ("both_high_freeze_embedding", ETA_HIGH, ETA_HIGH, frozenset({"embedding.W"})),
    ]

    results = {}
    sampler_anchor = {"rng": trainer.train_rng.bit_generator.state,
                      "shard_idx": trainer.current_train_shard_idx,
                      "source_idx": dict(trainer.current_train_source_shard_idx)}

    def reset_sampler():
        trainer.train_rng.bit_generator.state = sampler_anchor["rng"]
        trainer.current_train_shard_idx = sampler_anchor["shard_idx"]
        for key, value in sampler_anchor["source_idx"].items():
            trainer.current_train_source_shard_idx[key] = value

    def make_step(muon_eta: float, adam_eta: float, frozen: frozenset):
        if muon_eta == 0.0 and adam_eta == 0.0:
            def _noop(lr=None, active_parameter_names=None):
                return None
            return _noop

        def _step(lr=None, active_parameter_names=None):
            if active_parameter_names is None:
                active_adam = (adam_names - set(frozen)) if frozen else None
                active_muon = (muon_names - set(frozen)) if frozen else None
            else:
                active = {str(n) for n in active_parameter_names} - set(frozen)
                active_adam = active & adam_names
                active_muon = active & muon_names
            opt.adam.step(lr=adam_eta, active_parameter_names=active_adam)
            opt.muon.step(eta=muon_eta,
                          active_parameter_names=(active_muon if active_muon else None))
            opt._step_index += 1
        return _step

    def grad_stats():
        """Read the accumulated FP32 parameter gradients (post-clip, pre-step).

        With ``grad_clip=1000`` the clip scale is exactly 1.0, so these are the
        raw accumulated gradients.
        """
        buckets: dict[str, dict] = {}
        per_name = {}
        sub_vectors = {}
        group_totals = {"muon": 0.0, "adamw": 0.0, "all": 0.0}
        for p in opt.parameters:
            grad = p.grad
            values = grad if grad is not None else np.zeros_like(p.data)
            n2 = norm2(values)
            group = "muon" if p.name in muon_names else "adamw"
            group_totals[group] += n2
            group_totals["all"] += n2
            component = component_of(p.name)
            bucket = buckets.setdefault(component, {"n2": 0.0, "elements": 0, "arrays": 0})
            bucket["n2"] += n2
            bucket["elements"] += int(values.size)
            bucket["arrays"] += 1
            per_name[p.name] = {"group": group, "component": component,
                                "grad_rms": rms(values), "grad_n2": n2}
            sub_vectors[p.name] = subsample(values)
        for component, bucket in buckets.items():
            bucket["grad_norm"] = math.sqrt(bucket["n2"])
            bucket["grad_rms"] = math.sqrt(bucket["n2"] / bucket["elements"])
        for key in group_totals:
            group_totals[key] = math.sqrt(group_totals[key])
        flat_sub = np.concatenate([sub_vectors[n] for n in sorted(sub_vectors)])
        return {"buckets": buckets, "per_name": per_name, "groups": group_totals,
                "sub_vector": flat_sub}

    def master_snapshot():
        """Group-relative snapshot: adam/muon master lists are indexed per group."""
        snapshot = {}
        for group in (opt.adam, opt.muon):
            for index, p in enumerate(group.parameters):
                snapshot[p.name] = np.array(group.master_weights[index],
                                            dtype=np.float32, copy=True)
        return snapshot

    captured: dict[str, object] = {}
    production_zero_grad = opt.zero_grad

    def capture_then_zero_grad():
        """The trainer zeroes the accumulated gradients at the end of train_step,
        so the pre-update gradient statistics must be read before that happens.
        """
        captured["grad_stats"] = grad_stats()
        production_zero_grad()

    opt.zero_grad = capture_then_zero_grad

    for (label, muon_eta, adam_eta, frozen) in branches:
        restore(trainer, state)
        trainer.optimizer.step = make_step(muon_eta, adam_eta, frozen)
        reset_sampler()
        before = master_snapshot()
        captured.clear()
        loss, grad_norm = trainer.train_step()
        pre_stats = captured["grad_stats"]
        after = master_snapshot()
        delta_vectors = {}
        delta_buckets: dict[str, dict] = {}
        for p in opt.parameters:
            name = p.name
            d = after[name] - before[name]
            n2 = norm2(d)
            component = component_of(name)
            bucket = delta_buckets.setdefault(component, {"n2": 0.0, "elements": 0})
            bucket["n2"] += n2
            bucket["elements"] += int(d.size)
            delta_vectors[name] = subsample(d)
        delta_flat = np.concatenate([delta_vectors[n] for n in sorted(delta_vectors)])
        for bucket in delta_buckets.values():
            bucket["delta_norm"] = math.sqrt(bucket["n2"])
            bucket["delta_rms"] = math.sqrt(bucket["n2"] / bucket["elements"])

        # Fixed probe: identical batch (sampler state rewound), no update applied.
        trainer.optimizer.step = make_step(0.0, 0.0, frozenset())
        reset_sampler()
        probe_loss, probe_grad_norm = trainer.train_step()
        probe_stats = captured["grad_stats"]

        results[label] = {
            "muon_eta": muon_eta, "adam_eta": adam_eta,
            "frozen": sorted(frozen),
            "train_loss": loss, "train_grad_norm": grad_norm,
            "probe_loss": probe_loss, "probe_grad_norm": probe_grad_norm,
            "pre_groups": pre_stats["groups"],
            "probe_groups": probe_stats["groups"],
            "probe_components": {c: {k: v[k] for k in ("grad_norm", "grad_rms", "elements")}
                                for c, v in probe_stats["buckets"].items()},
            "pre_components": {c: {k: v[k] for k in ("grad_norm", "grad_rms", "elements")}
                              for c, v in pre_stats["buckets"].items()},
            "delta_components": {c: {k: v[k] for k in ("delta_norm", "delta_rms", "elements")}
                                for c, v in delta_buckets.items()},
            "probe_sub_vector": probe_stats["sub_vector"],
            "delta_sub_vector": delta_flat,
        }
        print(f"[{label}] muon_eta={muon_eta:.4e} adam_eta={adam_eta:.4e} "
              f"train_grad_norm={grad_norm:.6f} probe_grad_norm={probe_grad_norm:.6f} "
              f"train_loss={loss:.6f} probe_loss={probe_loss:.6f} "
              f"({time.time() - t0:.0f}s)", flush=True)



    reference = results["base_no_update"]
    for label, record in results.items():
        record["probe_vs_base_grad_ratio"] = (
            record["probe_groups"]["all"] / max(reference["probe_groups"]["all"], 1e-300))
        record["cos_probe_grad_vs_base"] = cosine_of(
            record["probe_sub_vector"], reference["probe_sub_vector"])
        record["cos_update_vs_both_high"] = cosine_of(
            record["delta_sub_vector"], results["both_high"]["delta_sub_vector"])

    print("\n== fixed-probe gradient response (all values on identical probe batch) ==")
    base_norm = reference["probe_groups"]["all"]
    print(f"{'branch':32} {'probe|g|':>10} {'ratio vs no-update':>12} "
          f"{'cos(dir, no-update)':>10} {'cos(dir, both_high)':>10}")
    for label, record in results.items():
        print(f"{label:32} {record['probe_groups']['all']:10.6f} "
              f"{record['probe_vs_base_grad_ratio']:12.4f} "
              f"{record['cos_probe_grad_vs_base']:10.4f} "
              f"{record['cos_update_vs_both_high']:10.4f}")

    print("\n== per-component probe gradient norms ==")
    components = sorted(reference["probe_components"])
    print(f"{'component':20} " + " ".join(f"{label[:14]:>14}" for label in results))
    for component in components:
        cells = []
        for label in results:
            entry = results[label]["probe_components"].get(component)
            cells.append(f"{entry['grad_norm']:14.6f}" if entry else f"{'-':>14}")
        print(f"{component:20} " + " ".join(cells))

    print("\n== realised update displacement per component ==")
    print(f"{'component':20} {'delta_norm(high)':>18} {'delta_rms(high)':>16} "
          f"{'0.2*eta_h':>10} {'eta_h':>10}")
    for component in sorted(reference["delta_components"]):
        entry = results["both_high"]["delta_components"][component]
        print(f"{component:20} {entry['delta_norm']:18.6e} {entry['delta_rms']:16.6e} "
              f"{0.2 * ETA_HIGH:10.4e} {ETA_HIGH:10.4e}")

    serialisable = {label: {k: v for k, v in record.items()
                            if k not in ("probe_sub_vector", "delta_sub_vector")}
                    for label, record in results.items()}
    (OUT / f"ablation_seq{args.seq}_accum{args.accum}.json").write_text(
        json.dumps(serialisable, indent=2))
    print(f"\nwrote {OUT / f'ablation_seq{args.seq}_accum{args.accum}.json'} "
          f"in {time.time() - t0:.0f}s")


if __name__ == "__main__":
    main()
