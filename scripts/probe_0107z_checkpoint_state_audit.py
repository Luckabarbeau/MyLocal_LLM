"""Read-only audit of the real Model B checkpoints (step 16200 branch origin).

Questions answered with trained tensors only (no synthetic initialisation):

1. Are the FP32 optimizer master weights in a checkpoint directory consistent
   with the BF16 ``model_params`` in the same directory, or are they offset in
   training time (the ``optimizer_step = train_step + 50`` anomaly seen in
   ``optimizer_state/scalars.json``)?
2. What is the real per-component ``RMS(W)`` of the trained weights, and the
   real momentum scale ``RMS(muon_m)`` / ``RMS(m / sqrt(v))``?
3. What parameter displacement did the two learning-rate branches *actually*
   produce?  ``..._16200`` -> ``..._16250_deterioration...`` is 50 steps of the
   high-LR branch; ``..._16200`` -> ``..._long`` (step 16500) is 300 steps of the
   low-LR branch.

Nothing here writes to the checkpoints or the active run.
"""

from __future__ import annotations

import csv
import hashlib
import json
import math
from pathlib import Path

import numpy as np
from ml_dtypes import bfloat16

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "logs" / "lr_gradient_audit_0107z"

BRANCHES = {
    "16200_origin": ROOT / "checkpoints" / "model_b_batch1_8k_norecompute_long_16200",
    "high_16250": ROOT / "checkpoints" /
                  "model_b_batch1_8k_norecompute_long_16250_deterioration_after_change_in_LR",
    "low_16500": ROOT / "checkpoints" / "model_b_batch1_8k_norecompute_long",
}

# Learning rates read from logs/model_b_batch1_8k_norecompute_test.csv.
ETA_HIGH = 2.841753146565134e-04
ETA_LOW = 7.104382866412835e-05
STEPS_HIGH = 50
STEPS_LOW = 300


def read_array(path: Path) -> np.ndarray:
    """Load a saved .npy, decoding BF16 (stored as 2-byte void) faithfully."""
    raw = np.load(path, allow_pickle=False)
    if raw.dtype.kind == "V" and raw.dtype.itemsize == 2:
        flat = np.frombuffer(np.ascontiguousarray(raw).tobytes(), dtype=bfloat16)
        return flat.reshape(raw.shape)
    return raw


def as_f32(values: np.ndarray) -> np.ndarray:
    return np.asarray(values, dtype=np.float32)


def rms(values: np.ndarray) -> float:
    flat = np.asarray(values, dtype=np.float64).reshape(-1)
    if flat.size == 0:
        return 0.0
    return float(math.sqrt(np.dot(flat, flat) / flat.size))


def dot(a: np.ndarray, b: np.ndarray) -> float:
    fa = np.asarray(a, dtype=np.float64).reshape(-1)
    fb = np.asarray(b, dtype=np.float64).reshape(-1)
    return float(np.dot(fa, fb))


def cosine(a: np.ndarray, b: np.ndarray) -> float:
    left = dot(a, a)
    right = dot(b, b)
    if left <= 0.0 or right <= 0.0:
        return 0.0
    return dot(a, b) / math.sqrt(left * right)


def classify(name: str) -> str:
    if "embedding" in name:
        return "embedding_tied"
    if "matrix_pool" in name:
        return "matrix_pool"
    if ".global." in name:
        return "compressed_global_attn"
    if "attention" in name:
        return "local_attn_proj"
    if "ffn" in name:
        return "ffn"
    if "norm" in name:
        return "norm"
    return "other"


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    origin = BRANCHES["16200_origin"]
    names = sorted(p.stem for p in (origin / "model_params").glob("*.npy"))
    muon_names = {p.stem for p in (origin / "optimizer_state" / "muon_m").glob("*.npy")}
    adam_names = {p.stem for p in (origin / "optimizer_state" / "m").glob("*.npy")}

    rows = []
    parity_worst = (0.0, "")
    drift_worst = (0.0, "")
    group_accum: dict[str, dict[str, float]] = {}

    for name in names:
        param = read_array(origin / "model_params" / f"{name}.npy")
        master = read_array(origin / "optimizer_state" / "master_weights" / f"{name}.npy")
        param_f32 = as_f32(param)
        master_rounded = as_f32(as_f32(master).astype(bfloat16))

        # (a) exact production parity: is the compute parameter bf16(master)?
        parity = rms(param_f32 - master_rounded)
        # (b) total master<->parameter gap: includes the master-only sub-ULP
        #     accumulation, i.e. the difference between the FP32 master and the
        #     value actually used by the forward pass.
        drift = rms(param_f32 - as_f32(master))
        if parity > parity_worst[0]:
            parity_worst = (parity, name)
        if drift > drift_worst[0]:
            drift_worst = (drift, name)

        high_master = read_array(
            BRANCHES["high_16250"] / "optimizer_state" / "master_weights" / f"{name}.npy")
        low_master = read_array(
            BRANCHES["low_16500"] / "optimizer_state" / "master_weights" / f"{name}.npy")

        disp_high = as_f32(high_master) - as_f32(master)
        disp_low = as_f32(low_master) - as_f32(master)

        is_muon = name in muon_names
        group = "muon" if is_muon else ("adamw" if name in adam_names else "untracked")

        muon_m_rms = m_rms = mv_rms = float("nan")
        if is_muon:
            momentum = read_array(origin / "optimizer_state" / "muon_m" / f"{name}.npy")
            muon_m_rms = rms(momentum)
        elif name in adam_names:
            m = read_array(origin / "optimizer_state" / "m" / f"{name}.npy")
            v = read_array(origin / "optimizer_state" / "v" / f"{name}.npy")
            m_rms = rms(m)
            # Adam's elementwise step multiplier m/sqrt(v) is scale-free; RMS of
            # it tells us the realised Adam step size in units of eta.
            mv_rms = rms(as_f32(m) / np.sqrt(np.maximum(as_f32(v), 1e-30)))

        weight_rms = rms(master)
        row = {
            "name": name,
            "shape": "x".join(str(d) for d in param.shape),
            "group": group,
            "component": classify(name),
            "rms_W": weight_rms,
            "rms_param_minus_bf16_master": parity,
            "rms_param_minus_master": drift,
            "rms_muon_m": muon_m_rms,
            "rms_m": m_rms,
            "rms_m_over_sqrtv": mv_rms,
            "rms_disp_high50": rms(disp_high),
            "rms_disp_low300": rms(disp_low),
            "cos_disp_high_vs_low": cosine(disp_high, disp_low),
        }
        rows.append(row)

        bucket = group_accum.setdefault(row["component"], {
            "arrays": 0.0, "elements": 0.0, "params": 0.0,
            "rms_W": 0.0, "rms_W2": 0.0,
            "disp_high2": 0.0, "disp_low2": 0.0,
            "muon_m2": 0.0, "adam_step2": 0.0, "n_muon": 0.0, "n_adam": 0.0,
        })
        elements = float(param.size)
        bucket["arrays"] += 1.0
        bucket["elements"] += elements
        bucket["params"] += elements * weight_rms ** 2
        bucket["rms_W2"] += elements * weight_rms ** 2
        bucket["disp_high2"] += elements * row["rms_disp_high50"] ** 2
        bucket["disp_low2"] += elements * row["rms_disp_low300"] ** 2
        if is_muon:
            bucket["n_muon"] += 1.0
            bucket["muon_m2"] += elements * muon_m_rms ** 2
        else:
            bucket["n_adam"] += 1.0
            if mv_rms == mv_rms:
                bucket["adam_step2"] += elements * mv_rms ** 2

    with (OUT / "checkpoint_component_arrays.csv").open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)

    print("== per-component summary (trained tensors) ==")
    header = (f"{'component':26} {'arrays':>6} {'elems':>11} {'RMS(W)':>10} "
              f"{'RMS(dW)50hi':>11} {'RMS(dW)300lo':>11} {'ratio/eta-pred':>14}")
    print(header)
    summary = {}
    for component in sorted(group_accum, key=lambda c: -group_accum[c]["elements"]):
        b = group_accum[component]
        n_el = b["elements"]
        rms_W = math.sqrt(b["rms_W2"] / n_el)
        d_hi = math.sqrt(b["disp_high2"] / n_el)
        d_lo = math.sqrt(b["disp_low2"] / n_el)
        # Design predictions for the realised displacement after S steps.
        pred_hi = 0.2 * ETA_HIGH * STEPS_HIGH if b["n_muon"] and not b["n_adam"] else ETA_HIGH * STEPS_HIGH
        pred_lo = 0.2 * ETA_LOW * STEPS_LOW if b["n_muon"] and not b["n_adam"] else ETA_LOW * STEPS_LOW
        print(f"{component:26} {int(b['arrays']):>6} {int(n_el):>11} {rms_W:10.4e} "
              f"{d_hi:11.4e} {d_lo:11.4e} "
              f"hi/pred={d_hi / max(pred_hi, 1e-300):6.3f} lo/pred={d_lo / max(pred_lo, 1e-300):6.3f}")
        summary[component] = {
            "arrays": int(b["arrays"]),
            "elements": int(n_el),
            "rms_W": rms_W,
            "rms_disp_high50": d_hi,
            "rms_disp_low300": d_lo,
            "rms_muon_m": (math.sqrt(b["muon_m2"] / n_el) if b["n_muon"] else None),
            "rms_adam_step_in_eta": (math.sqrt(b["adam_step2"] / n_el) if b["n_adam"] else None),
            "displacement_hi_vs_lo_ratio": d_hi / max(d_lo, 1e-300),
            "eta_step_ratio_expected": (ETA_HIGH * STEPS_HIGH) / (ETA_LOW * STEPS_LOW),
        }

    all_elements = sum(b["elements"] for b in group_accum.values())
    total_W2 = sum(b["rms_W2"] for b in group_accum.values())
    total_hi = sum(b["disp_high2"] for b in group_accum.values())
    total_lo = sum(b["disp_low2"] for b in group_accum.values())
    print("\n== global (all 302 arrays) ==")
    print(f"RMS(W)           = {math.sqrt(total_W2 / all_elements):.6e}")
    print(f"RMS(dW) 50 hi-LR = {math.sqrt(total_hi / all_elements):.6e}  "
          f"(design 0.2*eta_h*S = {0.2 * ETA_HIGH * STEPS_HIGH:.4e}, "
          f"S*eta_h = {ETA_HIGH * STEPS_HIGH:.4e})")
    print(f"RMS(dW) 300 lo-LR= {math.sqrt(total_lo / all_elements):.6e}  "
          f"(design 0.2*eta_l*S = {0.2 * ETA_LOW * STEPS_LOW:.4e})")
    print(f"hi/lo displacement ratio = "
          f"{math.sqrt(total_hi / total_lo):.4f}; "
          f"expected if displacement is LR-linear: "
          f"{(ETA_HIGH * STEPS_HIGH) / (ETA_LOW * STEPS_LOW):.4f}; "
          f"if LR-independent: 1.0")

    print("\n== master <-> parameter parity (the '+50' question) ==")
    print(f"worst RMS(param - bf16(master)) = {parity_worst[0]:.3e}  [{parity_worst[1]}]")
    print(f"worst RMS(param - master)       = {drift_worst[0]:.3e}  [{drift_worst[1]}]")
    print("Interpretation test: a bf16 rounding gap is ~1e-3 x RMS(W); a 50-step "
          f"training offset would be >= {0.2 * ETA_HIGH * STEPS_HIGH:.2e} (muon) / "
          f"{ETA_HIGH * STEPS_HIGH:.2e} (adam).")

    print("\n== update-vs-gradient scale on trained state ==")
    for component in sorted(group_accum, key=lambda c: -group_accum[c]["elements"]):
        b = group_accum[component]
        n_el = b["elements"]
        if b["n_muon"]:
            grad_rms = math.sqrt(b["muon_m2"] / n_el)
            print(f"{component:26} RMS(muon_m)={grad_rms:10.4e}  "
                  f"MuoN element step RMS=0.2*eta={0.2 * ETA_HIGH:8.3e}  "
                  f"amplification vs raw grad = {0.2 * ETA_HIGH / max(grad_rms, 1e-300):8.2f}x")
        else:
            step_in_eta = math.sqrt(b["adam_step2"] / n_el) if b["n_adam"] else float("nan")
            print(f"{component:26} AdamW element step = {step_in_eta:8.4f} x eta  "
                  f"(RMS(m/sqrt(v))={step_in_eta:10.4e} per element)")

    report = {"branches": {k: str(v) for k, v in BRANCHES.items()},
              "etas": {"high": ETA_HIGH, "low": ETA_LOW},
              "steps": {"high": STEPS_HIGH, "low": STEPS_LOW},
              "parity_worst": {"param_minus_bf16_master": parity_worst,
                               "param_minus_master": drift_worst},
              "components": summary,
              "global": {"rms_W": math.sqrt(total_W2 / all_elements),
                         "rms_disp_high50": math.sqrt(total_hi / all_elements),
                         "rms_disp_low300": math.sqrt(total_lo / all_elements)}}
    (OUT / "checkpoint_state_audit.json").write_text(json.dumps(report, indent=2))
    print(f"\nwrote {OUT / 'checkpoint_component_arrays.csv'}")
    print(f"wrote {OUT / 'checkpoint_state_audit.json'}")


if __name__ == "__main__":
    main()
