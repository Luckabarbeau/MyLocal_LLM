"""Read-only component gradient and realized optimizer-update diagnostics.

The monitor samples bounded slices from representative early/middle/final Model B
parameters. It reads normalized/clipped gradients and FP32 optimizer masters before
the real step, then measures the realized master-weight delta afterward. It never
writes model gradients, weights, optimizer state, or sampling state.
"""
from __future__ import annotations

import csv
import math
import re
from pathlib import Path

import numpy as np

from mini_llm.diagnostics.gradient_noise import _sample_slices
from mini_llm.optim.hybrid_muon import HybridMuonAdamW


FIELDS = (
    "step", "component", "layer_role", "parameter_name", "optimizer",
    "parameter_count", "sample_count", "learning_rate", "effective_learning_rate",
    "global_clip_scale", "parameter_rms", "gradient_rms",
    "gradient_to_parameter_rms", "update_rms", "update_to_parameter_rms",
    "update_to_gradient_rms", "descent_gradient_cosine",
)


def component_name(parameter_name):
    """Stable Model B component taxonomy derived from canonical parameter names."""
    name = str(parameter_name)
    if name == "embedding.W":
        return "tied_embedding_unembedding"
    if name == "final_norm.gamma":
        return "final_norm"
    if ".norm" in name:
        return "block_norm"
    if ".attention.matrix_pool.score" in name:
        return "matrix_pool_score"
    if ".attention.matrix_pool.position_bias" in name:
        return "matrix_pool_position_bias"
    if ".attention.global." in name:
        return "compressed_global_attention"
    if ".attention.Wq" in name or ".attention.Wk" in name or ".attention.Wv" in name:
        return "local_attention_qkv"
    if ".attention.Wo" in name:
        return "local_attention_output"
    if ".dense_ffn.expert.W_gate" in name or ".dense_ffn.expert.W_up" in name:
        return "ffn_gate_up"
    if ".dense_ffn.expert.W_down" in name:
        return "ffn_down"
    return "other"


def _representative_layers(parameters):
    layers = sorted({
        int(match.group(1))
        for parameter in parameters
        if (match := re.match(r"^blocks\.(\d+)\.", parameter.name or ""))
    })
    if not layers:
        return {}
    positions = (("early", 0), ("middle", len(layers) // 2), ("final", len(layers) - 1))
    return {layers[index]: role for role, index in positions}


def select_component_parameters(parameters, sample_elements=16_384):
    """Select every component in representative layers plus top-level tensors."""
    parameters = [parameter for parameter in parameters if parameter.grad is not None]
    layer_roles = _representative_layers(parameters)
    selected = []
    for parameter in parameters:
        name = parameter.name or ""
        match = re.match(r"^blocks\.(\d+)\.", name)
        if match:
            layer = int(match.group(1))
            if layer not in layer_roles:
                continue
            role = layer_roles[layer]
        elif name in {"embedding.W", "final_norm.gamma"}:
            role = "top_level"
        else:
            continue
        selected.append((
            component_name(name), role, parameter,
            _sample_slices(parameter.shape, limit=int(sample_elements)),
        ))
    return selected


def _host(value):
    return np.array(value.get() if hasattr(value, "get") else value,
                    dtype=np.float32, copy=True)


def _rms(value):
    array = np.asarray(value, dtype=np.float64)
    return float(np.sqrt(np.mean(array * array))) if array.size else 0.0


def _cosine(left, right):
    a = np.asarray(left, dtype=np.float64).reshape(-1)
    b = np.asarray(right, dtype=np.float64).reshape(-1)
    denom = float(np.linalg.norm(a) * np.linalg.norm(b))
    return float(np.dot(a, b) / denom) if denom > 0 else None


class ComponentUpdateMonitor:
    """Measure sampled component gradients and actual FP32 master deltas."""

    def __init__(self, parameters, optimizer, *, interval=1000,
                 output="./logs/component_updates.csv", sample_elements=16_384):
        if int(interval) < 1 or int(sample_elements) < 1:
            raise ValueError("component update interval/sample size must be positive")
        self.optimizer = optimizer
        self.interval = int(interval)
        self.output = Path(output)
        self.sample_elements = int(sample_elements)
        self.step = None
        self.snapshots = []
        self.refresh(parameters)

    def _optimizer_locations(self):
        locations = {}
        if isinstance(self.optimizer, HybridMuonAdamW):
            for algorithm, group in (("adamw", self.optimizer.adam),
                                     ("muon", self.optimizer.muon)):
                for index, parameter in enumerate(group.parameters):
                    locations[id(parameter)] = (
                        algorithm, group.master_weights[index], parameter,
                    )
        else:
            for index, parameter in enumerate(self.optimizer.parameters):
                locations[id(parameter)] = (
                    "adamw", self.optimizer.master_weights[index], parameter,
                )
        return locations

    def refresh(self, parameters):
        self.selected = select_component_parameters(
            parameters, sample_elements=self.sample_elements,
        )
        self.locations = self._optimizer_locations()
        unknown = [parameter.name for _, _, parameter, _ in self.selected
                   if id(parameter) not in self.locations]
        if unknown:
            raise ValueError(f"component monitor parameters absent from optimizer: {unknown}")
        self.snapshots = []

    def begin(self, completed_step):
        self.step = (
            int(completed_step) + 1
            if (int(completed_step) + 1) % self.interval == 0 else None
        )
        self.snapshots = []
        return self.step is not None

    def capture_before(self, active_parameters, *, learning_rate, clip_scale):
        if self.step is None:
            return
        active = {id(parameter) for parameter in active_parameters}
        eta = float(learning_rate)
        if not math.isfinite(eta) or eta < 0:
            raise ValueError("component monitor learning rate must be finite nonnegative")
        for component, role, parameter, selector in self.selected:
            if id(parameter) not in active:
                continue
            algorithm, master, _ = self.locations[id(parameter)]
            effective_eta = eta
            if algorithm == "adamw" and parameter.name == getattr(
                    self.optimizer, "unembedding_parameter_name", None):
                effective_eta *= float(self.optimizer.unembedding_lr_ratio)
            self.snapshots.append({
                "component": component,
                "layer_role": role,
                "parameter": parameter,
                "selector": selector,
                "optimizer": algorithm,
                "master": master,
                "before": _host(master[selector]),
                "gradient": _host(parameter.grad[selector]),
                "learning_rate": eta,
                "effective_learning_rate": effective_eta,
                "clip_scale": float(clip_scale),
            })

    def finish(self):
        if self.step is None:
            return []
        rows = []
        for snapshot in self.snapshots:
            parameter = snapshot["parameter"]
            before = snapshot["before"]
            gradient = snapshot["gradient"]
            after = _host(snapshot["master"][snapshot["selector"]])
            descent = before - after
            parameter_rms = _rms(before)
            gradient_rms = _rms(gradient)
            update_rms = _rms(descent)
            rows.append({
                "step": self.step,
                "component": snapshot["component"],
                "layer_role": snapshot["layer_role"],
                "parameter_name": parameter.name,
                "optimizer": snapshot["optimizer"],
                "parameter_count": int(parameter.size),
                "sample_count": int(before.size),
                "learning_rate": snapshot["learning_rate"],
                "effective_learning_rate": snapshot["effective_learning_rate"],
                "global_clip_scale": snapshot["clip_scale"],
                "parameter_rms": parameter_rms,
                "gradient_rms": gradient_rms,
                "gradient_to_parameter_rms": (
                    gradient_rms / parameter_rms if parameter_rms > 0 else None
                ),
                "update_rms": update_rms,
                "update_to_parameter_rms": (
                    update_rms / parameter_rms if parameter_rms > 0 else None
                ),
                "update_to_gradient_rms": (
                    update_rms / gradient_rms if gradient_rms > 0 else None
                ),
                "descent_gradient_cosine": _cosine(descent, gradient),
            })
        self.snapshots = []
        return rows

    def cancel(self):
        self.step = None
        self.snapshots = []

    def commit(self, rows):
        if not rows:
            return
        self.output.parent.mkdir(parents=True, exist_ok=True)
        need_header = not self.output.exists() or self.output.stat().st_size == 0
        if not need_header:
            with self.output.open(newline="") as handle:
                if tuple(next(csv.reader(handle), ())) != FIELDS:
                    raise ValueError("existing component-update CSV header differs")
        with self.output.open("a", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=FIELDS)
            if need_header:
                writer.writeheader()
            writer.writerows(rows)
