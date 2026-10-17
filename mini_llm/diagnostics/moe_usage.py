"""0080AQ observational expert-routing utilization diagnostics.

Only reduced device statistics are retained.  Sampling a few actual TRAINING
forwards avoids CUDA synchronization/large CPU transfers inside each layer.
Indices are treated as categorical observations, never as differentiable data.
This code is not involved in inference, router gradients, or checkpoint state.
"""

import math

from mini_llm.backend import xp, asnumpy


EXPERT_USAGE_COLUMNS = (
    "step", "layer", "expert", "n_experts", "top_k",
    "sampled_microbatches", "positions", "selections",
    "selection_share", "token_hit_rate", "mean_router_probability",
    "normalized_entropy", "mean_selected_probability_mass",
)


class ExpertUsageCollector:
    """Accumulate a small device-side summary for each Transformer layer."""

    def __init__(self):
        self.layers = {}
        self.sampled_microbatches = 0

    def observe(self, layer, indices, full_probs):
        """Read routing decisions without copying retained activation tensors.

        Called only for sampled primary forward passes, *not* backward replays.
        Includes all positions actually executed by the Transformer, including
        sequence padding; these counts measure computational expert routing,
        not the supervised-label distribution.
        """
        n_experts = int(full_probs.shape[-1])
        top_k = int(indices.shape[-1])
        if full_probs.shape[:-1] != indices.shape[:-1]:
            raise ValueError("expert utilization: routing indices/probability shape mismatch")
        count = int(indices.size // top_k)
        if count == 0:
            return
        # All operations run on the current device; there is no .get()/item()
        # or host synchronization in the forward path.
        selections = xp.bincount(
            indices.reshape(-1).astype(xp.int32, copy=False),
            minlength=n_experts,
        ).astype(xp.float64)
        preference = xp.sum(full_probs, axis=tuple(range(full_probs.ndim - 1)),
                            dtype=xp.float64)
        p = full_probs.astype(xp.float32, copy=False)
        ent_sum = -xp.sum(p * xp.log(xp.maximum(p, 1e-20)),
                          dtype=xp.float64)
        chosen = xp.take_along_axis(p, indices, axis=-1)
        mass_sum = xp.sum(chosen, dtype=xp.float64)

        layer = int(layer)
        existing = self.layers.get(layer)
        if existing is None:
            self.layers[layer] = {
                "n_experts": n_experts, "top_k": top_k, "positions": count,
                "selections": selections, "preference": preference,
                "entropy": ent_sum, "selected_mass": mass_sum,
            }
        else:
            if existing["n_experts"] != n_experts or existing["top_k"] != top_k:
                raise ValueError("expert utilization: expert layout changed inside a layer")
            existing["positions"] += count
            existing["selections"] += selections
            existing["preference"] += preference
            existing["entropy"] += ent_sum
            existing["selected_mass"] += mass_sum

    def rows(self, step):
        """One bounded D2H transfer for all equally sized per-layer summaries."""
        if not self.layers:
            return []
        ordered = sorted(self.layers.items())
        # All layers of one MoE Transformer have the same number of experts.
        # A single stacked array avoids 32 separate device synchronizations.
        n_experts = ordered[0][1]["n_experts"]
        if any(r["n_experts"] != n_experts for _, r in ordered):
            raise ValueError("expert utilization expects equal expert counts per layer")
        packs = []
        for _, stats in ordered:
            packs.append(xp.concatenate((
                stats["selections"], stats["preference"],
                xp.stack((stats["entropy"], stats["selected_mass"])),
            )))
        data = asnumpy(xp.stack(packs, axis=0))
        rows = []
        for (layer, stats), cpu in zip(ordered, data):
            n = stats["n_experts"]
            k = stats["top_k"]
            positions = stats["positions"]
            entropy = float(cpu[2 * n]) / (positions * math.log(n)) if n > 1 else 0.0
            selected_mass = float(cpu[2 * n + 1]) / positions
            for expert in range(n):
                selections = int(round(float(cpu[expert])))
                rows.append({
                    "step": int(step), "layer": layer, "expert": expert,
                    "n_experts": n, "top_k": k,
                    "sampled_microbatches": self.sampled_microbatches,
                    "positions": positions, "selections": selections,
                    "selection_share": selections / (positions * k),
                    "token_hit_rate": selections / positions,
                    "mean_router_probability": float(cpu[n + expert]) / positions,
                    "normalized_entropy": entropy,
                    "mean_selected_probability_mass": selected_mass,
                })
        return rows


def summarize_usage(rows):
    """Concise per-layer worst-case overview, not an auxiliary loss."""
    if not rows:
        return None
    by_layer = {}
    for row in rows:
        by_layer.setdefault(row["layer"], []).append(row)
    summaries = []
    for layer, layer_rows in sorted(by_layer.items()):
        summaries.append({
            "layer": layer,
            "min_share": min(r["selection_share"] for r in layer_rows),
            "max_share": max(r["selection_share"] for r in layer_rows),
            "entropy": layer_rows[0]["normalized_entropy"],
            "selected_mass": layer_rows[0]["mean_selected_probability_mass"],
            "under_one_percent": sum(r["token_hit_rate"] < 0.01 for r in layer_rows),
        })
    return {
        "layers": len(by_layer),
        "min_expert_share": min(s["min_share"] for s in summaries),
        "max_expert_share": max(s["max_share"] for s in summaries),
        "mean_entropy": sum(s["entropy"] for s in summaries) / len(summaries),
        "mean_selected_mass": sum(s["selected_mass"] for s in summaries) / len(summaries),
        "under_one_percent": sum(s["under_one_percent"] for s in summaries),
        "worst_layers": sorted(summaries, key=lambda s: s["min_share"])[:3],
    }
