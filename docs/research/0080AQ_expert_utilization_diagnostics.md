# 0080AQ — Training expert-utilization diagnostics (observational)

The 32-layer, two-expert top-1 model introduced in 0080AO and unified in
0080AP needs evidence that **both expert networks actually receive tokens**.
This patch **does not** change routing, expert weights, loss, or gradients.

## Enable and sampling cost

- `MINI_LLM_EXPERT_USAGE_INTERVAL=10`: collect every tenth **completed optimizer step**. Set `0` to disable completely or `1` to collect every step.
- `MINI_LLM_EXPERT_USAGE_MICROBATCHES=1`: number of accumulated training microbatches sampled per selected optimizer step. Set `32` to cover a full `grad-accum-steps=32` optimizer step, at increased diagnostic cost.
- The 0080AO deep32 launcher enables the default `10,1` combination.
- Counters collect only primary training forwards, not evaluation or activation-checkpoint backward replay. They are accumulated as small device arrays and transferred once at the finished-step logging boundary. No per-token values or full activations are retained by the diagnostics.
- Samples count **all positions executed by the main Transformer**, including nonsupervised padding and prefix positions. These are computational-load statistics, **not** a histogram of only supervised target tokens. Padding can bias utilization; review observed compact working length and valid-target fill alongside this metric.

## Output

With `--log-file ./logs/example.csv`, a separate file is written:

    ./logs/example.expert_usage.csv

One row **per layer and per expert** is appended for each observed optimizer step; there are 64 rows/step for the deep32 top-1 architecture. Older files are appended to after resume. Columns:

- `step`: completed optimizer update number.
- `layer`: zero-based main Transformer layer index (`0..31` for deep32).
- `expert`: zero-based expert index.
- `n_experts`, `top_k`, `sampled_microbatches`, `positions`: denominator/provenance.
- `selections`: number of routed token-to-expert assignments for that expert.
- `selection_share`: `selections / (positions * top_k)`; shares sum to 1 per layer.
- `token_hit_rate`: `selections / positions`; shares sum to `top_k` per layer.
- `mean_router_probability`: **unselected included**; mean full-softmax probability assigned to the expert before top-k. Useful to separate probabilities from discrete expert selection.
- `normalized_entropy`: `mean[-sum_i(p_i log p_i)] / log(n_experts)`; 1 means diffuse/equal, 0 highly confident. This is independent of selection and is repeated for each expert row in its layer.
- `mean_selected_probability_mass`: mean `sum_selected p_i`; for top-1 with two experts it measures mean winning-expert probability (initially often near 0.5). This matters under 0080AP's non-renormalized gating and is repeated per layer.

At observation steps the console prints a compact summary including the three worst-balanced layers. `experts_below_1pct` is a **diagnostic warning threshold** (it does not trigger load-balancing losses or otherwise affect training).

## Interpretation

For 2-expert top-1, the ideal *utilization* is not necessarily exactly 50/50, but consistent 99/1 or 100/0 utilization means one expert may receive almost no learning signal. Compare `selection_share` with `mean_router_probability`: the model can express soft uncertainty while assigning nearly every token to the same expert. A high normalized entropy with strongly unbalanced hard assignments can arise from small systematic logit differences or repeated padding. Low entropy and a skewed selection indicate confident expert preference.

The new CSV does **not** include router gradient norm or an auxiliary balancing loss. Those should be considered separately only after the data shows persistent collapse on representative *valid* text.

## Tests

```bash
MINI_LLM_BACKEND=numpy OPENBLAS_NUM_THREADS=1 python -m pytest -q \
    tests/test_0080aq_expert_usage_logging.py \
    tests/test_0080ap_unified_expert_routing.py \
    tests/test_moe_router.py
```

CUDA tests and end-to-end 32-layer throughput measurements have not been run here; those remain workstation checks.
