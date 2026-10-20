# 0080AP — Unified MoE expert routing (no top-1-specific switches)

## Mathematical definition

For **all** `n_experts=E` and `1 <= top_k=k <= E`:

1. Compute `z = x @ W_router + b_router`.
2. Calculate `p = softmax(z)` **over all E experts**, in FP32 for BF16/FP16.
3. Select the indices of the k largest logits (stable existing `argsort` convention).
4. Execute only these k experts and combine with their original probabilities:
   `y = sum_{i in top-k} p[i] * expert_i(x)`.

The selected weights **do not sum to one in general**. They sum to the softmax
probability mass of the chosen experts; this is exactly 1 only if k=E. This
is deliberate, because renormalizing a singleton selected set erases all
router gradients when k=1.

The discrete top-k indices are treated as locally fixed for the Jacobian. For
selected-weight derivatives `g_i = dL / dp_i` (0 for unselected experts),

`dL / dz_j = p_j * (g_j - sum_{i in top-k} p_i*g_i)`.

So every router logit receives an exact gradient (away from top-k ties),
including logits of unselected experts. Only executed expert FFN weights get
expert-parameter gradients. This rule is **identical for k=1,2,...,E**.

## Implementation

- `mini_llm/ops/router.py`: single shared `expert_topk_forward` and
  `expert_topk_backward` functions. No top-1-specific branch, field or flag.
- `mini_llm/ops/router_inference.py`: imports that same forward primitive.
- Remove `top1_router_full_softmax` from ModelConfig and all constructor
  plumbing; no per-preset selection.
- Keep the separate `selected_topk_softmax_*` primitive **unchanged** for the
  learned historical-context router. This is an MoE routing redesign, not a
  change to sparse-attention selection or 64K memory training.
- No change to the model parameter shapes, `top_k` settings, expert counts,
  activation offload implementation, 4-token decoder, or learning objective.

## Training, checkpoint and numerical compatibility

**The unified definition intentionally changes MoE behavior for existing
legacy top-k>1 networks**, which previously normalized *only among the
selected experts*. Even though tensor shapes and saved weights match, output
values and gradients do not. It is NOT bitwise/optimization equivalent to an
old k=2 checkpoint, and old frozen baseline losses necessarily change.
Retain the legacy training branch for old-model continuation. Start fresh
when comparing with the 0080AP unified-routing model.

For the new 0080AO top-1 model, the forward/backward mathematical rule is
unchanged. However, any checkpoint previously saved under 0080AO can include
the now-removed `top1_router_full_softmax` field in `config.json`; `ModelConfig`
strict JSON deserialization will not accept it. **Do not edit or overwrite
an existing checkpoint in place without a deliberate metadata migration and
a copy of the original.** New 0080AP scratch runs write the clean config.

Because selected probabilities no longer renormalize, the initial MoE
residual signal has a smaller expected magnitude when k<E: approximately
`k/E` for nearly uniform router scores (e.g. ~0.5 for k=1,E=2;
~0.33 for k=2,E=6), though top-k selection biases that estimate upward.
Monitor training throughput, gradient norm, expert utilization and validation
loss; a universal best scaling factor is not assumed.

## Tests

NumPy tests cover E/k=(2/1, 6/1, 6/2, 3/3), exact forward probabilities,
finite differences for logits and inputs, unselected-logit gradients, shared
inference/training parity, 32-layer backward, no special model flags and
existing MoE block tests.

The original **historical frozen** `tests/fixtures/0080l_cpu_reference.json`
remains intact. The test harness now reads the separately versioned
`tests/fixtures/0080ap_cpu_reference.json` for the unified mathematical
rule; the new fixture captures the new CPU loss, gradient signatures and
AdamW updates. Three old golden loss checks initially differed by about
2e-4, exactly because their router normalization changed. The 0080L
methodology now passes again against the 0080AP fixture, with both versions
available for historical comparison. CuPy BF16 GPU and integrated 32-layer
training are not tested here.
