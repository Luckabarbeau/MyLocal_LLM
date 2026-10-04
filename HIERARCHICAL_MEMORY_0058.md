# 0058 Hierarchical Memory Context Routing

Patch 0058 separates the amount of history that can be searched from the number
of tokens that enter the expensive Transformer trunk.  The invariant is:

> **addressable memory horizon != deep Transformer context length**

A memory model performs one cheap top-level search over distant history,
reopens only selected full-resolution token blocks, appends a contiguous recent
context and bounded target region, and sends only that active sequence through
the existing Transformer stack.

## Length semantics

`MemoryContextConfig.memory_length` is the **total pre-target history horizon**.
It includes the contiguous recent region.  Therefore:

```text
distant_memory_length = memory_length - recent_length
retrieved_length       = block_size * top_k_blocks
active_length          = retrieved_length + recent_length + target_length
source_input_length    = memory_length + target_length
```

For the production `wide-500m-memory-{16k,32k,64k}` presets the active deep
sequence stays at 7,168 tokens:

```text
retrieved historical tokens = 16 * 128 = 2,048
recent contiguous tokens                       = 4,096
target input rows                              = 1,024
active Transformer rows                        = 7,168
```

Only the number of distant blocks scanned by the shallow router grows when
`memory_length` grows.  The existing `wide-500m-context-*` presets remain
unchanged direct-sequence experiments.

## Sample and target semantics

The packed-shard sampler reads a contiguous window of
`memory_length + target_length + 1` corpus tokens.  It returns:

```text
source input rows: [0 : memory_length + target_length]
target labels:     [memory_length + 1 : memory_length + target_length + 1]
```

The final `target_length` **input rows** therefore predict the following
`target_length` next tokens.  The historical and recent rows are conditioning
only and receive no direct LM-head loss.

The router query is formed only from the last `router_query_length` rows of the
recent pre-target context.  Target input rows are not accepted by the router
API, so changing them cannot change the selected memory blocks.

## Top-level router

`HierarchicalMemoryRouter` reuses the existing handwritten primitives:

- `HistoryBlockPooler` for cheap block summaries;
- `CausalQueryPooler` for the recent-context query;
- selected top-k softmax and its handwritten backward;
- the same fixed-top-k backward convention already used elsewhere in the repo.

The default history pool is a mean.  A 64k horizon with 4,096 recent tokens and
128-token blocks searches 480 complete distant blocks.  Block summaries are
projected from `d_model` to `router_dim=64` only after pooling; there is no deep
64k memory encoder.

Top-k identities are discrete.  The selected probabilities remain trainable
through a neutral-at-uniform gate on each reopened exact block:

```text
gate_i = 1 + router_weight_scale * (K * p_i - 1)
```

At the default scale 1 this is `K*p_i`; uniform selected probabilities produce
a gate of exactly one.  This gives target LM loss a differentiable path into
selected router scores without replacing token content by pooled summaries.
There is intentionally no gradient through changes in the discrete selected
block identities.

## Active context and positions

Top-k returns blocks in score order.  0058 sorts selected block IDs by source
position before reopening them, yielding:

```text
[selected distant blocks in chronological order]
[recent contiguous context]
[target input rows]
```

The active tensor is compact, but RoPE does **not** use compact indices.
`position_ids` carry the original sample-local source positions through
`DecoderLanguageModel -> TransformerBlock -> GQAAttention -> rope_forward`.
Block checkpoint replay retains and reuses the exact same position-ID tensor.
The ordinary model path remains backward compatible: omitting `position_ids`
still uses `0..T-1`.

Because selected blocks are sorted chronologically and all retrieved rows occur
before recent/target rows, compact causal order agrees with source-time causal
order.  Current local/dilated/global sparse kernels still define neighborhood
geometry in compact active-array coordinates.  Consequently, tokens on two
retrieved blocks separated by a large source gap can be adjacent for a local
head.  RoPE distances remain correct, but a future patch may make local masks
source-distance-aware.  This is a known 0058 limitation, not a causality leak.

## Target-only LM head

Both full-logit and chunked-LM-head paths project only the target hidden rows in
a hierarchical-memory model.  During backward the LM-head gradient is scattered
into a zero-initialized active hidden tensor at the target slice.  Transformer
backward then naturally propagates target loss into selected/recent conditioning
states.  Conditioning rows are never assigned their own language-model labels.

## Memory lifecycle

The long history is embedded only for shallow routing.  With the default mean
history pool, the pooler backward cache retains shapes rather than the full
history embedding tensor.  After top-k selection the selected exact token IDs
are re-embedded, and the temporary full distant-history embedding is released
before the deep Transformer stack is evaluated.  Handwritten backward retains
source token IDs and replays selected embedding lookup as needed.

Using `history_pooling="learned"` is supported, but the existing learned
`HistoryBlockPooler` cache retains its full block tensor.  The production 0058
presets deliberately use `history_pooling="mean"` to preserve the intended
memory lifecycle.

## Configuration and runtime overrides

Canonical architecture values are stored in `MemoryContextConfig`.  The new
presets are:

```text
wide-500m-memory-16k
wide-500m-memory-32k
wide-500m-memory-64k
```

They are shape-compatible with each other; changing the memory horizon does not
change Transformer, embedding, LM-head, or router parameter shapes.

Optional runtime overrides are:

```text
MINI_LLM_HIERARCHICAL_MEMORY
MINI_LLM_MEMORY_LENGTH
MINI_LLM_MEMORY_BLOCK_SIZE
MINI_LLM_MEMORY_TOP_K_BLOCKS
MINI_LLM_MEMORY_RECENT_TOKENS
MINI_LLM_MEMORY_TARGET_TOKENS
MINI_LLM_MEMORY_QUERY_TOKENS
MINI_LLM_MEMORY_ROUTER_DIM
```

`--context-length` remains a direct-sequence control and is rejected for an
enabled hierarchical-memory preset to avoid conflating memory horizon with deep
context length.

0058 remains compatible with full optimizer offload, chunked LM-head training,
and block activation checkpointing.  The top-level memory route is selected
once per sample; Transformer block replay never reruns top-k selection.

## Diagnostics and profiling

Startup logging distinguishes addressable, distant, retrieved, recent, target,
active, query, and source-sample lengths.  Performance scopes separately expose
memory embedding/router/active-build work and the existing per-layer/model,
LM-head, backward, and optimizer scopes.  Profiled memory runs report target,
active, and source tokens/s rather than one ambiguous throughput number.

Set `MINI_LLM_MEMORY_ROUTER_DIAGNOSTICS=1` to periodically report selected-block
entropy, uniqueness, duplicate count, recent-block fraction, source distance,
and score statistics.  No auxiliary routing/load-balancing loss is introduced
in 0058.

## Validation

Reference correctness:

```bash
MINI_LLM_BACKEND=numpy python validate_0058_hierarchical_memory.py
```

Actual CuPy/BF16 path:

```bash
MINI_LLM_BACKEND=cupy python validate_0058_hierarchical_memory.py --bf16
```

The validator checks fixed active-length scaling, target-independent memory
selection, chronological reopened positions, a finite full forward/backward,
nonzero selected-route router gradients, and block-checkpoint equivalence.
