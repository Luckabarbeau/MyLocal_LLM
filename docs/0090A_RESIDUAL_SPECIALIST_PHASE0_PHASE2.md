# 0090A — Modular residual specialists: audit, interface, and CPU prototype

## Research goal and corrected size target

The **two-layer specialist is a smoke-test only**. A **12-layer, d=1024,
FFN=4096 specialist (~182M dense-block parameters, excluding adapters)** is
currently the favored size for actual research, subject to measured VRAM,
throughput, and held-out performance. The shared foundation is ~338M (20 ×
1024) and **remains frozen**. All specialists must be independently trainable,
checkpointed, combinable, and potentially deeper or wider than each other.

A specialist is **not inherently limited to the foundation's historical
retrieval**. Some should eventually own their own history readers, possibly
covering a greater addressable horizon or retrieving different source blocks.
This requires explicit causal source tokens and positions; the foundation's
final H0 alone cannot reconstruct information omitted by foundation retrieval.

## Repository-grounded Phase 0 audit

| Current component | Reusable contract | Issue / adaptation |
|---|---|---|
| `mini_llm/model/decoder_lm.py` | `forward_body(..., return_cache=False, return_target_slice=True)` produces normalized H0 without logits | Need frozen-loader preserving checkpoint `memory_context`; forward-only historical encoder should avoid backward caches |
| `mini_llm/parameter.py` | `Parameter` owns array and gradient | Default FP32 gradient allocation costs approximately 1.26 GiB for the 338M-parameter frozen foundation. Added *opt-in scoped* no-grad construction. |
| `mini_llm/ops/linear.py` | `output_proj.W.data`, `backward_input()` | Reuse shared frozen head, do **not** call head `.backward()` because it accumulates frozen head gradients |
| `DecoderLanguageModel.chunked_lm_head_*` | Existing tiled vocab CE and dH computation, accepts `target_slice`, `loss_mask` | Added `frozen_output_projection=True` to backwards, default remains False |
| `mini_llm/blocks/transformer_block.py` | Optimized attention + dense SwiGLU, forward/backward, replay cache support | New specialist wrapper manages its own blocks, not a second decoder embedding or vocab head |
| `mini_llm/config.py` | `AttentionLayerConfig` with `LocalAttentionConfig`, `ModelConfig` | For 12-layer specialists reuse 1 full-window / 15 local-512 heads as a practical initial schedule; no private-history schedule exists yet |
| `mini_llm/ops/historical_schedule.py` | Source-coordinate deterministic/learned schedules | **12 × 16 is currently rejected**; cannot instantiate foundation `HistoricalReadMemory` unchanged for a 12-layer specialist |
| `mini_llm/ops/historical_read.py` | Compact historical encoder and historical K/V reuse | `open(..., inference=True)` releases backward encoder intermediates. Need specialist-owned learned K/V and separate schedules later |
| `mini_llm/collective/online_training.py` | `load_frozen_teacher()` example and independent logits fusion | This loader **disables** long history and must **not** be used for specialists |
| `mini_llm/optim/hybrid_muon.py` | Hybrid optimizer and full CPU offload | Eligibility depends on parameter names; now recognizes `specialists.<id>.blocks.<n>.*` |
| `mini_llm/checkpoint/` | `save_checkpoint` / `load_checkpoint` arrays + optimizer | Specialist-only checkpoints need configuration/fingerprint/tokenizer validation in a subsequent phase |
| `mini_llm/data/mixed_shards.py` | `load_weighted_shard_sources`, source-based splitting | Reuse Python-Edu and its held-out validation shards without new tokenization |
| `mini_llm/train_extended.py`, `train_model.py` | Existing accumulation and logging | Do **not** insert specialist training into production autoregressive loop until isolated trainer tested |

### Architecture contract

The frozen checkpoint is authoritative. `FrozenFoundation.from_checkpoint(path)`
reads its `config.json`, instantiates the exact architecture **without any
parameter gradients**, loads all named weights, and keeps historical memory
active. It exposes normalized H0 and the original `target_slice`. It never
builds a foundation backward tape. The shared frozen output projection is used
once per tile, with gradients propagated to the corrected H, never W_U.

For each independent specialist:

    H0 = frozen_foundation(input_ids)
    delta = specialist(H0, source_positions)                # independent params
    H_corrected = H0 + delta                               # no new post-add norm
    z = H_corrected @ W_U                                  # frozen W_U

The last correction projection W_delta is zero-initialized. This gives **exact
zero-specialist baseline parity** in the same arithmetic path. With a zero
W_delta, early block gradients are zero until W_delta takes a first step; this
is an expected property of the initialization, and is tested.

Specialist parameter names are `specialists.<id>.blocks.*` and
`specialists.<id>.correction_proj.W`. Only these parameters should enter its
optimizer; the underlying frozen foundation object is *never* added.

### Historical access — explicit long-term contract

Specialists should carry a per-specialist `memory_mode`:

1. `foundation_only` — implemented prototype. Specialist sees historical
   information already encoded into H0. It may use its own causal processing
   of the current working window, but has no independent history reader.
2. `private` — **not yet implemented; fails with `NotImplementedError`**.
   Specialist has its own compact history encoder and K/V reader over source
   tokens, possibly with different `history_horizon`, `history_block_size`,
   `history_retrieval_blocks`, number of historical heads and schedule.

For private memory later, a clean interface is:

    specialist.forward_delta(
        H0, position_ids,
        history_context=HistoricalSourceView(
            token_ids, valid_start, valid_mask, source_positions, ...),
    )

`HistoricalSourceView` is **design-only** and is not a current API. The
foundation can consume 65,536 source tokens while a specialist independently
has a 128K or smaller store, provided same-document causal sampling supplies
those tokens. Do not equate history *available for retrieval* with history
*selected and processed*; they can differ among specialists.

The specialist's history may be built from its own token-embedding projection,
from frozen foundation embeddings, or from frozen/learned compact features.
These choices affect expressivity and VRAM. Private learned encoders/readers
must receive gradients, but never propagate into frozen foundation parameters.
A separately trained specialist checkpoint must include its own memory
parameters and routing configuration. No padded `[B,65K,65K]` masks.

**Before private memory is added, extend the historical head schedule** to
support `12 × 16`, without modifying the existing production `20 × 16`
foundation schedule. Confirm RoPE absolute positions, causal eligibility,
nonfuture block selection, ragged documents, replay correctness, and routing
interference. A specialist with more history than the foundation additionally
requires a data-sampler contract for the extra earlier tokens, correct EOS
boundaries, and independent memory validity masks.

### Important caution: positional information

In `historical_read` the frozen working positions begin at `memory_length -
target_length`; the current Phase 2 `forward_delta` accepts optional
`position_ids`. Its local-only attention works causally with default relative
positions; when private historical keys are introduced, **absolute source
positions and validity metadata are mandatory**. The current H0-only API does
not export the irregular selected-token positions for `routed_prefix`; that
mode needs an explicit source-position adapter before use with independent
specialist memory.

### Reference prototype (new code)

* `mini_llm/specialists/frozen_foundation.py`: strict frozen checkpoint loader,
  forward H0, shared vocabulary head, target-only tiled CE and backward dH.
* `mini_llm/specialists/residual_specialist.py`: independent blocks, hybrid
  causal attention layout, zero correction projection, no-history forward,
  backward and block recomputation. Defaults to 2 layers for the smoke test;
  `SpecialistConfig(n_layers=12, ...)` creates the intended 12-layer target.
* `tests/test_residual_specialist_phase1.py` and
  `tests/test_residual_specialist_phase2.py`: parity, no foundation gradients,
  tiled frozen head, target slice, first/second step gradient behavior,
  checkpoint-based initialization, recompute and prefix-causality tests.

### CPU results (Oct 26, 2026)

Command (repository root):

```bash
MINI_LLM_BACKEND=numpy python -m pytest \
  tests/test_hybrid_muon.py tests/test_checkpoint.py \
  tests/test_0083c_hybrid_current_attention.py \
  tests/test_residual_specialist_phase1.py \
  tests/test_residual_specialist_phase2.py -q
```

Result: **43 passed, 1 skipped** (NumPy-only). The frozen loader tests use a
synthetic **small** checkpoint, not an actual 338M model. No RTX 5080, CuPy,
BF16 or 64K-production test has been performed here.

### Remaining work before real specialist pretraining

1. Validate frozen full 338M checkpoint load and short-context/H0 parity on
   RTX 5080; repeat with enabled `historical_read` and real sampler metadata.
2. Validate 2/4-layer specialist BF16 CuPy forward/backward/recompute, and
   strict causal-prefix invariance using real historical source positions.
3. Measure peak reserved/allocated memory, GPU forward vs backward time,
   wall-time/token throughput at microbatches 1, 2, and 3. Do this separately
   at **12 layers**, with and without recompute, before treating 12L as feasible.
4. Implement isolated train/validation sampler reusing `python_edu` shards,
   matching supervised-target masks and accumulation weights; avoid taking over
   the production `ExtendedTrainer` unmodified, which expects a decoder model.
5. Add specialist-only checkpoint save/resume with immutable foundation and
   tokenizer **cryptographic** fingerprints, separate optimizer state, source
   provenance and exact training step tracking.
6. Run controlled Python-Edu held-out CE and executable Python tasks; compare
   2/4/12L vs residual MLP and LoRA, matched by supervised tokens and compute.
7. Implement per-specialist private historical readers and longer horizons as
   an independent later experiment, then test their scaling and value.
8. Only after independent learning is established: multi-specialist combiner
   with explicit foundation fallback and context-dependent gating.

### Memory expectations (heuristics, not measured)

Foundation BF16 weights: `338,134,308*2/2**30 ≈ 0.63 GiB`. Original FP32
foundation gradient allocation: `338,134,308*4/2**30 ≈ 1.26 GiB`. The scoped
no-grad constructor avoids this *allocation*, not just deallocating it later.
For a 12-layer specialist, BF16 weights alone are ~0.34 GiB; its FP32 gradients
are ~0.68 GiB, plus activations, CUDA workspaces, temporary GEMMs and offloaded
optimizer state/staging. The handoff's previous **~9.2 GiB** 12-layer total is
only an early heuristic; independent historical readers will add memory. The
priority is to measure peak allocated **and** peak reserved VRAM at realistic
batch sizes, not extrapolate from two-layer CPU results.

### Incompatibilities and limitations

* `FrozenFoundation` deliberately offers no optimizer and no foundation
  backward route. The underlying `DecoderLanguageModel` object still exposes
  methods that could be misused; callers must treat it as strictly immutable.
* The private historical specialist reader is not implemented. A 12-layer
  specialist can be instantiated for local-context CPU tests, but no
  production performance claim follows.
* No specialist-only checkpoint serializer, CLI training command, integrated
  dataset loader, end-to-end GPU check, generation evaluation, or benchmark
  suite has yet been completed.
* `chunked_lm_head_backward()` adds a strictly opt-in keyword and retains the
  original autoregressive default. `Parameter` no-grad construction is also
  strictly opt-in; ordinary training gradient allocation is unchanged.
* The existing fused CUDA attention and FFN paths need real hardware parity
  and performance checks. Do not switch to generic PyTorch implementations.
