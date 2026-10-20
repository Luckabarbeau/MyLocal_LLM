# 0080AE repository audit and interface contract (15 October 2026)

**Source:** uploaded repository archive, not a speculative design; 0080AE changes
only configuration and tests. Its `full_vocab` and `dec_residual` branches are
**not operational** until validated later patches. No new weights, changes to
old parameter ordering, changes to RNG draws, v2 checkpoints, or CUDA kernels
are introduced in 0080AE. The legacy expectation + boundary-off mode is the
only executable setting in this patch.

## Existing data flow, shapes, causal contract

- Production main model: source `[B,D+T]`, one `forward_body`, hidden
  `[B,P+T,d]` after routed-prefix compaction, where `P` is the actually
  selected prefix (not the entire historical input). The normalized block
  boundaries `h = main_hidden[rows, P+b*K, :]` have shape `[G,d]`.
  `block_schedule` packs prefix-contiguous EOS/document tails and counts each
  valid target once; only `h` of each block is gathered. The model's baseline
  large configuration in the handoff uses `d=768`, `T=4096`, `K=4`.
- `HorizonProposalHead.forward(h)` computes K proposal hidden rows and sends
  them through the shared output projection, producing `[G,K,V]` logits with
  `V=65,280` in production. The K=1 path bypasses future-horizon adapters.
- `SoftTokenExpectation` converts all `[G,K,V]` proposal distributions into
  `[G,K,d]` expected embeddings; `plan_projection` maps these to
  `[G,K,width]`. The committed-prefix path uses `shifted_decoder_tokens`:
  DEC input is `[BOS,y1,...,y(K-1)]`, not future labels. PLAN rows may read
  all PLAN rows; DEC row j reads all PLAN plus DEC rows through j only. There
  is **one** masked decoder pass over `[G,2K,width]` (3 layers, width 512 and
  4 heads in the handoff's production configuration). The normalized DEC
  outputs map back through correction projection and shared vocabulary head;
  `L_final = L_proposal + delta`.
- Existing `SoftTokenExpectation` already provides an all-vocabulary
  numerically stable `stable_softmax`, fused CuPy softmax/VJP kernels, and
  2-D BF16 matrix multiplications with FP32 Parameter gradient accumulation.
  The cached FP32 probability buffer can be **consumed destructively** by
  `MINI_LLM_INPLACE_EXPECTATION_VJP` when fused mode is enabled. Reuse the
  numerical reductions where appropriate, but a new query-dependent read
  needs its own score, key/value, and streaming-gradient implementation;
  its memory bank cannot alias an expectation buffer overwritten by VJP.
- `TwoStageBlockModel.backward` returns `d_h` from **only** the proposal
  pathway today. It sums the residual-prior CE, lambda-weighted proposal CE,
  and the differentiable expectation's logit gradient. `SinglePassBlockTrainer`
  scatters `[G,d]` into `[B,P+T,d]` and performs just one `backward_body`.
  A future direct boundary adapter must add its independent `d_h` contribution
  **before** this scatter, preserving supervised-token normalization.
- In cached generation, `BlockPrefixDecoder.prepare_plan_cache(L)` builds K
  PLAN keys/values per decoder layer; `cached_step(state, previous_token_ids)`
  decodes one DEC position with persistent KV (BOS for the first, previously
  committed token otherwise). `DirectCachedBlockGenerator` supports dense
  main KV with chunk append. `SequentialCommitBlockGenerator` uses original
  `InferenceModel.decode_one` for sparse/routed main KV, one token at a time;
  both call the decoder's same PLAN/cache hooks. A future dynamic read must
  be inserted in the teacher-forced **and** cached layer loops.

## Existing checkpoint and optimizer semantics

- `architecture_metadata` strictly fingerprints the v1 architecture string
  `probabilistic_block_proposal_prefix_decoder_v1` and exact shape/config.
  `load_two_stage_checkpoint` checks the entire architecture dictionary,
  tokenizer/provenance when supplied, parameter names/shapes, and optimizer
  master/moment groups (including Hybrid Muon) before loading.
- `train_model.py --resume-from` reconstructs the objective from strict v1
  metadata and restores all main, proposal, decoder parameters. In contrast,
  `--init-from` calls `initialize_matching_parameters` with **only**
  `model.parameters()`: it does **not** warm-start the proposal or decoder.
  New v2 checkpoint tag and complete weights-only migration belong to 0080AL,
  not this patch; nondefault new options are explicitly rejected rather than
  creating ambiguous v1 training artifacts.
- **0080AB packed-QKV refresh:** after optimizer step,
  `ExtendedTrainer.train_step` explicitly calls
  `layer.attn.refresh_compute_buffers()` on **each decoder layer**, in
  addition to main-model refresh. `load_two_stage_checkpoint` and the
  `train_model.py` production resume path likewise refresh decoder packed
  attention buffers. A fresh or updated projection cache must have an equally
  explicit refresh contract; avoid new stale train/inference copies.
- The final CE kernel can destructively reuse its exclusive logits buffer;
  other logit/expectation paths can reuse FP32 buffers via env flags. Any
  full-vocab read must enforce buffer ownership before destructive CE/VJP.

## Memory and parameter risks

At G=64, K=4, H=4, V=65,280, a single full FP32 attention-score tensor
`[G,K,H,V]` costs ~255 MiB; an all-source `[G,K,H,K,V]` tensor costs
~1,020 MiB **per copy**. A materialized `[G,K,H,K,V,192]` costs ~191 GiB.
Production training reported ~14.5-14.6 GiB already used on a 16 GiB GPU.
Do not replicate attention probabilities per decoder layer, allocate the
vocab×embedding broadcast, or retain multiple probability/VJP temporaries.
Later patches must stream vocabulary tiles, recompute scores in backward,
retain only FP32 online reductions, and benchmark realistic group sizes.
The CPU tiny-vocab reference does not establish production GPU viability.

## Source archive deviations and tests

- The archive **does include** the 0080AB packed-QKV regression and the
  0080AC/AD test sources, but **omits**
  `mini_llm/diagnostics/block_proposal_roundtrip.py`,
  `mini_llm/diagnostics/block_proposal_live_audit.py`,
  `benchmarks/audit_decoder_checkpoint_discontinuity.py`, and
  `benchmarks/audit_block_proposal_checkpoint_roundtrip.py`,
  `benchmarks/audit_block_proposal_forward_parity.py`, and
  `benchmarks/audit_block_proposal_training_contract.py` used by these
  tests/certain production paths. 0080AD and related tests cannot import in
  this exact archive. Restore those original files from the working branch
  before claiming the full 0080A–AD suite passes; **do not replace the missing
  tests with stubs**.
- The focused *prepatch* 0080A/0080L/0080AB CPU suite passed (20 tests).
  The available CPU block-proposal suite after 0080AE ran
  **202 passed, 12 skipped** (the incomplete audit-file cases were excluded
  from collection). Older metadata remains byte-identical because this patch
  adds no metadata keys. The production 16 GiB RTX 5080 path was **not**
  benchmarked in this environment; the first full-vocabulary CUDA kernel is
  intentionally postponed.

## Later implementation checkpoints (not part of 0080AE)

0080AF: tiny-V CPU own-source full read + exact explicit VJP; 0080AG:
teacher-forced decoder insertion; 0080AH: bounded CuPy path; 0080AI:
all-source per-horizon log-softmax and cross-horizon VJP; 0080AJ: boundary
conditioning and main h VJP; 0080AK: generation parity; 0080AL: strict v2
save/resume and decoder-inclusive warm start; 0080AM: controlled matching
experiments. Cached inference and v2 round-trip must precede long 500M runs.
