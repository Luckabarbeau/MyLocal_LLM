# 0080AG — Teacher-forced full-distribution retrieval: CPU-only integration

Parent: 0080AF, which introduced `SoftVocabularyRead` and independent finite-difference VJP tests. This patch is the **first small-model teacher-forced vertical slice**, not a 500M production mode. It adds `experimental_cpu_full_vocab=True` exclusively to the direct Python `BlockPrefixDecoder` constructor, *not* the CLI. The original v1 execution, initialized weight tensors and order, CUDA path, and cached-generation hooks remain unchanged when that explicit opt-in is absent.

## Forward and causality

The legacy PLAN prefix remains `softmax(L) @ E`, with exactly the original `prefix_lm_allow_mask` and shifted DEC tokens. After the configured decoder layer (default 0), split its `[G,2K,width]` hidden rows into PLAN and DEC. Compute `read(DEC_hidden,L)` with the **complete tiny vocabulary** using the 0080AF own-source reference, and add its correction only to DEC rows. Resume the remaining decoder layers and final residual logits `L + delta`. DEC_j can query alternatives from its own P_j after processing earlier committed tokens; future true token labels never enter this read. There is no second decoder pass.

The output projection is initialized to zero, so the new network starts with the *same* function as v1. `SoftVocabularyRead` uses an independent seeded RNG after all old layers/projections are allocated; no old parameter name, order or random draw changes. The shared token embedding Parameter is registered once through the main model; two new projections are appended to decoder parameters.

## Reverse-mode contract

At the insertion boundary, backward propagates through later decoder layers first. For the read residual `DEC_next = DEC_layer + read(DEC_layer, L)`, add the identity gradient to the read input VJP; leave PLAN rows untouched. `SoftVocabularyRead.backward` returns `d_DEC` and `dL_read` and accumulates its learned projection gradients plus key/value embedding gradients. Continue backward through the same masked decoder layer and the existing PLAN expectation VJP. The returned full proposal gradient is exactly

```
dL = dL_residual_prior + dL_legacy_PLAN_expectation + dL_query_read
```

The existing `TwoStageBlockModel.backward` adds the proposal CE gradient, calls the proposal head VJP, and returns one boundary gradient; `SinglePassBlockTrainer` performs its existing single-body backward. There are no detaches or stop-gradient approximations.

## Deliberate safety limits

- The **production `train_model.py` CLI continues to reject** `full_vocab` and `dec_residual` through 0080AE's `require_implemented` gate. The reference constructor opt-in is only for CPU NumPy float32/float64, `V <= 512`, own-source, boundary off; the source module also enforces its bounded score budget. The 65,280-vocab BF16/CuPy path requires tiled production kernels in 0080AH; do **not** use this integration on RTX 5080 yet.
- All-source log-probability normalization/cross-horizon source reads remain for 0080AI; direct h conditioning remains for 0080AJ.
- `prepare_plan_cache` and `cached_step` fail explicitly in experiment mode until correct parity (0080AK). Cached generation in the v1 model is unchanged.
- `architecture_metadata()` refuses non-v1 interfaces rather than mislabeling experimental parameters and enabling unsafe v1 checkpoint/resume. Strict v2 metadata, optimizer roundtrip and complete warm-start are scheduled for 0080AL.
- CPU float64 finite differences are a correctness reference, not a measurement of large-vocab memory, GPU time or final CE improvement.

## Reproducible test commands

```bash
MINI_LLM_BACKEND=numpy OPENBLAS_NUM_THREADS=1 \
python -m pytest -q tests/test_block_proposal_0080ag_decoder_integration.py

MINI_LLM_BACKEND=numpy OPENBLAS_NUM_THREADS=1 \
python -m pytest -q tests/test_block_proposal_0080ae_interface_contract.py \
                         tests/test_block_proposal_0080af_soft_vocab_read.py \
                         tests/test_block_proposal_0080ag_decoder_integration.py \
                         tests/test_block_proposal_0080l_baselines.py
```

The new suite checks exact zero-output initialization parity, original parameter names/order and ownership, proposal/embedding/read-query/read-output/manual decoder parameter finite differences **through the integrated decoder**, input-label causality, `read_layer=0` and `1`, masked committed-prefix mode, zero query scale, single main-body forward/backward, and safe rejection of checkpointing/generation/unimplemented source modes.

The uploaded repository archive omits several *preexisting* diagnostic helper modules. Do not interpret their import failures as regressions caused by 0080AG; do not fabricate their contents. On the available `test_block_proposal*` subset, excluding those five incomplete modules, the CPU suite is **228 passed, 12 skipped**. No RTX 5080 execution or large-vocabulary memory benchmark was performed.
