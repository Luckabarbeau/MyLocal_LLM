# 0080 — Probabilistic K-block proposals + prefix-LM decoder

## Verified repository baseline

The supplied `MyLocal_LLM-main.zip` includes 0079Q4 (`tests/test_predictive_refinement_0079q4.py`, `examples/train_predictive_refinement_routed_text.py`) and its original one-pass shared-memory Q3/Q4 production integration. **0079Q4a is not present** in the supplied archive. No git metadata or old patch series accompanies the ZIP, so exact git ancestry beyond files/tests cannot be established.

Production code reused unchanged for training:

- `mini_llm/model/decoder_lm.py`: `forward_body`, `backward_body`, normal LM head, embedding and full Transformer.
- `mini_llm/blocks/transformer_block.py`, `mini_llm/ops/attention.py`, `mini_llm/ops/hierarchical_memory.py`: existing original routed-prefix learned router and per-layer learned retrieval attention, including causal RoPE positions. No replacement of retrieval heads by static local heads.
- `mini_llm/predictive_refinement/soft_embeddings.py`: full-vocabulary `SoftTokenExpectation`/softmax VJP into `embedding.W` and the proposal logits; copied *functionality*, not the old recurrent architecture.
- `mini_llm/data/token_shards.py`: original document-first, token-weighted packed-document sampler, EOS, valid lengths, history-valid-starts; original tokenizer/mixture provenance.
- `mini_llm/ops/loss.py`, `mini_llm/optim/adamw.py`, `mini_llm/checkpoint.py`: original BF16-safe CE, AdamW with optional CPU state offload, optimizer serializers.
- `mini_llm/inference_model.py`: original causal KV-cache inference, extended with explicit most-recent normalized hidden state and updating it when an accepted direct-dense block is appended.

## Crucial alignment detail

Unlike 0079Q4's shifted-label setup, 0080 trains with the **normal source/next-token labels**: source active window `[x0,x1,...]`, target labels `[x1,x2,...]`. For the target block beginning at label index `b*K`, use the normal causal main hidden at **active working position `b*K`**, which has already seen source token `x_(b*K)` and predicts the first token of that block. Add the actual retrieved-prefix length to locate this hidden vector in the full active sequence. The first block therefore has a real causal boundary token even if top-level routing selected remote historical blocks. Q4's shifted labels are NOT directly reusable for these hidden-state proposals. This policy is tested.

The main forward is run once on all actual source inputs. Later input tokens do pass through the main trunk, but a block receives **only its own strictly causal boundary hidden**. The top-level router is explicitly forced to select memory using a **pre-window query**, and each original per-layer retrieval kernel preserves its existing causal policy. No dense 64K mask, no block-sized repeated main history.

## Architecture / gradients

1. `HorizonProposalHead`: K configurable horizon distributions using the original untied `output_proj.W`; horizon 1 uses exactly the conventional normalized AR hidden; other horizons use a shared context-dependent tanh MLP and learned horizon embeddings. This is not a bank of 65K-wide output heads.
2. `BlockPrefixDecoder`: independent small dense Transformer blocks with RMSNorm, GQA, **single SwiGLU FFN (no MoE)**. The K PLAN positions are bidirectional among themselves and never read DEC; the K DEC positions read all PLAN and only themselves/earlier DEC positions. Shifted token inputs are BOS, y1,...,y(K-1), not current/future labels.
3. PLAN inputs use **full** differentiable `P @ E_input`; DEC inputs use the actual previous token's input embedding, not raw main hidden states. Final logits use the shared original `W_out` to form a small projected residual correction to `L_prop`.
4. Loss = final CE + configurable positive lambda * directly supervised proposal CE, both normalized by **unique valid supervised tokens**. Invalid tails never train/pollute earlier decisions. Backward includes both residual-prior identity gradient and probability-expectation VJP, plus direct proposal CE.
5. `SinglePassBlockTrainer`: groups nonempty K blocks into bounded decoder microgroups, locally averages by the unique number of target tokens and scales by each group's actual fraction of the window; accumulates gradients at their original causal main boundaries; performs **one** main `backward_body` for the whole window. Works with existing learned per-layer retrieval heads; grouping does not change the mathematical objective. Large model activations are not recomputed K times.
6. Cached small decoder: one bidirectional PLAN prefill and K sequential DEC KV updates; parallel teacher forcing and cached DEC logits pass tight FP64 numerical comparisons. Direct dense `InferenceModel` additionally supports one batched K-token causal append and normalized new boundary state. Partial tails, EOS, greedy/seeded sampling and repeated blocks are handled for direct dense mode only.
7. Strict 0080 checkpoints save original and new trainable parameter tensors plus FP32 AdamW moments/master weights, config, training step, unique token count, tokenizer fingerprint, provenance, and training RNG. Strict exact CPU AdamW resumed-step test passes. The new real-data CLI uses unchanged original production sampling and a fixed independent validation record set.

## Patch sequence

- **0080A**: block/target alignment, prefix mask, reference attention and full-vocabulary expectation finite differences.
- **0080B**: learned nonlinear K-horizon proposal head with handwritten backward and baseline equivalence tests.
- **0080C**: small prefix-LM, full `P@E`, shared-head correction, dual objective, CPU end-to-end finite differences and cached prefix-decoder mathematical path.
- **0080D**: real shared causal trunk training with learned 64K routed-prefix machinery, chunk equivalence and source sampler preserving documents/EOS.
- **0080E**: direct-dense KV-cached generation and chunk-vs-sequential main cache parity; **routed long-memory generation is deliberately not yet implemented**.
- **0080F**: strict joint checkpoint/AdamW resume, source-weighted fixed validation, per-horizon metrics and real 500M preset training CLI. **GPU profiling and full real-data runs remain unverified**.

## Usage

Apply in order with `git apply` (or `patch -p1`), from the original archive root:

```
git apply /path/to/PATCH_0080A.patch
# ... 0080B, C, D, E, F, in order
python -m pytest -q tests/test_block_proposal_0080*.py
python -m pytest -q tests/test_predictive_refinement_0079q{2,3,4}.py
```

Preflight source and tokenizer without allocating the 500M model:

```
python -m examples.train_block_proposal_routed_text \
  --mixed-shard-root /path/to/existing/mixed-shards --preflight-only
```

An **experimental** BF16 one-step run, only after checking 16-GB GPU feasibility:

```
MINI_LLM_BACKEND=cupy MINI_LLM_OPTIMIZER_OFFLOAD=full \
  python -m examples.train_block_proposal_routed_text \
  --mixed-shard-root /path/to/existing/mixed-shards \
  --steps 1 --batch-size 1 --grad-accum 1 --block-group-size 1 \
  --block-size 8 --decoder-width 384 --decoder-layers 3
```

The shard root, tokenizer and val split must exist. No dataset paths were guessed. A real CUDA kernel and true peak-VRAM audit is required *before* scaling group sizes or training duration.

## Explicit remaining acceptance gaps (do not claim production completion)

- **RTX 5080 BF16 smoke/throughput/true peak VRAM** were not executed in the available CPU environment. BF16 fused pipeline and 16-GB feasibility are **unverified**; memory/throughput estimates are not results. Chunked full-vocab buffers must be tuned on real GPU.
- **Full 64K routed main-model inference generation**: direct-dense cached generation is implemented, but the main inference routed-prefix circular buffer needs a K-token append/route-refresh implementation preserving per-layer learned retrieval and absolute RoPE positions. Do not silently route via a dense/static surrogate or re-run the entire main transformer per decoder token.
- **True complete 500M real-text run**, validation/generation quality comparisons, first-token vs AR baseline, K ablations, and measured FLOPs/wall-clock benchmarks await the real corpus/hardware.
- **Proposal-plan ablations** (uniform/shuffled input vs real plans) are not integrated in the CLI yet. CPU tests verify the information-flow restriction, not empirical benefit.
- **Checkpoint sampling cursors**: the original local packed-shard dataset and NumPy sampling RNG are recorded. Remote streaming dataset cursor/segment hashes are not part of this CLI; if the optional remote streaming subsystem is later connected, exact per-source segment/cursor provenance must be added before claiming reproducible remote-data resume.
- **Profiler counters**: current CLI reports actual wall-time and unique supervised tok/s, source-weighted held-out CE and per-horizon CE/gain. Stage-level GPU timing and *actual* device high-water mark must be added and validated on CUDA (allocator snapshots alone are not true physical VRAM peaks).

This implementation adds opt-in `mini_llm/block_proposal` modules. The original 0079 nine-pass experiment remains in place and is not imported by the 0080 trainer, except for the mathematically reusable `SoftTokenExpectation` and the original packed-document loader.
