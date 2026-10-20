# 0080AA: Production live train/eval metric parity

The 500M block-proposal checkpoint was reported to have strongly positive decoder
improvements (+0.34, +1.48, +1.86, +2.12 CE in training) while independent
validation was negative at horizon 1 and much less positive at later horizons.
Prior same-batch benchmark checks showed training/eval forward parity within
~0.001 CE, but those tests did **not** replay the actual microbatches selected
by `ExtendedTrainer` during training.

This patch adds only an opt-in production diagnostic; normal training is
unchanged. When `MINI_LLM_BLOCK_PROPOSAL_LIVE_AUDIT=1`, each microbatch in the
next training step is evaluated a second time with **the exact same inputs,
targets, validity masks, and routing metadata**, before the optimizer update.
It records per-source/per-horizon proposal CE, final CE, and decoder gain for
both the live training forward/backward and the forward-only replay. It also
checks the production logger's accumulated horizon sums against independent
host-side aggregation.

Run it on a **copy** of a known checkpoint, since the command still performs
one genuine optimizer update as usual. From the repository root, first copy a
fully saved checkpoint while it is not being written, then reuse the **exact
same flags and environment as the original 500M training command**. Add:

    MINI_LLM_BLOCK_PROPOSAL_LIVE_AUDIT=1
    --max-run-steps 1

If the loaded trainer requires a new checkpoint destination, create a scratch
run directory; do not overwrite the reference checkpoint.

The diagnostic intentionally doubles forward compute and introduces CUDA
synchronization on *every* microbatch. Do not leave the flag enabled for
normal training.

Interpretation:

* If same-batch train and eval are separated by ~1 CE, inspect the actual
  training metadata, live low-precision kernels and in-place buffer behavior.
* If same-batch scores agree but the production accumulator differs, inspect
  logging/metric reduction.
* If both agree but **every source's live training gains** exceed independent
  train-shard benchmark gains by 1+ CE, compare actual sampled windows,
  provenance, stochastic routing and the benchmark's sampling distribution.
* If the per-source live training gains are similar to independent evaluation,
  but the logged global gains are much larger, the old production aggregation
  or loaded checkpoint/step pairing is suspect.

No changes are made to the optimizer, architecture, datasets, gradients, or
checkpoint format when the flag is off.
