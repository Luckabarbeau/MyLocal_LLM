MyLocal_LLM remaining code-efficiency micro-patches
====================================================

These patches were generated against the same fast post-MoE/post-attention base and
verified to apply sequentially in this order:

08_adamw_disable_hotpath_host_checks.patch
09_cross_entropy_single_fp32_workspace.patch
10_bounded_memmap_shard_cache.patch
11_adamw_single_update_workspace.patch

Recommended workflow
--------------------
Apply ONE patch, run tests, benchmark, and keep it only if beneficial.

Apply:
  git apply --check /path/to/PATCH.patch
  git apply /path/to/PATCH.patch

Revert most recently applied patch:
  git apply -R /path/to/PATCH.patch

Patch 08
--------
Moves AdamW per-parameter finite-value host copies/checks behind numerical_debug.
Normal training no longer copies gradients, m/v, and master weights to CPU every
optimizer step. ExtendedTrainer passes its numerical_debug flag through to AdamW.
This is expected to be the largest raw training throughput improvement.

Patch 09
--------
Reworks FP16 cross entropy to use one FP32 work/probability buffer in place of
separate logits_f32/shifted/exp/probs arrays. This targets VRAM and memory bandwidth.

Patch 10
--------
Adds memory-mapped token-shard loading and a bounded LRU cache in ExtendedTrainer.
This primarily targets host RAM growth and shard I/O robustness, not GPU math.
Default cache size is 2 shards.

Patch 11
--------
Apply only after patch 08. Rewrites Adam bias correction/update algebra so one
full-sized scratch array is used instead of materializing m_hat, v_hat, and update.
Mathematics are unchanged and a reference-formula regression test is included.

Validation performed here
-------------------------
All four patches apply sequentially with git apply --check.
Adam/loss targeted tests after all patches: 15 passed.
Python compilation succeeded for modified trainer/data files.
Full data/trainer test collection could not run here because pyarrow is not installed.
