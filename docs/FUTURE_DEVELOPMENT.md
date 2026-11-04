# Current roadmap

This file is only the forward-looking roadmap. Completed work and negative results belong in [`DEVELOPMENT_LOG.md`](DEVELOPMENT_LOG.md).

## 1. Verify and protect the current baseline

- Preserve Model B's **348,261,376-parameter** matrix-pool/stride-16 architecture, tokenizer pairing, causal document isolation and full/incremental agreement.
- Treat external **`step165`** as **user-reported/unverified** until checkpoint/config/tokenizer hashes, training state, logs, finite metrics and a read-only resume/load check are recorded.
- Run correctness/performance gates only on a sufficiently free GPU; blocked/OOM is not pass.

## 2. Close specialist production gates

- Validate the 0107–0108 exact-Model-B specialist path on the target GPU without mutating the foundation.
- Measure two-layer smoke first, then 12-layer context/memory/throughput and deterministic resume.
- Compare specialist and frozen foundation on matched supervised tokens and compute; add composed incremental generation only after training parity is established.

## 3. Establish quality evidence

- Run Model B long-retrieval, held-out LM and capability suites from immutable checkpoint copies.
- Run the SmolLM2 control only within its native context, with its own tokenizer and exact model hash.
- Report negative and null results, unsupported cases, confusion/answer bias and counterfactual sensitivity—not aggregate accuracy alone.

## 4. Simplify safely

- Continue moving callers onto model-centered loading/generation/evaluation interfaces only when outputs and CLI behavior match.
- Keep process-global backend limitations explicit.
- Keep the consolidated migration ledger accurate; do not recreate standalone patch-report Markdown files.
- Retain research `.tex`/`.pdf` artifacts and never treat runtime data as a cleanup target.