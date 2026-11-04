# Benchmarks and evaluation

Run every command in this document from the repository root. Activate the project
environment and expose the repository root before using direct script paths:

```bash
source .venv/bin/activate
export PYTHONPATH=.
```

Commands then use paths such as `python benchmarks/run_unified.py`; paths in suite
JSON files are interpreted relative to the process working directory. If another
environment already provides the dependencies, it may be used instead, but
`PYTHONPATH=.` is still required for these direct benchmark-script invocations.

> **GPU safety:** the protected external training run is user-reported at step 165;
> its checkpoint and host are not available in this workspace. On any machine where
> training or inference owns the CUDA device, do **not** run competing CuPy
> evaluation, long-retrieval, performance benchmarks, or GPU tests. CPU-only
> inspection, `--dry-run`, and sandbox diagnostics are safe with
> `MINI_LLM_BACKEND=numpy`. Before any GPU run, confirm ownership and free memory
> with `nvidia-smi`. A benchmark output lock prevents concurrent writers; it does
> not reserve the GPU or make concurrent training safe.

## Primary entry point: unified generation and evaluation

`benchmarks/run_unified.py` is the common launcher for native checkpoints,
registered local GGUF/Safetensors models, and registered OpenAI-compatible remote
models. It uses the shared model-loading and evaluation engine. Existing specialist
WikiText, code, capability, and retrieval scorers remain authoritative and are
available through `--legacy-suite`; the adapter does not change their scoring.

Inspect a native generation run without loading the model:

```bash
MINI_LLM_BACKEND=numpy python benchmarks/run_unified.py \
  --checkpoint checkpoints/model_b_348m \
  --prompt-ids math_general_short,code_python_short \
  --tokens 128 --seeds 42,123 \
  --output-dir benchmarks/results/model_b_generation \
  --dry-run
```

When the GPU is free, remove `--dry-run` and select the CuPy backend:

```bash
MINI_LLM_BACKEND=cupy python benchmarks/run_unified.py \
  --checkpoint checkpoints/model_b_348m \
  --prompt-ids math_general_short,code_python_short \
  --tokens 128 --seeds 42,123 \
  --output-dir benchmarks/results/model_b_generation
```

Use a model catalog for registered local or remote models:

```bash
python benchmarks/run_unified.py \
  --registry serving_models.json --model MODEL_ID \
  --tokens 128 --seeds 42 \
  --output-dir benchmarks/results/MODEL_ID_generation
```

Set `MINI_LLM_BACKEND=cupy` only for a local GPU model on an unoccupied device.
Remote credentials belong in the environment variable named by the registry's
`api_key_env`; never put a secret in the manifest. Raw text completion is the
remote default. Use `--remote-mode chat` only when changing the prompt format is
intentional. Providers may reject completion mode rather than silently switching
to chat, and the unified runner rejects `--top-k` for remote models because that
API is not portable.

Generation results are resumable with the identical command plus `--resume`.
The resume guard rejects changes to model source metadata, suite bytes, prompt
selection, seeds, context, execution policy, or sampling. Each output directory
contains:

- `manifest.json`: source metadata, prompt-suite hash, configuration, and metric
  conventions. Local source identity uses file size and mtime, not a cryptographic
  hash of tensor contents; remote aliases do not prove immutable upstream weights.
- `results.jsonl`: one flushed row per `(prompt_id, seed)` with text, available
  token IDs, execution report, timing, and repetition metrics when token IDs exist.
- `summary.json`: completion counts and domain-level mean end-to-end throughput.
- `.benchmark.lock`: an advisory writer lock; the file may remain after the lock
  has been released.

Use a new output directory for a changed configuration. A torn, duplicate, or
unexpected resume row is an error and is not silently skipped.

## Existing checkpoint suites through the unified launcher

Run the existing quality evaluators against a native checkpoint with the same JSON
suite used during live training:

```bash
MINI_LLM_BACKEND=numpy python benchmarks/run_unified.py \
  --checkpoint checkpoints/model_b_348m \
  --legacy-suite benchmarks/configs/foundation_capabilities_0097_smoke.json \
  --benchmark-ids math_ladders_0097,logic_ladders_0097 \
  --output-dir benchmarks/results/model_b_capability_smoke \
  --dry-run
```

When the GPU is free, use `MINI_LLM_BACKEND=cupy` and remove `--dry-run`. The
legacy adapter writes the evaluator's existing artifacts under the selected output
root and adds `unified_legacy_index.json` with checkpoint and suite identity.
`--resume` is only for unified generation JSONL; legacy suite results implement
their own reuse guard and are reused only when the suite hash and benchmark
configuration match.

For options not exposed by the adapter—especially historical supervised-token
metadata—use the checkpoint-suite entry point directly:

```bash
MINI_LLM_BACKEND=cupy python benchmarks/run_checkpoint_suite.py \
  --checkpoint checkpoints/model_b_348m \
  --suite benchmarks/configs/foundation_capabilities_0097_progressive.json \
  --benchmarks long_retrieval_0097 \
  --output-root benchmarks/results/model_b_checkpoint_suite
```

Multiple `--checkpoint` arguments are evaluated sequentially. Use
`--supervised-tokens VERIFIED_TOTAL` for one old checkpoint, or
`--token-map path/to/verified_token_totals.json` for several. If neither the
checkpoint nor the command supplies a total, the result is explicitly labeled
`unknown_legacy_checkpoint` and uses zero; source-capacity counters are not
silently substituted.

## Live in-process benchmarks during training

Add the current suite to the normal `train_model.py` launch:

```text
--benchmark-suite benchmarks/configs/foundation_capabilities_0097_progressive.json
```

For a resumed checkpoint created before `supervised_tokens_total` was saved, also
supply the verified historical total:

```text
--benchmark-supervised-tokens-offset VERIFIED_TOTAL
```

If a checkpoint already contains the counter, a disagreeing explicit offset is an
error. Without an offset, an old checkpoint starts measuring from zero and records
`since_0092_only` provenance instead of pretending the count is absolute.

Live benchmark semantics are deliberately strict:

- Evaluation runs in the training process against the live model after a successful
  optimizer update. It does **not** save a checkpoint, reload weights, or perform a
  checkpoint round trip. At a save interval, evaluation happens before that step's
  checkpoint save.
- Every benchmark has an independent optimizer-step or cumulative
  supervised-token interval. On startup or resume, the next threshold is the next
  interval after the current counter; missed historical thresholds are not replayed.
- Seeded benchmark generation uses temporary NumPy and CuPy global RNG state and
  restores the original global RNG objects afterward. Trainer-owned RNGs are not
  used, so benchmark sampling does not consume the training random stream.
- The default `on_error: "raise"` makes a benchmark failure fatal. A failed job
  writes `error.json`; only an atomically written `result.json` is complete.
  `on_error: "continue"` must be an explicit suite decision.
- Evaluation pauses training and shares its process and device. RNG isolation does
  not remove benchmark runtime or VRAM cost.

Live and checkpoint suite results are rooted at the suite's `output_root`, or at
`--output-root` when overridden. A normal result is stored as:

```text
benchmarks/results/<suite>/step_<step>[_checkpoint_<id>_suite_<hash>]/<benchmark-id>/result.json
```

Depending on benchmark type, the same directory may also contain `report.html`,
`reference_per_case.csv`, progression CSVs, or `retrieval_by_case.csv`. The suite
root contains the rebuilt longitudinal `summary.csv`.

## Model B long-retrieval evaluation

The current Model B path is selected when `dual_resolution.enabled` is true and
`compression_mode` is `matrix_pool`. It runs the complete sequence with
`return_cache=False` and projects only the final hidden state through the vocabulary
head, avoiding a full `[sequence, vocabulary]` logits tensor. Its evaluation
horizon is `max_blocks * block_size`, which may exceed the checkpoint's training
`context_length`; weights and training configuration are not changed.

Run the bounded smoke selection first, then the progressive suite when the GPU is
free:

```bash
MINI_LLM_BACKEND=cupy python benchmarks/run_unified.py \
  --checkpoint checkpoints/model_b_348m \
  --legacy-suite benchmarks/configs/foundation_capabilities_0097_smoke.json \
  --benchmark-ids long_retrieval_0097 \
  --output-dir benchmarks/results/model_b_long_retrieval_smoke

MINI_LLM_BACKEND=cupy python benchmarks/run_unified.py \
  --checkpoint checkpoints/model_b_348m \
  --legacy-suite benchmarks/configs/foundation_capabilities_0097_progressive.json \
  --benchmark-ids long_retrieval_0097 \
  --output-dir benchmarks/results/model_b_long_retrieval_progressive
```

The progressive selection evaluates the shared synthetic fact bank at 1K, 4K, 8K,
16K, 24K, and 32K requested distances with two deterministic associations per base
case. The disabled exhaustive and counterfactual entries in the suite add 6K/12K
coverage or causal fact-replacement checks and must be selected explicitly with
`--benchmark-ids` if desired. Full 32K attention is expensive even with
last-token-only projection.

No retrieval prompt is silently truncated. Unsupported cases either fail or, when
the suite explicitly enables `skip_unsupported_distances`, appear in
`unsupported_cases` with the prompt length and reason. Results include exact
fact-end-to-query distance, four-choice prediction and margin, correct-token
full-vocabulary rank/log-probability/probability, source phase modulo 16,
per-distance summaries, elapsed time, and optional counterfactual shifts. Rank can
improve with general language modeling; sensitivity to replacing the historical
fact is stronger retrieval evidence. Repeated filler and synthetic success do not
establish general long-document retrieval.

### External GGUF control

The standalone control uses the same text task bank, prompt builder, four-choice
scorer, and rank calculation with the GGUF's embedded tokenizer:

```bash
MINI_LLM_BACKEND=cupy python benchmarks/evaluate_external_long_retrieval.py \
  --model pretrained_models/SmolLM2-360M-Instruct-F16.gguf \
  --distances 1000,4000 \
  --repetitions 2 \
  --output-dir benchmarks/results/smollm2_long_retrieval
```

It requires CuPy and must also wait for a free GPU. It never extends RoPE or
silently truncates: prompts beyond the model-native context are listed as explicit
exclusions. Long external prefill retains full causal KV history but can still
allocate large attention workspaces. Its output directory contains `result.json`,
`retrieval_by_case.csv`, and `report.html`.

Tokenizer comparisons require care. Each model uses its own vocabulary, native
positions, and context limit, so identical source text and requested distances do
not imply identical token IDs or filler bytes. The GGUF adapter suppresses
per-fragment BOS insertion and adds at most the model's single required BOS after
assembly. A target that is not one exact token at the query boundary is rejected.

## Timing and comparison rules

- **Local unified generation:** end-to-end wall time includes tokenization,
  sampling, host overhead, and decode. Prefill is separately synchronized where
  available. Decode time is derived from total wall time minus prefill; it is not
  isolated CUDA-kernel throughput.
- **Remote generation:** wall time includes HTTP transport and provider execution.
  Token counts are provider-reported when available; prefill, exact token IDs,
  token repetition, and reliable per-token decode timing may be `null`. Upstream
  seed determinism is not guaranteed.
- **Long retrieval:** reports prompt tokens per elapsed second for the full scoring
  work, including a second prompt when counterfactual mode is enabled. Do not
  compare this directly with autoregressive decode tokens/second.
- **Cross-model quality:** use the same raw prompts and sampling settings, but label
  tokenizer, context, chat-template, and local/remote execution differences.
  Repetition metrics do not measure factuality, code correctness, or narrative
  coherence; use the specialist scorers and inspect generated text.

## Dataset preparation and specialist scripts

External snapshots are created under a caller-selected root. This step may use the
network and the optional `datasets` dependency:

```bash
python benchmarks/prepare_external_benchmarks.py \
  --root benchmarks/data \
  --datasets ptb pg19 c4 lambada
```

Direct evaluators and microbenchmarks remain available for focused work, for
example:

```bash
MINI_LLM_BACKEND=numpy python benchmarks/benchmark_tokenizer.py --smoke-test
MINI_LLM_BACKEND=numpy python benchmarks/benchmark_performance.py \
  --mode both --model micro --iterations 5 --warmup 2
```

Use each script's `--help` before a production run. Keep generated datasets,
benchmark outputs, profiles, and model files out of source control.

## Generated-code sandbox preflight

Python/C++ correctness tasks that execute generated code require the Bubblewrap
sandbox. Diagnose the host non-invasively before enabling `run_tests`:

```bash
MINI_LLM_BACKEND=numpy python benchmarks/sandbox_diagnostics.py
```

The diagnostic reports paths for `bwrap`, `g++`, `python3`, and
`apparmor_parser`; relevant user-namespace sysctls; a temporary isolated Python
preflight; `sandbox_ready`; and a suggested next action. It does not execute
model-generated code or change host security settings.

- If `bwrap` is absent, install the distribution's Bubblewrap package and rerun.
- On Ubuntu systems where the detail contains `Failed RTM_NEWADDR`, inspect the
  packaged unprivileged-user-namespace/AppArmor policy and host security logs.
- If the preflight fails or times out, do not bypass it. In particular, do not
  disable AppArmor globally and do not use shared networking for generated code.

Research history and superseded benchmark decisions belong in
[`docs/DEVELOPMENT_LOG.md`](../docs/DEVELOPMENT_LOG.md); this README documents only
the current operational paths and semantics.
