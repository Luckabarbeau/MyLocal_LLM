# 0210 — explicit Qwen4Exp generation lifecycle

## API

```python
from mini_llm.external.qwen4exp_model import Qwen4ExpGGUFModel

# path can be either original numbered shard; the store discovers its sibling.
with Qwen4ExpGGUFModel.open(path, max_length=32, cpu_workers=2) as model:
    state = model.create_generation_state(batch_size=1, max_length=32)
    logits = model.prefill([1, 2, 3], state)  # FP32 [1,vocab], last prompt position
    logits = model.step(4, state)           # consume token 4 exactly once
    logits = model.decode_one([5], state)   # old lifecycle [1] or [[5]] spelling
```

`prefill` accepts a nonempty integer `[T]` or `[1,T]` sequence and executes slowly
and sequentially. All token IDs and whole-chunk capacity are checked before the
first step. `step(token,state)` consumes one integer and returns `[1,vocab]`.
`forward(sequence)` creates a bounded temporary state and returns `[1,T,vocab]`;
it is intended for small diagnostic fixtures, not full-context prefill.

State owns length, max_length, model identity, recurrent raw conv history and
key-by-value matrix state, QSA keys/values/raw index/pooled keys, and PLE predecessor
and normalized conv history. There is no implicit model request history. Only
batch size 1 and zero-origin contiguous scalar text positions are supported.

Reset by discarding the state and creating a **fresh** state. In a budgeted
factory model, live states collectively share the state reservation approved at
construction. Two shorter states may coexist within that reservation; a second
full-capacity state is rejected. Drop unused state references before allocating
replacement capacity. Operator state and model ownership cannot be interchanged.

A step advances model length exactly once after successful logits and diagnostics.
If an operator/callback fails after a partial forward, `state.failed` is set and
all subsequent use is rejected. The runtime does not roll back recurrence or
silently retry. Argument/capacity failures before execution do not poison state.
Calls on a model are serial; concurrent forward/close is rejected. Callers sharing
a borrowed executor across models must serialize those models themselves.

## Greedy helper

```python
with Qwen4ExpGGUFModel.open(path, max_length=32, cpu_workers=2) as model:
    generated_ids = model.greedy([1, 2, 3], max_new_tokens=4)
```

Returns generated IDs only, including a stopping ID if selected. The first output
is selected from the final prompt logits; it is not fed back unless another output
is needed. Stop IDs come from tokenizer `eos_token_id`, `eot_token_id`, and
`eom_token_id` metadata when present, never PLE EOS/padding IDs. In the audited
checkpoint tokenizer EOS is 248046 and PLE EOS is 248044. The helper adds no
sampling, tokenizer, server or chat-template policy.

## Explicit original-model CPU probe

From the repository root (the local path below is the path recorded in the audit):

```sh
MINI_LLM_BACKEND=numpy OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 \
QWEN4EXP_MODEL_PATH=/home/lucka/softwares/local_llm_lab_north_v7/local_llm_lab/models/IQ3_S/Qwen3.8-Flash-Next-GSQ-RCO-IQ3_S-00001-of-00002.gguf \
.venv/bin/python -m pytest -q -s tests/test_0209_qwen4exp_model.py::test_actual_original_factory_opt_in
```

Optional `QWEN4EXP_PROBE_TOKEN` selects the initial token (default 1). This loads
the actual model, consumes that token, greedily selects one next token, consumes
it, and checks finite logits/length. It may be slow on CPU and pages selected
expert bytes from disk. It does not expand all experts or the PLE table. Native
packed kernels require a local C++ compiler on first use; existing native build
cache behavior is reused (`MINI_LLM_NATIVE_BUILD_DIR` can choose a writable cache).
No network access or weight conversion is required.

This is a runnable smoke probe, **not an independent numerical parity test**.
For later llama.cpp comparison, supply the per-layer diagnostic callback during
`open`, compare identically tokenized text and FP32/quantized-weight settings,
and locate the first divergent layer/stage before drawing full-model conclusions.

## Recorded original-checkpoint execution

The original factory smoke passed (`1 passed in26.76s`). The additional opt-in `tests/test_0210_qwen4exp_generation.py` passed under NumPy (`1 passed in40.68s`) and CuPy/RTX5080 (`1 passed in30.06s`). InputIDs[1,2], three consecutive greedyIDs[220,100561,198] on both; fresh-request incremental logits equal sequential prefill logits exactly within each backend. This test performs six token forwards and verifies all recurrent/QSA positions plus zero decoded-expert host cache. Run it with the same QWEN4EXP_MODEL_PATH and backend environment shown above.

PeakRSS CPU~9.58GiB /CuPy~8.04GiB; final CuPy pool~4.12GiB. RSS includes mmap pages and is not limited by owned-allocation plan estimates. No dedicated throughput measurement is claimed.

**Independent model numerical gate remains unresolved.** The safe no-prefetch linked reference now executes, but fixed-prefix logit relativeL2 differences are6.1643% and5.0675%, maxabs0.958172 and1.006777. Matching argmax is not sufficient. The first boundary association is suspect; the next experiment must verify tensor/op/source identities before a production correction. See implementation_status.md and0210_reference_diagnostics.md. No serving/optimized-inference completion is claimed.
