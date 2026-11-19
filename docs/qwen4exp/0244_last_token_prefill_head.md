# 0244 — Skip unused prompt logits

`Qwen4ExpGGUFModel.prefill()` returns only the last prompt position's logits. The output hyper-connection collapse and vocabulary projection do not advance recurrent, QSA, or PLE state, so untraced prefill now runs them once instead of T times. All decoder blocks still run for every token. `step`, `decode_one`, and all-position `forward` retain their output contracts; a diagnostic callback retains both output boundaries at every position. Packed weights, dtypes, accumulation and routing are unchanged. Qwen3-Next was not modified.

Modified `mini_llm/external/qwen4exp_model.py` and `tests/test_0209_qwen4exp_model.py`. Tests verify projection count, diagnostic boundaries, exact synthetic last-position FP32 logits and the subsequent decode output. The existing native projection-count regression now expects twelve block projections and one vocabulary projection for its two-token prompt.

Validation: `env OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 .venv/bin/python -m pytest -q tests/test_0209_qwen4exp_model.py tests/test_0210_qwen4exp_generation.py tests/test_0239_qwen4exp_chat.py` — 27 passed, 2 skipped. No real-checkpoint speed, peak RAM/VRAM or independent-reference results were measured for this patch. This is a prefill savings, not a decode throughput claim.

Next dependency: time matched prefill runs and isolate Strata's native GPU Q8_1 packed projection against the same packed weights and FP32 inputs. Treat arithmetic changes as a distinct opt-in patch; keep full routing and the existing numerical oracle.
