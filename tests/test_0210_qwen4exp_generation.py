"""Opt-in real-model lifecycle smoke, never a reference-logit parity claim."""
import gc
import json
import os
import resource
import time

import numpy as np
import pytest

from mini_llm.backend import BACKEND_NAME, asnumpy, xp
from mini_llm.external.qwen4exp_model import Qwen4ExpGGUFModel


@pytest.mark.slow
@pytest.mark.integration
def test_original_incremental_reset_and_three_greedy_tokens():
    path = os.environ.get("QWEN4EXP_MODEL_PATH")
    if not path:
        pytest.skip("Set QWEN4EXP_MODEL_PATH to explicitly opt into original-model execution")
    start = time.perf_counter()
    with Qwen4ExpGGUFModel.open(path, max_length=8, cpu_workers=4) as model:
        state = model.create_generation_state(max_length=8)
        baseline = asnumpy(model.prefill([1, 2], state)).copy()
        assert state.length == 2 and np.isfinite(baseline).all()
        del state
        gc.collect()
        # A new request must not inherit recurrence, convolution, PLE or QSA.
        state = model.create_generation_state(max_length=8)
        model.step(1, state)
        logits = model.decode_one([2], state)
        repeated = asnumpy(logits)
        np.testing.assert_array_equal(repeated, baseline)
        generated = []
        for i in range(3):
            token = int(asnumpy(xp.argmax(logits[0])))
            generated.append(token)
            if i != 2:
                logits = model.step(token, state)
                assert np.isfinite(asnumpy(logits)).all()
        assert len(generated) == 3 and state.length == 4 and not state.failed
        assert all(s.conv.length == s.delta.length == 4 for s in state.recurrent.values())
        assert all(s.length == 4 for s in state.qsa.values())
        stats = model.expert_executor.statistics()
        assert stats["host_dequant"]["used_bytes"] == 0
        if BACKEND_NAME == "cupy":
            xp.cuda.get_current_stream().synchronize()
            device = dict(pool_used_bytes=xp.get_default_memory_pool().used_bytes(),
                          pool_reserved_bytes=xp.get_default_memory_pool().total_bytes())
        else:
            device = dict(pool_used_bytes=0, pool_reserved_bytes=0)
        print(json.dumps(dict(scope="lifecycle smoke, NOT independent reference numerical parity",
            backend=BACKEND_NAME, input_ids=[1, 2], greedy_ids=generated,
            reset_and_sequential_prefill_exact=True,
            peak_rss_bytes=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss * 1024,
            elapsed_seconds=time.perf_counter()-start,
            memory_plan=model.plan, expert_statistics=stats, device_snapshot=device)))
