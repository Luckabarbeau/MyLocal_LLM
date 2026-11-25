"""Fresh-process WP7 oracle: avoids unloading/reloading a GGML backend DSO."""
import sys
import numpy as np
import cupy as cp
from mini_llm.external.qwen4exp_model import Qwen4ExpGGUFModel


def main(path, destination):
    traces = {}
    saved = {}
    with Qwen4ExpGGUFModel.open(path, max_length=4, expert_cpu_backend='ggml_pool',
        expert_promote_after=1000000, cache_bytes=0, host_expert_bytes=0, cpu_workers=2,
        static_device_budget_bytes=7*1024**3, device_budget_bytes=10*1024**3,
        host_budget_bytes=4*1024**3,
        diagnostic_callback=lambda i,s,x:traces.__setitem__((i,s),cp.asnumpy(x).copy())) as model:
        assert model.plan['cache_bytes'] == 0
        state = model.create_generation_state(max_length=4)
        for t,token in enumerate((1,7)):
            saved[f'{t}_logits'] = cp.asnumpy(model.step(token,state))
            for (i,s),value in traces.items():
                saved[f'{t}_{i}_{s}'] = value
    np.savez(destination,**saved)


if __name__ == '__main__':
    main(*sys.argv[1:])
