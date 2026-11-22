"""Whole-chunk router using the established per-token GEMV and top-k graph."""
from functools import lru_cache

import numpy as np

from mini_llm.backend import BACKEND_NAME, xp
from . import router_decode as native
from .inference_matmul import matmul_inference


@lru_cache(maxsize=24)
def _kernel(device, warp_select, warp_reduce):
    source = native._REDUCE_SOURCE if warp_reduce else native._WARP_SOURCE if warp_select else native._SOURCE
    if source.count('    int tid = threadIdx.x;') != 1:
        raise RuntimeError('Native router source contract changed; review chunk adapter')
    source = source.replace('    int tid = threadIdx.x;',
        '    logits += (long long)blockIdx.x*n; weights += (long long)blockIdx.x*k;\n'
        '    ids += (long long)blockIdx.x*k;\n    int tid = threadIdx.x;', 1)
    with xp.cuda.Device(device):
        return xp.RawKernel(source, 'normalized_topk', options=native._OPTIONS)


def try_router_chunk(router, x):
    if (BACKEND_NAME != 'cupy' or not getattr(router, '_capturable_logits', False)
            or not router.renormalize_selected or not router._decode_fastpath_enabled
            or x.dtype != xp.float32 or not 1 <= router.n_experts <= 4096
            or not 1 <= router.k <= min(32, router.n_experts)
            or not np.finfo(np.float32).tiny <= router.normalization_floor <= np.finfo(np.float32).max):
        return None
    count = x.shape[0]*x.shape[1]
    if not 1 <= count <= 65535:
        return None
    rows = x.reshape(count, 1, router.d_model)
    logits = matmul_inference(rows, router.W_router).reshape(count, router.n_experts) + router.b_router
    weights = xp.empty(x.shape[:2] + (router.k,), xp.float32)
    ids = xp.empty(x.shape[:2] + (router.k,), xp.int64)
    select, reduce = native._selection_modes(None, None, environment=True)
    kernel = _kernel(xp.cuda.runtime.getDevice(), select, reduce)
    kernel((count,), (128,), (logits, weights, ids, np.int32(router.n_experts),
                             np.int32(router.k), np.float32(router.normalization_floor)))
    return weights, ids
