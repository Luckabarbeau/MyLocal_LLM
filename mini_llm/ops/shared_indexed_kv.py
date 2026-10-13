"""0079P2: read-only indexed views of a shared causal K/V cache.

The causal encoder owns one [document, time, KV-head, head-width] tensor per
layer. Refinement rows reference that tensor through (document, key) indices;
only their K provisional keys are newly projected for each refinement pass.
Historical gradients accumulate directly into the causal encoder's FP32 buffers.

This path is experimental, opt-in, and independent of the normal AR kernels.
"""
import numpy as np

from mini_llm.backend import xp, BACKEND_NAME


class SharedIndexedKV:
    """Virtual [refinement_row, history_slot + active_slot, KV-head, width].

    No historical K/V is copied on construction or retained by each pass.
    Context gradient buffers are shared across all passes and groups.
    """
    _is_shared_indexed_kv = True

    def __init__(self, historical, history_gradient, doc_rows, history_index,
                 active=None):
        self.historical = historical
        self.history_gradient = history_gradient
        self.doc_rows = doc_rows
        self.history_index = history_index
        self.active = active
        self.history_width = int(history_index.shape[1])
        self.n_rows = int(history_index.shape[0])
        self.dtype = historical.dtype
        self.ndim = 4
        self.shape = (self.n_rows, self.history_width + (0 if active is None else active.shape[1]),
                      historical.shape[2], historical.shape[3])
        if historical.ndim != 4 or doc_rows.shape != (self.n_rows,):
            raise ValueError('invalid shared K/V history view')
        if history_gradient is not None and history_gradient.shape != historical.shape:
            raise ValueError('historical K/V gradient must match causal cache')
        if active is not None and (active.shape[0] != self.n_rows or
                                   active.shape[2:] != historical.shape[2:]):
            raise ValueError('active K/V does not match historical heads')

    def with_active(self, active):
        return SharedIndexedKV(self.historical, self.history_gradient,
                               self.doc_rows, self.history_index, active)

    def gather(self, batch_ids, key_indices, kv_ids):
        """Gather only selected keys, never a [rows, history, heads, width] copy."""
        if self.active is None:
            raise RuntimeError('a shared K/V view requires provisional K/V')
        if BACKEND_NAME == 'cupy':
            return _gather_split_cuda(self, key_indices, kv_ids)
        c = self.history_width
        if c == 0:
            return self.active[batch_ids, key_indices, kv_ids, :]
        if not self.active.shape[1]:
            source = self.history_index[batch_ids, key_indices]
            return self.historical[self.doc_rows[batch_ids], source, kv_ids, :]
        is_history = key_indices < c
        historical_slot = xp.minimum(key_indices, c - 1)
        actual_key = self.history_index[batch_ids, historical_slot]
        from_context = self.historical[self.doc_rows[batch_ids], actual_key, kv_ids, :]
        from_active = self.active[batch_ids, xp.maximum(key_indices - c, 0), kv_ids, :]
        return xp.where(is_history[..., None], from_context, from_active)

    def add_gradient(self, indices, valid, kv_indices, contributions, active_grad):
        """Portable exact reference scatter; GPU BF16 uses the fused kernel."""
        b, h, q, slots, _ = contributions.shape
        bidx = xp.broadcast_to(xp.arange(b)[:, None, None, None], indices.shape)
        heads = xp.broadcast_to(kv_indices[None, :, None, None], indices.shape)
        allowed = valid
        if self.history_width:
            historical = allowed & (indices < self.history_width)
            source = self.history_index[bidx, xp.minimum(indices, self.history_width - 1)]
            values = xp.where(historical[..., None], contributions, 0)
            if self.history_gradient is None:
                raise RuntimeError('shared K/V gradient destination is missing')
            xp.add.at(self.history_gradient,
                      (self.doc_rows[bidx], source, heads), values)
        active = allowed & (indices >= self.history_width)
        slot = xp.maximum(indices - self.history_width, 0)
        xp.add.at(active_grad, (bidx, slot, heads),
                  xp.where(active[..., None], contributions, 0))


_SPLIT_GATHER_KERNELS = {}


def _get_split_gather_kernel(itemsize):
    if itemsize in _SPLIT_GATHER_KERNELS:
        # Keep RawModule alive in the cache, but return its callable RawKernel.
        return _SPLIT_GATHER_KERNELS[itemsize][1]
    if itemsize not in (2, 4, 8):
        raise TypeError('split gather supports BF16/FP16/FP32/FP64')
    word = {2: 'unsigned short', 4: 'unsigned int', 8: 'unsigned long long'}[itemsize]
    code = r"""
    extern "C" __global__ void gather_split(
        const WORD* history, const WORD* active,
        const long long* key_indices, const long long* head_map,
        const long long* doc_rows, const long long* history_index,
        WORD* out, long long elements, int n_heads, int n_queries,
        int n_slots, int d_head, int n_history, int n_active,
        int n_context, int n_kv_heads)
    {
        for (long long i=(long long)blockIdx.x*blockDim.x+threadIdx.x;
             i<elements; i+=(long long)gridDim.x*blockDim.x) {
            int d=(int)(i%d_head); long long t=i/d_head;
            int slot=(int)(t%n_slots); t/=n_slots;
            int query=(int)(t%n_queries); t/=n_queries;
            int head=(int)(t%n_heads); int row=(int)(t/n_heads);
            long long pair=(((long long)row*n_heads+head)*n_queries+query)*n_slots+slot;
            long long key=key_indices[pair];
            int kv_head=(int)head_map[head];
            if (key < n_history) {
                int doc=(int)doc_rows[row];
                long long pos=history_index[(long long)row*n_history+key];
                long long off=(((long long)doc*n_context+pos)*n_kv_heads+kv_head)*d_head+d;
                out[i]=history[off];
            } else {
                int index=(int)(key-n_history);
                long long off=(((long long)row*n_active+index)*n_kv_heads+kv_head)*d_head+d;
                out[i]=active[off];
            }
        }
    }
    """.replace('WORD', word)
    module = xp.RawModule(code=code, options=('--std=c++11',),
                          name_expressions=('gather_split',))
    func = module.get_function('gather_split')
    _SPLIT_GATHER_KERNELS[itemsize] = (module, func)
    return func


def _gather_split_cuda(source, key_indices, kv_ids):
    b, h, q, slots = map(int, key_indices.shape)
    dh = int(source.shape[-1])
    dest = xp.empty((b,h,q,slots,dh), dtype=source.dtype)
    total = dest.size
    if not total:
        return dest
    if not (source.historical.flags.c_contiguous and source.active.flags.c_contiguous):
        raise ValueError('resident K/V CUDA gather requires contiguous source tensors')
    kernel=_get_split_gather_kernel(source.dtype.itemsize)
    threads=256
    grid=min(65535,(total+threads-1)//threads)
    kernel((grid,), (threads,), (
        source.historical, source.active,
        xp.ascontiguousarray(key_indices, dtype=xp.int64),
        xp.ascontiguousarray(kv_ids.reshape(-1),dtype=xp.int64),
        xp.ascontiguousarray(source.doc_rows,dtype=xp.int64),
        xp.ascontiguousarray(source.history_index,dtype=xp.int64),
        dest, np.int64(total), np.int32(h),np.int32(q),np.int32(slots),
        np.int32(dh),np.int32(source.history_width),
        np.int32(source.active.shape[1]),np.int32(source.historical.shape[1]),
        np.int32(source.historical.shape[2]),
    ))
    return dest


_SPLIT_GRAD_KERNEL = None


def _get_split_grad_kernel():
    global _SPLIT_GRAD_KERNEL
    if _SPLIT_GRAD_KERNEL is not None:
        return _SPLIT_GRAD_KERNEL
    if BACKEND_NAME != 'cupy':
        raise RuntimeError('split gradient CUDA kernel requires CuPy')
    code = r'''
    extern "C" __global__ void shared_split_kv_backward(
        const float* q, const float* dc, const float* probs, const float* ds,
        const long long* indices, const bool* valid, const long long* kv_map,
        const long long* doc_rows, const long long* history_index,
        float* dk_context, float* dv_context, float* dk_active, float* dv_active,
        long long elements, int n_heads, int n_query, int n_slots,
        int dh, int n_history, int n_active, int n_context_tokens,
        int n_kv_heads, float scale)
    {
        for (long long i=(long long)blockIdx.x*blockDim.x+threadIdx.x;
             i<elements; i+=(long long)gridDim.x*blockDim.x) {
            int d = (int)(i % dh);
            long long idx = i / dh;
            int slot = (int)(idx % n_slots); idx /= n_slots;
            int query = (int)(idx % n_query); idx /= n_query;
            int head = (int)(idx % n_heads);
            int row = (int)(idx / n_heads);
            long long pair = ((((long long)row*n_heads+head)*n_query+query)*n_slots+slot);
            if (!valid[pair]) continue;
            long long key = indices[pair];
            int kvh = (int)kv_map[head];
            long long qi = ((((long long)row*n_heads+head)*n_query+query)*dh+d);
            long long dest;
            float* dk; float* dv;
            if (key < n_history) {
                int doc = (int)doc_rows[row];
                long long physical_key = history_index[(long long)row*n_history+key];
                if (doc < 0 || physical_key < 0 || physical_key >= n_context_tokens) continue;
                dest = (((long long)doc*n_context_tokens+physical_key)*n_kv_heads+kvh)*dh+d;
                dk = dk_context; dv = dv_context;
            } else {
                key -= n_history;
                if (key < 0 || key >= n_active) continue;
                dest = (((long long)row*n_active+key)*n_kv_heads+kvh)*dh+d;
                dk = dk_active; dv = dv_active;
            }
            atomicAdd(&dk[dest], ds[pair]*q[qi]*scale);
            atomicAdd(&dv[dest], probs[pair]*dc[qi]);
        }
    }
    '''
    module = xp.RawModule(code=code, options=('--std=c++11',),
                          name_expressions=('shared_split_kv_backward',))
    _SPLIT_GRAD_KERNEL = module.get_function('shared_split_kv_backward')
    return _SPLIT_GRAD_KERNEL


def scatter_split_pair_cuda(q, dc, probs, dscores, indices, valid, kv_map,
                            key_source, value_source, dk_active, dv_active, scale):
    """FP32 atomic K/V accumulation directly into shared history and active K/V."""
    n_rows, n_heads, n_query, n_slots = map(int, probs.shape)
    d_head = int(q.shape[-1])
    if key_source.history_gradient is None or value_source.history_gradient is None:
        raise RuntimeError('shared gradient destinations required for backward')
    if not (key_source.history_gradient.flags.c_contiguous and
            value_source.history_gradient.flags.c_contiguous and
            dk_active.flags.c_contiguous and dv_active.flags.c_contiguous):
        raise ValueError('split backward needs contiguous gradient destinations')
    n_elements = n_rows * n_heads * n_query * n_slots * d_head
    if not n_elements:
        return
    kernel = _get_split_grad_kernel()
    block = 256
    grid = min(65535, (n_elements + block - 1) // block)
    kernel((grid,), (block,), (
        xp.ascontiguousarray(q, dtype=xp.float32),
        xp.ascontiguousarray(dc, dtype=xp.float32),
        xp.ascontiguousarray(probs, dtype=xp.float32),
        xp.ascontiguousarray(dscores, dtype=xp.float32),
        xp.ascontiguousarray(indices, dtype=xp.int64),
        xp.ascontiguousarray(valid, dtype=xp.bool_),
        xp.ascontiguousarray(kv_map, dtype=xp.int64),
        xp.ascontiguousarray(key_source.doc_rows, dtype=xp.int64),
        xp.ascontiguousarray(key_source.history_index, dtype=xp.int64),
        key_source.history_gradient, value_source.history_gradient,
        dk_active, dv_active, np.int64(n_elements), np.int32(n_heads),
        np.int32(n_query), np.int32(n_slots), np.int32(d_head),
        np.int32(key_source.history_width), np.int32(key_source.active.shape[1]),
        np.int32(key_source.historical.shape[1]),
        np.int32(key_source.historical.shape[2]), np.float32(scale),
    ))
