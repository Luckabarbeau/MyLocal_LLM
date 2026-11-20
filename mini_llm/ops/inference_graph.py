"""Bounded current-stream replay of an array-only inference computation.

Warmup always uses private input/state buffers. Capture never executes caller
state updates; explicitly bound mutable operands update in place only on replay.
A private pool owns capture intermediates, even
when their Python array wrappers disappear; it is never used for replay-time
external allocations. Architecture adapters select replay for eligible calls.
"""

from functools import lru_cache
import os
import numpy as np

from mini_llm.backend import BACKEND_NAME, xp


@lru_cache(maxsize=8)
def _device_copy_kernel(device):
    with xp.cuda.Device(device):
        return xp.RawKernel(r'''
extern "C" __global__ void copy_graph_bytes(const unsigned char* src,unsigned char* dst,
                                           unsigned long long n) {
    unsigned long long i=(blockIdx.x*256ull+threadIdx.x)*16ull;
    if(i>=n) return;
    if(i+16<=n && !(reinterpret_cast<unsigned long long>(src)&15ull) &&
                     !(reinterpret_cast<unsigned long long>(dst)&15ull))
        *reinterpret_cast<uint4*>(dst+i)=*reinterpret_cast<const uint4*>(src+i);
    else for(unsigned long long j=i;j<n && j<i+16;++j) dst[j]=src[j];
}
''','copy_graph_bytes',options=('--std=c++17',))


class CapturedArrayInference:
    def __init__(self, enqueue, x, mutable_arrays, *, pool_limit_bytes=32 * 1024**2,
                 copy_back_mutable=True, bound_mutable=()):
        if BACKEND_NAME != 'cupy':
            raise RuntimeError('CUDA graph replay requires CuPy')
        if not isinstance(copy_back_mutable, bool):
            raise TypeError('copy_back_mutable must be bool')
        self.copy_back_mutable = copy_back_mutable
        mutable_arrays=tuple(mutable_arrays)
        bound_mutable=tuple(bound_mutable)
        if (len(set(bound_mutable))!=len(bound_mutable) or
                any(isinstance(i,bool) or not isinstance(i,int) or not 0<=i<len(mutable_arrays)
                    for i in bound_mutable)):
            raise ValueError('Bound mutable indices must be distinct valid integers')
        self.bound_mutable=bound_mutable
        self.closed=False
        arrays = (x, *mutable_arrays)
        if not arrays or any(not isinstance(a, xp.ndarray) or not a.flags.c_contiguous
                             for a in arrays):
            raise ValueError('Graph operands must be contiguous backend arrays')
        device = xp.cuda.runtime.getDevice()
        if any(a.device.id != device for a in arrays):
            raise ValueError('Graph operands must be on the current device')
        for i in bound_mutable:
            a=mutable_arrays[i]
            for j,b in enumerate(arrays):
                if j==i+1: continue
                if a.nbytes and b.nbytes and a.data.ptr<b.data.ptr+b.nbytes and b.data.ptr<a.data.ptr+a.nbytes:
                    raise ValueError('Bound graph state must not alias other operands')
        self.device = device
        self.device_copy=os.environ.get('MINI_LLM_GRAPH_DEVICE_COPY','0')=='1'
        self.copy_kernel=_device_copy_kernel(device) if self.device_copy else None
        if self.copy_kernel is not None:
            self.copy_kernel.compile() # Prepare outside any graph capture.
        self.bound_pointers=tuple(mutable_arrays[i].data.ptr for i in bound_mutable)
        self.signature = tuple((a.shape, a.dtype.str) for a in arrays)
        self.pool = xp.cuda.MemoryPool()
        self.pool.set_limit(size=int(pool_limit_bytes))
        self.capture_stream = xp.cuda.Stream(non_blocking=True)
        ready = xp.cuda.Event(disable_timing=True)
        ready.record(xp.cuda.get_current_stream())
        self.capture_stream.wait_event(ready)
        with self.capture_stream, xp.cuda.using_allocator(self.pool.malloc):
            self.input = x.copy()
            self.mutable = tuple(a.copy() for a in mutable_arrays)
            # Warm every high-level operation/toolchain on private state first.
            try:
                warm_output = enqueue(self.input, self.mutable)
            except BaseException:
                # Graph-owned warmup buffers may already have queued consumers.
                self.capture_stream.synchronize()
                raise
            self.capture_stream.synchronize()
            # Capture starts from caller values, not the warmed private state.
            for target, source in zip(self.mutable, mutable_arrays):
                xp.copyto(target, source)
            self.capture_stream.synchronize()
            # Warmup only touched private copies. Capture, which does not execute
            # the graph, may now bind explicitly retained request-owned state.
            warm_mutable=self.mutable
            self.mutable=tuple(a if i in bound_mutable else private
                               for i,(a,private) in enumerate(zip(mutable_arrays,self.mutable)))
            self.capture_stream.begin_capture()
            try:
                self.output = enqueue(self.input, self.mutable)
                self.graph = self.capture_stream.end_capture()
            except Exception:
                if self.capture_stream.is_capturing():
                    self.capture_stream.end_capture()
                raise
            self.graph.upload(stream=self.capture_stream)
            self.capture_stream.synchronize()
            # Keep warm arrays live until capture/upload is complete.
            del warm_output,warm_mutable
        self.calls = 0

    def matches(self, x, mutable_arrays):
        arrays = (x, *mutable_arrays)
        return (not self.closed and xp.cuda.runtime.getDevice() == self.device and
                len(arrays) == len(self.signature) and
                all(isinstance(a, xp.ndarray) and a.flags.c_contiguous and
                    a.device.id == self.device and (a.shape, a.dtype.str) == spec
                    for a, spec in zip(arrays, self.signature)) and
                all(mutable_arrays[i].data.ptr==pointer for i,pointer in zip(self.bound_mutable,self.bound_pointers)))

    def accepts_current_stream(self):
        """Metadata-only eligibility check; never migrate live graph scratch."""
        return not self.calls or xp.cuda.get_current_stream().ptr == self._replay_stream

    def forward(self, x, mutable_arrays, *, borrow_output=False):
        """Replay; borrowed arrays expire at the next replay on this stream.

        Tuple outputs support internal multi-boundary segments. Fresh ownership
        remains the default; borrowing requires serialized downstream consumers.
        """
        if not isinstance(borrow_output, bool):
            raise TypeError('borrow_output must be bool')
        if not self.matches(x, mutable_arrays):
            raise ValueError('Graph operand shape/dtype/device/layout or bound state ownership changed')
        stream = xp.cuda.get_current_stream()
        # The adapter's serial execution contract orders graph replay on one
        # stream. Reject cross-stream reuse rather than overwrite live scratch.
        if not self.accepts_current_stream():
            raise ValueError('A captured inference call must reuse its replay stream')
        self._replay_stream = stream.ptr
        self._replay_stream_owner=stream
        self._copy(self.input, x)
        for i,(target, source) in enumerate(zip(self.mutable, mutable_arrays)):
            if i not in self.bound_mutable: self._copy(target, source)
        self.graph.launch(stream=stream)
        if self.copy_back_mutable:
            for i,(target, source) in enumerate(zip(mutable_arrays, self.mutable)):
                if i not in self.bound_mutable: self._copy(target, source)
        if borrow_output:
            out = self.output
        elif isinstance(self.output, tuple):
            out = tuple(self._clone(a) for a in self.output)
        else:
            out = self._clone(self.output)
        self.calls += 1
        return out

    def _copy(self,target,source):
        if self.copy_kernel is None:
            xp.copyto(target,source)
        elif target.nbytes and target.data.ptr!=source.data.ptr:
            self.copy_kernel(((target.nbytes+4095)//4096,),(256,),
                (source,target,np.uint64(target.nbytes)))

    def _clone(self,array):
        if self.copy_kernel is None: return array.copy()
        result=xp.empty_like(array)
        self._copy(result,array)
        return result

    def bound_arrays(self):
        """Actual retained external allocations, not private-pool reservations."""
        return () if self.closed else tuple(self.mutable[i] for i in self.bound_mutable)

    def close(self):
        """Retire capture/replay readers before releasing bound state or scratch."""
        if self.closed: return
        self.capture_stream.synchronize()
        if self.calls:
            self._replay_stream_owner.synchronize()
        self.graph=None
        self.output=None
        self.input=None
        self.mutable=()
        self.pool.free_all_blocks()
        self._replay_stream_owner=None
        self.closed=True

    def statistics(self):
        return dict(calls=self.calls, copy_back_mutable=self.copy_back_mutable,
                    bound_mutable=self.bound_mutable,closed=self.closed,device_copy=self.device_copy,
                    pool_reserved_bytes=int(self.pool.total_bytes()),
                    pool_used_bytes=int(self.pool.used_bytes()),
                    pool_limit_bytes=int(self.pool.get_limit()))
