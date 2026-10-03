"""Expert FFN layers for Mixture of Experts.

Two implementations:
1. Dense (reference): Runs all selected experts on full batch
2. Sparse (optimized): True token-to-expert dispatch
"""

from ..backend import xp, BACKEND_NAME, is_bfloat16_dtype
import os
import numpy as np
from .routing_plan import RoutingPlan
from ..performance_profiler import moe_detail_scope


from .silu import silu, silu_prime


_FUSED_SWIGLU_MODULE = None
_FUSED_SWIGLU_DISABLED = False


def _fused_swiglu_enabled(dtype):
    raw = os.environ.get("MINI_LLM_FUSED_SWIGLU", "0").strip().lower()
    return (
        BACKEND_NAME == "cupy"
        and is_bfloat16_dtype(dtype)
        and raw not in {"0", "false", "off", "no"}
    )


def _get_fused_swiglu_module():
    """Compile header-free BF16 SwiGLU CUDA kernels lazily."""
    global _FUSED_SWIGLU_MODULE, _FUSED_SWIGLU_DISABLED
    if BACKEND_NAME != "cupy" or _FUSED_SWIGLU_DISABLED:
        return None
    if _FUSED_SWIGLU_MODULE is not None:
        return _FUSED_SWIGLU_MODULE

    code = r"""
    __device__ __forceinline__ float bf16_to_float(unsigned short x) {
        union { unsigned int u; float f; } v;
        v.u = ((unsigned int)x) << 16;
        return v.f;
    }

    __device__ __forceinline__ unsigned short float_to_bf16(float x) {
        union { unsigned int u; float f; } v;
        v.f = x;
        unsigned int bits = v.u;
        unsigned int lsb = (bits >> 16) & 1u;
        bits += 0x7fffu + lsb;
        return (unsigned short)(bits >> 16);
    }

    __device__ __forceinline__ float sigmoidf_stable(float x) {
        return 1.0f / (1.0f + expf(-x));
    }

    extern "C" __global__
    void bf16_swiglu_fwd(
        const unsigned short* g,
        const unsigned short* u,
        unsigned short* h,
        long long n) {
        long long idx = ((long long)blockIdx.x) * blockDim.x + threadIdx.x;
        long long stride = ((long long)gridDim.x) * blockDim.x;
        for (; idx < n; idx += stride) {
            float gf = bf16_to_float(g[idx]);
            float uf = bf16_to_float(u[idx]);
            float s = sigmoidf_stable(gf);
            h[idx] = float_to_bf16((gf * s) * uf);
        }
    }

    extern "C" __global__
    void bf16_swiglu_bwd(
        const unsigned short* dh,
        const unsigned short* g,
        const unsigned short* u,
        unsigned short* dg,
        unsigned short* du,
        long long n) {
        long long idx = ((long long)blockIdx.x) * blockDim.x + threadIdx.x;
        long long stride = ((long long)gridDim.x) * blockDim.x;
        for (; idx < n; idx += stride) {
            float dhf = bf16_to_float(dh[idx]);
            float gf = bf16_to_float(g[idx]);
            float uf = bf16_to_float(u[idx]);
            float s = sigmoidf_stable(gf);
            float a = gf * s;
            float silu_prime = s + gf * s * (1.0f - s);
            dg[idx] = float_to_bf16((dhf * uf) * silu_prime);
            du[idx] = float_to_bf16(dhf * a);
        }
    }
    """
    try:
        module = xp.RawModule(
            code=code,
            options=("--std=c++11",),
            name_expressions=("bf16_swiglu_fwd", "bf16_swiglu_bwd"),
        )
        module.get_function("bf16_swiglu_fwd")
        module.get_function("bf16_swiglu_bwd")
        _FUSED_SWIGLU_MODULE = module
    except Exception:
        strict = os.environ.get("MINI_LLM_FUSED_SWIGLU_STRICT", "0").strip().lower()
        if strict not in {"0", "false", "off", "no"}:
            raise
        _FUSED_SWIGLU_DISABLED = True
        return None
    return _FUSED_SWIGLU_MODULE


def _fused_swiglu_forward(g, u):
    if not _fused_swiglu_enabled(g.dtype):
        return None
    if g.shape != u.shape or not g.flags.c_contiguous or not u.flags.c_contiguous:
        return None
    module = _get_fused_swiglu_module()
    if module is None:
        return None
    h = xp.empty_like(g)
    n = int(g.size)
    if n == 0:
        return h
    threads = 256
    blocks = min(65535, max(1, (n + threads - 1) // threads))
    module.get_function("bf16_swiglu_fwd")((blocks,), (threads,), (
        g, u, h, np.int64(n)
    ))
    return h


def _fused_swiglu_backward(dh, g, u):
    if not _fused_swiglu_enabled(g.dtype) or not is_bfloat16_dtype(dh.dtype):
        return None
    if dh.dtype != g.dtype or u.dtype != g.dtype:
        return None
    if dh.shape != g.shape or u.shape != g.shape:
        return None
    if not dh.flags.c_contiguous or not g.flags.c_contiguous or not u.flags.c_contiguous:
        return None
    module = _get_fused_swiglu_module()
    if module is None:
        return None
    dg = xp.empty_like(g)
    du = xp.empty_like(u)
    n = int(g.size)
    if n == 0:
        return dg, du
    threads = 256
    blocks = min(65535, max(1, (n + threads - 1) // threads))
    module.get_function("bf16_swiglu_bwd")((blocks,), (threads,), (
        dh, g, u, dg, du, np.int64(n)
    ))
    return dg, du


def _concurrent_experts_enabled(dtype):
    """Whether to overlap independent sparse experts on CUDA streams.

    This path is deliberately BF16/CuPy-only.  Experts have disjoint
    parameters, so their FFN math and parameter-gradient updates can execute
    independently.  Token/output scatters remain on the caller stream after
    explicit event waits because top-k experts may target the same token.
    """
    raw = os.environ.get("MINI_LLM_CONCURRENT_EXPERTS", "0").strip().lower()
    return (
        BACKEND_NAME == "cupy"
        and is_bfloat16_dtype(dtype)
        and raw not in {"0", "false", "off", "no", ""}
    )


def _concurrent_expert_stream_count(n_experts):
    raw = os.environ.get("MINI_LLM_EXPERT_STREAMS", "2").strip()
    try:
        count = int(raw)
    except ValueError:
        count = 2
    return max(1, min(int(n_experts), count))


def _fused_expert_gemm_enabled():
    """Enable the checkpoint-compatible fused SwiGLU projection path.

    The optimization packs the two independent gate/up matrices transiently
    and evaluates them with one larger GEMM.  It can be disabled at runtime
    for A/B performance checks without changing model parameters or
    checkpoints.
    """
    raw = os.environ.get("MINI_LLM_FUSED_EXPERT_GEMM", "1").strip().lower()
    return raw not in {"0", "false", "off", "no"}


class ExpertFFN:
    """
    Single expert FFN using SwiGLU structure.
    
    This is a standalone expert that can be composed into a Mixture of Experts.
    """

    def __init__(
        self, d_model, d_ff, input_std, output_std, rng, name="expert", dtype="float32"
    ):
        """
        Initialize the expert FFN.
        
        Args:
            d_model: Model dimension
            d_ff: FFN hidden dimension
            input_std: Standard deviation for weight initialization
            output_std: Standard deviation for down projection
            rng: Random stream
            name: Name prefix for parameters
            dtype: Data type
        """
        from ..parameter import Parameter

        self.d_model = int(d_model)
        self.d_ff = int(d_ff)
        std = input_std
        self.W_gate = Parameter(
            xp.asarray(rng.normal((d_model, d_ff), std=std, dtype=dtype)),
            name=f"{name}.W_gate"
        )
        self.W_up = Parameter(
            xp.asarray(rng.normal((d_model, d_ff), std=std, dtype=dtype)),
            name=f"{name}.W_up"
        )
        self.W_down = Parameter(
            xp.asarray(rng.normal((d_ff, d_model), std=output_std, dtype=dtype)),
            name=f"{name}.W_down"
        )

    def parameters(self):
        """Return trainable parameters."""
        return [self.W_gate, self.W_up, self.W_down]

    def zero_grad(self):
        """Zero out gradients."""
        self.W_gate.zero_grad()
        self.W_up.zero_grad()
        self.W_down.zero_grad()

    @classmethod
    def reset_forward_count(cls):
        """Reset the forward call counter."""
        cls._forward_call_count = 0

    @classmethod
    def get_forward_count(cls):
        """Get the total number of forward calls since last reset."""
        return cls._forward_call_count

    _forward_call_count = 0  # Class-level counter for profiling

    def forward(self, x, return_cache=True):
        """
        Forward pass through the expert.
        
        G = X W_gate
        U = X W_up
        A = SiLU(G)
        H = A * U
        Y = H W_down
        
        Args:
            x: Input tensor of shape (B, T, d_model)
            
        Returns:
            y: Output tensor of shape (B, T, d_model)
            cache: Dictionary for backward pass (contains minimal intermediates)
        """
        # Increment forward call counter
        ExpertFFN._forward_call_count += 1
        
        if _fused_expert_gemm_enabled():
            # W_gate and W_up remain independent Parameters so checkpoint and
            # optimizer state layout stay unchanged.  Packing is only a
            # transient contiguous copy; the larger GEMM amortizes launch
            # overhead and gives cuBLAS more work per invocation.
            with moe_detail_scope("moe.expert.pack_gate_up"):
                w_gate_up = xp.concatenate((self.W_gate.data, self.W_up.data), axis=1)
            with moe_detail_scope("moe.expert.gate_up_gemm"):
                gate_up = x @ w_gate_up
            g = gate_up[..., : self.d_ff]
            u = gate_up[..., self.d_ff :]
        else:
            with moe_detail_scope("moe.expert.gate_gemm"):
                g = x @ self.W_gate.data
            with moe_detail_scope("moe.expert.up_gemm"):
                u = x @ self.W_up.data
        with moe_detail_scope("moe.expert.swiglu"):
            h = _fused_swiglu_forward(g, u)
            if h is None:
                a = silu(g)
                h = a * u
        with moe_detail_scope("moe.expert.down_gemm"):
            y = h @ self.W_down.data
        
        if not return_cache:
            return y

        # Memory-oriented training cache.  ``a = SiLU(g)`` and
        # ``h = a * u`` are cheap elementwise intermediates and are therefore
        # recomputed in backward rather than retained for every routed token.
        # At long context this removes two d_ff-sized BF16 activation caches
        # per expert assignment without adding any extra GEMM work.
        cache = {
            "x": x,
            "g": g,
            "u": u,
        }
        
        return y, cache

    def backward(self, dy, cache):
        """
        Backward pass through the expert.
        
        Args:
            dy: Gradient w.r.t. output, same shape as y
            cache: Dictionary from forward pass
            
        Returns:
            dx: Gradient w.r.t. input, same shape as x
        """
        x = cache["x"]
        g = cache["g"]
        u = cache["u"]
        # Recompute only the down-projection input.  The fused BF16 path
        # produces H directly from cached G/U and avoids materializing A.
        with moe_detail_scope("moe.expert.recompute"):
            h = _fused_swiglu_forward(g, u)
            fused_swiglu = h is not None
            if not fused_swiglu:
                a = silu(g)
                h = a * u

        dy_2d = dy.reshape(-1, dy.shape[-1])
        x_2d = x.reshape(-1, x.shape[-1])
        h_2d = h.reshape(-1, h.shape[-1])

        with moe_detail_scope("moe.expert.down_wgrad"):
            self.W_down.grad += h_2d.T @ dy_2d

        with moe_detail_scope("moe.expert.down_dx"):
            dh = dy_2d @ self.W_down.data.T
        with moe_detail_scope("moe.expert.swiglu_bwd"):
            fused_grads = (
                _fused_swiglu_backward(
                    dh,
                    g.reshape(dh.shape),
                    u.reshape(dh.shape),
                )
                if fused_swiglu
                else None
            )
            if fused_grads is not None:
                dg, du = fused_grads
            else:
                if fused_swiglu:
                    a = silu(g)
                da = dh * u.reshape(-1, u.shape[-1])
                du = dh * a.reshape(-1, a.shape[-1])
                dg = da * silu_prime(g.reshape(-1, g.shape[-1]))
        
        if _fused_expert_gemm_enabled():
            # The concatenated d[G,U] matrix lets both parameter gradients be
            # produced by one GEMM, and the same packed weights turn the two
            # input-gradient GEMMs plus add into one GEMM.
            with moe_detail_scope("moe.expert.pack_dgate_up"):
                dgate_up = xp.concatenate((dg, du), axis=-1)
            with moe_detail_scope("moe.expert.gate_up_wgrad"):
                dweight_gate_up = x_2d.T @ dgate_up
                split = self.d_ff
                self.W_gate.grad += dweight_gate_up[:, :split]
                self.W_up.grad += dweight_gate_up[:, split:]

            with moe_detail_scope("moe.expert.pack_weights_bwd"):
                w_gate_up = xp.concatenate((self.W_gate.data, self.W_up.data), axis=1)
            with moe_detail_scope("moe.expert.gate_up_dx"):
                dx = dgate_up @ w_gate_up.T
        else:
            with moe_detail_scope("moe.expert.gate_wgrad"):
                self.W_gate.grad += x_2d.T @ dg
            with moe_detail_scope("moe.expert.up_wgrad"):
                self.W_up.grad += x_2d.T @ du
            with moe_detail_scope("moe.expert.gate_dx"):
                dx_gate = dg @ self.W_gate.data.T
            with moe_detail_scope("moe.expert.up_dx"):
                dx_up = du @ self.W_up.data.T
            with moe_detail_scope("moe.expert.dx_add"):
                dx = dx_gate + dx_up
        dx = dx.reshape(dy.shape)
        
        return dx


class Experts:
    """
    Mixture of Expert FFN layers with sparse expert evaluation.
    
    Only evaluates the unique experts that are selected, not all experts.
    Uses vectorized operations for each expert's full batch computation.
    """

    def __init__(
        self, d_model, d_ff, n_experts, input_std, output_std, rng, name="experts", dtype="float32"
    ):
        """
        Initialize the experts.
        
        Args:
            d_model: Model dimension
            d_ff: FFN hidden dimension for each expert
            n_experts: Number of experts
            input_std: Standard deviation for weight initialization
            output_std: Standard deviation for down projections
            rng: Random stream
            name: Name prefix for parameters
            dtype: Data type
        """
        self.d_model = d_model
        self.d_ff = d_ff
        self.n_experts = n_experts
        # Lazily-created non-blocking CUDA streams used only by the opt-in
        # concurrent-expert path.  Keeping them on the Experts instance avoids
        # stream creation in the hot loop.
        self._expert_streams = []
        
        # Create n_experts ExpertFFN layers
        self.experts = []
        for i in range(n_experts):
            expert = ExpertFFN(
                d_model=d_model,
                d_ff=d_ff,
                input_std=input_std,
                output_std=output_std,
                rng=rng,
                name=f"{name}.expert_{i}",
                dtype=dtype
            )
            self.experts.append(expert)

    def parameters(self):
        """Return all trainable parameters from all experts."""
        params = []
        for expert in self.experts:
            params.extend(expert.parameters())
        return params

    def zero_grad(self):
        """Zero out gradients for all experts."""
        for expert in self.experts:
            expert.zero_grad()

    def reset_forward_count(self):
        """Reset forward call counters for all experts."""
        for expert in self.experts:
            ExpertFFN.reset_forward_count()

    def get_forward_count(self):
        """Get the total number of forward calls from all experts."""
        # The counter is shared across all instances via class variable
        return ExpertFFN.get_forward_count()

    def _get_expert_streams(self):
        """Return a persistent pool of non-blocking CUDA streams."""
        count = _concurrent_expert_stream_count(self.n_experts)
        if len(self._expert_streams) != count:
            self._expert_streams = [
                xp.cuda.Stream(non_blocking=True) for _ in range(count)
            ]
        return self._expert_streams

    @staticmethod
    def _record_ready_event():
        """Record all caller-stream work that expert streams must depend on."""
        event = xp.cuda.Event()
        event.record()
        return event

    @staticmethod
    def _wait_for_expert_events(events):
        """Make the caller stream depend on every launched expert stream."""
        current = xp.cuda.get_current_stream()
        for event in events:
            current.wait_event(event)

    def _forward_concurrent(
        self, x, weights, expert_indices, routing_plan, return_cache
    ):
        """Sparse expert forward with independent experts overlapped on streams.

        Only expert-local gather + FFN work is concurrent.  The weighted token
        scatter is intentionally performed on the caller stream after waiting
        for every expert event because different top-k experts can contribute
        to the same output token.
        """
        batch_size, seq_len, d_model = x.shape
        k = weights.shape[-1]
        N = batch_size * seq_len
        x_flat = x.reshape(N, d_model)

        with moe_detail_scope("moe.dispatch.fwd_alloc"):
            y_flat = xp.zeros((N, d_model), dtype=x.dtype)

        streams = self._get_expert_streams()
        ready = self._record_ready_event()
        for stream in streams:
            stream.wait_event(ready)

        pending = []
        expert_outputs = {}
        expert_caches = {}

        for exp_idx in range(self.n_experts):
            token_indices, slot_indices, expert_weights = (
                routing_plan.get_expert_assignments(exp_idx)
            )
            if routing_plan.get_assignment_count(exp_idx) == 0:
                continue

            stream = streams[exp_idx % len(streams)]
            with stream:
                with moe_detail_scope("moe.dispatch.fwd_gather"):
                    expert_x = x_flat[token_indices]
                if return_cache:
                    with moe_detail_scope("moe.expert.forward_total"):
                        expert_out, expert_cache = self.experts[exp_idx].forward(
                            expert_x
                        )
                else:
                    with moe_detail_scope("moe.expert.forward_total"):
                        expert_out = self.experts[exp_idx].forward(
                            expert_x, return_cache=False
                        )
                    expert_cache = None
                done = xp.cuda.Event()
                done.record()

            pending.append(
                (exp_idx, token_indices, expert_weights, expert_out, expert_cache, done)
            )

        self._wait_for_expert_events([item[-1] for item in pending])

        # Preserve the existing deterministic serialized scatter semantics.
        for exp_idx, token_indices, expert_weights, expert_out, expert_cache, _ in pending:
            if return_cache:
                expert_outputs[exp_idx] = {
                    "outputs": expert_out,
                    "token_indices": token_indices,
                    "weights": expert_weights,
                }
                expert_caches[exp_idx] = expert_cache
            with moe_detail_scope("moe.dispatch.fwd_weight"):
                weighted_out = expert_weights[:, xp.newaxis] * expert_out
            with moe_detail_scope("moe.dispatch.fwd_scatter"):
                y_flat[token_indices] += weighted_out

        y = y_flat.reshape(batch_size, seq_len, d_model)
        if not return_cache:
            return y
        return y, {
            "x": x,
            "weights": weights,
            "expert_indices": expert_indices,
            "y": y,
            "expert_outputs": expert_outputs,
            "expert_caches": expert_caches,
            "routing_plan": routing_plan,
        }

    def _backward_concurrent(self, dy, cache, return_dweights):
        """Sparse expert backward with independent expert math overlapped.

        Each expert owns disjoint parameters, so its weight-gradient updates are
        safe on a private stream.  Token dX and router-weight scatters remain on
        the caller stream after event waits, avoiding cross-stream write races.
        """
        x = cache["x"]
        weights = cache["weights"]
        expert_indices = cache["expert_indices"]
        routing_plan = cache.get("routing_plan")
        if routing_plan is None:
            # Legacy caches keep the original serial implementation.
            return None

        batch_size, seq_len, d_model = x.shape
        N = batch_size * seq_len
        k = weights.shape[-1]
        dy_flat = dy.reshape(N, d_model)

        with moe_detail_scope("moe.dispatch.bwd_alloc"):
            dx_flat = xp.zeros((N, d_model), dtype=x.dtype)
            dweights_flat = (
                xp.zeros((N, k), dtype=dy.dtype) if return_dweights else None
            )

        expert_outputs = cache.get("expert_outputs", {})
        expert_caches = cache.get("expert_caches", {})
        active_experts = [
            exp_idx
            for exp_idx in range(self.n_experts)
            if routing_plan.get_assignment_count(exp_idx) > 0
        ]
        if any(
            exp_idx not in expert_outputs or exp_idx not in expert_caches
            for exp_idx in active_experts
        ):
            # New optimized forward always caches these.  Retain the serial
            # fallback for any old/hand-built cache used by tests or tools.
            return None

        streams = self._get_expert_streams()
        ready = self._record_ready_event()
        for stream in streams:
            stream.wait_event(ready)

        pending = []
        for exp_idx in active_experts:
            token_indices, slot_indices, expert_weights = (
                routing_plan.get_expert_assignments(exp_idx)
            )
            expert_out = expert_outputs[exp_idx]["outputs"]
            expert_cache = expert_caches[exp_idx]
            stream = streams[exp_idx % len(streams)]
            with stream:
                with moe_detail_scope("moe.dispatch.bwd_gather"):
                    raw_expert_dy = dy_flat[token_indices]

                if return_dweights:
                    with moe_detail_scope("moe.dispatch.router_wgrad"):
                        dweight_values = xp.sum(
                            raw_expert_dy * expert_out, axis=-1
                        )
                else:
                    dweight_values = None

                with moe_detail_scope("moe.dispatch.bwd_weight"):
                    expert_dy = raw_expert_dy * expert_weights[:, xp.newaxis]
                with moe_detail_scope("moe.expert.backward_total"):
                    expert_dx = self.experts[exp_idx].backward(
                        expert_dy, expert_cache
                    )
                done = xp.cuda.Event()
                done.record()

            pending.append(
                (token_indices, slot_indices, dweight_values, expert_dx, done)
            )

        self._wait_for_expert_events([item[-1] for item in pending])

        for token_indices, slot_indices, dweight_values, expert_dx, _ in pending:
            if return_dweights:
                with moe_detail_scope("moe.dispatch.router_wgrad_scatter"):
                    dweights_flat[token_indices, slot_indices] = dweight_values
            with moe_detail_scope("moe.dispatch.bwd_scatter"):
                dx_flat[token_indices] += expert_dx

        dx = dx_flat.reshape(batch_size, seq_len, d_model)
        if return_dweights:
            return dx, dweights_flat.reshape(batch_size, seq_len, k)
        return dx

    def forward(self, x, weights, expert_indices, routing_plan: RoutingPlan = None, n_experts: int = None, return_cache=True):
        """
        Forward pass through experts with true sparse token-to-expert dispatch.
        
        Flattens tokens and dispatches each token to its selected experts.
        Each expert only processes tokens routed to it (sparse computation).
        
        Args:
            x: Input tensor of shape (B, T, d_model)
            weights: Expert weights from router, shape (B, T, k)
            expert_indices: Indices of selected experts, shape (B, T, k)
            routing_plan: Pre-computed routing plan (optional, creates if not provided)
            n_experts: Total number of experts (uses self.n_experts if not provided)
            
        Returns:
            y: Weighted combination of expert outputs, shape (B, T, d_model)
            cache: Dictionary for backward pass
        """
        batch_size, seq_len, d_model = x.shape
        k = weights.shape[-1]
        N = batch_size * seq_len  # Total tokens
        
        # Flatten inputs: [B, T, D] -> [N, D]
        x_flat = x.reshape(N, d_model)
        weights_flat = weights.reshape(N, k)
        expert_indices_flat = expert_indices.reshape(N, k)
        
        # For each (token, slot) pair, we have: token_idx, expert_idx, weight
        # Total assignments = N * k

        # Determine n_experts and build the routing plan before allocating the
        # serial output buffer so the concurrent path does not allocate it twice.
        if n_experts is None:
            n_experts = self.n_experts
        if routing_plan is None:
            routing_plan = RoutingPlan.from_router_outputs(
                expert_indices, weights, k, n_experts=self.n_experts
            )

        if _concurrent_experts_enabled(x.dtype):
            return self._forward_concurrent(
                x, weights, expert_indices, routing_plan, return_cache
            )

        # Prepare output: [N, D]
        with moe_detail_scope("moe.dispatch.fwd_alloc"):
            y_flat = xp.zeros((N, d_model), dtype=x.dtype)

        # Cache for expert outputs and caches (to avoid recomputation in backward)
        expert_outputs = {}  # exp_idx -> {"outputs": [...], "token_indices": [...], "weights": [...]}
        expert_caches = {}   # exp_idx -> cache from expert.forward()
        
        # Group assignments by expert using the routing plan
        for exp_idx in range(self.n_experts):
            # Get token, slot, and weight indices for this expert from plan
            token_indices, slot_indices, expert_weights = routing_plan.get_expert_assignments(exp_idx)
            
            if len(token_indices) > 0:
                # Gather tokens: [n_assigned, D]
                with moe_detail_scope("moe.dispatch.fwd_gather"):
                    expert_x = x_flat[token_indices]

                # Forward through this expert.  Reference inference does not
                # retain activations or routed outputs needed only by backward.
                if return_cache:
                    with moe_detail_scope("moe.expert.forward_total"):
                        expert_out, expert_cache = self.experts[exp_idx].forward(expert_x)
                    expert_outputs[exp_idx] = {
                        "outputs": expert_out,
                        "token_indices": token_indices,
                        "weights": expert_weights,
                    }
                    expert_caches[exp_idx] = expert_cache
                else:
                    with moe_detail_scope("moe.expert.forward_total"):
                        expert_out = self.experts[exp_idx].forward(
                            expert_x, return_cache=False
                        )

                # Weight and accumulate to output.  token_indices are unique
                # within a single expert because top-k cannot select the same
                # expert twice for one token, so atomics are unnecessary.
                with moe_detail_scope("moe.dispatch.fwd_weight"):
                    weighted_out = expert_weights[:, xp.newaxis] * expert_out
                with moe_detail_scope("moe.dispatch.fwd_scatter"):
                    y_flat[token_indices] += weighted_out
        
        # Reshape: [N, D] -> [B, T, D]
        y = y_flat.reshape(batch_size, seq_len, d_model)
        
        if not return_cache:
            return y

        # Store cache for backward - includes expert outputs to avoid recomputation
        cache = {
            "x": x,
            "weights": weights,
            "expert_indices": expert_indices,
            "y": y,
            "expert_outputs": expert_outputs,  # For router gradient computation
            "expert_caches": expert_caches,    # For expert backward pass
            "routing_plan": routing_plan,      # For backward pass
        }
        
        return y, cache

    def backward(self, dy, cache, return_dweights=False):
        """
        Backward pass through sparsely-routed experts.

        The routing plan and expert outputs cached in forward are reused here.
        When ``return_dweights`` is true, the router-weight gradient is
        computed in this SAME expert loop, avoiding the second dispatch pass
        that previously lived in ``MoE.backward``.

        Args:
            dy: Gradient w.r.t. MoE output, shape (B, T, d_model).
            cache: Dictionary returned by ``Experts.forward``.
            return_dweights: Also return dL/d(selected routing weights).

        Returns:
            dx, or ``(dx, dweights)`` when ``return_dweights=True``.
        """
        x = cache["x"]
        weights = cache["weights"]
        expert_indices = cache["expert_indices"]
        batch_size, seq_len, d_model = x.shape
        N = batch_size * seq_len
        k = weights.shape[-1]

        dy_flat = dy.reshape(N, d_model)
        weights_flat = weights.reshape(N, k)
        expert_indices_flat = expert_indices.reshape(N, k)

        with moe_detail_scope("moe.dispatch.bwd_alloc"):
            dx_flat = xp.zeros((N, d_model), dtype=x.dtype)
            dweights_flat = (
                xp.zeros((N, k), dtype=dy.dtype) if return_dweights else None
            )

        expert_outputs = cache.get("expert_outputs", {})
        expert_caches = cache.get("expert_caches", {})
        routing_plan = cache.get("routing_plan")

        if _concurrent_experts_enabled(x.dtype):
            concurrent = self._backward_concurrent(
                dy, cache, return_dweights=return_dweights
            )
            if concurrent is not None:
                return concurrent

        # Fallback metadata is built lazily only for legacy caches.  The normal
        # optimized path always receives a RoutingPlan from forward.
        fallback_token_ids = None
        fallback_slot_ids = None
        fallback_weights = None

        for exp_idx in range(self.n_experts):
            if routing_plan is not None:
                token_indices, slot_indices, expert_weights = (
                    routing_plan.get_expert_assignments(exp_idx)
                )
                if routing_plan.get_assignment_count(exp_idx) == 0:
                    continue
            else:
                expert_selected = (expert_indices_flat == exp_idx)
                if fallback_token_ids is None:
                    fallback_token_ids = xp.repeat(xp.arange(N), k)
                    fallback_slot_ids = xp.tile(xp.arange(k), N)
                    fallback_weights = weights_flat.reshape(-1)

                flat_mask = expert_selected.reshape(-1)
                token_indices = fallback_token_ids[flat_mask]
                slot_indices = fallback_slot_ids[flat_mask]
                expert_weights = fallback_weights[flat_mask]
                if len(token_indices) == 0:
                    continue

            # Raw dy is needed for dL/d(router weight).  The expert parameter
            # gradient receives dy multiplied by the selected routing weight.
            with moe_detail_scope("moe.dispatch.bwd_gather"):
                raw_expert_dy = dy_flat[token_indices]

            if exp_idx in expert_outputs and exp_idx in expert_caches:
                expert_out = expert_outputs[exp_idx]["outputs"]
                expert_cache = expert_caches[exp_idx]
            else:
                # Legacy-cache fallback only.  New forward passes always cache
                # both expert output and activation cache.
                expert_x = x.reshape(N, d_model)[token_indices]
                expert_out, expert_cache = self.experts[exp_idx].forward(expert_x)

            if return_dweights:
                # For y = sum_s w_s E_s(x):
                #   dL/dw_s = dot(dL/dy, E_s(x)).
                # The cached output order is exactly the routing-plan order for
                # this expert, so no token search / xp.where is necessary.
                with moe_detail_scope("moe.dispatch.router_wgrad"):
                    dweight_values = xp.sum(
                        raw_expert_dy * expert_out, axis=-1
                    )
                    dweights_flat[token_indices, slot_indices] = dweight_values

            with moe_detail_scope("moe.dispatch.bwd_weight"):
                expert_dy = raw_expert_dy * expert_weights[:, xp.newaxis]
            with moe_detail_scope("moe.expert.backward_total"):
                expert_dx = self.experts[exp_idx].backward(
                    expert_dy, expert_cache
                )

            # top-k selection contains each expert at most once per token, so
            # token_indices are unique within this expert.  The expert loop is
            # serialized on the same stream; atomics are unnecessary here.
            with moe_detail_scope("moe.dispatch.bwd_scatter"):
                dx_flat[token_indices] += expert_dx

        dx = dx_flat.reshape(batch_size, seq_len, d_model)

        if return_dweights:
            return dx, dweights_flat.reshape(batch_size, seq_len, k)
        return dx
