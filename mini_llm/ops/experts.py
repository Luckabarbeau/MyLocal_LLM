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

    def forward(self, x, return_cache=True, cache_input=True):
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
        # 0055A: the parent Experts dispatcher already retains the original
        # MoE input and routing plan.  Keeping every gathered expert_x here
        # duplicates all routed tokens (top-k times) until backward.  Cache only
        # the two expensive SwiGLU projection activations; the dispatcher
        # regathers expert_x from the original input during backward.
        cache = {
            "g": g,
            "u": u,
        }
        if cache_input:
            cache["x"] = x
        
        return y, cache

    def backward(
        self, dy, cache, *, x=None, route_weights=None, return_route_dweights=False
    ):
        """Backward pass through the expert.

        ``0055A`` optionally accepts unweighted routed ``dy`` together with the
        selected ``route_weights``.  This lets the same down-projection GEMM
        provide the expert-path gradient while computing ``dL/d(route_weight)``
        from the cached hidden activation, so the full expert output no longer
        has to be retained from forward.

        Args:
            dy: Gradient w.r.t. expert output.  When ``route_weights`` is
                supplied this is the *unweighted* upstream MoE gradient.
            cache: Expert cache containing the expensive G/U projections.
            x: Regathered expert input.  Legacy caches may omit this argument
               and retain ``cache["x"]`` instead.
            route_weights: Optional selected router weights for this expert.
            return_route_dweights: Also return dL/d(selected router weight).

        Returns:
            dx, or ``(dx, droute_weights)`` when requested.
        """
        if x is None:
            x = cache.get("x")
        if x is None:
            raise ValueError("ExpertFFN.backward requires x or a legacy cache['x']")
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

        raw_dy_2d = dy.reshape(-1, dy.shape[-1])
        x_2d = x.reshape(-1, x.shape[-1])
        h_2d = h.reshape(-1, h.shape[-1])
        route_dweights = None

        if route_weights is not None:
            route_weights_2d = route_weights.reshape(-1, 1)
            # Parameter gradients must see the same weighted branch gradient as
            # before 0055A.  The expert-output cache is avoided by deriving the
            # router gradient from H and the down-projection input gradient:
            #   dL/dw = <raw_dy, H W_down>
            #          = <raw_dy W_down^T, H>.
            weighted_dy_2d = raw_dy_2d * route_weights_2d
        else:
            route_weights_2d = None
            weighted_dy_2d = raw_dy_2d

        with moe_detail_scope("moe.expert.down_wgrad"):
            self.W_down.grad += h_2d.T @ weighted_dy_2d

        with moe_detail_scope("moe.expert.down_dx"):
            if route_weights_2d is None:
                dh = weighted_dy_2d @ self.W_down.data.T
            else:
                # Compute the unweighted down-projection input gradient once.
                # It supplies dL/dw and is then weighted for the expert path.
                dh_raw = raw_dy_2d @ self.W_down.data.T
                if return_route_dweights:
                    with moe_detail_scope("moe.dispatch.router_wgrad"):
                        route_dweights = xp.sum(
                            dh_raw * h_2d, axis=-1, dtype=xp.float32
                        )
                dh = dh_raw * route_weights_2d

        if return_route_dweights and route_weights_2d is None:
            raise ValueError(
                "return_route_dweights=True requires route_weights"
            )

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

        if return_route_dweights:
            return dx, route_dweights
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

        0055A keeps only the expensive G/U expert activations.  Gathered expert
        inputs and expert outputs are intentionally not retained; both can be
        reconstructed from the original MoE input/routing plan during backward.
        """
        batch_size, seq_len, d_model = x.shape
        N = batch_size * seq_len
        x_flat = x.reshape(N, d_model)

        with moe_detail_scope("moe.dispatch.fwd_alloc"):
            y_flat = xp.zeros((N, d_model), dtype=x.dtype)

        streams = self._get_expert_streams()
        ready = self._record_ready_event()
        for stream in streams:
            stream.wait_event(ready)

        pending = []
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
                            expert_x, cache_input=False
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

        # Preserve deterministic serialized scatter semantics.  Do not put
        # expert_out in the backward cache; after the scatter it can die here.
        for exp_idx, token_indices, expert_weights, expert_out, expert_cache, _ in pending:
            if return_cache:
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
            "expert_caches": expert_caches,
            "routing_plan": routing_plan,
        }

    def _backward_concurrent(
        self, dy, cache, return_dweights, output_scale=None,
        return_output_scale_grad=False,
    ):
        """Sparse expert backward with independent expert math overlapped.

        ``output_scale`` gates the complete MoE output.  Ordinary expert and
        router gradients see the scaled branch, while raw route-weight
        derivatives are reused to accumulate dL/d(output_scale) without
        retaining expert outputs from forward.
        """
        x = cache["x"]
        routing_plan = cache.get("routing_plan")
        if routing_plan is None:
            return None

        batch_size, seq_len, d_model = x.shape
        N = batch_size * seq_len
        k = int(routing_plan.k)
        x_flat = x.reshape(N, d_model)
        dy_flat = dy.reshape(N, d_model)

        with moe_detail_scope("moe.dispatch.bwd_alloc"):
            dx_flat = xp.zeros((N, d_model), dtype=x.dtype)
            dweights_flat = (
                xp.zeros((N, k), dtype=xp.float32) if return_dweights else None
            )
            output_scale_grad = (
                xp.asarray(0.0, dtype=xp.float32)
                if return_output_scale_grad else None
            )

        expert_caches = cache.get("expert_caches", {})
        active_experts = [
            exp_idx
            for exp_idx in range(self.n_experts)
            if routing_plan.get_assignment_count(exp_idx) > 0
        ]
        if any(exp_idx not in expert_caches for exp_idx in active_experts):
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
            expert_cache = expert_caches[exp_idx]
            stream = streams[exp_idx % len(streams)]
            with stream:
                with moe_detail_scope("moe.dispatch.bwd_gather"):
                    expert_x = x_flat[token_indices]
                    raw_expert_dy = dy_flat[token_indices]

                with moe_detail_scope("moe.expert.backward_total"):
                    need_raw_dweights = return_dweights or return_output_scale_grad
                    effective_weights = (
                        expert_weights
                        if output_scale is None
                        else expert_weights * output_scale
                    )
                    if need_raw_dweights:
                        expert_dx, raw_dweight_values = self.experts[exp_idx].backward(
                            raw_expert_dy,
                            expert_cache,
                            x=expert_x,
                            route_weights=effective_weights,
                            return_route_dweights=True,
                        )
                        dweight_values = (
                            raw_dweight_values
                            if output_scale is None
                            else raw_dweight_values * output_scale
                        ) if return_dweights else None
                    else:
                        expert_dx = self.experts[exp_idx].backward(
                            raw_expert_dy,
                            expert_cache,
                            x=expert_x,
                            route_weights=effective_weights,
                        )
                        raw_dweight_values = None
                        dweight_values = None
                done = xp.cuda.Event()
                done.record()

            pending.append(
                (token_indices, slot_indices, expert_weights, dweight_values,
                 raw_dweight_values, expert_dx, done)
            )

        self._wait_for_expert_events([item[-1] for item in pending])

        for (
            token_indices, slot_indices, expert_weights, dweight_values,
            raw_dweight_values, expert_dx, _,
        ) in pending:
            if return_dweights:
                with moe_detail_scope("moe.dispatch.router_wgrad_scatter"):
                    dweights_flat[token_indices, slot_indices] = dweight_values
            if return_output_scale_grad:
                output_scale_grad += xp.sum(
                    expert_weights.astype(xp.float32, copy=False)
                    * raw_dweight_values.astype(xp.float32, copy=False),
                    dtype=xp.float32,
                )
            with moe_detail_scope("moe.dispatch.bwd_scatter"):
                dx_flat[token_indices] += expert_dx

        dx = dx_flat.reshape(batch_size, seq_len, d_model)
        if return_dweights and return_output_scale_grad:
            return dx, dweights_flat.reshape(batch_size, seq_len, k), output_scale_grad
        if return_dweights:
            return dx, dweights_flat.reshape(batch_size, seq_len, k)
        if return_output_scale_grad:
            return dx, output_scale_grad
        return dx

    def forward(
        self, x, weights, expert_indices, routing_plan: RoutingPlan = None,
        n_experts: int = None, return_cache=True
    ):
        """Forward pass through sparsely-routed experts.

        0055A's training cache stores only the original MoE input, routing plan,
        and each expert's expensive G/U activations.  It deliberately avoids
        retaining gathered expert_x, expert outputs, or the MoE output itself.
        """
        batch_size, seq_len, d_model = x.shape
        k = weights.shape[-1]
        N = batch_size * seq_len
        x_flat = x.reshape(N, d_model)

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

        with moe_detail_scope("moe.dispatch.fwd_alloc"):
            y_flat = xp.zeros((N, d_model), dtype=x.dtype)

        expert_caches = {}

        for exp_idx in range(self.n_experts):
            token_indices, slot_indices, expert_weights = (
                routing_plan.get_expert_assignments(exp_idx)
            )
            if routing_plan.get_assignment_count(exp_idx) == 0:
                continue

            with moe_detail_scope("moe.dispatch.fwd_gather"):
                expert_x = x_flat[token_indices]

            if return_cache:
                with moe_detail_scope("moe.expert.forward_total"):
                    expert_out, expert_cache = self.experts[exp_idx].forward(
                        expert_x, cache_input=False
                    )
                expert_caches[exp_idx] = expert_cache
            else:
                with moe_detail_scope("moe.expert.forward_total"):
                    expert_out = self.experts[exp_idx].forward(
                        expert_x, return_cache=False
                    )

            with moe_detail_scope("moe.dispatch.fwd_weight"):
                weighted_out = expert_weights[:, xp.newaxis] * expert_out
            with moe_detail_scope("moe.dispatch.fwd_scatter"):
                y_flat[token_indices] += weighted_out

        y = y_flat.reshape(batch_size, seq_len, d_model)
        if not return_cache:
            return y

        return y, {
            "x": x,
            "expert_caches": expert_caches,
            "routing_plan": routing_plan,
        }

    def backward(
        self, dy, cache, return_dweights=False, *, output_scale=None,
        return_output_scale_grad=False,
    ):
        """Backward pass through sparsely-routed experts.

        When ``output_scale`` is provided, the whole MoE branch is interpreted
        as ``alpha * MoE(x)``.  Expert-path and router gradients are scaled by
        alpha, while the raw per-route output derivatives are reused to obtain
        dL/d(alpha).
        """
        x = cache["x"]
        routing_plan = cache.get("routing_plan")
        batch_size, seq_len, d_model = x.shape
        N = batch_size * seq_len
        x_flat = x.reshape(N, d_model)
        dy_flat = dy.reshape(N, d_model)

        if routing_plan is not None:
            k = int(routing_plan.k)
            weights = None
            expert_indices = None
            weights_flat = None
            expert_indices_flat = None
        else:
            weights = cache["weights"]
            expert_indices = cache["expert_indices"]
            k = int(weights.shape[-1])
            weights_flat = weights.reshape(N, k)
            expert_indices_flat = expert_indices.reshape(N, k)

        with moe_detail_scope("moe.dispatch.bwd_alloc"):
            dx_flat = xp.zeros((N, d_model), dtype=x.dtype)
            dweights_flat = (
                xp.zeros((N, k), dtype=xp.float32) if return_dweights else None
            )
            output_scale_grad = (
                xp.asarray(0.0, dtype=xp.float32)
                if return_output_scale_grad else None
            )

        expert_caches = cache.get("expert_caches", {})

        if _concurrent_experts_enabled(x.dtype):
            concurrent = self._backward_concurrent(
                dy,
                cache,
                return_dweights=return_dweights,
                output_scale=output_scale,
                return_output_scale_grad=return_output_scale_grad,
            )
            if concurrent is not None:
                return concurrent

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

            with moe_detail_scope("moe.dispatch.bwd_gather"):
                expert_x = x_flat[token_indices]
                raw_expert_dy = dy_flat[token_indices]

            expert_cache = expert_caches.get(exp_idx)
            if expert_cache is None:
                _, expert_cache = self.experts[exp_idx].forward(expert_x)

            with moe_detail_scope("moe.expert.backward_total"):
                need_raw_dweights = return_dweights or return_output_scale_grad
                effective_weights = (
                    expert_weights
                    if output_scale is None
                    else expert_weights * output_scale
                )
                if need_raw_dweights:
                    expert_dx, raw_dweight_values = self.experts[exp_idx].backward(
                        raw_expert_dy,
                        expert_cache,
                        x=expert_x,
                        route_weights=effective_weights,
                        return_route_dweights=True,
                    )
                    if return_dweights:
                        dweight_values = (
                            raw_dweight_values
                            if output_scale is None
                            else raw_dweight_values * output_scale
                        )
                        dweights_flat[token_indices, slot_indices] = dweight_values
                    if return_output_scale_grad:
                        output_scale_grad += xp.sum(
                            expert_weights.astype(xp.float32, copy=False)
                            * raw_dweight_values.astype(xp.float32, copy=False),
                            dtype=xp.float32,
                        )
                else:
                    expert_dx = self.experts[exp_idx].backward(
                        raw_expert_dy,
                        expert_cache,
                        x=expert_x,
                        route_weights=effective_weights,
                    )

            with moe_detail_scope("moe.dispatch.bwd_scatter"):
                dx_flat[token_indices] += expert_dx

        dx = dx_flat.reshape(batch_size, seq_len, d_model)
        if return_dweights and return_output_scale_grad:
            return dx, dweights_flat.reshape(batch_size, seq_len, k), output_scale_grad
        if return_dweights:
            return dx, dweights_flat.reshape(batch_size, seq_len, k)
        if return_output_scale_grad:
            return dx, output_scale_grad
        return dx
