"""0076A: Hybrid Moonlight-scaled Muon/AdamW, with optional CPU offload.

Muon is applied only to named hidden-layer matrices. Every other tensor uses
unchanged AdamW. No persistent second-moment tensor exists for Muon matrices.
"""

import math
import os

import numpy as np

from ..backend import xp, BACKEND_NAME, is_low_precision_dtype
from .adamw import AdamW, _offload_mode


def muon_eligible(parameter):
    """Conservative, name-based eligibility for the current decoder architecture."""
    name = parameter.name or ""
    if len(parameter.data.shape) != 2 or min(parameter.data.shape) < 2:
        return False
    if name.startswith("block_proposal."):
        # The learned horizon embedding represents token position, not a
        # hidden-to-hidden transform. Only the two proposal projections qualify.
        return name in {"block_proposal.horizon_context", "block_proposal.horizon_back"}
    if not (name.startswith("blocks.") or name.startswith("block_decoder.")):
        return False
    # Router tensors are intentionally left to AdamW in the first comparison.
    if any(token in name.lower() for token in (
        "router", "norm", "embedding", "output_proj", "memory", "gate_param"
    )):
        return False
    # Expert W_gate is a SwiGLU projection, not a router; allow it.
    return bool(name.endswith((
        ".W", ".Wq", ".Wk", ".Wv", ".Wo", ".W_q", ".W_k", ".W_v", ".W_o",
        ".W_gate", ".W_up", ".W_down",
    )))


def newton_schulz_orthogonalize(gradient, *, steps=5, dtype=None):
    """Quintic Newton–Schulz approx. to the polar factor, without in-place input edits.

    Uses BF16/FP16 GEMMs on compatible GPU tensors and float32 on CPU.
    Zero gradients produce zero output. A full matrix is needed for orthogonalization.
    """
    if gradient.ndim != 2 or min(gradient.shape) < 2:
        raise ValueError("Muon requires matrices with both dimensions >= 2")
    if steps < 1:
        raise ValueError("Muon Newton-Schulz iterations must be positive")
    compute_dtype = dtype if dtype is not None else "float32"
    g = gradient.astype("float32", copy=False)
    norm = xp.sqrt(xp.sum(g * g))
    x = (g / xp.maximum(norm, xp.asarray(1e-7, dtype="float32"))).astype(compute_dtype)
    transposed = x.shape[0] > x.shape[1]
    if transposed:
        x = x.T
    a, b, c = 3.4445, -4.7750, 2.0315
    for _ in range(steps):
        A = x @ x.T
        B = (b * A + c * (A @ A)).astype(compute_dtype)
        x = (a * x + B @ x).astype(compute_dtype)
    if transposed:
        x = x.T
    # The input-norm guard makes zero -> zero for every supported backend.
    return x.astype("float32")


class _MuonMatrices:
    """One FP32 momentum and one FP32 master per Muon parameter.

    In full offload mode, both live in pinned host RAM and matrix-sized reusable
    GPU staging buffers are used for a whole-matrix orthogonalization. Moments
    mode keeps the FP32 masters on device while offloading only momentum.
    """

    def __init__(self, parameters, *, momentum, ns_steps, weight_decay,
                 matmul_precision="bf16", numerical_debug=False):
        self.parameters = list(parameters)
        self.momentum = float(momentum)
        self.ns_steps = int(ns_steps)
        self.weight_decay = float(weight_decay)
        self.matmul_precision = str(matmul_precision)
        self.numerical_debug = bool(numerical_debug)
        self.offload_mode = _offload_mode() if BACKEND_NAME == "cupy" else "none"
        self.moments_offloaded = self.offload_mode in {"moments", "full"}
        self.master_weights_offloaded = self.offload_mode == "full"
        self._pinned_handles = []
        self.master_weights = []
        self.m = []
        self.offload_host_bytes = 0
        self.offload_stage_bytes = 0
        self.step_index = 0
        self._m_stage = None
        self._w_stage = None
        self._add_state(self.parameters)

    def _pinned(self, p, *, zeros):
        import cupy
        nbytes = int(p.data.size) * 4
        mem = cupy.cuda.alloc_pinned_memory(nbytes)
        self._pinned_handles.append(mem)
        # CuPy pinned-memory pools may return more bytes than requested.
        # Map only the parameter payload, not allocator padding.
        arr = np.frombuffer(mem, dtype=np.float32, count=int(p.data.size)).reshape(p.data.shape)
        if zeros:
            arr.fill(0.0)
        self.offload_host_bytes += nbytes
        return arr

    def _add_state(self, parameters):
        parameters = list(parameters)
        for p in parameters:
            if self.moments_offloaded:
                self.m.append(self._pinned(p, zeros=True))
            else:
                self.m.append(xp.zeros_like(p.grad, dtype="float32"))
            if self.master_weights_offloaded:
                self.master_weights.append(self._pinned(p, zeros=False))
            else:
                self.master_weights.append(p.data.astype("float32", copy=True))
        if self.master_weights_offloaded and parameters:
            stream = xp.cuda.get_current_stream()
            for p, dst in zip(parameters, self.master_weights[-len(parameters):]):
                # BF16 data transferred to a bounded FP32 staging view.
                flat = p.data.reshape(-1)
                out = dst.reshape(-1)
                chunk = max(1, int(os.environ.get("MINI_LLM_OPTIMIZER_OFFLOAD_CHUNK_MB", "16").split(".")[0])) * (1024**2 // 4)
                for start in range(0, int(p.data.size), chunk):
                    stop = min(int(p.data.size), start + chunk)
                    scratch = flat[start:stop].astype("float32")
                    scratch.get(out=out[start:stop], stream=stream, blocking=False)
                    stream.synchronize()
        if self.moments_offloaded:
            self._allocate_staging()

    def _allocate_staging(self):
        max_size = max((int(p.data.size) for p in self.parameters), default=1)
        if self._m_stage is not None and self._m_stage.size >= max_size:
            return
        self._m_stage = xp.empty((max_size,), dtype="float32")
        self._w_stage = (
            xp.empty((max_size,), dtype="float32")
            if self.master_weights_offloaded else None
        )
        self.offload_stage_bytes = max_size * 4 * (2 if self.master_weights_offloaded else 1)

    def add_parameters(self, parameters):
        new_params = list(parameters)
        self.parameters.extend(new_params)
        self._add_state(new_params)
        return len(new_params)

    def restore_master_weight(self, index, value):
        dst = self.master_weights[index]
        if tuple(dst.shape) != tuple(value.shape):
            return False
        if self.master_weights_offloaded:
            np.copyto(dst, np.asarray(value.get() if hasattr(value, "get") else value, dtype=np.float32))
        else:
            dst[...] = xp.asarray(value, dtype="float32")
        return True

    def restore_momentum(self, index, value):
        dst = self.m[index]
        if tuple(dst.shape) != tuple(value.shape):
            return False
        if self.moments_offloaded:
            np.copyto(dst, np.asarray(value.get() if hasattr(value, "get") else value, dtype=np.float32))
        else:
            dst[...] = xp.asarray(value, dtype="float32")
        return True

    def step(self, eta):
        self.step_index += 1
        for i, p in enumerate(self.parameters):
            shape = p.data.shape
            g = p.grad.astype("float32", copy=False)
            if self.moments_offloaded:
                m = self._m_stage[:int(p.data.size)].reshape(shape)
                m.set(self.m[i])
            else:
                m = self.m[i]
            m *= self.momentum
            m += (1.0 - self.momentum) * g
            nesterov = self.momentum * m + (1.0 - self.momentum) * g
            ns_dtype = "float32"
            if (BACKEND_NAME == "cupy" and self.matmul_precision == "bf16"
                    and is_low_precision_dtype(p.data.dtype)):
                ns_dtype = p.data.dtype
            direction = newton_schulz_orthogonalize(
                nesterov, steps=self.ns_steps, dtype=ns_dtype,
            )
            direction *= 0.2 * math.sqrt(max(int(shape[0]), int(shape[1])))
            if self.master_weights_offloaded:
                w = self._w_stage[:int(p.data.size)].reshape(shape)
                w.set(self.master_weights[i])
            else:
                w = self.master_weights[i]
            if p.decay and self.weight_decay:
                w *= 1.0 - float(eta) * self.weight_decay
            w -= float(eta) * direction
            p.data[...] = w
            if self.moments_offloaded:
                stream = xp.cuda.get_current_stream()
                m.get(out=self.m[i], stream=stream, blocking=False)
                if self.master_weights_offloaded:
                    w.get(out=self.master_weights[i], stream=stream, blocking=False)
                stream.synchronize()
            if self.numerical_debug:
                result = self.master_weights[i]
                if not np.all(np.isfinite(result.get() if hasattr(result, "get") else result)):
                    raise FloatingPointError(f"Nonfinite Muon master weight: {p.name}")


class HybridMuonAdamW:
    """Trainer-facing optimizer adapter, preserving AdamW-only default elsewhere."""

    kind = "hybrid-muon"

    def __init__(self, parameters, *, lr, beta1, beta2, eps, weight_decay,
                 muon_momentum=0.95, muon_ns_steps=5, muon_matmul_precision="bf16",
                 numerical_debug=False):
        self.parameters = list(parameters)
        if not 0 <= muon_momentum < 1 or not math.isfinite(muon_momentum):
            raise ValueError("Muon momentum must be finite and in [0,1)")
        if not 1 <= muon_ns_steps <= 20:
            raise ValueError("Muon Newton-Schulz steps must lie in [1,20]")
        if muon_matmul_precision not in {"bf16", "fp32"}:
            raise ValueError("Muon matmul precision must be bf16 or fp32")
        self.muon_names = {p.name for p in self.parameters if muon_eligible(p)}
        self.adam = AdamW(
            [p for p in self.parameters if p.name not in self.muon_names],
            lr=lr, beta1=beta1, beta2=beta2, eps=eps,
            weight_decay=weight_decay, numerical_debug=numerical_debug,
        )
        self.muon = _MuonMatrices(
            [p for p in self.parameters if p.name in self.muon_names],
            momentum=muon_momentum, ns_steps=muon_ns_steps,
            weight_decay=weight_decay, matmul_precision=muon_matmul_precision,
            numerical_debug=numerical_debug,
        )
        self.beta1, self.beta2, self.eps = self.adam.beta1, self.adam.beta2, self.adam.eps
        self.weight_decay = float(weight_decay)
        self.momentum = self.muon.momentum
        self.ns_steps = self.muon.ns_steps
        self.matmul_precision = self.muon.matmul_precision
        self._step_index = 0
        self.lr = lr
        self.moments_offloaded = self.adam.moments_offloaded
        self.master_weights_offloaded = self.adam.master_weights_offloaded
        self.offload_host_bytes = self.adam.offload_host_bytes + self.muon.offload_host_bytes
        self.offload_stage_bytes = self.adam.offload_stage_bytes + self.muon.offload_stage_bytes
        self._rebuild_indices()

    def _rebuild_indices(self):
        self._locations = {}
        for group, opt in (("adam", self.adam), ("muon", self.muon)):
            for idx, p in enumerate(opt.parameters):
                self._locations[p.name] = (group, idx)

    @property
    def step_index(self):
        return self._step_index

    @step_index.setter
    def step_index(self, value):
        self._step_index = int(value)
        self.adam.step_index = int(value)
        self.muon.step_index = int(value)

    def step(self, lr=None):
        if lr is not None:
            self.lr = float(lr)
        self.adam.step(lr=self.lr)
        self.muon.step(eta=self.lr)
        self._step_index += 1

    def zero_grad(self):
        for p in self.parameters:
            p.zero_grad()

    def add_parameters(self, parameters):
        existing = {p.name for p in self.parameters}
        new = [p for p in parameters if p.name not in existing]
        if not new:
            return 0
        new_adam = [p for p in new if not muon_eligible(p)]
        new_muon = [p for p in new if muon_eligible(p)]
        self.adam.add_parameters(new_adam)
        self.muon.add_parameters(new_muon)
        self.muon_names.update(p.name for p in new_muon)
        self.parameters.extend(new)
        self._rebuild_indices()
        self.offload_host_bytes = self.adam.offload_host_bytes + self.muon.offload_host_bytes
        self.offload_stage_bytes = self.adam.offload_stage_bytes + self.muon.offload_stage_bytes
        return len(new)

    def parameter_birth_step_state(self):
        return self.adam.parameter_birth_step_state()

    def restore_parameter_birth_steps(self, state):
        return self.adam.restore_parameter_birth_steps(state)

    def checkpoint_state(self):
        return {
            "step": self.step_index,
            "master_weights": {
                p.name: opt.master_weights[i]
                for opt in (self.adam, self.muon)
                for i, p in enumerate(opt.parameters)
            },
            "m": {p.name: self.adam.m[i] for i, p in enumerate(self.adam.parameters)},
            "v": {p.name: self.adam.v[i] for i, p in enumerate(self.adam.parameters)},
            "muon_m": {p.name: self.muon.m[i] for i, p in enumerate(self.muon.parameters)},
        }

    def restore_checkpoint(self, state, *, from_hybrid):
        if from_hybrid:
            expected = {p.name for p in self.muon.parameters}
            saved = set(state.get("muon_m", {}))
            if expected != saved:
                raise ValueError("Muon group assignment differs from saved checkpoint")
        master = state.get("master_weights", {})
        count_master = 0
        for group in (self.adam, self.muon):
            for i, p in enumerate(group.parameters):
                value = master.pop(p.name, None)
                if value is not None:
                    if not group.restore_master_weight(i, value):
                        raise ValueError("master weight shape mismatch: " + p.name)
                    count_master += 1
        if count_master != len(self.parameters):
            raise ValueError("hybrid resume requires complete FP32 master weight checkpoint")
        m = state.get("m", {})
        v = state.get("v", {})
        restored_adam = 0
        for i, p in enumerate(self.adam.parameters):
            if p.name in m and p.name in v:
                if not self.adam.restore_moments(i, m.pop(p.name), v.pop(p.name)):
                    raise ValueError("AdamW moment shape mismatch: " + p.name)
                restored_adam += 1
        if restored_adam != len(self.adam.parameters):
            raise ValueError("hybrid resume is missing AdamW states")
        restored_muon = 0
        if from_hybrid:
            mm = state.get("muon_m", {})
            for i, p in enumerate(self.muon.parameters):
                if not self.muon.restore_momentum(i, mm.pop(p.name)):
                    raise ValueError("Muon momentum shape mismatch: " + p.name)
                restored_muon += 1
        self.step_index = int(state.get("step", 0))
        return count_master, restored_adam, restored_muon
