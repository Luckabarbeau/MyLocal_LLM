"""AdamW optimizer with mixed precision and optional CPU optimizer-state offload."""

import os

try:
    import numpy as np
except ImportError:
    np = None

from ..backend import xp, BACKEND_NAME


def _offload_mode():
    raw = os.environ.get("MINI_LLM_OPTIMIZER_OFFLOAD", "none").strip().lower()
    aliases = {"0": "none", "off": "none", "false": "none", "": "none"}
    raw = aliases.get(raw, raw)
    if raw not in {"none", "moments", "full"}:
        raise ValueError(
            "MINI_LLM_OPTIMIZER_OFFLOAD must be 'none', 'moments', or 'full'"
        )
    return raw


class AdamW:
    """
    AdamW optimizer with explicit mixed precision support.

    The normal path keeps FP32 master weights and Adam first/second moments on
    the active backend.  0055B adds ``MINI_LLM_OPTIMIZER_OFFLOAD=moments`` for
    CuPy.  0055C extends this with ``MINI_LLM_OPTIMIZER_OFFLOAD=full`` so the
    FP32 master weights also live in CUDA-pinned system RAM.  A bounded reusable
    GPU staging window performs the exact same FP32 Adam arithmetic while
    accumulated gradients and BF16 model weights remain on GPU.
    """

    def __init__(
        self, parameters, lr=3e-4, beta1=0.9, beta2=0.95,
        eps=1e-8, weight_decay=0.1, numerical_debug=False
    ):
        self.parameters = list(parameters)
        self.lr = float(lr)
        self.beta1 = float(beta1)
        self.beta2 = float(beta2)
        self.eps = float(eps)
        self.weight_decay = float(weight_decay)
        self.numerical_debug = bool(numerical_debug)
        self.step_index = 0
        # 0064B: Adam age is tracked per parameter so a block activated after
        # many global updates starts with correct step-1 bias correction instead
        # of inheriting the age of the already-trained parameters.  Initial
        # parameters preserve historical behavior with birth step zero.
        self.parameter_birth_steps = [0 for _ in self.parameters]

        self.offload_mode = _offload_mode()
        self.moments_offloaded = (
            self.offload_mode in {"moments", "full"} and BACKEND_NAME == "cupy"
        )
        self.master_weights_offloaded = (
            self.offload_mode == "full" and BACKEND_NAME == "cupy"
        )
        if self.offload_mode in {"moments", "full"} and BACKEND_NAME != "cupy":
            # NumPy already stores optimizer state in system RAM, so offload is
            # semantically a no-op on the reference backend.
            self.offload_mode = "none"

        # 0055C optionally stores FP32 master weights in pinned host RAM.
        self.master_weights = []
        if not self.master_weights_offloaded:
            self.master_weights = [
                p.data.astype("float32", copy=True) for p in self.parameters
            ]

        self.m = []
        self.v = []
        self._m_host_storage = None
        self._v_host_storage = None
        self._master_host_storage = None
        self._m_host_memory = None
        self._v_host_memory = None
        self._master_host_memory = None
        # Dynamically added progressive-depth parameters use additional pinned
        # slabs rather than reallocating/copying the established optimizer
        # state.  Keeping the memory handles alive preserves the NumPy views.
        self._extra_m_host_memory = []
        self._extra_v_host_memory = []
        self._extra_master_host_memory = []
        self._m_stage = None
        self._v_stage = None
        self._w_stage = None
        self._update_stage = None
        self.offload_host_bytes = 0
        self.offload_stage_bytes = 0

        if self.moments_offloaded:
            if np is None:
                raise RuntimeError("NumPy is required for optimizer moment offload")
            self._init_pinned_moments()
            self._init_offload_staging()
            if self.master_weights_offloaded:
                self._init_pinned_master_weights()
        else:
            for p in self.parameters:
                self.m.append(xp.zeros_like(p.data, dtype="float32"))
                self.v.append(xp.zeros_like(p.data, dtype="float32"))

    def _init_pinned_moments(self):
        """Allocate two contiguous CUDA-pinned FP32 host slabs for m and v."""
        import cupy

        total_numel = sum(int(p.data.size) for p in self.parameters)
        nbytes = total_numel * np.dtype(np.float32).itemsize
        self._m_host_memory = cupy.cuda.alloc_pinned_memory(nbytes)
        self._v_host_memory = cupy.cuda.alloc_pinned_memory(nbytes)
        self._m_host_storage = np.frombuffer(
            self._m_host_memory, dtype=np.float32, count=total_numel
        )
        self._v_host_storage = np.frombuffer(
            self._v_host_memory, dtype=np.float32, count=total_numel
        )
        self._m_host_storage.fill(0.0)
        self._v_host_storage.fill(0.0)

        offset = 0
        for p in self.parameters:
            n = int(p.data.size)
            shape = tuple(int(x) for x in p.data.shape)
            self.m.append(self._m_host_storage[offset:offset + n].reshape(shape))
            self.v.append(self._v_host_storage[offset:offset + n].reshape(shape))
            offset += n
        self.offload_host_bytes = 2 * nbytes

    def _init_offload_staging(self):
        """Create bounded reusable GPU staging buffers for streamed Adam."""
        raw_mb = os.environ.get("MINI_LLM_OPTIMIZER_OFFLOAD_CHUNK_MB", "32")
        try:
            chunk_mb = max(1.0, float(raw_mb))
        except ValueError as exc:
            raise ValueError(
                "MINI_LLM_OPTIMIZER_OFFLOAD_CHUNK_MB must be numeric"
            ) from exc
        bytes_per = int(chunk_mb * 1024 * 1024)
        elems = max(1, bytes_per // np.dtype(np.float32).itemsize)
        max_param = max((int(p.data.size) for p in self.parameters), default=1)
        elems = min(elems, max_param)
        self._offload_chunk_elems = elems
        self._m_stage = xp.empty((elems,), dtype="float32")
        self._v_stage = xp.empty((elems,), dtype="float32")
        if self.master_weights_offloaded:
            self._w_stage = xp.empty((elems,), dtype="float32")
        # Reused first for scaled gradient products and then for sqrt(v)/update.
        self._update_stage = xp.empty((elems,), dtype="float32")
        n_stages = 4 if self.master_weights_offloaded else 3
        self.offload_stage_bytes = n_stages * elems * np.dtype(np.float32).itemsize


    def _init_pinned_master_weights(self):
        """0055C: allocate and initialize contiguous pinned FP32 masters."""
        import cupy

        total_numel = sum(int(p.data.size) for p in self.parameters)
        nbytes = total_numel * np.dtype(np.float32).itemsize
        self._master_host_memory = cupy.cuda.alloc_pinned_memory(nbytes)
        self._master_host_storage = np.frombuffer(
            self._master_host_memory, dtype=np.float32, count=total_numel
        )

        offset = 0
        for p in self.parameters:
            n = int(p.data.size)
            shape = tuple(int(x) for x in p.data.shape)
            self.master_weights.append(
                self._master_host_storage[offset:offset + n].reshape(shape)
            )
            offset += n

        # Initialize from the BF16/FP16/FP32 model parameters through the same
        # bounded FP32 staging window used during optimizer steps.  This avoids
        # ever materializing a full FP32 parameter copy on the GPU.
        stream = xp.cuda.get_current_stream()
        for i, p in enumerate(self.parameters):
            src = p.data.reshape(-1)
            dst = self.master_weights[i].reshape(-1)
            size = int(p.data.size)
            for start in range(0, size, self._offload_chunk_elems):
                stop = min(size, start + self._offload_chunk_elems)
                n = stop - start
                w_dev = self._w_stage[:n]
                w_dev[...] = src[start:stop]
                w_dev.get(out=dst[start:stop], stream=stream, blocking=False)
                stream.synchronize()

        self.offload_host_bytes += nbytes

    def restore_master_weight(self, index, value):
        """Restore one exact FP32 master weight without a persistent duplicate."""
        if self.master_weights_offloaded:
            host = value.get() if hasattr(value, "get") else np.asarray(value)
            if host.shape != self.master_weights[index].shape:
                return False
            np.copyto(
                self.master_weights[index], np.asarray(host, dtype=np.float32)
            )
            return True

        arr = self.master_weights[index]
        if tuple(value.shape) != tuple(arr.shape):
            return False
        if BACKEND_NAME == "cupy" and not hasattr(value, "get"):
            arr.set(np.asarray(value, dtype=np.float32))
        else:
            arr[...] = xp.asarray(value, dtype="float32")
        return True

    def restore_moments(self, index, m_value, v_value):
        """Restore one pair of checkpoint moments without unwanted GPU copies."""
        if self.moments_offloaded:
            m_host = m_value.get() if hasattr(m_value, "get") else np.asarray(m_value)
            v_host = v_value.get() if hasattr(v_value, "get") else np.asarray(v_value)
            if m_host.shape != self.m[index].shape or v_host.shape != self.v[index].shape:
                return False
            np.copyto(self.m[index], np.asarray(m_host, dtype=np.float32))
            np.copyto(self.v[index], np.asarray(v_host, dtype=np.float32))
            return True

        m_arr = xp.asarray(m_value)
        v_arr = xp.asarray(v_value)
        if m_arr.shape != self.m[index].shape or v_arr.shape != self.v[index].shape:
            return False
        self.m[index][...] = m_arr
        self.v[index][...] = v_arr
        return True

    def _allocate_pinned_views(self, parameters, *, zero=False):
        """Allocate one pinned FP32 slab and return per-parameter views."""
        import cupy

        params = list(parameters)
        total_numel = sum(int(p.data.size) for p in params)
        if total_numel <= 0:
            return None, []
        nbytes = total_numel * np.dtype(np.float32).itemsize
        memory = cupy.cuda.alloc_pinned_memory(nbytes)
        storage = np.frombuffer(memory, dtype=np.float32, count=total_numel)
        if zero:
            storage.fill(0.0)
        views = []
        offset = 0
        for p in params:
            n = int(p.data.size)
            shape = tuple(int(x) for x in p.data.shape)
            views.append(storage[offset:offset + n].reshape(shape))
            offset += n
        return memory, views

    def _initialize_offloaded_masters(self, parameters, master_views):
        """Copy newly activated model weights into pinned FP32 masters."""
        stream = xp.cuda.get_current_stream()
        for p, dst_view in zip(parameters, master_views):
            src = p.data.reshape(-1)
            dst = dst_view.reshape(-1)
            size = int(p.data.size)
            for start in range(0, size, self._offload_chunk_elems):
                stop = min(size, start + self._offload_chunk_elems)
                n = stop - start
                w_dev = self._w_stage[:n]
                w_dev[...] = src[start:stop]
                w_dev.get(out=dst[start:stop], stream=stream, blocking=False)
                stream.synchronize()

    def add_parameters(self, parameters):
        """Add newly activated parameters without disturbing existing Adam state.

        New parameters receive fresh FP32 masters, zero first/second moments and
        a birth step equal to the number of already-completed optimizer updates.
        The next optimizer update therefore uses local Adam step 1 for them.
        """
        existing_ids = {id(p) for p in self.parameters}
        existing_names = {p.name for p in self.parameters}
        new_params = []
        for p in parameters:
            if id(p) in existing_ids:
                continue
            if p.name in existing_names:
                raise ValueError(f"duplicate optimizer parameter name: {p.name}")
            existing_ids.add(id(p))
            existing_names.add(p.name)
            new_params.append(p)
        if not new_params:
            return 0

        if self.master_weights_offloaded:
            master_memory, master_views = self._allocate_pinned_views(new_params)
            self._extra_master_host_memory.append(master_memory)
            self.master_weights.extend(master_views)
            self._initialize_offloaded_masters(new_params, master_views)
            self.offload_host_bytes += sum(
                int(p.data.size) for p in new_params
            ) * np.dtype(np.float32).itemsize
        else:
            self.master_weights.extend(
                p.data.astype("float32", copy=True) for p in new_params
            )

        if self.moments_offloaded:
            m_memory, m_views = self._allocate_pinned_views(new_params, zero=True)
            v_memory, v_views = self._allocate_pinned_views(new_params, zero=True)
            self._extra_m_host_memory.append(m_memory)
            self._extra_v_host_memory.append(v_memory)
            self.m.extend(m_views)
            self.v.extend(v_views)
            state_bytes = sum(int(p.data.size) for p in new_params) * np.dtype(np.float32).itemsize
            self.offload_host_bytes += 2 * state_bytes
        else:
            self.m.extend(xp.zeros_like(p.data, dtype="float32") for p in new_params)
            self.v.extend(xp.zeros_like(p.data, dtype="float32") for p in new_params)

        self.parameters.extend(new_params)
        self.parameter_birth_steps.extend(
            [int(self.step_index)] * len(new_params)
        )
        return len(new_params)

    def parameter_birth_step_state(self):
        """Return JSON-safe per-parameter Adam birth steps keyed by name."""
        return {
            p.name: int(self.parameter_birth_steps[i])
            for i, p in enumerate(self.parameters)
        }

    def restore_parameter_birth_steps(self, state):
        """Restore per-parameter Adam ages from checkpoint training state."""
        if not state:
            return 0
        restored = 0
        for i, p in enumerate(self.parameters):
            if p.name not in state:
                continue
            birth = int(state[p.name])
            if birth < 0 or birth > int(self.step_index):
                raise ValueError(
                    f"invalid optimizer birth step {birth} for {p.name} "
                    f"at optimizer step {self.step_index}"
                )
            self.parameter_birth_steps[i] = birth
            restored += 1
        return restored

    def _bias_correction(self, index, eta):
        local_step = int(self.step_index) - int(self.parameter_birth_steps[index])
        if local_step <= 0:
            raise RuntimeError("optimizer parameter has not reached local step 1")
        c1 = 1.0 - self.beta1 ** local_step
        c2 = 1.0 - self.beta2 ** local_step
        sqrt_c2 = c2 ** 0.5
        return eta * sqrt_c2 / c1, self.eps * sqrt_c2

    def _debug_check_gradient(self, g, i, p):
        if self.numerical_debug and np is not None:
            g_cpu = g.get() if hasattr(g, "get") else g
            if not np.all(np.isfinite(g_cpu)):
                raise FloatingPointError(
                    f"Nonfinite gradient in {p.name or f'parameter_{i}'} "
                    f"at step {self.step_index}"
                )

    def _debug_check_master(self, i, p):
        if self.numerical_debug and np is not None:
            w = self.master_weights[i]
            w_cpu = w.get() if hasattr(w, "get") else w
            if not np.all(np.isfinite(w_cpu)):
                raise FloatingPointError(
                    f"Nonfinite master weight in {p.name or f'parameter_{i}'} "
                    f"at step {self.step_index}"
                )

    def _step_device_moments(self, eta):
        """Established all-device Adam path with per-parameter Adam age."""
        for i, p in enumerate(self.parameters):
            step_size, eps_scaled = self._bias_correction(i, eta)
            g = p.grad.astype("float32", copy=False)
            self._debug_check_gradient(g, i, p)

            self.m[i] *= self.beta1
            self.m[i] += (1.0 - self.beta1) * g
            self.v[i] *= self.beta2
            self.v[i] += (1.0 - self.beta2) * (g * g)

            if self.numerical_debug and np is not None:
                m_cpu = self.m[i].get() if hasattr(self.m[i], "get") else self.m[i]
                v_cpu = self.v[i].get() if hasattr(self.v[i], "get") else self.v[i]
                if not (np.all(np.isfinite(m_cpu)) and np.all(np.isfinite(v_cpu))):
                    raise FloatingPointError(
                        f"Nonfinite moment in {p.name or f'parameter_{i}'} "
                        f"at step {self.step_index}"
                    )

            # ``xp.sqrt`` of a NumPy 0-D array returns a scalar, which cannot
            # be used as an ``out`` buffer by the following divide.  Learned
            # progressive residual gates are scalar Parameters, so keep an
            # explicit array buffer here for both scalar and tensor parameters.
            update = xp.empty_like(self.v[i])
            xp.sqrt(self.v[i], out=update)
            update += eps_scaled
            xp.divide(self.m[i], update, out=update)
            update *= step_size

            if p.decay and self.weight_decay != 0.0:
                self.master_weights[i] *= (1.0 - eta * self.weight_decay)
            self.master_weights[i] -= update
            self._debug_check_master(i, p)
            p.data[...] = self.master_weights[i].astype(p.data.dtype)

    def _step_offloaded_moments(self, eta):
        """0055B/0055C streamed Adam with pinned host optimizer state."""
        stream = xp.cuda.get_current_stream()
        one_minus_b1 = 1.0 - self.beta1
        one_minus_b2 = 1.0 - self.beta2
        decay_factor = 1.0 - eta * self.weight_decay

        for i, p in enumerate(self.parameters):
            step_size, eps_scaled = self._bias_correction(i, eta)
            g_flat = p.grad.reshape(-1)
            p_flat = p.data.reshape(-1)
            m_host = self.m[i].reshape(-1)
            v_host = self.v[i].reshape(-1)
            if self.master_weights_offloaded:
                w_host = self.master_weights[i].reshape(-1)
                w_flat = None
            else:
                w_host = None
                w_flat = self.master_weights[i].reshape(-1)
            self._debug_check_gradient(p.grad, i, p)

            size = int(p.data.size)
            for start in range(0, size, self._offload_chunk_elems):
                stop = min(size, start + self._offload_chunk_elems)
                n = stop - start
                m_dev = self._m_stage[:n]
                v_dev = self._v_stage[:n]
                scratch = self._update_stage[:n]
                mh = m_host[start:stop]
                vh = v_host[start:stop]

                # Pinned H2D copies enqueue on the same stream as the Adam math.
                m_dev.set(mh, stream=stream)
                v_dev.set(vh, stream=stream)
                if self.master_weights_offloaded:
                    wh = w_host[start:stop]
                    w = self._w_stage[:n]
                    w.set(wh, stream=stream)
                else:
                    wh = None
                    w = w_flat[start:stop]
                g = g_flat[start:stop]

                # m = beta1*m + (1-beta1)*g, reusing scratch for the scaled g.
                xp.multiply(g, one_minus_b1, out=scratch)
                m_dev *= self.beta1
                m_dev += scratch

                # v = beta2*v + (1-beta2)*g^2, same scratch reused.
                xp.multiply(g, g, out=scratch)
                scratch *= one_minus_b2
                v_dev *= self.beta2
                v_dev += scratch

                # Adam update in the established algebraically folded form.
                xp.sqrt(v_dev, out=scratch)
                scratch += eps_scaled
                xp.divide(m_dev, scratch, out=scratch)
                scratch *= step_size

                if p.decay and self.weight_decay != 0.0:
                    w *= decay_factor
                w -= scratch
                # Direct assignment casts FP32 master -> model dtype without a
                # persistent full-parameter conversion buffer.
                p_flat[start:stop] = w

                # Queue updated host state and synchronize once before this
                # bounded staging window is reused.
                m_dev.get(out=mh, stream=stream, blocking=False)
                v_dev.get(out=vh, stream=stream, blocking=False)
                if self.master_weights_offloaded:
                    w.get(out=wh, stream=stream, blocking=False)
                stream.synchronize()

            if self.numerical_debug and np is not None:
                if not (
                    np.all(np.isfinite(m_host)) and np.all(np.isfinite(v_host))
                ):
                    raise FloatingPointError(
                        f"Nonfinite moment in {p.name or f'parameter_{i}'} "
                        f"at step {self.step_index}"
                    )
            self._debug_check_master(i, p)

    def step(self, lr=None):
        """Perform one optimizer step with mixed precision."""
        self.step_index += 1
        eta = self.lr if lr is None else float(lr)

        if self.moments_offloaded:
            self._step_offloaded_moments(eta)
        else:
            self._step_device_moments(eta)

    def zero_grad(self):
        """Zero out all gradients."""
        for p in self.parameters:
            p.zero_grad()
