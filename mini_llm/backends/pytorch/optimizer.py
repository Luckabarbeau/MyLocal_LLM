"""Optional PyTorch implementation of the native hybrid Muon/AdamW optimizer.

PyTorch is intentionally imported only by this opt-in backend module.  The
optimizer consumes named parameters so Muon assignment and checkpoint state use
the same canonical names as the native implementation.
"""

from __future__ import annotations

import math
import re
from collections.abc import Callable, Iterable, Mapping
from typing import Any

import torch


_MUON_SUFFIXES = (
    ".W", ".Wq", ".Wk", ".Wv", ".Wo", ".W_q", ".W_k", ".W_v", ".W_o",
    ".W_gate", ".W_up", ".W_down",
)
_MUON_EXCLUDED_TOKENS = (
    "router", "norm", "embedding", "output_proj", "memory", "gate_param",
)


def muon_eligible(name: str, parameter: torch.Tensor) -> bool:
    """Return the native optimizer's conservative name/shape Muon decision."""
    if parameter.ndim != 2 or min(parameter.shape) < 2:
        return False
    if name.startswith("block_proposal."):
        return name in {
            "block_proposal.horizon_context",
            "block_proposal.horizon_back",
        }
    specialist_block = re.match(
        r"^specialists\.[A-Za-z0-9_-]+\.blocks\.\d+\.", name
    )
    if not (
        name.startswith("blocks.")
        or name.startswith("block_decoder.")
        or specialist_block
    ):
        return False
    if any(token in name.lower() for token in _MUON_EXCLUDED_TOKENS):
        return False
    return name.endswith(_MUON_SUFFIXES)


def newton_schulz_orthogonalize(
    gradient: torch.Tensor,
    *,
    steps: int = 5,
    dtype: torch.dtype = torch.float32,
) -> torch.Tensor:
    """Apply the native quintic Newton--Schulz polar-factor approximation."""
    if gradient.ndim != 2 or min(gradient.shape) < 2:
        raise ValueError("Muon requires matrices with both dimensions >= 2")
    if steps < 1:
        raise ValueError("Muon Newton-Schulz iterations must be positive")

    g = gradient.float()
    norm = torch.sqrt(torch.sum(g * g))
    x = (g / torch.clamp(norm, min=1e-7)).to(dtype=dtype)
    transposed = x.shape[0] > x.shape[1]
    if transposed:
        x = x.transpose(0, 1)

    a, b, c = 3.4445, -4.7750, 2.0315
    for _ in range(steps):
        gram = x @ x.transpose(0, 1)
        polynomial = (b * gram + c * (gram @ gram)).to(dtype=dtype)
        x = (a * x + polynomial @ x).to(dtype=dtype)

    if transposed:
        x = x.transpose(0, 1)
    return x.float()


class HybridMuonAdamW(torch.optim.Optimizer):
    """Hybrid Muon/AdamW for canonical PyTorch named parameters.

    ``named_parameters`` accepts ``(name, parameter)`` pairs or
    ``(name, parameter, decay)`` triples.  A mapping or callable
    ``decay_policy`` may instead provide the native per-parameter decay flag;
    without either, a parameter's optional ``decay`` attribute is honored and
    otherwise defaults to ``True``, matching ``mini_llm.parameter.Parameter``.

    Offload modes are:

    * ``none``: FP32 masters and moments stay on each parameter's device.
    * ``moments``: moments are CPU tensors; masters stay on parameter devices.
    * ``full``: moments and FP32 masters are CPU tensors.

    CPU offload tensors use pinned memory whenever CUDA is available.  State is
    explicitly transferred to the parameter device for each update and copied
    back before ``step`` returns.
    """

    kind = "hybrid-muon"

    def __init__(
        self,
        named_parameters: Iterable[tuple[str, torch.Tensor]],
        *,
        lr: float = 3e-4,
        betas: tuple[float, float] = (0.9, 0.95),
        beta1: float | None = None,
        beta2: float | None = None,
        eps: float = 1e-8,
        weight_decay: float = 0.1,
        muon_momentum: float = 0.95,
        muon_ns_steps: int = 5,
        muon_matmul_precision: str = "bf16",
        offload: str = "none",
        decay_policy: Mapping[str, bool]
        | Callable[[str, torch.Tensor], bool]
        | None = None,
    ) -> None:
        resolved_beta1 = float(betas[0] if beta1 is None else beta1)
        resolved_beta2 = float(betas[1] if beta2 is None else beta2)
        self._validate_hyperparameters(
            lr=lr,
            beta1=resolved_beta1,
            beta2=resolved_beta2,
            eps=eps,
            weight_decay=weight_decay,
            muon_momentum=muon_momentum,
            muon_ns_steps=muon_ns_steps,
            muon_matmul_precision=muon_matmul_precision,
            offload=offload,
        )

        entries = self._resolve_named_parameters(named_parameters, decay_policy)
        if not entries:
            raise ValueError("optimizer got an empty parameter list")

        muon_entries = [entry for entry in entries if muon_eligible(entry[0], entry[1])]
        adam_entries = [entry for entry in entries if not muon_eligible(entry[0], entry[1])]
        groups: list[dict[str, Any]] = []
        for algorithm, selected in (("adamw", adam_entries), ("muon", muon_entries)):
            if selected:
                groups.append(
                    {
                        "params": [parameter for _, parameter, _ in selected],
                        "param_names": [name for name, _, _ in selected],
                        "algorithm": algorithm,
                    }
                )

        defaults = {
            "lr": float(lr),
            "beta1": resolved_beta1,
            "beta2": resolved_beta2,
            "eps": float(eps),
            "weight_decay": float(weight_decay),
            "muon_momentum": float(muon_momentum),
            "muon_ns_steps": int(muon_ns_steps),
            "muon_matmul_precision": str(muon_matmul_precision),
        }
        super().__init__(groups, defaults)

        self.names = tuple(name for name, _, _ in entries)
        self.parameters = tuple(parameter for _, parameter, _ in entries)
        self.decay_by_name = {name: decay for name, _, decay in entries}
        self.muon_names = frozenset(name for name, _, _ in muon_entries)
        self.adam_names = frozenset(name for name, _, _ in adam_entries)
        self._parameter_by_name = {
            name: parameter for name, parameter, _ in entries
        }
        self._name_by_parameter = {
            parameter: name for name, parameter, _ in entries
        }
        self._algorithm_by_name = {
            name: ("muon" if name in self.muon_names else "adamw")
            for name in self.names
        }
        self.offload_mode = str(offload)
        self.moments_offloaded = self.offload_mode in {"moments", "full"}
        self.master_weights_offloaded = self.offload_mode == "full"
        self.beta1 = resolved_beta1
        self.beta2 = resolved_beta2
        self.eps = float(eps)
        self.weight_decay = float(weight_decay)
        self.momentum = float(muon_momentum)
        self.ns_steps = int(muon_ns_steps)
        self.matmul_precision = str(muon_matmul_precision)
        self._step_index = 0
        self._pin_offload = bool(torch.cuda.is_available())
        self.offload_host_bytes = 0
        self._initialize_state()

    @staticmethod
    def _validate_hyperparameters(**values: Any) -> None:
        lr = float(values["lr"])
        weight_decay = float(values["weight_decay"])
        if not math.isfinite(lr) or lr < 0.0:
            raise ValueError("learning rate must be finite and nonnegative")
        if not math.isfinite(weight_decay) or weight_decay < 0.0:
            raise ValueError("weight decay must be finite and nonnegative")
        for name in ("beta1", "beta2"):
            value = float(values[name])
            if not math.isfinite(value) or not 0.0 <= value < 1.0:
                raise ValueError(f"AdamW {name} must be finite and in [0, 1)")
        eps = float(values["eps"])
        if not math.isfinite(eps) or eps <= 0.0:
            raise ValueError("AdamW eps must be finite and > 0")
        momentum = float(values["muon_momentum"])
        if not math.isfinite(momentum) or not 0.0 <= momentum < 1.0:
            raise ValueError("Muon momentum must be finite and in [0, 1)")
        if not 1 <= int(values["muon_ns_steps"]) <= 20:
            raise ValueError("Muon Newton-Schulz steps must lie in [1, 20]")
        if values["muon_matmul_precision"] not in {"bf16", "fp32"}:
            raise ValueError("Muon matmul precision must be bf16 or fp32")
        if values["offload"] not in {"none", "moments", "full"}:
            raise ValueError("offload must be 'none', 'moments', or 'full'")

    @staticmethod
    def _resolve_named_parameters(
        named_parameters: Iterable[tuple[str, torch.Tensor]],
        decay_policy: Mapping[str, bool]
        | Callable[[str, torch.Tensor], bool]
        | None,
    ) -> list[tuple[str, torch.Tensor, bool]]:
        entries: list[tuple[str, torch.Tensor, bool]] = []
        seen_names: set[str] = set()
        seen_parameters: set[int] = set()
        for item in named_parameters:
            if not isinstance(item, (tuple, list)) or len(item) not in {2, 3}:
                raise TypeError(
                    "named parameters must be (name, parameter) pairs or "
                    "(name, parameter, decay) triples"
                )
            name, parameter = item[0], item[1]
            if not isinstance(name, str) or not name:
                raise ValueError("every optimizer parameter requires a canonical name")
            if not isinstance(parameter, torch.Tensor):
                raise TypeError(f"parameter {name!r} is not a torch Tensor")
            if name in seen_names:
                raise ValueError(f"duplicate optimizer parameter name: {name}")
            if id(parameter) in seen_parameters:
                raise ValueError(f"parameter appears more than once: {name}")

            if len(item) == 3:
                if decay_policy is not None:
                    raise ValueError(
                        "decay must be supplied either in named triples or by "
                        "decay_policy, not both"
                    )
                decay = bool(item[2])
            elif isinstance(decay_policy, Mapping):
                if name not in decay_policy:
                    raise ValueError(f"decay policy is missing parameter: {name}")
                decay = bool(decay_policy[name])
            elif decay_policy is not None:
                decay = bool(decay_policy(name, parameter))
            else:
                decay = bool(getattr(parameter, "decay", True))

            seen_names.add(name)
            seen_parameters.add(id(parameter))
            entries.append((name, parameter, decay))
        return entries

    def _cpu_state_tensor(
        self, parameter: torch.Tensor, *, zeros: bool
    ) -> torch.Tensor:
        result = torch.empty(
            tuple(parameter.shape),
            dtype=torch.float32,
            device="cpu",
            pin_memory=self._pin_offload,
        )
        if zeros:
            result.zero_()
        else:
            result.copy_(parameter.detach().to(device="cpu", dtype=torch.float32))
        self.offload_host_bytes += result.numel() * result.element_size()
        return result

    def _device_state_tensor(
        self, parameter: torch.Tensor, *, zeros: bool
    ) -> torch.Tensor:
        if zeros:
            return torch.zeros_like(
                parameter, dtype=torch.float32, memory_format=torch.preserve_format
            )
        return parameter.detach().to(dtype=torch.float32).clone()

    def _initialize_state(self) -> None:
        for name in self.names:
            parameter = self._parameter_by_name[name]
            state = self.state[parameter]
            if self.master_weights_offloaded:
                state["master_weights"] = self._cpu_state_tensor(
                    parameter, zeros=False
                )
            else:
                state["master_weights"] = self._device_state_tensor(
                    parameter, zeros=False
                )

            if self._algorithm_by_name[name] == "muon":
                if self.moments_offloaded:
                    state["muon_m"] = self._cpu_state_tensor(parameter, zeros=True)
                else:
                    state["muon_m"] = self._device_state_tensor(
                        parameter, zeros=True
                    )
            else:
                for key in ("m", "v"):
                    if self.moments_offloaded:
                        state[key] = self._cpu_state_tensor(parameter, zeros=True)
                    else:
                        state[key] = self._device_state_tensor(
                            parameter, zeros=True
                        )

    @property
    def step_index(self) -> int:
        return self._step_index

    @step_index.setter
    def step_index(self, value: int) -> None:
        value = int(value)
        if value < 0:
            raise ValueError("optimizer step must be nonnegative")
        self._step_index = value

    def _compute_tensor(
        self, tensor: torch.Tensor, device: torch.device
    ) -> torch.Tensor:
        if tensor.device == device:
            return tensor
        return tensor.to(
            device=device,
            dtype=torch.float32,
            non_blocking=tensor.device.type == "cpu" and tensor.is_pinned(),
        )

    @staticmethod
    def _copy_state_back(destination: torch.Tensor, source: torch.Tensor) -> bool:
        if destination.data_ptr() == source.data_ptr() and destination.device == source.device:
            return False
        asynchronous = (
            destination.device.type == "cpu"
            and source.device.type == "cuda"
            and destination.is_pinned()
        )
        destination.copy_(source, non_blocking=asynchronous)
        return asynchronous

    @staticmethod
    def _bf16_ns_supported(device: torch.device) -> bool:
        return (
            device.type == "cuda"
            and torch.cuda.is_available()
            and bool(torch.cuda.is_bf16_supported())
        )

    def _group_for(self, algorithm: str) -> dict[str, Any]:
        return next(
            group for group in self.param_groups if group["algorithm"] == algorithm
        )

    def _step_adamw(
        self,
        name: str,
        parameter: torch.Tensor,
        group: dict[str, Any],
    ) -> bool:
        gradient = parameter.grad
        if gradient is None:
            return False
        if gradient.is_sparse:
            raise RuntimeError("HybridMuonAdamW does not support sparse gradients")

        state = self.state[parameter]
        device = parameter.device
        master = self._compute_tensor(state["master_weights"], device)
        first = self._compute_tensor(state["m"], device)
        second = self._compute_tensor(state["v"], device)
        grad = gradient.detach().to(device=device, dtype=torch.float32)

        beta1 = float(group["beta1"])
        beta2 = float(group["beta2"])
        first.mul_(beta1).add_(grad, alpha=1.0 - beta1)
        second.mul_(beta2).addcmul_(grad, grad, value=1.0 - beta2)

        correction1 = 1.0 - beta1 ** self._step_index
        correction2 = 1.0 - beta2 ** self._step_index
        sqrt_correction2 = math.sqrt(correction2)
        step_size = float(group["lr"]) * sqrt_correction2 / correction1
        eps_scaled = float(group["eps"]) * sqrt_correction2
        update = first / (torch.sqrt(second) + eps_scaled)
        update.mul_(step_size)

        if self.decay_by_name[name] and float(group["weight_decay"]) != 0.0:
            master.mul_(1.0 - float(group["lr"]) * float(group["weight_decay"]))
        master.sub_(update)
        parameter.copy_(master.to(dtype=parameter.dtype))

        pending = self._copy_state_back(state["m"], first)
        pending |= self._copy_state_back(state["v"], second)
        pending |= self._copy_state_back(state["master_weights"], master)
        return pending

    def _step_muon(
        self,
        name: str,
        parameter: torch.Tensor,
        group: dict[str, Any],
    ) -> bool:
        gradient = parameter.grad
        if gradient is None:
            return False
        if gradient.is_sparse:
            raise RuntimeError("HybridMuonAdamW does not support sparse gradients")

        state = self.state[parameter]
        device = parameter.device
        master = self._compute_tensor(state["master_weights"], device)
        momentum = self._compute_tensor(state["muon_m"], device)
        grad = gradient.detach().to(device=device, dtype=torch.float32)

        coefficient = float(group["muon_momentum"])
        momentum.mul_(coefficient).add_(grad, alpha=1.0 - coefficient)
        nesterov = momentum.mul(coefficient).add(grad, alpha=1.0 - coefficient)
        ns_dtype = torch.float32
        if (
            group["muon_matmul_precision"] == "bf16"
            and self._bf16_ns_supported(device)
        ):
            ns_dtype = torch.bfloat16
        direction = newton_schulz_orthogonalize(
            nesterov,
            steps=int(group["muon_ns_steps"]),
            dtype=ns_dtype,
        )
        direction.mul_(
            0.2 * math.sqrt(max(int(parameter.shape[0]), int(parameter.shape[1])))
        )

        if self.decay_by_name[name] and float(group["weight_decay"]) != 0.0:
            master.mul_(1.0 - float(group["lr"]) * float(group["weight_decay"]))
        master.add_(direction, alpha=-float(group["lr"]))
        parameter.copy_(master.to(dtype=parameter.dtype))

        pending = self._copy_state_back(state["muon_m"], momentum)
        pending |= self._copy_state_back(state["master_weights"], master)
        return pending

    def step(self, closure: Callable[[], Any] | None = None) -> Any:
        """Perform one exact native-form update and return an optional closure loss."""
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()

        with torch.no_grad():
            self._step_index += 1
            pending_devices: set[torch.device] = set()
            for algorithm in ("adamw", "muon"):
                if not any(
                    group["algorithm"] == algorithm for group in self.param_groups
                ):
                    continue
                group = self._group_for(algorithm)
                for name, parameter in zip(group["param_names"], group["params"]):
                    pending = (
                        self._step_adamw(name, parameter, group)
                        if algorithm == "adamw"
                        else self._step_muon(name, parameter, group)
                    )
                    if pending:
                        pending_devices.add(parameter.device)
            for device in pending_devices:
                torch.cuda.current_stream(device).synchronize()
        return loss

    def state_dict(self) -> dict[str, Any]:
        """Return the native name-keyed checkpoint schema."""
        return {
            "master_weights": {
                name: self.state[self._parameter_by_name[name]]["master_weights"]
                for name in self.names
            },
            "m": {
                name: self.state[self._parameter_by_name[name]]["m"]
                for name in self.names
                if name in self.adam_names
            },
            "v": {
                name: self.state[self._parameter_by_name[name]]["v"]
                for name in self.names
                if name in self.adam_names
            },
            "muon_m": {
                name: self.state[self._parameter_by_name[name]]["muon_m"]
                for name in self.names
                if name in self.muon_names
            },
            "step": self.step_index,
        }

    checkpoint_state = state_dict

    @staticmethod
    def _restore_tensor(
        destination: torch.Tensor, value: Any, *, name: str
    ) -> None:
        source = value.detach() if isinstance(value, torch.Tensor) else torch.as_tensor(value)
        if tuple(source.shape) != tuple(destination.shape):
            raise ValueError(f"optimizer state shape mismatch: {name}")
        destination.copy_(source.to(device=destination.device, dtype=torch.float32))

    def load_state_dict(self, state_dict: Mapping[str, Any]) -> None:
        """Restore a complete native name-keyed checkpoint."""
        expected = {
            "master_weights": set(self.names),
            "m": set(self.adam_names),
            "v": set(self.adam_names),
            "muon_m": set(self.muon_names),
        }
        for key, expected_names in expected.items():
            values = state_dict.get(key)
            if not isinstance(values, Mapping) or set(values) != expected_names:
                raise ValueError(
                    f"checkpoint {key} names do not match optimizer parameters"
                )
            for name in self.names:
                if name in expected_names:
                    self._restore_tensor(
                        self.state[self._parameter_by_name[name]][key],
                        values[name],
                        name=name,
                    )
        self.step_index = int(state_dict.get("step", 0))

    def restore_checkpoint(self, state_dict: Mapping[str, Any]) -> tuple[int, int, int]:
        self.load_state_dict(state_dict)
        return len(self.names), len(self.adam_names), len(self.muon_names)
