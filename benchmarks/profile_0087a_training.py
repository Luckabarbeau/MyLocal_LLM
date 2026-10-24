#!/usr/bin/env python3
"""Bounded production-CuPy forward/backward timeline probe.

This process never constructs an optimizer, updates parameters, or writes a
checkpoint. Data setup, checkpoint loading, gradient clearing, and warmup are
outside each measured microbatch window. Grouped attention remains the default;
0089A permits an explicit strict Triton comparison on the same workload.
"""

from __future__ import annotations

import argparse
import contextlib
import ctypes
import hashlib
import json
import os
from pathlib import Path
import platform
import shutil
import subprocess
import sys
import tempfile
import threading
import time
from typing import Callable, Iterable

import numpy as np

_REPO_ROOT = Path(__file__).resolve().parents[1]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

CAPTURE_NAME = "PROFILE_0087A_MICROBATCH"
CAPTURE_DOMAIN = "MyLocal_LLM_0087A"
REQUIRED_RANGES = (
    "microbatch_total",
    "forward",
    "attention_full",
    "attention_local",
    "historical_retrieval",
    "ffn",
    "rmsnorm_residual",
    "lm_head_loss",
    "backward",
    "attention_backward_recompute",
    "ffn_backward",
    "gradient_accumulation",
)

# Exact flags supplied with the production invocation. setdefault preserves
# deliberate A/B overrides passed to this standalone process.
_PRODUCTION_BENCH_DEFAULTS = {
    "MINI_LLM_BACKEND": "cupy",
    "MINI_LLM_TRAIN_ATTN_BACKEND": "grouped",
    "MINI_LLM_LOCAL_DETAIL_PROFILE": "0",
    "MINI_LLM_MOE_DETAIL_PROFILE": "0",
    "MINI_LLM_ACTIVATION_CHECKPOINT": "none",
    "MINI_LLM_HISTORY_RECOMPUTE": "1",
    "MINI_LLM_HISTORY_QUERY_CHUNK": "1024",
    "MINI_LLM_LOCAL_ATTN_RECOMPUTE_PROBS": "1",
    "MINI_LLM_FUSED_EXPERT_GEMM": "0",
    "MINI_LLM_LOCAL_ATTN_QUERY_CHUNK": "auto",
    "MINI_LLM_LM_HEAD_CHUNK_TOKENS": "8192",
    "MINI_LLM_AR_COMPACT_TARGET": "1",
    "MINI_LLM_AR_ACCUM_LENGTH_BUCKETS": "1",
    "MINI_LLM_CUBLAS_GROUPED_LOCAL_GEMM": "1",
    "MINI_LLM_CUBLAS_GROUPED_LOCAL_GEMM_STRICT": "0",
    "MINI_LLM_CUBLAS_PACKED_POINTER_UPLOAD": "1",
    "MINI_LLM_CUBLAS_COMPACT_DESCRIPTORS": "1",
    "MINI_LLM_ROPE_PREPARED_POSITIONS": "1",
}
_REPORT_ENV_NAMES = tuple(sorted({
    *_PRODUCTION_BENCH_DEFAULTS,
    "MINI_LLM_TRAIN_ATTN_STRICT",
    "MINI_LLM_BF16_MIXED_CONTEXT",
}))


def _env_enabled(name: str, default: str = "0") -> bool:
    raw = os.environ.get(name, default).strip().lower()
    return raw not in {"", "0", "false", "off", "no"}


def _scope_label(name: str) -> str | None:
    """Translate existing synchronized profiler scopes to coarse NVTX labels."""
    if name.startswith("attention.dense."):
        return "attention_full"
    if name.startswith("attention.local.") or name.startswith(
        "attention.triton_mixed."
    ):
        return "attention_local"
    if (
        name.startswith("attention.historical.")
        or name.startswith("attention.retrieval.")
        or name.startswith("memory_context.")
    ):
        return "historical_retrieval"
    if name == "block.moe.forward":
        return "ffn"
    if name == "block.moe.backward":
        return "ffn_backward"
    if name.startswith("block.norm") or name.startswith("model.final_norm."):
        return "rmsnorm_residual"
    if name == "block.attention.backward" or (
        name.startswith("model.layer") and name.endswith(".recompute")
    ):
        return "attention_backward_recompute"
    if name.startswith("model.output_projection.") or name.startswith("model.loss."):
        return "lm_head_loss"
    return None


class RangeController:
    """CPU-safe balanced range controller used by the probe and unit tests."""

    def __init__(
        self,
        enabled: bool,
        push: Callable[[str], None] | None = None,
        pop: Callable[[], None] | None = None,
        translator: Callable[[str], str | None] = _scope_label,
    ) -> None:
        self.enabled = bool(enabled)
        self._push = push
        self._pop = pop
        self._translator = translator
        self._local = threading.local()

    def _stack(self) -> list[str | None]:
        stack = getattr(self._local, "stack", None)
        if stack is None:
            stack = []
            self._local.stack = stack
        return stack

    @contextlib.contextmanager
    def range(self, label: str):
        if not self.enabled:
            yield
            return
        if self._push is None or self._pop is None:
            raise RuntimeError("enabled NVTX controller has no emitter")
        stack = self._stack()
        pushed = not stack or stack[-1] != label
        stack.append(label if pushed else None)
        if pushed:
            self._push(label)
        try:
            yield
        finally:
            marker = stack.pop()
            if marker is not None:
                self._pop()

    def scope(self, name: str):
        label = self._translator(str(name))
        return contextlib.nullcontext() if label is None else self.range(label)


_NVTX_HELPER_SOURCE = r"""
#include <stdint.h>
#include <nvtx3/nvToolsExt.h>
static nvtxDomainHandle_t domain_handle;
static nvtxStringHandle_t range_name;
static void init_handles(void) {
    if (!domain_handle) {
        domain_handle = nvtxDomainCreateA("MyLocal_LLM_0087A");
        range_name = nvtxDomainRegisterStringA(
            domain_handle, "PROFILE_0087A_MICROBATCH");
    }
}
uint64_t profile_0087a_start(void) {
    nvtxEventAttributes_t event = {0};
    init_handles();
    event.version = NVTX_VERSION;
    event.size = NVTX_EVENT_ATTRIB_STRUCT_SIZE;
    event.messageType = NVTX_MESSAGE_TYPE_REGISTERED;
    event.message.registered = range_name;
    return (uint64_t)nvtxDomainRangeStartEx(domain_handle, &event);
}
void profile_0087a_end(uint64_t id) {
    init_handles();
    nvtxDomainRangeEnd(domain_handle, (nvtxRangeId_t)id);
}
"""


class RegisteredCaptureRange:
    """Registered-string NVTX domain range required by Nsight capture filters."""

    def __init__(self, library: ctypes.CDLL):
        self._library = library
        library.profile_0087a_start.restype = ctypes.c_uint64
        library.profile_0087a_end.argtypes = [ctypes.c_uint64]
        library.profile_0087a_end.restype = None
        self._range_id: int | None = None

    def start(self) -> None:
        if self._range_id is not None:
            raise RuntimeError("capture range already active")
        self._range_id = int(self._library.profile_0087a_start())

    def end(self) -> None:
        if self._range_id is None:
            raise RuntimeError("capture range is not active")
        range_id, self._range_id = self._range_id, None
        self._library.profile_0087a_end(range_id)

    def __enter__(self):
        self.start()
        return self

    def __exit__(self, exc_type, exc, tb):
        self.end()
        return False


def _nvtx_header_dir() -> Path:
    candidates = []
    cuda_path = os.environ.get("CUDA_PATH") or os.environ.get("CUDA_HOME")
    if cuda_path:
        candidates.append(Path(cuda_path) / "include")
    candidates.extend(
        [
            Path("/usr/local/cuda/include"),
            Path("/usr/local/cuda-13.0/targets/x86_64-linux/include"),
            Path("/usr/local/cuda-13.4/targets/x86_64-linux/include"),
        ]
    )
    for candidate in candidates:
        if (candidate / "nvtx3" / "nvToolsExt.h").is_file():
            return candidate
    raise RuntimeError("NVTX3 header nvtx3/nvToolsExt.h was not found")


def build_registered_capture() -> RegisteredCaptureRange:
    compiler = shutil.which("cc") or shutil.which("gcc") or shutil.which("clang")
    if compiler is None:
        raise RuntimeError("a C compiler is required for the registered NVTX capture range")
    include_dir = _nvtx_header_dir()
    digest = hashlib.sha256(
        (_NVTX_HELPER_SOURCE + str(include_dir)).encode("utf-8")
    ).hexdigest()[:16]
    cache_dir = Path(tempfile.gettempdir()) / "mylocal_llm_0087a_nvtx"
    cache_dir.mkdir(parents=True, exist_ok=True)
    source = cache_dir / f"nvtx_capture_{digest}.c"
    library = cache_dir / f"nvtx_capture_{digest}.so"
    if not library.is_file():
        source.write_text(_NVTX_HELPER_SOURCE, encoding="utf-8")
        completed = subprocess.run(
            [
                compiler,
                "-shared",
                "-fPIC",
                "-O2",
                f"-I{include_dir}",
                str(source),
                "-o",
                str(library),
                "-ldl",
            ],
            text=True,
            capture_output=True,
            check=False,
        )
        if completed.returncode != 0:
            raise RuntimeError(
                "failed to compile registered NVTX helper: "
                + (completed.stderr.strip() or completed.stdout.strip())
            )
    return RegisteredCaptureRange(ctypes.CDLL(str(library)))


def _percentile(values: Iterable[float], q: float) -> float:
    ordered = sorted(float(value) for value in values)
    if not ordered:
        raise ValueError("cannot summarize an empty timing sequence")
    position = (len(ordered) - 1) * q
    lower = int(position)
    upper = min(lower + 1, len(ordered) - 1)
    weight = position - lower
    return ordered[lower] * (1.0 - weight) + ordered[upper] * weight


def summarize_ms(values: Iterable[float]) -> dict[str, float | int]:
    samples = [float(value) for value in values]
    return {
        "count": len(samples),
        "median_ms": _percentile(samples, 0.5),
        "p10_ms": _percentile(samples, 0.1),
        "p90_ms": _percentile(samples, 0.9),
        "min_ms": min(samples),
        "max_ms": max(samples),
    }


def _array_sha256(value) -> str:
    array = np.ascontiguousarray(np.asarray(value))
    digest = hashlib.sha256()
    digest.update(array.dtype.str.encode("ascii"))
    digest.update(json.dumps(array.shape).encode("ascii"))
    digest.update(array.view(np.uint8))
    return digest.hexdigest()


def _install_scope_translation(controller: RangeController) -> None:
    # These modules import performance_scope directly, so patch only this
    # standalone process after all production modules have loaded.
    import mini_llm.blocks.transformer_block as transformer_block
    import mini_llm.model.decoder_lm as decoder_lm
    import mini_llm.ops.attention as attention

    for module in (decoder_lm, transformer_block, attention):
        module.performance_scope = controller.scope


def _active_head_patterns(config, metadata: dict) -> dict:
    from mini_llm.ops.historical_schedule import historical_slots, source_indices

    schedule = historical_slots(config.n_layers, config.n_q_heads)
    memory_length = int(config.memory_context.distant_memory_length)
    valid_starts = np.asarray(metadata["history_valid_starts"], dtype=np.int64)
    row_has_history = valid_starts < memory_length
    all_rows_have_valid_history = bool(
        row_has_history.size and np.all(row_has_history)
    )
    per_layer = []
    pattern_counts: dict[str, int] = {}
    for layer, slots in enumerate(schedule):
        scheduled_heads = [int(slot.head) for slot in slots]
        any_valid_heads = []
        for slot in slots:
            if slot.kind == "learned":
                valid = np.any(row_has_history)
            else:
                valid = any(
                    bool(source_indices(
                        slot, memory_length, valid_start=int(start),
                        block_size=int(config.memory_context.block_size),
                    ))
                    for start in valid_starts
                )
            if valid:
                any_valid_heads.append(int(slot.head))
        executed_skip_heads = (
            scheduled_heads if all_rows_have_valid_history else []
        )
        executed_local_heads = [
            head for head in range(1, int(config.n_q_heads))
            if head not in executed_skip_heads
        ]
        local_count = len(executed_local_heads)
        pattern_counts[str(local_count)] = pattern_counts.get(str(local_count), 0) + 1
        per_layer.append(
            {
                "layer": layer,
                "full_heads": [0],
                "scheduled_historical_heads": scheduled_heads,
                "historical_heads_with_any_valid_source": any_valid_heads,
                "all_batch_rows_have_valid_history": all_rows_have_valid_history,
                "executed_historical_skip_heads": executed_skip_heads,
                "executed_active_local_heads": executed_local_heads,
                "active_local_head_count": local_count,
                "any_source_active_local_head_count": (
                    int(config.n_q_heads) - 1 - len(any_valid_heads)
                ),
            }
        )
    return {
        "batch_all_rows_have_valid_history": all_rows_have_valid_history,
        "counts_by_active_local_heads": pattern_counts,
        "counts_by_executed_active_local_heads": pattern_counts,
        "layers": per_layer,
    }


def _sample_batch(
    *, sources, config, batch_size: int, rng: np.random.Generator,
    desired_working_length: int | None, max_attempts: int,
):
    from mini_llm.data.compact_target import ar_compact_metadata
    from mini_llm.data.token_shards import (
        create_hierarchical_memory_minibatch,
        load_or_build_packed_document_index,
        map_token_shard,
    )

    target_length = int(config.memory_context.target_length)
    bucket_tokens = min(128, max(1, target_length // 8))
    weights = np.asarray([source.weight for source in sources], dtype=np.float64)
    weights /= weights.sum()
    observed_lengths: dict[int, int] = {}
    for attempt in range(1, max_attempts + 1):
        source = sources[int(rng.choice(len(sources), p=weights))]
        path = source.train_paths[int(rng.integers(0, len(source.train_paths)))]
        tokens = map_token_shard(str(path))
        document_index = load_or_build_packed_document_index(
            path, tokens, 1, write_sidecar=False,
        )
        inputs, targets, metadata = create_hierarchical_memory_minibatch(
            tokens,
            batch_size,
            int(config.memory_context.distant_memory_length),
            target_length,
            rng=rng,
            eos_token_id=1,
            document_index=document_index,
            document_aware=True,
            return_metadata=True,
            min_history_tokens=0,
            sampling_policy="document_first",
        )
        metadata = ar_compact_metadata(
            metadata,
            target_length=target_length,
            batch_size=batch_size,
            bucket_tokens=bucket_tokens,
        )
        working_length = int(metadata.get("compact_working_length", target_length))
        observed_lengths[working_length] = observed_lengths.get(working_length, 0) + 1
        if desired_working_length is None or working_length == desired_working_length:
            metadata = dict(metadata)
            metadata["router_gumbel_seed"] = int(rng.integers(0, 2**31 - 1))
            loss_mask = metadata["target_loss_mask"][:, :working_length]
            targets = targets[:, :working_length]
            valid_lengths = np.asarray(metadata["target_valid_lengths"], dtype=np.int64)
            history_starts = np.asarray(metadata["history_valid_starts"], dtype=np.int64)
            sample_report = {
                "source": source.name,
                "shard_path": str(path),
                "sampling_attempts": attempt,
                "original_working_length": target_length,
                "compact_working_length": working_length,
                "active_rows": int(batch_size * working_length),
                "computed_rows": int(batch_size * working_length),
                "supervised_rows": int(loss_mask.sum()),
                "target_valid_lengths": valid_lengths.tolist(),
                "valid_history_tokens_per_sample": (
                    int(config.memory_context.distant_memory_length) - history_starts
                ).tolist(),
                "history_valid_per_sample": (history_starts < int(
                    config.memory_context.distant_memory_length
                )).tolist(),
                "route_validity": "not_applicable_historical_read",
                "document_sampling_policy": metadata["document_sampling_policy"],
                "document_spans": (
                    np.asarray(metadata["document_ends"], dtype=np.int64)
                    - np.asarray(metadata["document_starts"], dtype=np.int64)
                ).tolist(),
                "observed_compact_lengths_before_match": {
                    str(key): value for key, value in sorted(observed_lengths.items())
                },
                "router_gumbel_seed": int(metadata["router_gumbel_seed"]),
                "fixture_hashes": {
                    "inputs": _array_sha256(inputs),
                    "targets": _array_sha256(targets),
                    "loss_mask": _array_sha256(loss_mask),
                    "history_valid_starts": _array_sha256(history_starts),
                },
                "head_patterns": _active_head_patterns(config, metadata),
            }
            return inputs, targets, loss_mask, metadata, sample_report
        del tokens
    raise RuntimeError(
        f"no real document-first B={batch_size} batch with compact working length "
        f"{desired_working_length} found in {max_attempts} attempts; observed "
        f"{dict(sorted(observed_lengths.items()))}"
    )


def _run_microbatch(
    model, batch, controller: RangeController, *, activation_checkpoint: bool,
    chunk_tokens: int, grad_scale: float,
):
    inputs, targets, loss_mask, metadata, _ = batch
    with controller.range("microbatch_total"):
        with controller.range("forward"):
            head_input, cache = model.forward_body(
                inputs,
                finite_trace=None,
                activation_checkpoint=activation_checkpoint,
                memory_metadata=metadata,
            )
        with controller.range("lm_head_loss"):
            target_slice = cache.get("target_slice")
            loss, loss_cache = model.chunked_lm_head_loss_forward(
                head_input,
                targets,
                chunk_tokens=chunk_tokens,
                loss_mask=loss_mask,
                return_device_loss=True,
                finite_trace=None,
                target_slice=target_slice,
            )
            if target_slice is not None:
                del head_input
        with controller.range("gradient_accumulation"):
            with controller.range("backward"):
                with controller.range("lm_head_loss"):
                    dx_head = model.chunked_lm_head_backward(
                        loss_cache, grad_scale=grad_scale,
                    )
                model.backward_body(dx_head, cache)
                del dx_head
        if target_slice is None:
            del head_input
    return loss


def _parse_working_length(value: str) -> int | None:
    if value.strip().lower() == "auto":
        return None
    result = int(value)
    if result <= 0 or result % 128:
        raise argparse.ArgumentTypeError("working length must be auto or a positive multiple of 128")
    return result


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--mixed-shard-root", type=Path, required=True)
    parser.add_argument("--batch-size", type=int, default=3)
    parser.add_argument("--microbatches", type=int, default=1)
    parser.add_argument("--warmup", type=int, default=2)
    parser.add_argument("--repeats", type=int, default=20)
    parser.add_argument("--working-length", type=_parse_working_length, default=None)
    parser.add_argument("--max-sampling-attempts", type=int, default=10000)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--runtime-profile", default="gpu-fast")
    parser.add_argument(
        "--attention-backend",
        choices=("grouped", "triton"),
        default=None,
        help=(
            "explicit current-attention backend; default preserves the environment "
            "or the grouped production baseline"
        ),
    )
    parser.add_argument("--annotations", action="store_true")
    parser.add_argument("--capture", action="store_true")
    parser.add_argument("--no-optimizer", action="store_true", help="accepted safety assertion; optimizer is never created")
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    if args.batch_size <= 0 or args.microbatches <= 0 or args.warmup < 0 or args.repeats <= 0:
        parser.error("batch size, microbatches and repeats must be positive; warmup must be non-negative")
    if args.capture:
        args.annotations = True
        args.repeats = 1
    return args


def _write_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    temporary.replace(path)


def main() -> int:
    args = _parse_args()
    if args.attention_backend is not None:
        os.environ["MINI_LLM_TRAIN_ATTN_BACKEND"] = args.attention_backend
    for name, value in _PRODUCTION_BENCH_DEFAULTS.items():
        os.environ.setdefault(name, value)
    attention_backend = os.environ["MINI_LLM_TRAIN_ATTN_BACKEND"].strip().lower()
    if attention_backend not in {"grouped", "triton"}:
        raise RuntimeError(
            "0089A supports only grouped or triton current attention, got "
            f"{attention_backend!r}"
        )
    if attention_backend == "triton" and not _env_enabled(
        "MINI_LLM_TRAIN_ATTN_STRICT"
    ):
        raise RuntimeError(
            "0089A Triton measurement requires MINI_LLM_TRAIN_ATTN_STRICT=1"
        )

    # Backend selection and runtime defaults must precede model/kernel imports.
    import cupy as cp
    from mini_llm.backend import BACKEND_NAME
    from mini_llm.checkpoint import load_checkpoint
    from mini_llm.config import ModelConfig
    from mini_llm.data.mixed_shards import load_weighted_shard_sources
    from mini_llm.model.decoder_lm import DecoderLanguageModel
    from mini_llm.runtime_defaults import apply_runtime_defaults

    if BACKEND_NAME != "cupy":
        raise RuntimeError(f"0089A requires CuPy backend, got {BACKEND_NAME!r}")

    checkpoint = args.checkpoint.expanduser().resolve()
    mixed_root = args.mixed_shard_root.expanduser().resolve()
    config = ModelConfig(**json.loads((checkpoint / "config.json").read_text(encoding="utf-8")))
    runtime = apply_runtime_defaults(
        args.runtime_profile,
        backend_name=BACKEND_NAME,
        config=config,
        training=True,
        precision="bf16-mixed",
    )
    activation_checkpoint = os.environ["MINI_LLM_ACTIVATION_CHECKPOINT"].lower() == "block"
    chunk_tokens = int(os.environ["MINI_LLM_LM_HEAD_CHUNK_TOKENS"])

    sources, manifest = load_weighted_shard_sources(mixed_root)
    eos_id = int(manifest.get("eos_token_id", 1))
    if eos_id != 1:
        raise RuntimeError(f"fixture expects checkpoint tokenizer EOS 1, manifest reports {eos_id}")
    rng = np.random.default_rng(args.seed)
    batches = [
        _sample_batch(
            sources=sources,
            config=config,
            batch_size=args.batch_size,
            rng=rng,
            desired_working_length=args.working_length,
            max_attempts=args.max_sampling_attempts,
        )
        for _ in range(args.microbatches)
    ]

    model = DecoderLanguageModel(config, rng_seed=args.seed, dtype=config.dtype)
    parameters = model.parameters()
    parameter_names = [parameter.name for parameter in parameters]
    loaded, _, _ = load_checkpoint(
        checkpoint,
        param_names=parameter_names,
        skip_optimizer=True,
        skip_training=True,
    )
    missing = sorted(set(parameter_names) - set(loaded))
    if missing:
        raise RuntimeError(f"checkpoint is missing {len(missing)} model parameters: {missing[:5]}")
    for parameter in parameters:
        parameter.data[...] = loaded.pop(parameter.name)
    del loaded
    model.refresh_compute_buffers()
    cp.get_default_memory_pool().free_all_blocks()

    emitter = cp.cuda.nvtx
    controller = RangeController(
        args.annotations,
        push=emitter.RangePush,
        pop=emitter.RangePop,
    )
    if args.annotations:
        _install_scope_translation(controller)

    def run_group() -> object:
        last_loss = None
        for batch in batches:
            last_loss = _run_microbatch(
                model,
                batch,
                controller,
                activation_checkpoint=activation_checkpoint,
                chunk_tokens=chunk_tokens,
                grad_scale=1.0,
            )
        return last_loss

    for _ in range(args.warmup):
        model.zero_grad()
        run_group()
        cp.cuda.runtime.deviceSynchronize()

    gpu_ms: list[float] = []
    wall_ms: list[float] = []
    capture = build_registered_capture() if args.capture else None
    for _ in range(args.repeats):
        model.zero_grad()
        cp.cuda.runtime.deviceSynchronize()
        start_event = cp.cuda.Event()
        end_event = cp.cuda.Event()
        if capture is not None:
            capture.start()
        wall_start = time.perf_counter_ns()
        start_event.record()
        loss = run_group()
        end_event.record()
        end_event.synchronize()
        wall_end = time.perf_counter_ns()
        gpu_ms.append(float(cp.cuda.get_elapsed_time(start_event, end_event)))
        wall_ms.append((wall_end - wall_start) / 1_000_000.0)

        result = {
            "schema_version": "0089A-v1",
            "status": "captured" if args.capture else "measured",
            "safety": {
                "optimizer_constructed": False,
                "optimizer_update": False,
                "checkpoint_written": False,
                "weights_modified": False,
            },
            "system": {
                "platform": platform.platform(),
                "python": platform.python_version(),
                "cupy": cp.__version__,
                "cuda_runtime": int(cp.cuda.runtime.runtimeGetVersion()),
                "driver_version": int(cp.cuda.runtime.driverGetVersion()),
                "device": cp.cuda.runtime.getDeviceProperties(0)["name"].decode(),
                "compute_capability": list(cp.cuda.Device(0).compute_capability),
                "triton": (
                    __import__("triton").__version__
                    if attention_backend == "triton" else None
                ),
            },
            "model": {
                "checkpoint": str(checkpoint),
                "parameter_count": int(sum(parameter.data.size for parameter in parameters)),
                "layers": int(config.n_layers),
                "d_model": int(config.d_model),
                "d_ff": int(config.d_ff),
                "q_heads": int(config.n_q_heads),
                "kv_heads": int(config.n_kv_heads),
                "dtype": str(config.dtype),
                "activation_checkpoint": "block" if activation_checkpoint else "none",
                "lm_head_chunk_tokens": chunk_tokens,
            },
            "runtime": {
                "profile_requested": args.runtime_profile,
                "profile_resolved": runtime.resolved,
                "profile_applied": list(runtime.applied),
                "annotations": args.annotations,
                "capture": args.capture,
                "capture_name": CAPTURE_NAME,
                "capture_domain": CAPTURE_DOMAIN,
                "attention_backend": attention_backend,
                "attention_strict": _env_enabled("MINI_LLM_TRAIN_ATTN_STRICT"),
                "effective_flags": {
                    name: os.environ.get(name) for name in _REPORT_ENV_NAMES
                },
            },
            "workload": {
                "batch_size": args.batch_size,
                "microbatches_per_sample": args.microbatches,
                "warmup_samples": args.warmup,
                "measured_samples": len(gpu_ms),
                "requested_working_length": args.working_length,
                "batches": [batch[4] for batch in batches],
                "total_supervised_rows_per_sample": int(sum(batch[4]["supervised_rows"] for batch in batches)),
                "total_computed_rows_per_sample": int(sum(batch[4]["computed_rows"] for batch in batches)),
            },
            "timing": {
                "gpu_event": summarize_ms(gpu_ms),
                "host_wall_outer_synchronized": summarize_ms(wall_ms),
                "gpu_event_samples_ms": gpu_ms,
                "host_wall_samples_ms": wall_ms,
                "profiler_perturbed": bool(args.capture),
                "note": "GPU-event and synchronized host-wall samples cover complete forward plus numerical backward; setup, warmup and zero_grad are excluded.",
            },
            "last_loss_device_resident_until_boundary": float(loss.item()),
        }
        _write_json(args.out, result)
        if capture is not None:
            # JSON is durable before stop-shutdown terminates the target. The
            # analyzer uses the nested microbatch_total range as its measurement
            # boundary, excluding this post-measurement serialization interval.
            capture.end()
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
