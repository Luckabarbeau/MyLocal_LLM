#!/usr/bin/env python3
"""CPU-only smooth FP64 checks of trained Model B blocks, not GPU-kernel parity.

Export block-entry residuals as layer_0_input, layer_10_input, layer_19_input
[B,T,D] (or [T,D]), with shared integer position_ids [B,T] (or [T]). Optional
shared document_ids [B,T] and target_valid_lengths [B] reject boundaries/padding.
Without document_ids, --contiguous-prefix explicitly certifies one unpadded
single-document prefix. Only row --row and its first --max-tokens (<=256) are
used; more than block_size tokens are required to exercise compressed history.
This is an isolated prefix check, not a replay of the omitted full context.

The existing block graph and directional_derivative_check are reused. A scoped
AST dtype-only lift removes hard-coded FP32 casts/buffers in matrix-pool and
sparse reference backward/forward, and computes RoPE constants in FP64. No
formula is replaced, no production file is written, and every patch is restored.
The lift is reported: this validates the smooth equations, not BF16 arithmetic.

Examples (run from repository root):
  OPENBLAS_NUM_THREADS=1 .venv/bin/python scripts/probe_0107z_trained_block_gradcheck.py --self-test
  OPENBLAS_NUM_THREADS=1 .venv/bin/python scripts/probe_0107z_trained_block_gradcheck.py \
      --inputs logs/OUTPUT/block_inputs.npz --contiguous-prefix --weights masters
"""
from __future__ import annotations

import argparse
import ast
from contextlib import ExitStack, contextmanager
from dataclasses import replace
import inspect
import json
import os
from pathlib import Path
import sys
import textwrap
from unittest.mock import patch

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
EPSILONS = (1e-3, 1e-4, 1e-5, 1e-6, 1e-7)
# Same per-layer phase assignment as DecoderLanguageModel.__init__ (0102Z).
SPARSE_PHASES = (0, 8, 4, 12, 2, 10, 6, 14, 1, 9, 5, 13, 3, 11, 7, 15)


def configure_cpu():
    """Must precede all mini_llm imports; never inherit production fast paths."""
    sys.path.insert(0, str(ROOT))
    settings = {
        "MINI_LLM_BACKEND": "numpy",
        "MINI_LLM_DUAL_RESOLUTION_FAST_PROJECTIONS": "0",
        "MINI_LLM_DUAL_RESOLUTION_LOCAL_KERNEL": "cupy",  # NumPy reference branch
        "MINI_LLM_DUAL_RESOLUTION_GLOBAL_KERNEL": "cupy",
        "MINI_LLM_MATRIX_POOL_FUSED_FORWARD": "0",
        "MINI_LLM_MATRIX_POOL_FUSED_BACKWARD": "0",
        "MINI_LLM_MATRIX_POOL_GRAD_SCATTER_INPLACE": "0",
        "MINI_LLM_FUSED_EXPERT_GEMM": "0",
        "MINI_LLM_FUSED_SWIGLU": "0",
        "MINI_LLM_MOE_RECOMPUTE_GATE_UP": "0",
        "MINI_LLM_MOE_OFFLOAD_GATE_UP": "0",
    }
    os.environ.update(settings)
    from mini_llm.backend import BACKEND_NAME
    if BACKEND_NAME != "numpy":
        raise RuntimeError("Run in a fresh process: a non-NumPy backend is already imported")
    return settings


class _LiftDtype(ast.NodeTransformer):
    def __init__(self):
        self.count = 0

    def visit_Attribute(self, node):
        self.generic_visit(node)
        if isinstance(node.value, ast.Name) and node.value.id == "xp" and node.attr == "float32":
            node.attr = "float64"
            self.count += 1
        return node

    def visit_Constant(self, node):
        if node.value == "float32":
            node.value = "float64"
            self.count += 1
        return node


@contextmanager
def smooth_fp64_reference():
    """Precision-lift existing functions, retaining their exact dependency graph."""
    from mini_llm.ops import matrix_pool_dual_resolution_attention as matrix
    from mini_llm.ops import rope, sparse_stride_history_0102z as sparse
    targets = (
        (matrix.MatrixPoolDualResolutionAttention, "_pool_backward"),
        (matrix.MatrixPoolDualResolutionAttention, "backward"),
        (sparse, "sparse_forward"), (sparse, "sparse_backward"),
        (rope, "get_rope_cos_sin"), (rope, "prepare_rope_positions"),
    )
    records = []
    rope.clear_rope_cache()
    try:
        with ExitStack() as stack:
            for owner, name in targets:
                original = getattr(owner, name)
                tree = ast.parse(textwrap.dedent(inspect.getsource(original)))
                lift = _LiftDtype()
                tree = ast.fix_missing_locations(lift.visit(tree))
                if not lift.count:
                    raise RuntimeError(f"Expected explicit FP32 sites in {original.__qualname__}")
                namespace = {}
                exec(compile(tree, original.__code__.co_filename + " [diagnostic FP64]", "exec"),
                     original.__globals__, namespace)
                stack.enter_context(patch.object(owner, name, namespace[name]))
                records.append({"function": original.__module__ + "." + original.__qualname__,
                                "dtype_substitutions": lift.count})
            yield records
    finally:
        rope.clear_rope_cache()


def float64_array(value):
    """Decode NumPy's raw two-byte BF16 checkpoint storage without GPU imports."""
    array = np.asarray(value)
    if array.dtype.kind == "V" and array.dtype.itemsize == 2:
        bits = array.view(np.uint16).astype(np.uint32) << 16
        array = bits.view(np.float32)
    if array.dtype.kind != "f":
        raise TypeError(f"Expected floating/BF16 array, got {array.dtype}")
    result = np.array(array, dtype=np.float64, copy=True)
    if not np.isfinite(result).all():
        raise ValueError("Nonfinite checkpoint weights or activations")
    return result


def make_block(config, layer):
    from mini_llm.backend import RandomStream
    from mini_llm.blocks.transformer_block import TransformerBlock
    dual = config.dual_resolution
    if not (dual.enabled and dual.compression_mode == "matrix_pool" and dual.global_enabled):
        raise ValueError("Enabled matrix-pool global attention is required")
    if config.ffn_mode != "dense" or config.progressive_depth or config.full_history_layers:
        raise ValueError("This check requires the ordinary dense, ungated Model B block")
    if not 0 <= layer < config.n_layers:
        raise ValueError(f"Layer {layer} outside checkpoint architecture")
    if dual.sparse_retrieval_stride and layer in dual.sparse_retrieval_layers:
        index = dual.sparse_retrieval_layers.index(layer)
        dual = replace(dual, sparse_retrieval_phase=SPARSE_PHASES[index])
    return TransformerBlock(
        d_model=config.d_model, n_q_heads=config.n_q_heads, n_kv_heads=config.n_kv_heads,
        d_head=config.d_head, d_ff=config.d_ff, n_experts=config.n_experts,
        top_k=config.top_k, input_std=config.init_std, output_std=config.residual_init_std,
        rope_base=config.rope_base, rng=RandomStream(10700 + layer), eps=config.rms_eps,
        name=f"blocks.{layer}", dtype="float64", ffn_mode=config.ffn_mode,
        dual_resolution_config=dual,
    )


def load_block_weights(block, checkpoint, source):
    directory = checkpoint / ("optimizer_state/master_weights" if source == "masters" else "model_params")
    loaded = []
    for parameter in block.parameters():
        path = directory / f"{parameter.name}.npy"
        values = np.load(path, allow_pickle=False)
        dtype = str(values.dtype)
        values = float64_array(values)
        if values.shape != parameter.shape:
            raise ValueError(f"Checkpoint shape mismatch for {parameter.name}: {values.shape} != {parameter.shape}")
        parameter.data[...] = values
        loaded.append({"name": parameter.name, "shape": list(values.shape), "stored_dtype": dtype})
    return loaded


def prefix_input(archive, layer, width, max_tokens, row, contiguous_prefix, block_size):
    x = archive[f"layer_{layer}_input"]
    if x.ndim == 2:
        x = x[None, ...]
    if x.ndim != 3 or x.shape[-1] != width or not 0 <= row < x.shape[0]:
        raise ValueError(f"layer_{layer}_input must be [B,T,{width}] with a valid --row")
    batch, length = x.shape[:2]
    positions = archive["position_ids"]
    if positions.shape == (length,):
        positions = np.broadcast_to(positions, (batch, length))
    if positions.shape != (batch, length) or positions.dtype.kind not in "iu":
        raise ValueError("position_ids must be integer [T] or [B,T], matching the activations")
    count = min(max_tokens, length)
    if "target_valid_lengths" in archive:
        valid = archive["target_valid_lengths"]
        if valid.shape != (batch,) or valid.dtype.kind not in "iu" or np.any(valid > length):
            raise ValueError("target_valid_lengths must be integer [B] within the exported length")
        count = min(count, int(valid[row]))
    if count <= block_size:
        raise ValueError(f"Need >{block_size} valid prefix tokens for nontrivial global/pool gradients")
    pos = np.array(positions[row:row + 1, :count], copy=True)
    if np.any(pos < 0) or not np.all(np.diff(pos, axis=1) == 1):
        raise ValueError("Prefix positions must be consecutive and nonnegative, without document resets")
    if "document_ids" in archive:
        docs = archive["document_ids"]
        if docs.shape != (batch, length) or docs.dtype.kind not in "iu":
            raise ValueError("document_ids must be integer [B,T]")
        if not np.all(docs[row, :count] == docs[row, 0]):
            raise ValueError("Prefix crosses a document boundary")
    elif not contiguous_prefix:
        raise ValueError("Export document_ids or certify an unpadded single document with --contiguous-prefix")
    return float64_array(x[row:row + 1, :count]), pos


def tiny_config():
    from mini_llm.config import ModelConfig
    base = ModelConfig.dual_resolution_matrix_pool_sparse_1024_20l()
    return replace(base, context_length=32, d_model=128, d_ff=256,
                   n_q_heads=2, n_kv_heads=1, d_head=64, dtype="float64",
                   dual_resolution=replace(base.dual_resolution, block_size=4,
                       local_window=8, long_local_window=16, long_local_heads=1,
                       global_heads=1, global_head_dim=64, max_blocks=8, global_query_chunk=4))


def check_block(block, x, positions, *, seed, directions, atol, rtol, emit):
    from mini_llm.diagnostics.gradcheck import directional_derivative_check
    # This helper generates regular single-document complete-block metadata from
    # the actual positions, without inventing token IDs or importing a sampler.
    metadata = block.attention._prepare_metadata(None, x.shape[0], x.shape[1], positions)
    rng = np.random.default_rng(seed)
    upstream = rng.normal(size=x.shape)
    upstream[:, :block.attention.block_size] = 0.0
    upstream /= np.linalg.norm(upstream)
    block.zero_grad()
    output, cache = block.forward(x, dual_resolution_metadata=metadata)
    if output.dtype != np.float64 or cache["attn_cache"]["global"] is None:
        raise AssertionError("Expected FP64 full block with an active global branch")
    dx = block.backward(upstream, cache)
    if dx.dtype != np.float64:
        raise AssertionError("Input gradient was narrowed below FP64")
    base_value = float(np.sum(output * upstream))
    del output, cache

    def objective(z):
        value = block.forward(z, return_cache=False, dual_resolution_metadata=metadata)
        if value.dtype != np.float64:
            raise AssertionError("Finite-difference forward must stay FP64")
        return float(np.sum(value * upstream))

    selected = [block.attention.Wo, block.attention.Wv,
                block.attention.W_global_o, block.attention.W_global_v,
                block.attention.W_pool_score, block.attention.pool_position,
                block.moe.expert.W_gate, block.moe.expert.W_up, block.moe.expert.W_down]
    targets = [("input", None, dx, None),
               ("input_history_only", None, dx, block.attention.block_size)]
    targets += [(p.name, p, p.grad, None) for p in selected if p is not None]
    passed = True
    summaries = []
    for name, parameter, gradient, history_length in targets:
        base = x if parameter is None else parameter.data.copy()
        if gradient.dtype != np.float64 or not np.isfinite(gradient).all():
            raise AssertionError(f"Invalid analytical gradient dtype/values for {name}")
        restricted = gradient.copy()
        if history_length is not None:
            restricted[:, history_length:] = 0.0
        if np.linalg.norm(restricted) == 0.0:
            raise AssertionError(f"Zero gradient: {name} is not exercised by this prefix/objective")

        def scalar_function(z):
            if parameter is None:
                return objective(z)
            parameter.data[...] = z
            block.refresh_compute_buffers()
            return objective(x)

        for index in range(directions):
            # First use the gradient direction for signal, then independent random
            # directions. The existing helper L2-normalizes every direction.
            direction = restricted.copy() if index == 0 else rng.normal(size=base.shape)
            if history_length is not None:
                direction[:, history_length:] = 0.0
            rows = []
            try:
                for epsilon in EPSILONS:
                    result = directional_derivative_check(
                        scalar_function, base, gradient, direction=direction, eps=epsilon)
                    analytic, fd = result["analytic"], result["finite_difference"]
                    absolute = abs(fd - analytic)
                    threshold = atol + rtol * max(abs(fd), abs(analytic))
                    ok = bool(np.isfinite([analytic, fd]).all() and absolute <= threshold)
                    record = {"event": "epsilon", "layer": block.attention.Wo.name.split(".")[1],
                              "target": name, "direction_index": index,
                              "direction": "gradient" if index == 0 else "random",
                              "direction_normalization": "unit_L2", "dtype": "float64",
                              "epsilon": epsilon, "analytic": analytic, "finite_difference": fd,
                              "absolute_error": absolute, "relative_error": result["relative_error"],
                              "relative_error_definition": "abs(fd-an)/(abs(fd)+abs(an)+1e-12)",
                              "atol": atol, "rtol": rtol, "absolute_threshold": threshold,
                              "passed": ok}
                    rows.append(record)
                    emit(record)
            finally:
                if parameter is not None:
                    parameter.data[...] = base
                    block.refresh_compute_buffers()
            # Require a stable region, not success at one cherry-picked epsilon.
            stable = any(a["passed"] and b["passed"] for a, b in zip(rows, rows[1:]))
            passed = passed and stable
            summary = {"target": name, "direction_index": index,
                       "passed_two_adjacent_epsilons": stable,
                       "best_absolute_error": min(r["absolute_error"] for r in rows)}
            summaries.append(summary)
    block.zero_grad()
    return {"passed": passed, "objective": "sum(block(x) * fixed_unit_L2_upstream)",
            "upstream_zero_on_first_block": True, "baseline_objective": base_value,
            "input_shape": list(x.shape), "position_start": int(positions[0, 0]),
            "position_end": int(positions[0, -1]), "sparse_phase": block.attention.sparse_phase,
            "checks": summaries}


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--checkpoint", type=Path,
                        default=ROOT / "checkpoints/model_b_batch1_8k_norecompute_long_16200")
    parser.add_argument("--inputs", type=Path)
    parser.add_argument("--weights", choices=("masters", "model"), default="masters")
    parser.add_argument("--layers", nargs="+", type=int, default=[0, 10, 19])
    parser.add_argument("--max-tokens", type=int, default=256)
    parser.add_argument("--row", type=int, default=0)
    parser.add_argument("--contiguous-prefix", action="store_true")
    parser.add_argument("--directions", type=int, default=2)
    parser.add_argument("--seed", type=int, default=10716200)
    parser.add_argument("--atol", type=float, default=1e-8)
    parser.add_argument("--rtol", type=float, default=3e-5)
    parser.add_argument("--output", type=Path, help="Optional complete JSON report; stdout is JSONL")
    parser.add_argument("--self-test", action="store_true", help="Synthetic tiny blocks only; no checkpoint claim")
    args = parser.parse_args(argv)
    if not args.self_test and args.inputs is None:
        parser.error("--inputs is required unless --self-test is selected")
    if args.self_test and args.inputs is not None:
        parser.error("--self-test does not accept real --inputs")
    if not 1 <= args.max_tokens <= 256 or args.directions < 1 or args.row < 0:
        parser.error("Need 1<=max-tokens<=256, directions>=1, row>=0")
    if len(set(args.layers)) != len(args.layers):
        parser.error("--layers must be unique")
    if not np.isfinite([args.atol, args.rtol]).all() or min(args.atol, args.rtol) < 0:
        parser.error("Tolerances must be finite and nonnegative")
    return args


def main(argv=None):
    args = parse_args(argv)
    settings = configure_cpu()
    from mini_llm.model_loading import read_native_config
    config = tiny_config() if args.self_test else read_native_config(args.checkpoint)
    step = None if args.self_test else json.loads(
        (args.checkpoint / "training_state.json").read_text())["step"]
    records, blocks = [], []

    def emit(record):
        record = {"checkpoint_step": step, **record}
        records.append(record)
        print(json.dumps(record, allow_nan=False), flush=True)

    with ExitStack() as stack:
        lifts = stack.enter_context(smooth_fp64_reference())
        archive = None if args.self_test else stack.enter_context(np.load(args.inputs, allow_pickle=False))
        emit({"event": "setup", "self_test": args.self_test, "backend": "numpy", "dtype": "float64",
              "checkpoint": None if args.self_test else str(args.checkpoint.resolve()),
              "inputs": None if args.self_test else str(args.inputs.resolve()),
              "weights": "synthetic" if args.self_test else args.weights,
              "epsilon_sweep": EPSILONS, "atol": args.atol, "rtol": args.rtol,
              "scope": "isolated single-document prefix; both block residual paths",
              "precision_lifts": lifts, "environment_overrides": settings})
        for layer in args.layers:
            block = make_block(config, layer)
            if args.self_test:
                loaded = []
                x = np.random.default_rng(args.seed + layer).normal(scale=0.2, size=(1, 9, config.d_model))
                positions = np.arange(x.shape[1], dtype=np.int64)[None, :]
            else:
                loaded = load_block_weights(block, args.checkpoint, args.weights)
                x, positions = prefix_input(archive, layer, config.d_model, args.max_tokens,
                                            args.row, args.contiguous_prefix, config.dual_resolution.block_size)
            emit({"event": "block_loaded", "layer": layer, "parameters": loaded,
                  "input_shape": list(x.shape), "sparse_phase": block.attention.sparse_phase})
            result = check_block(block, x, positions, seed=args.seed + layer + 1,
                                 directions=args.directions, atol=args.atol, rtol=args.rtol, emit=emit)
            blocks.append({"layer": layer, **result})
            emit({"event": "block_summary", **blocks[-1]})
            del block, x
    passed = all(block["passed"] for block in blocks)
    emit({"event": "summary", "passed": passed, "self_test": args.self_test,
          "block_count": len(blocks), "criterion": "two adjacent epsilon passes for every target/direction"})
    if args.output is not None:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps({"records": records, "blocks": blocks, "passed": passed},
                                         indent=2, allow_nan=False) + "\n")
    return 0 if passed else 1


if __name__ == "__main__":
    raise SystemExit(main())
