#!/usr/bin/env python3
"""Run the 0097 foundation suite fairly across native Model B and HF baselines.

The derived suite uses a shared 1024-token horizon, keeps identical task text,
targets, seeds, sampling, and semantic scorers, and removes only Model-B-specific
CUDA-graph conformance requirements. Unsupported long-context jobs are recorded
as exclusions rather than truncated or silently redefined.
"""

from __future__ import annotations

import argparse
import copy
import csv
import json
import math
import os
import time
from dataclasses import dataclass
from pathlib import Path
from types import SimpleNamespace

import numpy as np


SHARED_CONTEXT = 1024


@dataclass(frozen=True)
class ReferenceModel:
    repo: str
    revision: str
    training_tokens: int | None
    optimizer_step: int


MODELS = {
    "pythia-410m-step6000": ReferenceModel(
        "EleutherAI/pythia-410m", "step6000", 6000 * 2_097_152, 6000),
    "pythia-410m-step7000": ReferenceModel(
        "EleutherAI/pythia-410m", "step7000", 7000 * 2_097_152, 7000),
    "pythia-410m-step8000": ReferenceModel(
        "EleutherAI/pythia-410m", "step8000", 8000 * 2_097_152, 8000),
    "cerebras-gpt-590m": ReferenceModel(
        "cerebras-archive/Cerebras-GPT-590M", "main", 11_800_000_000, 0),
    "gpt2-medium": ReferenceModel(
        "openai-community/gpt2-medium", "main", None, 0),
}

EXCLUDED_IDS = {
    "wikitext_4096": "configured 4096-token horizon exceeds the shared 1024-token limit",
    "long_retrieval_0097": "configured 1K-32K retrieval grid has no equal native coverage at 1024 tokens",
}
RENAMED_CORPORA = {
    "pg19_4096": "pg19_1024_shared",
    "c4_validation_4096": "c4_validation_1024_shared",
}


def atomic_json(path: Path, value) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    pending = path.with_name(path.name + ".tmp")
    pending.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    os.replace(pending, path)


def derive_fair_suite(source: dict) -> tuple[dict, list[dict]]:
    suite = copy.deepcopy(source)
    suite["suite_name"] = "foundation_capabilities_0097_cross_model_1024"
    suite["output_root"] = "unused_runner_override"
    suite["on_error"] = "continue"
    jobs = []
    exclusions = []
    for original in source["benchmarks"]:
        if not original.get("enabled", True):
            exclusions.append({"benchmark_id": original["id"], "reason": "disabled in source suite"})
            continue
        if original["id"] in EXCLUDED_IDS:
            exclusions.append({"benchmark_id": original["id"], "reason": EXCLUDED_IDS[original["id"]]})
            continue
        job = copy.deepcopy(original)
        if job["id"] in RENAMED_CORPORA:
            source_id = job["id"]
            job["id"] = RENAMED_CORPORA[source_id]
            job["parameters"]["context_length"] = SHARED_CONTEXT
            job["parameters"]["stride"] = SHARED_CONTEXT
            exclusions.append({
                "benchmark_id": source_id,
                "reason": f"replaced by {job['id']} at the shared 1024-token horizon",
            })
        parameters = job.get("parameters", {})
        for key in ("require_native_cuda_graph_decode", "prefer_native_cuda_graph_decode"):
            parameters.pop(key, None)
        if "max_context" in parameters:
            parameters["max_context"] = min(int(parameters["max_context"]), SHARED_CONTEXT)
        if "context_length" in parameters:
            parameters["context_length"] = min(int(parameters["context_length"]), SHARED_CONTEXT)
        if "stride" in parameters:
            parameters["stride"] = min(int(parameters["stride"]), SHARED_CONTEXT)
        jobs.append(job)
    suite["benchmarks"] = jobs
    return suite, exclusions


def _native_context(config) -> int:
    for name in ("max_position_embeddings", "n_positions", "n_ctx", "seq_length"):
        value = getattr(config, name, None)
        if isinstance(value, int) and 0 < value < 1_000_000:
            return value
    return SHARED_CONTEXT


class HFGenerator:
    def __init__(self, adapter, max_context=None):
        self.adapter = adapter
        self.max_context = min(adapter.context_length, int(max_context or adapter.context_length))
        self._state = None

    def generate(self, *, prompt, max_new_tokens, strategy="greedy", temperature=0.8,
                 top_k=None, top_p=None, seed=42, use_kv_cache=True,
                 log_speed=False, return_token_ids=False):
        del log_speed, return_token_ids
        torch = self.adapter.torch
        encoded = self.adapter.tokenizer(
            prompt, return_tensors="pt", add_special_tokens=True)["input_ids"]
        if int(encoded.shape[1]) + int(max_new_tokens) > self.max_context:
            raise ValueError(
                f"prompt plus generation exceeds shared context: {encoded.shape[1]}+"
                f"{max_new_tokens}>{self.max_context}")
        encoded = encoded.to(self.adapter.device)
        torch.manual_seed(int(seed))
        if self.adapter.device.type == "cuda":
            torch.cuda.manual_seed_all(int(seed))
            torch.cuda.synchronize()
        started = time.perf_counter()
        kwargs = {
            "max_new_tokens": int(max_new_tokens), "num_beams": 1,
            "do_sample": strategy != "greedy", "use_cache": bool(use_kv_cache),
            "pad_token_id": self.adapter.eos_token_id,
        }
        if strategy == "top-k":
            kwargs.update(temperature=float(temperature), top_k=int(top_k or 40), top_p=1.0)
        elif strategy == "top-p":
            kwargs.update(temperature=float(temperature), top_p=float(top_p or 1.0), top_k=0)
        elif strategy != "greedy":
            raise ValueError(f"unsupported generation strategy {strategy!r}")
        with torch.inference_mode():
            output = self.adapter.model.generate(input_ids=encoded, **kwargs)
        if self.adapter.device.type == "cuda":
            torch.cuda.synchronize()
        elapsed = time.perf_counter() - started
        ids = output[0, encoded.shape[1]:].detach().cpu().tolist()
        text = self.adapter.tokenizer.decode(ids, skip_special_tokens=True)
        rate = len(ids) / elapsed if elapsed > 0 else 0.0
        timing = {
            "generated_token_ids": [int(value) for value in ids],
            "eos_token_id": self.adapter.eos_token_id,
            "total_time_s": elapsed, "prefill_time_s": 0.0,
            "tokens_per_sec": rate, "decode_tokens_per_sec": rate,
            "backend_note": "HF generate; native CUDA graph not applicable",
        }
        return text, timing


class HFModelAdapter:
    """Expose HF causal logits to the repository's existing CuPy scorers."""

    def __init__(self, model, tokenizer, torch, context_length):
        from mini_llm.backend import xp
        self.model = model
        self.tokenizer = tokenizer
        self.torch = torch
        self.device = next(model.parameters()).device
        self.context_length = min(int(context_length), SHARED_CONTEXT)
        self.eos_token_id = int(tokenizer.eos_token_id)
        self.xp = xp
        self.historical_memory = None
        self.hierarchical_memory_enabled = False
        self.blocks = ()
        self.config = SimpleNamespace(
            context_length=self.context_length,
            memory_context=SimpleNamespace(enabled=False),
            full_history_layers=(),
        )

    def forward(self, inputs, return_cache=False):
        if return_cache:
            raise ValueError("HF fair adapter exposes stateless scoring only")
        if int(inputs.shape[-1]) > self.context_length:
            raise ValueError("input exceeds shared cross-model context")
        torch = self.torch
        if hasattr(inputs, "__dlpack__"):
            token_ids = torch.utils.dlpack.from_dlpack(inputs).to(
                device=self.device, dtype=torch.long)
        else:
            token_ids = torch.as_tensor(
                np.asarray(inputs), device=self.device, dtype=torch.long)
        with torch.inference_mode():
            logits = self.model(input_ids=token_ids, use_cache=False).logits.float()
        if self.xp is np:
            return logits.detach().cpu().numpy()
        return self.xp.from_dlpack(logits.detach())

    def create_benchmark_generator(self, max_context=None):
        return HFGenerator(self, max_context=max_context)


def _model_error_path(output_root: Path, label: str) -> Path:
    return output_root / label / "model_error.json"


def rebuild_comparison(output_root: Path) -> None:
    rows = []
    for model_dir in sorted(path for path in output_root.iterdir() if path.is_dir()):
        for result_path in model_dir.glob("step_*/**/result.json"):
            record = json.loads(result_path.read_text(encoding="utf-8"))
            row = {
                "model": model_dir.name,
                "benchmark_id": record.get("benchmark_id"),
                "benchmark_type": record.get("benchmark_type"),
                "status": record.get("status"),
                "supervised_tokens_total": record.get("supervised_tokens_total"),
                "elapsed_seconds": record.get("elapsed_seconds"),
            }
            row.update(record.get("summary_metrics", {}))
            rows.append(row)
    fields = sorted({key for row in rows for key in row})
    path = output_root / "comparison.csv"
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(sorted(rows, key=lambda row: (row["benchmark_id"], row["model"])))


def run_hf_model(label: str, spec: ReferenceModel, suite: dict, fair_suite_path: Path,
                 output_root: Path, device: str) -> None:
    import torch  # type: ignore[import-not-found]
    from transformers import (  # type: ignore[import-not-found]
        AutoConfig, AutoModelForCausalLM, AutoTokenizer,
    )
    from benchmarks.live_suite import LiveBenchmarkScheduler

    model = None
    error_path = _model_error_path(output_root, label)
    if error_path.exists():
        error_path.unlink()
    print(f"Loading {label}: {spec.repo}@{spec.revision}", flush=True)
    try:
        hf_config = AutoConfig.from_pretrained(spec.repo, revision=spec.revision)
        native_context = _native_context(hf_config)
        if native_context < SHARED_CONTEXT:
            raise ValueError(f"native context {native_context} is below shared {SHARED_CONTEXT}")
        tokenizer = AutoTokenizer.from_pretrained(spec.repo, revision=spec.revision)
        if tokenizer.eos_token_id is None:
            raise ValueError("tokenizer has no EOS token")
        if tokenizer.pad_token_id is None:
            tokenizer.pad_token = tokenizer.eos_token
        target_device = torch.device(device)
        dtype = torch.bfloat16 if target_device.type == "cuda" else torch.float32
        model = AutoModelForCausalLM.from_pretrained(
            spec.repo, revision=spec.revision, dtype=dtype,
            low_cpu_mem_usage=True).to(target_device).eval()
        adapter = HFModelAdapter(model, tokenizer, torch, native_context)
        scheduler = LiveBenchmarkScheduler(
            suite, tokenizer, int(spec.training_tokens or 0), suite_path=fair_suite_path)
        scheduler.output_root = output_root / label
        scheduler.run_selected(
            adapter, step=spec.optimizer_step,
            supervised_tokens_total=float(spec.training_tokens or 0),
            provenance="published_training_budget" if spec.training_tokens else "not_independently_specified",
            checkpoint_id=f"{label}-{spec.revision}", checkpoint_path=f"hf://{spec.repo}@{spec.revision}")
    except Exception as exc:
        atomic_json(error_path, {
            "model": label, "repo": spec.repo, "revision": spec.revision,
            "error": f"{type(exc).__name__}: {exc}",
        })
        print(f"{label} failed: {type(exc).__name__}: {exc}", flush=True)
    finally:
        if model is not None:
            del model
        if torch.cuda.is_available():
            torch.cuda.empty_cache()


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--suite", type=Path,
                        default=Path("benchmarks/configs/foundation_capabilities_0097_progressive.json"))
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--model", action="append", choices=tuple(MODELS), dest="models")
    parser.add_argument("--native-checkpoint", type=Path)
    parser.add_argument("--device", choices=("cuda", "cpu"), default="cuda")
    parser.add_argument("--dry-run", action="store_true")
    return parser.parse_args(argv)


def main(argv=None):
    args = parse_args(argv)
    source = json.loads(args.suite.read_text(encoding="utf-8"))
    fair_suite, exclusions = derive_fair_suite(source)
    args.output_root.mkdir(parents=True, exist_ok=True)
    fair_suite_path = args.output_root / "fair_suite_1024.json"
    atomic_json(fair_suite_path, fair_suite)
    selected = args.models or list(MODELS)
    manifest = {
        "format_version": "foundation_cross_model_1024_v1",
        "source_suite": str(args.suite.resolve()),
        "fair_suite": str(fair_suite_path.resolve()),
        "shared_context": SHARED_CONTEXT,
        "models": selected,
        "native_checkpoint": str(args.native_checkpoint.resolve()) if args.native_checkpoint else None,
        "exclusions": exclusions,
        "fairness_contract": {
            "same_text_targets_seeds_and_semantic_scorers": True,
            "model_native_tokenizers": True,
            "corpus_primary_cross_tokenizer_metric": "bits_per_byte",
            "native_cuda_graph_requirement_removed": True,
            "unsupported_contexts_never_truncated": True,
        },
    }
    atomic_json(args.output_root / "manifest.json", manifest)
    if args.dry_run:
        print(json.dumps(manifest, indent=2))
        print(f"Derived {len(fair_suite['benchmarks'])} runnable benchmarks")
        return 0

    if args.native_checkpoint:
        from benchmarks.run_checkpoint_suite import run_checkpoints
        run_checkpoints(
            checkpoints=[args.native_checkpoint], suite_path=fair_suite_path,
            output_root=args.output_root / "model-b-current")
        rebuild_comparison(args.output_root)

    for label in selected:
        run_hf_model(label, MODELS[label], fair_suite, fair_suite_path,
                     args.output_root, args.device)
        rebuild_comparison(args.output_root)
    print(f"Cross-model results: {args.output_root / 'comparison.csv'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
