"""PyTorch autograd training lifecycle with token-weighted accumulation."""

from __future__ import annotations

from dataclasses import dataclass
import os
from pathlib import Path
import time
from typing import Any, Iterable

import numpy as np

from .optimizer import HybridMuonAdamW


def _torch():
    try:
        import torch
    except ImportError as exc:
        raise RuntimeError("PyTorch training requires torch") from exc
    return torch


def chunked_lm_head_loss_sum(
    hidden, projection_weight, targets, weights, *, chunk_tokens: int,
    weight_is_vocab_by_hidden: bool,
):
    """Memory-bounded BF16 projection and FP32 CE with backward recomputation."""
    torch = _torch()
    if hidden.ndim != 3 or targets.shape != hidden.shape[:2] or weights.shape != targets.shape:
        raise ValueError("hidden, targets, and weights have incompatible shapes")
    rows = int(hidden.shape[0] * hidden.shape[1])
    chunk_tokens = max(1, min(rows, int(chunk_tokens)))
    flat_hidden = hidden.reshape(rows, hidden.shape[-1])
    flat_targets = targets.reshape(rows)
    flat_weights = weights.reshape(rows)

    def chunk_loss(chunk_hidden, weight, chunk_targets, chunk_weights):
        logits = (
            chunk_hidden @ weight.transpose(0, 1)
            if weight_is_vocab_by_hidden else chunk_hidden @ weight
        )
        losses = torch.nn.functional.cross_entropy(
            logits.float(), chunk_targets, reduction="none"
        )
        return torch.sum(losses * chunk_weights)

    loss_sum = hidden.new_zeros((), dtype=torch.float32)
    for start in range(0, rows, chunk_tokens):
        stop = min(start + chunk_tokens, rows)
        arguments = (
            flat_hidden[start:stop], projection_weight,
            flat_targets[start:stop], flat_weights[start:stop],
        )
        if torch.is_grad_enabled() and (
            arguments[0].requires_grad or projection_weight.requires_grad
        ):
            tile_loss = torch.utils.checkpoint.checkpoint(
                chunk_loss, *arguments, use_reentrant=False,
                preserve_rng_state=False,
            )
        else:
            tile_loss = chunk_loss(*arguments)
        loss_sum = loss_sum + tile_loss
    return loss_sum


def prepare_dual_resolution_metadata(inputs, valid_lengths, block_size):
    """Build the same per-row complete-block contract used by native Model B."""
    batch, length = inputs.shape
    block_size = int(block_size)
    counts = np.asarray(valid_lengths, dtype=np.int64) // block_size
    block_count = int(counts.max(initial=0))
    indices = np.zeros((batch, block_count, block_size), dtype=np.int64)
    valid = np.zeros((batch, block_count), dtype=np.bool_)
    documents = np.zeros((batch, length), dtype=np.int64)
    block_documents = np.zeros((batch, block_count), dtype=np.int64)
    ends = np.full((batch, block_count), -1, dtype=np.int64)
    positions = np.broadcast_to(np.arange(length, dtype=np.int64), (batch, length)).copy()
    for row in range(batch):
        for slot in range(int(counts[row])):
            start = slot * block_size
            indices[row, slot] = np.arange(start, start + block_size)
            valid[row, slot] = True
            ends[row, slot] = start + block_size - 1
    return {
        "document_ids": documents,
        "position_ids": positions,
        "block_indices": indices,
        "block_valid": valid,
        "block_document_ids": block_documents,
        "block_end_positions": ends,
    }


class TorchTrainer:
    """Exact supervised-token-weighted accumulation over PyTorch autograd."""
    def __init__(self, model, *, lr=1e-3, beta1=0.9, beta2=0.95, eps=1e-8,
                 weight_decay=0.1, muon_momentum=0.95, muon_ns_steps=5,
                 muon_matmul_precision="bf16", grad_accum_steps=1,
                 max_grad_norm=1.0, optimizer_offload="none", optimizer=None,
                 lm_head_chunk_tokens=None):
        self.model = model
        self.grad_accum_steps = int(grad_accum_steps)
        self.max_grad_norm = float(max_grad_norm)
        self.lm_head_chunk_tokens = int(
            lm_head_chunk_tokens
            if lm_head_chunk_tokens is not None
            else os.environ.get("MINI_LLM_LM_HEAD_CHUNK_TOKENS", "1024")
        )
        if self.grad_accum_steps < 1:
            raise ValueError("grad_accum_steps must be positive")
        named = list(model.canonical_named_parameters())
        self.optimizer = optimizer or HybridMuonAdamW(
            named, lr=lr, beta1=beta1, beta2=beta2, eps=eps,
            weight_decay=weight_decay, muon_momentum=muon_momentum,
            muon_ns_steps=muon_ns_steps,
            muon_matmul_precision=muon_matmul_precision,
            decay_policy=model.decay_policy(), offload=optimizer_offload,
        )
        self.step = 0
        self.supervised_tokens_total = 0.0
        self.processed_tokens_total = 0
        self.optimizer.zero_grad(set_to_none=True)

    def _loss_sum(self, batch):
        torch = _torch()
        token_ids = torch.as_tensor(
            batch["input_ids"], device=self.model.embedding.W.device, dtype=torch.long
        )
        targets = torch.as_tensor(
            batch["targets"], device=token_ids.device, dtype=torch.long
        )
        mask = batch.get("loss_mask")
        weights = (
            torch.ones_like(targets, dtype=torch.float32)
            if mask is None else torch.as_tensor(mask, device=token_ids.device, dtype=torch.float32)
        )
        count = weights.sum()
        if not bool(count > 0):
            raise ValueError("microbatch has no supervised tokens")
        dual_resolution_metadata = batch.get("dual_resolution_metadata")
        if isinstance(dual_resolution_metadata, dict):
            dual_resolution_metadata.pop("_pytorch_block_mask_cache", None)
        hidden = self.model.forward_body(
            token_ids,
            position_ids=batch.get("position_ids"),
            document_ids=batch.get("document_ids"),
            dual_resolution_metadata=dual_resolution_metadata,
        )
        tied = bool(self.model.config.tie_embeddings)
        projection_weight = (
            self.model.embedding.W if tied else self.model.output_proj.W
        )
        loss_sum = chunked_lm_head_loss_sum(
            hidden, projection_weight, targets, weights,
            chunk_tokens=self.lm_head_chunk_tokens,
            weight_is_vocab_by_hidden=tied,
        )
        return loss_sum, count, token_ids.numel()

    def train_step(self, microbatches: Iterable[dict[str, Any]]):
        torch = _torch()
        self.model.train()
        total_loss = torch.zeros((), device=self.model.embedding.W.device)
        total_supervised = torch.zeros((), device=total_loss.device)
        total_executed = 0
        batches = list(microbatches)
        if len(batches) != self.grad_accum_steps:
            raise ValueError(
                f"expected {self.grad_accum_steps} microbatches, got {len(batches)}"
            )
        for batch in batches:
            loss_sum, supervised, executed = self._loss_sum(batch)
            loss_sum.backward()
            total_loss = total_loss + loss_sum.detach()
            total_supervised = total_supervised + supervised.detach()
            total_executed += int(executed)
        supervised_value = float(total_supervised.item())
        for parameter in self.model.parameters():
            if parameter.grad is not None:
                parameter.grad.div_(supervised_value)
        grad_norm = torch.nn.utils.clip_grad_norm_(
            self.model.parameters(), self.max_grad_norm, error_if_nonfinite=True
        )
        self.optimizer.step()
        self.optimizer.zero_grad(set_to_none=True)
        self.step += 1
        self.supervised_tokens_total += supervised_value
        self.processed_tokens_total += total_executed
        return {
            "loss": float((total_loss / total_supervised).item()),
            "grad_norm": float(grad_norm.item()),
            "supervised_tokens": supervised_value,
            "executed_tokens": total_executed,
            "step": self.step,
        }

    def validate(self, batches: Iterable[dict[str, Any]]):
        torch = _torch()
        self.model.eval()
        total_loss = 0.0
        total_tokens = 0.0
        with torch.no_grad():
            for batch in batches:
                loss_sum, count, _ = self._loss_sum(batch)
                total_loss += float(loss_sum.item())
                total_tokens += float(count.item())
        if total_tokens == 0:
            raise ValueError("validation has no supervised tokens")
        return total_loss / total_tokens

    def checkpoint_state(self):
        return {
            "step": self.step,
            "processed_tokens": self.processed_tokens_total,
            "cumulative_supervised_tokens": self.supervised_tokens_total,
            "backend": "pytorch",
            "gradient_accumulation_weighting": "supervised_tokens",
        }


@dataclass
class TorchTrainingRun:
    model: Any
    trainer: TorchTrainer
    train_shards: list[Any]
    val_shards: list[Any]
    rng: np.random.Generator
    batch_size: int
    seq_length: int
    eos_token_id: int | None
    checkpoint_dir: Path | None
    tokenizer: Any = None

    @classmethod
    def from_model(cls, facade, *, train_shard_paths, val_shard_paths,
                   batch_size=1, seq_length=None, grad_accum_steps=1,
                   eos_token_id=None, checkpoint_dir=None, seed=0, **options):
        from mini_llm.data.token_shards import map_token_shard
        if not train_shard_paths or not val_shard_paths:
            raise ValueError("PyTorch training requires train and validation shards")
        model = facade._native_model
        length = int(seq_length or model.config.context_length)
        train = [map_token_shard(str(path), length) for path in train_shard_paths]
        val = [map_token_shard(str(path), length) for path in val_shard_paths]
        trainer = TorchTrainer(
            model, grad_accum_steps=grad_accum_steps,
            **{key: value for key, value in options.items() if key in {
                "lr", "beta1", "beta2", "eps", "weight_decay",
                "muon_momentum", "muon_ns_steps", "muon_matmul_precision",
                "max_grad_norm", "optimizer_offload",
            }}
        )
        return cls(
            facade, trainer, train, val, np.random.default_rng(seed),
            int(batch_size), length, eos_token_id,
            None if checkpoint_dir is None else Path(checkpoint_dir), facade.tokenizer,
        )

    @property
    def step(self):
        return self.trainer.step

    @property
    def supervised_tokens(self):
        return self.trainer.supervised_tokens_total

    @property
    def grad_accum_steps(self):
        return self.trainer.grad_accum_steps

    def _sample(self, shard):
        from mini_llm.data.token_shards import (
            create_dual_resolution_document_minibatch, create_minibatch,
        )
        config = self.model._native_model.config
        if config.dual_resolution.enabled:
            if self.eos_token_id is None:
                raise ValueError("dual-resolution PyTorch training requires eos_token_id")
            inputs, targets, metadata = create_dual_resolution_document_minibatch(
                shard, self.batch_size, self.seq_length, self.rng,
                eos_token_id=self.eos_token_id, return_metadata=True,
            )
            dual = prepare_dual_resolution_metadata(
                inputs, metadata["target_valid_lengths"], config.dual_resolution.block_size
            )
            return {
                "input_ids": inputs, "targets": targets,
                "loss_mask": metadata["target_loss_mask"],
                "dual_resolution_metadata": dual,
            }
        inputs, targets = create_minibatch(
            shard, self.batch_size, self.seq_length, self.rng
        )
        return {"input_ids": inputs, "targets": targets}

    def train_steps(self, steps: int, *, log_interval: int = 10):
        results = []
        for _ in range(int(steps)):
            batches = [
                self._sample(self.train_shards[(self.step + index) % len(self.train_shards)])
                for index in range(self.grad_accum_steps)
            ]
            torch = _torch()
            if torch.cuda.is_available():
                torch.cuda.reset_peak_memory_stats()
                torch.cuda.synchronize()
            started = time.perf_counter()
            result = self.trainer.train_step(batches)
            if torch.cuda.is_available():
                torch.cuda.synchronize()
            result["time_s"] = time.perf_counter() - started
            result["executed_tokens_per_s"] = result["executed_tokens"] / result["time_s"]
            result["supervised_tokens_per_s"] = result["supervised_tokens"] / result["time_s"]
            result["peak_vram_bytes"] = (
                int(torch.cuda.max_memory_allocated()) if torch.cuda.is_available() else 0
            )
            results.append(result)
            if log_interval and self.step % int(log_interval) == 0:
                print(
                    f"step {self.step} loss={result['loss']:.6f} "
                    f"time_s={result['time_s']:.3f} "
                    f"executed_tok_s={result['executed_tokens_per_s']:.1f} "
                    f"supervised_tok_s={result['supervised_tokens_per_s']:.1f} "
                    f"peak_vram_gib={result['peak_vram_bytes'] / 2**30:.3f}"
                )
        return results

    def save_checkpoint(self):
        if self.checkpoint_dir is None:
            raise ValueError("TorchTrainingRun has no checkpoint_dir")
        from .checkpoint import save_checkpoint
        return save_checkpoint(
            self.checkpoint_dir, self.model._native_model,
            self.trainer.optimizer.state_dict(), self.trainer.checkpoint_state(),
            tokenizer=self.tokenizer if not (self.checkpoint_dir / "tokenizer.json").exists() else None,
        )

    def inspect(self):
        return {
            "model_id": self.model.model_id,
            "step": self.step,
            "supervised_tokens": self.supervised_tokens,
            "grad_accum_steps": self.grad_accum_steps,
            "checkpoint_dir": None if self.checkpoint_dir is None else str(self.checkpoint_dir),
            "backend": "pytorch",
        }


__all__ = [
    "TorchTrainer", "TorchTrainingRun", "chunked_lm_head_loss_sum",
    "prepare_dual_resolution_metadata",
]
