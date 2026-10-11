"""Micro-model CPU/GPU wall-time benchmark for optional 0074A monitor.

Usage: MINI_LLM_BACKEND=numpy python benchmarks/benchmark_gradient_noise.py
Results are only an integration smoke/performance proxy, NOT 500M GPU numbers.
"""
import argparse
import statistics
import tempfile
import time
from pathlib import Path

import numpy as np

from mini_llm.backend import xp, synchronize, BACKEND_NAME
from mini_llm.config import ModelConfig
from mini_llm.model.decoder_lm import DecoderLanguageModel
from mini_llm.train_extended import ExtendedTrainer


def measure(root, interval, steps, warmup):
    model = DecoderLanguageModel(ModelConfig.tiny_inspection(), rng_seed=74, dtype="float32")
    shard = root / "train_shard_00000.bin"
    trainer = ExtendedTrainer(
        model, [str(shard)], [str(shard)], batch_size=1, seq_length=16,
        grad_accum_steps=8, total_steps=1000, warmup_steps=5,
        gradient_noise_monitor=interval > 0,
        gradient_noise_interval=max(1, interval),
        gradient_noise_accum_points="4,8",
        gradient_noise_output=str(root / f"noise_{interval}.csv"),
    )
    for _ in range(warmup):
        trainer.train_step()
    synchronize()
    samples = []
    for _ in range(steps):
        start = time.perf_counter()
        trainer.train_step()
        synchronize()
        samples.append(time.perf_counter() - start)
    pool = getattr(xp, "get_default_memory_pool", None)
    pool_reserved = None if pool is None else pool().total_bytes()
    return statistics.mean(samples), statistics.median(samples), pool_reserved


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--steps", type=int, default=30)
    parser.add_argument("--warmup", type=int, default=4)
    args = parser.parse_args()
    with tempfile.TemporaryDirectory() as td:
        root = Path(td)
        np.random.default_rng(74).integers(0, 256, size=2048, dtype=np.uint16).tofile(
            root / "train_shard_00000.bin"
        )
        results = {}
        for interval in (0, 10, 1):
            avg, med, mem = measure(root, interval, args.steps, args.warmup)
            results[interval] = avg
            print(f"backend={BACKEND_NAME}, interval={interval}, "
                  f"mean={avg * 1000:.3f} ms, median={med * 1000:.3f} ms, "
                  f"device_pool_reserved={mem}")
        for interval in (10, 1):
            print(f"interval={interval}, mean_overhead_percent="
                  f"{100.0 * (results[interval] / results[0] - 1):+.2f}%")


if __name__ == "__main__":
    main()
