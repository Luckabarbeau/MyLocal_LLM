"""0074B observational overhead benchmark (not a 500M GPU result).

Run: MINI_LLM_BACKEND=numpy python benchmarks/benchmark_adamw_update.py --steps 80
Use MINI_LLM_BACKEND=cupy on the RTX workstation for GPU estimates.
"""

import argparse
import statistics
import tempfile
import time
from pathlib import Path

import numpy as np

from mini_llm.backend import BACKEND_NAME, xp, synchronize
from mini_llm.config import ModelConfig
from mini_llm.model.decoder_lm import DecoderLanguageModel
from mini_llm.train_extended import ExtendedTrainer


def benchmark(root, mode, interval, steps, warmup):
    model = DecoderLanguageModel(ModelConfig.tiny_inspection(), rng_seed=74,
                                 dtype="float32")
    trainer = ExtendedTrainer(
        model, [str(root / "train_shard_00000.bin")],
        [str(root / "train_shard_00000.bin")],
        batch_size=1, seq_length=16, grad_accum_steps=8,
        total_steps=max(1000, steps + warmup + 10), warmup_steps=5,
        gradient_noise_monitor=mode >= 1,
        adamw_update_monitor=mode >= 2,
        gradient_noise_interval=interval,
        gradient_noise_accum_points="4,8",
        gradient_noise_output=str(root / f"grad_{mode}.csv"),
        adamw_update_output=str(root / f"adamw_{mode}.csv"),
    )
    for _ in range(warmup):
        trainer.train_step()
    synchronize()
    samples = []
    for _ in range(steps):
        begin = time.perf_counter()
        trainer.train_step()
        synchronize()
        samples.append(time.perf_counter() - begin)
    pool = getattr(xp, "get_default_memory_pool", None)
    memory = None if pool is None else pool().total_bytes()
    return statistics.mean(samples), statistics.median(samples), memory


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--steps", type=int, default=80)
    parser.add_argument("--warmup", type=int, default=5)
    parser.add_argument("--interval", type=int, default=10)
    args = parser.parse_args()
    with tempfile.TemporaryDirectory() as tmp:
        root = Path(tmp)
        np.random.default_rng(74).integers(0, 256, size=2048,
                 dtype=np.uint16).tofile(root / "train_shard_00000.bin")
        results = {}
        for mode in (0, 1, 2):
            mean, median, mem = benchmark(root, mode, args.interval,
                                         args.steps, args.warmup)
            results[mode] = mean
            print(f"backend={BACKEND_NAME} mode={mode} interval={args.interval} "
                  f"mean_ms={1000*mean:.3f} median_ms={1000*median:.3f} "
                  f"memory_pool_reserved={mem}")
        print("mode 0=off, 1=0074A only, 2=0074A + 0074B")
        print(f"74B incremental over 74A: {100 * (results[2]/results[1] - 1):+.2f}%")
        print(f"74B total over unmonitored: {100 * (results[2]/results[0] - 1):+.2f}%")


if __name__ == "__main__":
    main()
