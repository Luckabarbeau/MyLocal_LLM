"""CuPy performance benchmark comparing float32 vs float16."""

import time

import numpy as np

from mini_llm.config import ModelConfig
from mini_llm.model.decoder_lm import DecoderLanguageModel
from mini_llm.train import MiniTrainer, load_token_shard


def create_micro_shard(path, num_docs: int = 100, context_len: int = 512):
    """Create a micro-shard for benchmarking."""
    with open(path, "wb") as f:
        import numpy as np
        f.write(np.int64(num_docs).tobytes())
        f.write(np.int64(context_len).tobytes())
        data = np.random.randint(0, 256, size=(num_docs, context_len), dtype=np.uint16)
        f.write(data.tobytes())


def profile_cupy_training(dtype: str, model_config: ModelConfig, num_warmup: int = 5, num_steps: int = 20):
    """Profile training with specified dtype on CuPy."""
    from mini_llm.backend import xp
    import cupy as cp
    
    print(f"\n=== {dtype.upper()} Training ===")
    
    model = DecoderLanguageModel(model_config, rng_seed=42, dtype=dtype)
    
    # Create temporary shard
    import tempfile
    import os
    with tempfile.TemporaryDirectory() as tmpdir:
        shard_path = os.path.join(tmpdir, "train.bin")
        create_micro_shard(shard_path, num_docs=100, context_len=model_config.context_length)
        
        trainer = MiniTrainer(
            model=model,
            shard_paths=[shard_path],
            batch_size=2,
            seq_length=model_config.context_length - 8,
            warmup_steps=5,
            total_steps=num_steps + 10,
            peak_lr=1e-2,
            grad_clip=1.0,
        )
        
        # Warmup (5 steps)
        for _ in range(5):
            trainer.train_step()
        
        # Synchronize before timing
        if dtype == "float16":
            cp.cuda.Stream.null.synchronize()
        else:
            cp.cuda.Stream.null.synchronize()
        
        # Timed run
        start = time.perf_counter()
        losses = trainer.train(num_steps=num_steps, log_interval=num_steps)
        elapsed = time.perf_counter() - start
        
        # Synchronize after timing
        cp.cuda.Stream.null.synchronize()
        
        steps_per_sec = num_steps / elapsed
        tokens_per_sec = steps_per_sec * 2 * (model_config.context_length - 8)
        # Memory info (may not be available in all CuPy versions)
        try:
            pool = cp.cuda.get_default_memory_pool()
            memory_info = pool.used_bytes()
        except Exception:
            memory_info = 0
        
        print(f"\nResults:")
        print(f"  Steps/sec:       {steps_per_sec:.1f}")
        print(f"  ms/step:         {elapsed / num_steps * 1000:.2f}")
        print(f"  Tokens/sec:      {tokens_per_sec:.0f}")
        print(f"  VRAM used:       {memory_info / 1024**2:.1f} MB")
        print(f"  Loss range:      [{losses[0]:.4f}, {losses[-1]:.4f}]")
        
        return {
            "dtype": dtype,
            "steps_per_sec": steps_per_sec,
            "ms_per_step": elapsed / num_steps * 1000,
            "tokens_per_sec": tokens_per_sec,
            "vram_mb": memory_info / 1024**2,
            "loss_range": (losses[0], losses[-1]),
        }


def main():
    print("=" * 70)
    print("CuPy Performance Benchmark: FP32 vs FP16")
    print("=" * 70)
    
    # GPU info
    import cupy as cp
    props = cp.cuda.runtime.getDeviceProperties(0)
    print(f"\nGPU: {props['name'].decode()}")
    print(f"SMs: {props['multiProcessorCount']}")
    print(f"VRAM: {props['totalGlobalMem'] / 1024**3:.1f} GB")
    
    # Use micro model config (same as training examples)
    config = ModelConfig(
        vocab_size=64,
        context_length=64,
        n_layers=1,
        d_model=8,
        n_q_heads=1,
        n_kv_heads=1,
        d_head=8,
        d_ff=16,
        n_experts=2,
        top_k=1,
        init_std=0.1,
    )
    
    print(f"\nModel config:")
    print(f"  Vocab:      {config.vocab_size}")
    print(f"  Context:    {config.context_length}")
    print(f"  Layers:     {config.n_layers}")
    print(f"  d_model:    {config.d_model}")
    print(f"  d_ff:       {config.d_ff}")
    print(f"  Experts:    {config.n_experts}")
    
    # Run benchmarks
    results = {}
    for dtype in ["float32", "float16"]:
        result = profile_cupy_training(dtype, config)
        results[dtype] = result
    
    # Summary
    print("\n" + "=" * 70)
    print("SUMMARY")
    print("=" * 70)
    
    fp32 = results["float32"]
    fp16 = results["float16"]
    
    speedup = fp32["ms_per_step"] / fp16["ms_per_step"]
    memory_ratio = fp16["vram_mb"] / max(fp32["vram_mb"], 0.001)
    
    print(f"\nSpeed Comparison:")
    print(f"  FP32: {fp32['ms_per_step']:.2f} ms/step ({fp32['steps_per_sec']:.1f} steps/sec)")
    print(f"  FP16: {fp16['ms_per_step']:.2f} ms/step ({fp16['steps_per_sec']:.1f} steps/sec)")
    print(f"  Speedup: {speedup:.2f}x")
    
    print(f"\nMemory Comparison:")
    print(f"  FP32 VRAM: {fp32['vram_mb']:.1f} MB")
    print(f"  FP16 VRAM: {fp16['vram_mb']:.1f} MB")
    print(f"  Memory ratio: {memory_ratio:.2f}x ({(1-memory_ratio)*100:.1f}% reduction)")
    
    print("\n" + "=" * 70)
    print("Benchmark complete!")
    print("=" * 70)


if __name__ == "__main__":
    import sys
    sys.exit(main())
