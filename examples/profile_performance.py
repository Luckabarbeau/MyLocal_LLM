"""Profile float16 vs float32 training performance."""

import time
import tempfile
from pathlib import Path

import numpy as np

from mini_llm.config import ModelConfig
from mini_llm.model.decoder_lm import DecoderLanguageModel
from mini_llm.train import MiniTrainer, load_token_shard


def create_micro_shard(path: Path, num_docs: int = 10, context_len: int = 64):
    """Create a tiny micro-shard for profiling."""
    with open(path, "wb") as f:
        f.write(np.int64(num_docs).tobytes())
        f.write(np.int64(context_len).tobytes())
        data = np.tile(
            np.arange(context_len, dtype=np.uint16) % 64,
            (num_docs, 1)
        )
        f.write(data.tobytes())


def profile_training(dtype: str, num_steps: int = 50):
    """Profile training with specified dtype."""
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
    
    model = DecoderLanguageModel(config, rng_seed=42, dtype=dtype)
    
    with tempfile.TemporaryDirectory() as tmpdir:
        shard_path = Path(tmpdir) / "train.bin"
        create_micro_shard(shard_path, num_docs=10, context_len=64)
        
        trainer = MiniTrainer(
            model=model,
            shard_paths=[str(shard_path)],
            batch_size=2,
            seq_length=32,
            warmup_steps=5,
            total_steps=num_steps,
            peak_lr=1e-2,
            grad_clip=1.0,
        )
        
        # Warmup
        trainer.train_step()
        
        # Profile
        start = time.time()
        losses = trainer.train(num_steps=num_steps, log_interval=num_steps)
        elapsed = time.time() - start
        
        steps_per_sec = num_steps / elapsed
        return {
            "dtype": dtype,
            "steps_per_sec": steps_per_sec,
            "time_per_step_ms": elapsed / num_steps * 1000,
            "loss_range": (losses[0], losses[-1]),
        }


def main():
    print("=" * 60)
    print("Performance Profile: float16 vs float32")
    print("=" * 60)
    
    for dtype in ["float16", "float32"]:
        result = profile_training(dtype, num_steps=50)
        print(f"\n{dtype.upper()}:")
        print(f"  Steps/sec:       {result['steps_per_sec']:.1f}")
        print(f"  Time/step:       {result['time_per_step_ms']:.2f} ms")
        print(f"  Loss range:      [{result['loss_range'][0]:.4f}, {result['loss_range'][1]:.4f}]")
    
    print("\n" + "=" * 60)
    print("Profile complete!")
    print("=" * 60)


if __name__ == "__main__":
    import sys
    sys.exit(main())
