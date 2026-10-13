"""0079E reproducible *small dense* predictive-refinement research experiment.

Examples:
  python -m examples.train_predictive_refinement --steps 40 --task ambiguous
  python -m examples.train_predictive_refinement --block-size 1 --steps 40

This is a synthetic, CPU/short-context research harness, NOT the optimized
production document-first pretraining pipeline. No text from multiple
independent documents is concatenated into any single training example.
"""

import argparse
import time
import numpy as np

from mini_llm.backend import xp, asnumpy, synchronize
from mini_llm.config import ModelConfig
from mini_llm.inference_model import InferenceModel
from mini_llm.model.decoder_lm import DecoderLanguageModel
from mini_llm.optim.adamw import AdamW
from mini_llm.predictive_refinement.checkpoint import (
    load_refinement_checkpoint, save_refinement_checkpoint,
)
from mini_llm.predictive_refinement.training_reference import (
    DenseRefinementTrainingReference, RefinementReferenceConfig,
)
from mini_llm.predictive_refinement.window_training import WindowRefinementTrainer
from mini_llm.predictive_refinement.inference import (
    CachedPredictiveRefinementGenerator, RefinementGenerationConfig,
)


def synthetic_batch(rng, batch_size, context_len, block_size, task, vocab_size=64):
    """Independent complete examples with no cross-document token leakage."""
    total = context_len + block_size
    if task == "cycle":
        phases = rng.integers(0, 8, size=(batch_size, 1))
        sequence = (np.arange(total)[None, :] + phases) % 8 + 2
    elif task == "correlated":
        latent = rng.integers(0, 4, size=(batch_size, 1))
        context = rng.integers(2, 6, size=(batch_size, context_len))
        context[:, :1] = 10 + latent
        future = 20 + latent + (np.arange(block_size)[None, :] % 4) * 4
        sequence = np.concatenate((context, future), axis=1)
    elif task == "ambiguous":
        context = np.full((batch_size, context_len), 7, dtype=np.int64)
        choices = rng.integers(0, 2, size=(batch_size, 1))
        future = 20 + choices * 8 + np.arange(block_size)[None, :]
        sequence = np.concatenate((context, future), axis=1)
    elif task == "random":
        sequence = rng.integers(2, vocab_size, size=(batch_size, total))
    else:
        raise ValueError("unknown synthetic task")
    return xp.asarray(sequence[:, :context_len], dtype=xp.int64), xp.asarray(
        sequence[:, context_len:], dtype=xp.int64
    )


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--task", choices=["cycle", "correlated", "ambiguous", "random"], default="cycle")
    p.add_argument("--steps", type=int, default=40)
    p.add_argument("--batch-size", type=int, default=2)
    p.add_argument("--context-length", type=int, default=8)
    p.add_argument("--block-size", type=int, default=4)
    p.add_argument("--initial-loss-weight", type=float, default=0.1)
    p.add_argument("--learning-rate", type=float, default=0.01)
    p.add_argument("--seed", type=int, default=4)
    p.add_argument("--log-every", type=int, default=10)
    p.add_argument("--save", default=None, help="checkpoint directory")
    p.add_argument("--resume", default=None, help="strict refinement checkpoint directory")
    p.add_argument("--generate", type=int, default=8, help="tokens to generate after training")
    p.add_argument("--shared-context-training", action="store_true",
                   help="0079G: encode the causal context once, reuse its K/V across refinement passes")
    p.add_argument("--window-blocks", type=int, default=1,
                   help="0079H: nonoverlapping target blocks per independent sequence (default 1)")
    p.add_argument("--refinement-group-size", type=int, default=2,
                   help="0079H: maximum simultaneous target blocks per refinement batch")
    args = p.parse_args()
    if args.steps < 0 or args.batch_size < 1 or args.context_length < 1:
        p.error("steps must be >= 0 and batch/context lengths must be positive")
    if args.window_blocks < 1 or not 1 <= args.refinement_group_size <= 16:
        p.error("window-blocks must be positive and refinement-group-size must be [1,16]")
    if args.window_blocks > 1 and not args.shared_context_training:
        p.error("--window-blocks > 1 requires --shared-context-training")
    if args.context_length + args.window_blocks * args.block_size > 256:
        p.error("short-window research mode is limited to 256 source tokens")

    cfg = ModelConfig(
        tokenizer_vocab_size=64, context_length=max(64, args.context_length + args.window_blocks * args.block_size + args.generate + 16),
        n_layers=2, d_model=32, n_q_heads=4, n_kv_heads=2, d_head=8,
        n_experts=2, top_k=1, d_ff=64, dtype="float32",
    )
    model = DecoderLanguageModel(cfg, rng_seed=args.seed, dtype="float32")
    trainer_class = WindowRefinementTrainer if args.window_blocks > 1 else DenseRefinementTrainingReference
    ref = trainer_class(
        model, RefinementReferenceConfig(
            block_size=args.block_size,
            initial_loss_weight=args.initial_loss_weight,
            enable_shared_context_training=args.shared_context_training,
        ), rng_seed=args.seed + 1,
    )
    optimizer = AdamW(ref.parameters(), lr=args.learning_rate, weight_decay=0.0)
    start_step = 0
    supervised_tokens = 0
    provenance = {"task": args.task, "seed": args.seed, "synthetic": True}
    if args.window_blocks > 1:
        provenance["window_blocks"] = args.window_blocks
    if args.resume:
        state = load_refinement_checkpoint(args.resume, ref, optimizer=optimizer)
        saved_provenance = state["architecture"].get("data_provenance")
        if saved_provenance is not None and saved_provenance != provenance:
            raise ValueError("resume dataset/task/seed do not match checkpoint provenance")
        start_step = int(state["step"])
        supervised_tokens = int(state["supervised_tokens"])
    rng = np.random.default_rng(args.seed)
    # Reconstruct the same data stream after resume; this demo uses exactly
    # one independently sampled batch per optimizer step.
    for _ in range(start_step):
        synthetic_batch(rng, args.batch_size, args.context_length,
                        args.block_size * args.window_blocks, args.task)

    start = time.perf_counter()
    for step in range(start_step, args.steps):
        contexts, targets = synthetic_batch(
            rng, args.batch_size, args.context_length,
            args.block_size * args.window_blocks, args.task
        )
        ref.zero_grad()
        if args.window_blocks > 1:
            result = ref.train_window(
                xp.concatenate((contexts, targets), axis=1),
                prefix_length=args.context_length,
                group_size=args.refinement_group_size,
            )
        else:
            result, tape = ref.forward(contexts, targets)
            ref.backward(tape)
        optimizer.step()
        supervised_tokens += result["supervised_tokens"]
        if (step + 1) % args.log_every == 0 or step == start_step:
            synchronize()
            elapsed = max(time.perf_counter() - start, 1e-9)
            recent_tokens = (step + 1 - start_step) * args.batch_size * args.block_size * args.window_blocks
            print(
                f"step={step + 1} initial_ce={float(result['initial_ce']):.5f} "
                f"commit_ce={float(result['commitment_ce']):.5f} "
                f"total={float(result['total_loss']):.5f} "
                f"supervised_tok/s={recent_tokens / elapsed:.1f} "
                f"passes={result.get('full_model_passes', result.get('refinement_passes'))} "
                f"causal_encodes={result['causal_context_encodes']}"
            )
    if args.save:
        save_refinement_checkpoint(
            args.save, ref, optimizer=optimizer, training_step=args.steps,
            supervised_tokens=supervised_tokens,
            data_provenance=provenance,
        )

    if args.generate:
        inference = InferenceModel(model.config, dtype="float32")
        inference.set_weights(model)
        context, _ = synthetic_batch(
            rng, 1, args.context_length, args.block_size, args.task
        )
        state = inference.create_generation_state(1, cfg.context_length)
        inference.prefill(context.astype(xp.int32), state)
        generator = CachedPredictiveRefinementGenerator(
            inference, ref.states, RefinementGenerationConfig(block_size=args.block_size)
        )
        synchronize()
        gen_start = time.perf_counter()
        output = generator.generate(state, args.generate)
        synchronize()
        elapsed = max(time.perf_counter() - gen_start, 1e-9)
        print(
            f"prompt={asnumpy(context[0]).tolist()} "
            f"generated={output['tokens']} full_model_passes={output['passes']} "
            f"cached_gen_tok/s={len(output['tokens']) / elapsed:.2f}"
        )


if __name__ == "__main__":
    main()
