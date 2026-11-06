"""Training CLI adapter for ``train_model.py --backend pytorch``."""

from __future__ import annotations

import json
from pathlib import Path


def run(args, *, config_from_name):
    import torch

    from mini_llm.model_loading import read_native_config
    from mini_llm.tokenizer.tokenizer import FastBPETokenizer, SimpleBPETokenizer
    from .checkpoint import load_checkpoint
    from .model import TorchDecoderLanguageModel
    from .training import TorchTrainer, TorchTrainingRun

    if args.precision == "float16":
        raise ValueError(
            "The PyTorch backend does not use FP16; select --precision bf16-mixed or float32"
        )
    if args.objective not in {None, "ar"}:
        raise NotImplementedError("PyTorch currently supports the autoregressive objective only")
    if args.stream_registry:
        raise NotImplementedError(
            "PyTorch remote streaming source orchestration is not yet implemented"
        )
    if args.init_from or args.block_proposal_init_from_v1:
        raise NotImplementedError("PyTorch CLI supports new runs and strict resume only")
    if args.optimizer not in {None, "hybrid-muon"}:
        raise NotImplementedError("PyTorch CLI currently implements exact Hybrid Muon")
    if not torch.cuda.is_available():
        raise RuntimeError("PyTorch training requires CUDA for the requested BF16 configuration")

    device = torch.device("cuda")
    dtype = torch.bfloat16 if args.precision == "bf16-mixed" else torch.float32
    loaded = None
    if args.resume_from:
        checkpoint = Path(args.resume_from).expanduser().resolve()
        loaded = load_checkpoint(
            checkpoint,
            model_factory=lambda cfg: TorchDecoderLanguageModel(
                cfg, dtype=dtype, device=device
            ),
            device=device, dtype=dtype,
        )
        config = loaded.config
        saved_backend = loaded.training_state.get("backend")
        if saved_backend not in {None, "pytorch"} and not args.allow_nonexact_resume:
            raise ValueError(
                "Changing checkpoint backend to PyTorch is an intentional non-exact resume; "
                "pass --allow-nonexact-resume"
            )
        model, tokenizer = loaded.model, loaded.tokenizer
    else:
        config = config_from_name(args.model)
        model = TorchDecoderLanguageModel(config, dtype=dtype, device=device)
        shard_dir = Path(args.mixed_shard_root or args.shard_dir)
        tokenizer_path = Path(args.tokenizer_path) if args.tokenizer_path else shard_dir / "tokenizer.json"
        if not tokenizer_path.is_file():
            raise FileNotFoundError(f"tokenizer not found: {tokenizer_path}")
        try:
            tokenizer = FastBPETokenizer.load(str(tokenizer_path))
        except Exception:
            tokenizer = SimpleBPETokenizer.load(str(tokenizer_path))

    shard_dir = Path(args.mixed_shard_root or args.shard_dir)
    train_paths = sorted(shard_dir.rglob("train_shard_*.bin"))
    val_paths = sorted(shard_dir.rglob("val_shard_*.bin"))
    if not train_paths or not val_paths:
        legacy = sorted(shard_dir.rglob("shard_*.bin"))
        if len(legacy) < 2:
            raise FileNotFoundError(f"training/validation shards not found in {shard_dir}")
        train_paths, val_paths = legacy[:-1], legacy[-1:]
    token = getattr(tokenizer, "eos_token", None)
    mapping = getattr(tokenizer, "token_to_id", None)
    eos = mapping.get(token) if isinstance(mapping, dict) and token is not None else None
    if eos is None and getattr(tokenizer, "_tokenizer", None) is not None and token is not None:
        eos = tokenizer._tokenizer.token_to_id(token)

    facade = type("TorchFacade", (), {})()
    facade._native_model = model
    facade.tokenizer = tokenizer
    facade.model_id = args.model
    trainer = TorchTrainer(
        model,
        lr=args.peak_lr,
        beta1=0.9 if args.adam_beta1 is None else args.adam_beta1,
        beta2=0.95 if args.adam_beta2 is None else args.adam_beta2,
        eps=1e-8 if args.adam_eps is None else args.adam_eps,
        weight_decay=args.weight_decay,
        muon_momentum=0.95 if args.muon_momentum is None else args.muon_momentum,
        muon_ns_steps=5 if args.muon_ns_steps is None else args.muon_ns_steps,
        muon_matmul_precision=args.muon_matmul_precision or "bf16",
        grad_accum_steps=args.grad_accum_steps,
        max_grad_norm=args.grad_clip,
        optimizer_offload="full",
    )
    if loaded is not None:
        if loaded.optimizer_state is None:
            raise ValueError("resume checkpoint is missing optimizer state")
        trainer.optimizer.load_state_dict(loaded.optimizer_state)
        trainer.step = int(loaded.training_state.get("step", loaded.optimizer_state.get("step", 0)))
        trainer.processed_tokens_total = int(loaded.training_state.get("processed_tokens", 0))
        trainer.supervised_tokens_total = float(
            loaded.training_state.get("cumulative_supervised_tokens", 0.0)
        )

    run = TorchTrainingRun(
        facade, trainer,
        [__import__("mini_llm.data.token_shards", fromlist=["map_token_shard"]).map_token_shard(str(p), args.context_length or config.context_length) for p in train_paths],
        [__import__("mini_llm.data.token_shards", fromlist=["map_token_shard"]).map_token_shard(str(p), args.context_length or config.context_length) for p in val_paths],
        __import__("numpy").random.default_rng(args.seed),
        args.batch_size, int(args.context_length or config.context_length),
        None if eos is None else int(eos),
        None if args.checkpoint_dir is None else Path(args.checkpoint_dir), tokenizer,
    )
    steps = int(args.max_run_steps or args.total_steps)
    print(
        f"PyTorch backend: device={device}, dtype={dtype}, "
        f"parameters={sum(p.numel() for p in model.parameters()):,}, "
        f"train_shards={len(train_paths)}, val_shards={len(val_paths)}, "
        f"sparse_stride={config.dual_resolution.sparse_retrieval_stride}"
    )
    results = run.train_steps(steps, log_interval=args.log_interval)
    if run.checkpoint_dir is not None:
        run.save_checkpoint()
    return results


__all__ = ["run"]
