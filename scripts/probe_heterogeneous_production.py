#!/usr/bin/env python3
"""Small native BF16 CUDA/HIP production-bridge probe using only scratch data."""

from dataclasses import replace
import json
import math
import os
from pathlib import Path
import sys
from tempfile import TemporaryDirectory


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))


def main():
    smoke = json.loads((ROOT / "configs/heterogeneous_smoke.json").read_text())
    cuda_worker = next(w for w in smoke["workers"] if w["expected_runtime"] == "cuda")
    amd_worker = next(w for w in smoke["workers"] if w["expected_runtime"] == "hip")
    common_env = {**smoke["env"], "MINI_LLM_LM_HEAD_CHUNK_TOKENS": "16"}
    os.environ.update(common_env)
    os.environ.update(cuda_worker.get("env", {}))
    os.environ["MINI_LLM_BACKEND"] = "cupy"

    # The numerical profile must be installed before importing native modules.
    import numpy as np
    from mini_llm.backend import BACKEND_NAME, synchronize, validate_bfloat16_backend, xp
    from mini_llm.checkpoint import load_checkpoint
    from mini_llm.heterogeneous_production import ProductionPeerSession
    from mini_llm.model.decoder_lm import DecoderLanguageModel
    from mini_llm.model_registry import registry
    from mini_llm.train_extended import ExtendedTrainer

    assert BACKEND_NAME == "cupy", BACKEND_NAME
    assert not xp.cuda.runtime.is_hip, "owner must use native CUDA"
    xp.cuda.Device(int(cuda_worker.get("device", 0))).use()
    validate_bfloat16_backend()
    base = registry.config("model_b_348m_sparse16")
    config = replace(
        base, dtype="bfloat16", tokenizer_vocab_size=64, context_length=512,
        n_layers=2, d_ff=128,
        dual_resolution=replace(base.dual_resolution, sparse_retrieval_layers=(0, 1)),
    )
    assert (config.n_q_heads, config.n_kv_heads, config.d_head) == (16, 4, 64)
    assert config.dual_resolution == replace(
        base.dual_resolution, sparse_retrieval_layers=(0, 1)
    ), "production attention/pooling geometry must remain unchanged"

    with TemporaryDirectory(prefix="probe-heterogeneous-production-", dir="/tmp") as scratch:
        directory = Path(scratch)
        rng = np.random.default_rng(42)
        shards = []
        for split in ("train", "val"):
            path = directory / f"{split}_shard_00000.bin"
            tokens = rng.integers(0, 63, size=4096, dtype=np.uint16)
            tokens[511::512] = 63
            tokens.tofile(path)
            shards.append(str(path))

        peer_config = {
            "python": str(ROOT / amd_worker["python"]),
            "backend": "cupy", "expected_runtime": "hip",
            "device": int(amd_worker.get("device", 0)),
            "env": {**common_env, **amd_worker.get("env", {})},
            "context_length": 128, "batch_size": 1, "max_microbatches": 2,
            "warmup_microbatches": 2, "timeout_s": 120, "seed": 104729,
            "work_dir": scratch, "log_file": str(directory / "peer.log"),
        }
        peer_path = directory / "peer.json"
        peer_path.write_text(json.dumps(peer_config))
        session = None
        trainer = None
        try:
            model = DecoderLanguageModel(config, rng_seed=42, dtype="bfloat16")
            trainer = ExtendedTrainer(
                model, train_shard_paths=[shards[0]], val_shard_paths=[shards[1]],
                batch_size=1, seq_length=256, grad_accum_steps=2,
                gradient_accumulation_weighting="supervised_tokens",
                optimizer_kind="hybrid-muon", eos_token_id=63, rng_seed=42,
                total_steps=2000, warmup_steps=0, grad_clip=1.0,
                checkpoint_dir=str(directory / "checkpoint"), log_file=None,
            )
            assert trainer.lm_head_chunk_tokens == 16
            session = ProductionPeerSession(
                trainer, config_path=peer_path, training_state={},
                acknowledge_fork=True, tokenizer_path=None,
            )
            trainer.accumulation_peer = session
            loss, global_norm = trainer.train_step()
            trainer.save()
            synchronize()
            assert trainer.step == trainer.optimizer.step_index == 1
            assert math.isfinite(float(loss)), f"nonfinite loss: {loss}"
            assert math.isfinite(float(global_norm)), f"nonfinite global norm: {global_norm}"
            for parameter in model.parameters():
                assert bool(xp.isfinite(parameter.data).all().item()), parameter.name
            assert session.last_report["microbatches"] == 2, session.last_report
            assert session.process.poll() is None, "peer exited unexpectedly"

            _, optimizer_state, saved = load_checkpoint(
                trainer.checkpoint_dir, param_names=[p.name for p in model.parameters()]
            )
            assert saved["step"] == optimizer_state["step"] == 1
            heterogeneous = saved["heterogeneous_state"]
            assert heterogeneous["step"] == 1
            peer_sampler = heterogeneous["peer_sampler"]
            peer_tokens = float(session.last_report["supervised_tokens"])
            global_tokens = float(trainer._last_step_supervised_tokens)
            owner_tokens = global_tokens - peer_tokens
            assert peer_tokens > 0 and owner_tokens > 0
            source_tokens = sum(row["supervised_tokens"] for row in trainer.source_metrics.values())
            saved_source_tokens = sum(
                row["supervised_tokens"] for row in saved["source_metrics"]["sources"].values()
            )
            assert source_tokens == saved_source_tokens == global_tokens
            assert global_tokens >= owner_tokens
            assert sum(peer_sampler["train_source_supervised_tokens"].values()) == peer_tokens
            print(
                f"Verified BF16 update/checkpoint: loss={float(loss):.6f} "
                f"global_norm={float(global_norm):.6f} owner_tokens={owner_tokens:.0f} "
                f"peer_tokens={peer_tokens:.0f} global_source_tokens={source_tokens:.0f} "
                f"peer_sample_trace_microbatches={peer_sampler['sample_trace_microbatches']}",
                flush=True,
            )
            assert peer_sampler["tokens_processed"] == session.last_report["executed_tokens"]
            initial_rng = np.random.default_rng(peer_config["seed"]).bit_generator.state
            assert peer_sampler["train_rng_state"] != initial_rng, "peer sampling RNG did not advance"
            assert session.state_dict(trainer)["peer_sampler"] == peer_sampler
            print("PASS small native productionbridge", flush=True)
        except BaseException:
            peer_log = directory / "peer.log"
            if peer_log.exists():
                print("--- native HIP peer log ---", file=sys.stderr, flush=True)
                print(peer_log.read_text(errors="replace"), file=sys.stderr, flush=True)
            raise
        finally:
            if session is not None:
                try:
                    session.close()
                finally:
                    trainer.accumulation_peer = None


if __name__ == "__main__":
    main()
