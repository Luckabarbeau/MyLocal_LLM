"""Regression checks for independent 65K heads and checkpoint conversion."""

import json
import os
import shutil
from types import SimpleNamespace
import tempfile
import unittest
from dataclasses import asdict
from pathlib import Path

import numpy as np

from mini_llm.config import (
    ModelConfig, MemoryContextConfig, AttentionLayerConfig, LocalAttentionConfig,
)
from mini_llm.model.decoder_lm import DecoderLanguageModel
from mini_llm.inference_model import InferenceModel
from examples.migrate_full_history_heads import audit_migration, migrate
from mini_llm.ops.full_history_head import FullHistoryHead
from mini_llm.backend import RandomStream
from mini_llm.checkpoint import load_checkpoint
from mini_llm.optim.hybrid_muon import HybridMuonAdamW, muon_eligible
from mini_llm.parameter import Parameter


def fixture_config(layers=()):
    # Historical reader uses a fixed production schedule (20x12); narrow model
    # and 32-token working sequence make this suitable for CPU regression.
    return ModelConfig(
        tokenizer_vocab_size=128, context_length=32,
        n_layers=20, d_model=192, n_q_heads=12, n_kv_heads=3, d_head=16,
        n_experts=1, top_k=1, d_ff=64, ffn_mode="dense", dtype="float32",
        memory_context=MemoryContextConfig(
            enabled=True, integration_mode="historical_read", memory_length=128,
            recent_length=32, target_length=32, block_size=32,
            top_k_blocks=1, read_heads=2, read_kv_heads=1,
            router_query_length=32,
        ),
        attention_layers=(AttentionLayerConfig(
            heads=(LocalAttentionConfig(window=32),) * 12
        ),) * 20,
        full_history_layers=layers,
    )


class FullHistoryUpgrade(unittest.TestCase):
    def setUp(self):
        self.original = DecoderLanguageModel(fixture_config(), rng_seed=42)
        self.new = DecoderLanguageModel(fixture_config((0, 1, 2)), rng_seed=42)
        self.tokens = np.random.default_rng(15).integers(0, 128, (1, 128), dtype=np.int32)
        self.metadata = {"history_valid_starts": np.array([83], dtype=np.int64)}

    def test_parameter_count_and_exact_initialization(self):
        old = {p.name: p.data for p in self.original.parameters()}
        new = {p.name: p.data for p in self.new.parameters()}
        self.assertEqual(self.new.config.estimated_parameter_count(),
                         sum(p.size for p in self.new.parameters()))
        self.assertEqual(set(old) - set(new), set())
        for name, value in old.items():
            np.testing.assert_array_equal(value, new[name])
        for layer in self.new.config.full_history_layers:
            np.testing.assert_array_equal(
                new[f"blocks.{layer}.full_history.Wo"], 0
            )
        h0 = self.original.forward_body(
            self.tokens, memory_metadata=self.metadata, return_cache=False,
        )
        h1 = self.new.forward_body(
            self.tokens, memory_metadata=self.metadata, return_cache=False,
        )
        np.testing.assert_array_equal(h0, h1)

    def test_nonzero_head_backward_and_prefill_parity(self):
        for layer in self.new.config.full_history_layers:
            head = self.new.blocks[layer].full_history_head
            head.o.data[:] = np.random.default_rng(layer).normal(
                0, 0.02, head.o.data.shape
            )
        output, cache = self.new.forward_body(
            self.tokens, memory_metadata=self.metadata, return_cache=True,
        )
        self.new.backward_body(np.ones_like(output) * 0.01, cache)
        self.assertGreater(np.linalg.norm(
            self.new.blocks[0].full_history_head.q.grad), 0)
        reference = self.new.forward_body(
            self.tokens, memory_metadata=self.metadata, return_cache=False,
        )
        logits = reference[:, -1] @ self.new.output_proj.W.data
        inf = InferenceModel(self.new.config)
        inf.set_weights(self.new)
        state = inf.create_generation_state(batch_size=1, max_length=128)
        # History has 83 left pad rows, then 13 genuine old tokens and 32 current.
        actual = inf.prefill(self.tokens[:, 83:], state)
        np.testing.assert_allclose(actual, logits, rtol=2e-5, atol=2e-5)
        pred = inf.decode_one(np.array([[7]], dtype=np.int32), state)
        self.assertEqual(pred.shape, (1, 128))

    def test_causal_mask_and_invalid_history(self):
        embedding = Parameter(
            np.random.default_rng(100).normal(0, .2, (128, 32)).astype(np.float32),
            name="embedding.W",
        )
        head = FullHistoryHead(
            d_model=32, d_head=16, embedding=embedding, rng=RandomStream(3),
            init_std=.05, rope_base=10000, name="test.full_history",
            dtype="float32",
        )
        head.o.data[:] = np.random.default_rng(102).normal(
            0, .1, head.o.data.shape
        )
        context = np.random.default_rng(5).normal(0, .5, (1, 6, 32)).astype(np.float32)
        position = np.arange(10, 16, dtype=np.int64)[None, :]
        hist = SimpleNamespace(
            ids=np.array([[77, 82, 9, 10]], dtype=np.int32),
            valid=np.array([[False, False, True, True]]),
            positions=np.arange(6, 10, dtype=np.int64),
        )
        y0, _ = head.forward(context, hist, position, return_cache=False)
        # Changing future current rows may never affect earlier predictions.
        modified = context.copy()
        modified[0, 4:] *= -2
        y1, _ = head.forward(modified, hist, position, return_cache=False)
        np.testing.assert_allclose(y0[:, :4], y1[:, :4], atol=1e-7)
        # Left-padded old tokens are never used as keys or values.
        hist2 = SimpleNamespace(
            ids=np.array([[101, 99, 9, 10]], dtype=np.int32),
            valid=hist.valid, positions=hist.positions,
        )
        y2, _ = head.forward(context, hist2, position, return_cache=False)
        np.testing.assert_array_equal(y0, y2)

    def test_empty_and_mixed_validity_masks(self):
        embedding = Parameter(
            np.random.default_rng(201).normal(0, .2, (256, 24)).astype(np.float32),
            name="embedding.W",
        )
        head = FullHistoryHead(
            d_model=24, d_head=8, embedding=embedding, rng=RandomStream(9),
            init_std=.05, rope_base=10000, name="mask.full_history", dtype="float32",
        )
        head.o.data[:] = np.random.default_rng(202).normal(0, .1, head.o.data.shape)
        current = np.random.default_rng(203).normal(0, .3, (3, 6, 24)).astype(np.float32)
        positions = np.broadcast_to(np.arange(10, 16, dtype=np.int64), (3, 6))
        history = SimpleNamespace(
            ids=np.array([[90, 91, 92, 93], [100, 101, 102, 103],
                          [110, 111, 112, 113]], dtype=np.int32),
            valid=np.array([[False, False, False, False],
                            [False, False, False, True],
                            [False, True, True, True]]),
            positions=np.arange(6, 10, dtype=np.int64),
        )
        current_valid = np.array([[True] * 6, [True] * 4 + [False] * 2,
                                  [False] * 6])
        y0, cache = head.forward(
            current, history, positions, current_valid_mask=current_valid,
            return_cache=True,
        )
        self.assertTrue(np.all(np.isfinite(y0)))
        np.testing.assert_array_equal(y0[2], 0)
        np.testing.assert_array_equal(y0[1, 4:], 0)
        changed = current.copy()
        changed[1, 4:] = 1e4
        history_changed = SimpleNamespace(
            ids=history.ids.copy(), valid=history.valid, positions=history.positions,
        )
        history_changed.ids[0, :] = [1, 2, 3, 4]
        history_changed.ids[1, :3] = [5, 6, 7]
        y1, _ = head.forward(
            changed, history_changed, positions, current_valid_mask=current_valid,
            return_cache=False,
        )
        np.testing.assert_array_equal(y0[:2, :4], y1[:2, :4])
        embedding.zero_grad()
        dx = head.backward(np.ones_like(y0), cache)
        np.testing.assert_array_equal(dx[2], 0)
        np.testing.assert_array_equal(dx[1, 4:], 0)
        for token in (90, 91, 92, 93, 100, 101, 102):
            np.testing.assert_array_equal(embedding.grad[token], 0)

    def test_document_mask_and_fully_inactive_query(self):
        embedding = Parameter(
            np.random.default_rng(211).normal(0, .2, (64, 16)).astype(np.float32),
            name="embedding.W",
        )
        head = FullHistoryHead(
            d_model=16, d_head=8, embedding=embedding, rng=RandomStream(11),
            init_std=.05, rope_base=10000, name="docs.full_history", dtype="float32",
        )
        head.o.data[:] = np.random.default_rng(212).normal(0, .1, head.o.data.shape)
        current = np.random.default_rng(213).normal(0, .3, (1, 5, 16)).astype(np.float32)
        positions = np.arange(20, 25, dtype=np.int64)[None]
        history = SimpleNamespace(
            ids=np.array([[7, 8]], dtype=np.int32),
            valid=np.array([[True, True]]), positions=np.array([18, 19], dtype=np.int64),
        )
        # A occupies current rows 0..1 and history slot 0; B occupies rows 2..3
        # and history slot 1. Row 4 is fully inactive.
        allow = np.zeros((1, 5, 5), dtype=bool)
        allow[0, 0, 0] = True
        allow[0, 1, :2] = True
        allow[0, 2, 2] = True
        allow[0, 3, 2:4] = True
        history_allow = np.zeros((1, 5, 2), dtype=bool)
        history_allow[0, :2, 0] = True
        history_allow[0, 2:4, 1] = True
        valid = np.array([[True, True, True, True, False]])
        y0, _ = head.forward(
            current, history, positions, attention_allow_mask=allow,
            current_valid_mask=valid, historical_allow_mask=history_allow,
            return_cache=False,
        )
        modified = current.copy()
        modified[0, :2] *= -20
        original = embedding.data[7].copy()
        embedding.data[7] *= 50
        try:
            y1, _ = head.forward(
                modified, history, positions, attention_allow_mask=allow,
                current_valid_mask=valid, historical_allow_mask=history_allow,
                return_cache=False,
            )
        finally:
            embedding.data[7] = original
        np.testing.assert_allclose(y0[:, 2:4], y1[:, 2:4], atol=1e-7)
        np.testing.assert_array_equal(y0[:, 4], 0)

    def test_maximum_history_capacity_single_query(self):
        embedding = Parameter(
            np.random.default_rng(221).normal(0, .02, (32, 8)).astype(np.float32),
            name="embedding.W",
        )
        head = FullHistoryHead(
            d_model=8, d_head=4, embedding=embedding, rng=RandomStream(12),
            init_std=.02, rope_base=10000, name="max.full_history", dtype="float32",
        )
        head.o.data[:] = .01
        capacity = 61_440
        history = SimpleNamespace(
            ids=np.arange(capacity, dtype=np.int32)[None] % 32,
            valid=np.ones((1, capacity), dtype=bool),
            positions=np.arange(capacity, dtype=np.int64),
        )
        current = np.ones((1, 1, 8), dtype=np.float32)
        y, _ = head.forward(
            current, history, np.array([[capacity]], dtype=np.int64),
            return_cache=False,
        )
        self.assertTrue(np.all(np.isfinite(y)))

    def test_current_input_gradient_finite_difference(self):
        embedding = Parameter(
            np.random.default_rng(231).normal(0, .1, (16, 6)).astype(np.float32),
            name="embedding.W",
        )
        head = FullHistoryHead(
            d_model=6, d_head=4, embedding=embedding, rng=RandomStream(13),
            init_std=.04, rope_base=10000, name="grad.full_history", dtype="float32",
        )
        head.o.data[:] = np.random.default_rng(232).normal(0, .05, head.o.data.shape)
        current = np.random.default_rng(233).normal(0, .2, (1, 3, 6)).astype(np.float32)
        positions = np.arange(3, dtype=np.int64)[None]
        y, cache = head.forward(current, None, positions, return_cache=True)
        direction = np.random.default_rng(234).normal(0, .2, y.shape).astype(np.float32)
        dx = head.backward(direction, cache)
        epsilon = 1e-3
        plus = current.copy()
        minus = current.copy()
        plus[0, 1, 2] += epsilon
        minus[0, 1, 2] -= epsilon
        yp, _ = head.forward(plus, None, positions, return_cache=False)
        ym, _ = head.forward(minus, None, positions, return_cache=False)
        numerical = float(np.sum((yp - ym) * direction) / (2 * epsilon))
        self.assertAlmostEqual(float(dx[0, 1, 2]), numerical, delta=2e-4)

    def test_activation_checkpoint_gradient_parity(self):
        new2 = DecoderLanguageModel(self.new.config, rng_seed=42)
        for m in (self.new, new2):
            for layer in m.config.full_history_layers:
                head = m.blocks[layer].full_history_head
                head.o.data[:] = np.random.default_rng(layer).normal(
                    0, .02, head.o.data.shape
                )
        plain, plain_cache = self.new.forward_body(
            self.tokens, memory_metadata=self.metadata, return_cache=True,
        )
        replay, replay_cache = new2.forward_body(
            self.tokens, memory_metadata=self.metadata, return_cache=True,
            activation_checkpoint=True,
        )
        np.testing.assert_array_equal(plain, replay)
        grad = np.ones_like(plain) * 0.01
        self.new.backward_body(grad, plain_cache)
        new2.backward_body(grad, replay_cache)
        for a, b in zip(self.new.parameters(), new2.parameters()):
            np.testing.assert_allclose(a.grad, b.grad, atol=1e-5, rtol=1e-5,
                                       err_msg=a.name)

    def test_short_horizon_inference_uses_full_history_state(self):
        for layer in self.new.config.full_history_layers:
            head = self.new.blocks[layer].full_history_head
            head.o.data[:] = np.random.default_rng(300 + layer).normal(
                0, .02, head.o.data.shape
            )
        ids = self.tokens[:, -12:]
        source = np.zeros((1, 128), dtype=np.int32)
        source[:, 96:108] = ids
        reference = self.new.forward_body(
            source,
            memory_metadata={
                "history_valid_starts": np.array([96]),
                "target_valid_lengths": np.array([12]),
            },
            return_cache=False,
        )
        logits = reference[:, 11] @ self.new.output_proj.W.data
        inference = InferenceModel(self.new.config)
        inference.set_weights(self.new)
        state = inference.create_generation_state(batch_size=1, max_length=16)
        self.assertTrue(state.long_memory_enabled)
        actual = inference.prefill(ids, state)
        np.testing.assert_allclose(actual, logits, atol=2e-5, rtol=2e-5)

    def test_cached_decode_refresh_parity(self):
        cfg = self.new.config
        for layer in cfg.full_history_layers:
            head = self.new.blocks[layer].full_history_head
            head.o.data[:] = np.random.default_rng(layer).normal(
                0, .02, head.o.data.shape
            )
        old = os.environ.get("MINI_LLM_HISTORICAL_REFRESH_TOKENS")
        os.environ["MINI_LLM_HISTORICAL_REFRESH_TOKENS"] = "1"
        try:
            inf = InferenceModel(cfg)
            inf.set_weights(self.new)
            ids = np.random.default_rng(33).integers(0, 128, (1, 46), dtype=np.int32)
            state_a = inf.create_generation_state(batch_size=1, max_length=128)
            inf.prefill(ids[:, :-1], state_a)
            incremental = inf.decode_one(ids[:, -1:], state_a)
            state_b = inf.create_generation_state(batch_size=1, max_length=128)
            full = inf.prefill(ids, state_b)
            np.testing.assert_allclose(incremental, full, atol=2e-5, rtol=2e-5)
        finally:
            if old is None:
                os.environ.pop("MINI_LLM_HISTORICAL_REFRESH_TOKENS", None)
            else:
                os.environ["MINI_LLM_HISTORICAL_REFRESH_TOKENS"] = old

    def test_checkpoint_migration(self):
        with tempfile.TemporaryDirectory() as folder:
            src, dst = Path(folder) / "old", Path(folder) / "converted"
            (src / "model_params").mkdir(parents=True)
            for component in ("master_weights", "muon_m", "m", "v"):
                (src / "optimizer_state" / component).mkdir(parents=True)
            with (src / "config.json").open("w") as f:
                json.dump(asdict(fixture_config()), f)
            muon_names = sorted(
                p.name for p in self.original.parameters() if muon_eligible(p)
            )
            with (src / "training_state.json").open("w") as f:
                json.dump({
                    "step": 16400,
                    "optimizer_kind": "hybrid-muon",
                    "optimizer_parameter_birth_steps": {
                        p.name: 0 for p in self.original.parameters()
                        if not muon_eligible(p)
                    },
                    "muon_hyperparameters": {
                        "momentum": 0.95, "ns_steps": 5,
                        "matmul_precision": "bf16", "group_names": muon_names,
                    },
                }, f)
            with (src / "optimizer_state" / "scalars.json").open("w") as f:
                json.dump({"step": 16400}, f)
            for p in self.original.parameters():
                np.save(src / "model_params" / f"{p.name}.npy", p.data)
                np.save(src / "optimizer_state" / "master_weights" / f"{p.name}.npy",
                        p.data.astype(np.float32))
                component_names = ("muon_m",) if muon_eligible(p) else ("m", "v")
                for component in component_names:
                    np.save(src / "optimizer_state" / component / f"{p.name}.npy",
                            np.zeros(p.data.shape, np.float32))
            migrate(src, dst, layers=(1, 2, 3))
            with (dst / "config.json").open() as f:
                converted = ModelConfig(**json.load(f))
            self.assertEqual(converted.full_history_layers, (0, 1, 2))
            self.assertEqual(converted.estimated_parameter_count(),
                             sum(p.size for p in self.new.parameters()))
            with (dst / "training_state.json").open() as f:
                self.assertEqual(json.load(f)["step"], 16400)
            for p in self.original.parameters():
                np.testing.assert_array_equal(
                    np.load(src / "model_params" / f"{p.name}.npy"),
                    np.load(dst / "model_params" / f"{p.name}.npy"),
                )
            report = audit_migration(src, dst)
            self.assertEqual(report["training_step"], 16400)
            self.assertEqual(report["added_tensors"], 12)
            with (dst / "training_state.json").open() as f:
                migrated_state = json.load(f)
            for layer in range(3):
                name = f"blocks.{layer}.full_history.Wo.npy"
                np.testing.assert_array_equal(np.load(dst / "model_params" / name), 0)
                self.assertIn(name[:-4], migrated_state["muon_hyperparameters"]["group_names"])
                self.assertTrue((dst / "optimizer_state" / "muon_m" / name).exists())
                self.assertTrue((dst / "optimizer_state" / "master_weights" / name).exists())

            loaded_params, optimizer_state, loaded_training = load_checkpoint(dst)
            resumed = DecoderLanguageModel(converted, rng_seed=99)
            for parameter in resumed.parameters():
                parameter.data[...] = loaded_params[parameter.name]
            optimizer = HybridMuonAdamW(
                resumed.parameters(), lr=1e-4, beta1=.9, beta2=.95, eps=1e-8,
                weight_decay=.1, muon_momentum=.95, muon_ns_steps=5,
                muon_matmul_precision="fp32",
            )
            self.assertEqual(
                set(loaded_training["muon_hyperparameters"]["group_names"]),
                optimizer.muon_names,
            )
            masters, adam_count, muon_count = optimizer.restore_checkpoint(
                optimizer_state, from_hybrid=True,
            )
            self.assertEqual(masters, len(resumed.parameters()))
            self.assertEqual(adam_count, len(optimizer.adam.parameters))
            self.assertEqual(muon_count, len(optimizer.muon.parameters))
            new_output = next(
                p for p in resumed.parameters()
                if p.name == "blocks.0.full_history.Wo"
            )
            before = new_output.data.copy()
            new_output.grad[...] = .01
            optimizer.step(lr=1e-4)
            self.assertFalse(np.array_equal(before, new_output.data))
            adam_src, adam_dst = Path(folder) / "old_adam", Path(folder) / "converted_adam"
            shutil.copytree(src, adam_src)
            shutil.rmtree(adam_src / "optimizer_state" / "muon_m")
            adam_state = json.loads((adam_src / "training_state.json").read_text())
            adam_state["optimizer_kind"] = "adamw"
            adam_state["muon_hyperparameters"] = None
            adam_state["optimizer_parameter_birth_steps"] = {
                p.name: 0 for p in self.original.parameters()
            }
            (adam_src / "training_state.json").write_text(json.dumps(adam_state))
            for p in self.original.parameters():
                for component in ("m", "v"):
                    np.save(adam_src / "optimizer_state" / component / f"{p.name}.npy",
                            np.zeros(p.data.shape, np.float32))
            migrate(adam_src, adam_dst, layers=(1,))
            adam_report = audit_migration(adam_src, adam_dst)
            self.assertEqual(adam_report["optimizer_kind"], "adamw")
            migrated_adam = json.loads((adam_dst / "training_state.json").read_text())
            for suffix in ("Wq", "Wk", "Wv", "Wo"):
                self.assertEqual(
                    migrated_adam["optimizer_parameter_birth_steps"][
                        f"blocks.0.full_history.{suffix}"
                    ], 16400,
                )
            with self.assertRaises(FileExistsError):
                migrate(src, dst, layers=(1, 2, 3))


if __name__ == "__main__":
    unittest.main()
