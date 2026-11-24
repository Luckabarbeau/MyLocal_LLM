"""Model B's attention branch mode is persisted architecture, not ambient state.

Model B runs one of two local/global branch topologies:

* ``legacy``: the global branch consumes ``residual + local_delta`` (the
  historical sequential graph under which pre-fork weights were trained);
* ``parallel``: both branches consume the shared pre-attention normalized
  input (the canonical architecture for new Model B runs).

These tests pin the contract: the mode lives in the configuration (and
therefore in every new ``config.json``), a checkpoint that predates the fork
deserializes to ``legacy``, an explicit override may fork a legacy checkpoint
onto the parallel graph and that choice must then survive in the checkpoint
itself without any environment variable, and the environment is never a
process-global source of architecture.
"""

import json
import os
from dataclasses import asdict, replace

import numpy as np
import pytest

from mini_llm.backend import BACKEND_NAME
from mini_llm.checkpoint import (
    initialize_matching_parameters,
    load_checkpoint,
    save_checkpoint,
)
from mini_llm.config import (
    ATTENTION_BRANCH_MODE_ENV,
    DualResolutionConfig,
    ModelConfig,
    attention_branch_mode_override,
    resolve_attention_branch_mode,
)
from mini_llm.inference_model import InferenceModel
from mini_llm.model.decoder_lm import DecoderLanguageModel
from mini_llm.model_loading import read_native_config
from mini_llm.ops.dual_resolution_attention import build_dual_resolution_metadata


pytestmark = pytest.mark.skipif(
    BACKEND_NAME != "numpy", reason="CPU reference contracts; GPU graph paths "
    "are covered by tests/test_native_dual_cuda_graph_decode.py"
)


def _tiny_dual(mode):
    """Tiny-but-canonical Model B matrix-pool geometry for both topologies."""
    base = ModelConfig.dual_resolution_matrix_pool_1024_20l().dual_resolution
    return replace(
        base,
        block_size=4, local_window=8, long_local_window=8, long_local_heads=1,
        compression_dim=8, compression_heads=2, summaries_per_block=4,
        global_heads=2, global_head_dim=4, max_blocks=8, global_query_chunk=8,
        attention_branch_mode=mode,
    )


def _tiny_config(mode, dtype="float32"):
    base = ModelConfig.dual_resolution_matrix_pool_1024_20l()
    return replace(
        base,
        tokenizer_vocab_size=31, context_length=12, n_layers=2, d_model=16,
        d_ff=32, n_q_heads=4, n_kv_heads=2, d_head=4,
        dual_resolution=_tiny_dual(mode), dtype=dtype,
    )


def _prefork_checkpoint(tmp_path, name="checkpoint-prefork"):
    """A pre-fork Model B checkpoint: config.json without the branch key."""
    config = _tiny_config("legacy")
    model = DecoderLanguageModel(config, rng_seed=313, dtype="float32")
    path = tmp_path / name
    raw = asdict(config)
    raw["dual_resolution"].pop("attention_branch_mode")
    path.mkdir(parents=True)
    (path / "config.json").write_text(json.dumps(raw))
    save_checkpoint(path, {p.name: p.data for p in model.parameters()})
    return path, config


def _residual_sensitivity(operator, seed=5):
    """Output change when only the residual input changes (legacy couples them)."""
    rng = np.random.default_rng(seed)
    x = rng.normal(scale=0.35, size=(1, 12, 16))
    residual = rng.normal(scale=0.5, size=x.shape)
    metadata = build_dual_resolution_metadata(
        np.arange(12, dtype=np.int64)[None, :], eos_token_id=99, block_size=4,
    )
    first = operator.forward(
        x, residual_input=residual, dual_resolution_metadata=metadata,
        return_cache=False,
    )
    second = operator.forward(
        x, residual_input=residual * 9.0 + 1.0,
        dual_resolution_metadata=metadata, return_cache=False,
    )
    return first, second


def test_branch_mode_is_validated_and_overridable_without_ambient_state():
    with pytest.raises(ValueError, match="attention_branch_mode"):
        DualResolutionConfig(enabled=True, attention_branch_mode="sequential")
    with pytest.raises(TypeError):
        DualResolutionConfig(enabled=True, attention_branch_mode=None)  # type: ignore[arg-type]
    assert attention_branch_mode_override({}) is None
    assert attention_branch_mode_override({ATTENTION_BRANCH_MODE_ENV: "  "}) is None
    assert attention_branch_mode_override({ATTENTION_BRANCH_MODE_ENV: "1"}) == "parallel"
    assert attention_branch_mode_override({ATTENTION_BRANCH_MODE_ENV: "0"}) == "legacy"
    with pytest.raises(ValueError, match=ATTENTION_BRANCH_MODE_ENV):
        attention_branch_mode_override({ATTENTION_BRANCH_MODE_ENV: "sometimes"})
    # An explicit override travels through the configuration, not os.environ,
    # and resolution is idempotent on an already-resolved configuration.
    preset = ModelConfig.dual_resolution_matrix_pool_1024_20l()
    assert preset.dual_resolution.attention_branch_mode == "parallel"
    downgraded = resolve_attention_branch_mode(
        preset, environ={ATTENTION_BRANCH_MODE_ENV: "0"}
    )
    assert downgraded.dual_resolution.attention_branch_mode == "legacy"
    assert resolve_attention_branch_mode(downgraded, environ={}) is downgraded
    assert ATTENTION_BRANCH_MODE_ENV not in os.environ


def test_canonical_model_b_presets_select_parallel(tmp_path, monkeypatch):
    """Spec test B: new Model B configurations use the parallel graph."""
    monkeypatch.delenv(ATTENTION_BRANCH_MODE_ENV, raising=False)
    dense = ModelConfig.dual_resolution_matrix_pool_1024_20l()
    sparse = ModelConfig.dual_resolution_matrix_pool_sparse_1024_20l()
    assert dense.dual_resolution.attention_branch_mode == "parallel"
    assert sparse.dual_resolution.attention_branch_mode == "parallel"
    # Model A (transformer-compressed dual-resolution) is a different model and
    # deliberately keeps the legacy default.
    assert ModelConfig.dual_resolution_1024_20l().dual_resolution.attention_branch_mode == "legacy"
    # Canonical identity is unchanged by the fork: registry id, parameter
    # geometry, and the architecture contract all still construct identically.
    from mini_llm.model_contracts import ArchitectureSpec
    from mini_llm.model_registry import registry

    config = registry.config("model_b_348m_sparse16")
    assert config == sparse
    assert config.dual_resolution.attention_branch_mode == "parallel"
    assert config.estimated_parameter_count() == 348_261_376
    assert ArchitectureSpec.from_config(config) is not None


def test_prefork_checkpoint_keeps_legacy_architecture(tmp_path, monkeypatch, capsys):
    """Spec test A: absent key => legacy, and legacy semantics actually run."""
    monkeypatch.delenv(ATTENTION_BRANCH_MODE_ENV, raising=False)
    path, saved = _prefork_checkpoint(tmp_path)
    original_bytes = (path / "config.json").read_bytes()
    loaded = read_native_config(path)
    assert loaded.dual_resolution.attention_branch_mode == "legacy"
    assert saved.dual_resolution.attention_branch_mode == "legacy"
    notice = capsys.readouterr().out
    assert "predates the attention-branch fork" in notice
    model = DecoderLanguageModel(loaded, rng_seed=99, dtype="float32")
    loaded_names, missing, elements = initialize_matching_parameters(
        path, model.parameters()
    )
    assert not missing and len(loaded_names) > 0 and elements > 0
    operators = [block.attention for block in model.blocks]
    assert all(op.attention_branch_mode == "legacy" and not op.parallel_branches
               for op in operators)
    first, second = _residual_sensitivity(operators[0])
    assert np.max(np.abs(first - second)) > 1e-8
    # Reading a checkpoint never rewrites it.
    assert (path / "config.json").read_bytes() == original_bytes


def test_explicit_migration_forks_legacy_checkpoint_onto_parallel(
    tmp_path, monkeypatch, capsys
):
    """Spec test C + G: legacy ckpt + override => real parallel training graph,
    recorded provenance, and a correct env-free reload afterwards."""
    path, _saved = _prefork_checkpoint(tmp_path)
    original_bytes = (path / "config.json").read_bytes()

    monkeypatch.setenv(ATTENTION_BRANCH_MODE_ENV, "1")
    config = read_native_config(path)
    # The override is resolved during deserialization: the effective
    # configuration reports the graph the model will actually execute.
    assert config.dual_resolution.attention_branch_mode == "parallel"
    assert resolve_attention_branch_mode(config) is config
    notice = capsys.readouterr().out
    assert "Legacy Model B checkpoint detected" in notice
    assert "Original attention architecture: legacy" in notice
    assert "Selected attention architecture: parallel" in notice
    assert "intentional architectural fork" in notice

    model = DecoderLanguageModel(config, rng_seed=404, dtype="float32")
    _, missing, _ = initialize_matching_parameters(path, model.parameters())
    assert not missing
    assert all(block.attention.parallel_branches for block in model.blocks)
    first, second = _residual_sensitivity(model.blocks[0].attention)
    np.testing.assert_array_equal(first, second)

    # A continuation run saves config.json from dataclasses.asdict(config);
    # the resolved architecture must be in that payload.
    resumed = tmp_path / "checkpoint-parallel"
    raw = asdict(model.config)
    assert raw["dual_resolution"]["attention_branch_mode"] == "parallel"
    resumed.mkdir()
    (resumed / "config.json").write_text(json.dumps(raw))
    save_checkpoint(resumed, {p.name: p.data for p in model.parameters()})

    # The original checkpoint is never modified by a migration.
    assert (path / "config.json").read_bytes() == original_bytes

    # Compatible weights survived the architectural switch unchanged.
    original_params, _, _ = load_checkpoint(path)
    resumed_params, _, _ = load_checkpoint(resumed)
    assert set(original_params) == set(resumed_params)
    for name, array in original_params.items():
        np.testing.assert_array_equal(array, resumed_params[name])

    # Essential: after clearing the environment override the new checkpoint
    # reloads as parallel from its own recorded configuration.
    monkeypatch.delenv(ATTENTION_BRANCH_MODE_ENV)
    reloaded = read_native_config(resumed)
    assert reloaded.dual_resolution.attention_branch_mode == "parallel"
    followup = DecoderLanguageModel(reloaded, rng_seed=505, dtype="float32")
    assert all(block.attention.parallel_branches for block in followup.blocks)


@pytest.mark.parametrize("mode", ["legacy", "parallel"])
def test_checkpoint_roundtrip_preserves_attention_semantics(tmp_path, monkeypatch, mode):
    """Spec test D: save -> load -> same mode -> bitwise-same forward."""
    monkeypatch.delenv(ATTENTION_BRANCH_MODE_ENV, raising=False)
    config = _tiny_config(mode, dtype="float64")
    model = DecoderLanguageModel(config, rng_seed=21, dtype="float64")
    tokens = np.random.default_rng(22).integers(0, 31, size=(1, 12))
    reference, _ = model.forward(tokens)

    path = tmp_path / f"checkpoint-{mode}"
    path.mkdir()
    (path / "config.json").write_text(json.dumps(asdict(config)))
    save_checkpoint(path, {p.name: p.data for p in model.parameters()})

    loaded = read_native_config(path)
    assert loaded == config
    assert loaded.dual_resolution.attention_branch_mode == mode
    restored = DecoderLanguageModel(loaded, rng_seed=777, dtype="float64")
    _, missing, elements = initialize_matching_parameters(path, restored.parameters())
    assert not missing and elements > 0
    logits, _ = restored.forward(tokens)
    np.testing.assert_array_equal(logits, reference)


def test_branch_mode_change_alters_model_execution(tmp_path, monkeypatch):
    """The field is wired into the native model, not just the config."""
    monkeypatch.delenv(ATTENTION_BRANCH_MODE_ENV, raising=False)
    tokens = np.random.default_rng(23).integers(0, 31, size=(1, 12))
    outputs = {}
    for mode in ("legacy", "parallel"):
        model = DecoderLanguageModel(_tiny_config(mode, dtype="float64"),
                                     rng_seed=24, dtype="float64")
        logits, _ = model.forward(tokens)
        outputs[mode] = logits
    # 12 tokens over 4-token blocks activate the historical branch, which is
    # the only place the two topologies differ.
    assert np.max(np.abs(outputs["legacy"] - outputs["parallel"])) > 1e-4


@pytest.mark.parametrize("mode", ["legacy", "parallel"])
def test_incremental_decode_matches_full_forward_in_both_modes(monkeypatch, mode):
    """Spec test E (native side): training path vs incremental inference."""
    monkeypatch.delenv(ATTENTION_BRANCH_MODE_ENV, raising=False)
    monkeypatch.setenv("MINI_LLM_DUAL_RESOLUTION_FAST_PROJECTIONS", "0")
    config = _tiny_config(mode, dtype="float64")
    training = DecoderLanguageModel(config, rng_seed=1025, dtype="float64")
    tokens = np.random.default_rng(1026).integers(
        0, config.vocab_size, size=(1, config.context_length), dtype=np.int32
    )
    reference, _ = training.forward(tokens)

    engine = InferenceModel(config, dtype="float64")
    engine.set_weights(training)
    assert engine.blocks[0].attention.attention_branch_mode == mode
    state = engine.create_generation_state(1, config.context_length)
    crossed_history = False
    for position in range(tokens.shape[1]):
        logits = engine.decode_one(tokens[:, position], state)
        np.testing.assert_allclose(
            logits, reference[:, position], rtol=3e-11, atol=3e-12
        )
        if position + 1 >= int(config.dual_resolution.block_size):
            crossed_history = True
    assert crossed_history


def test_training_operator_and_inference_adapter_cannot_disagree(monkeypatch):
    """One model instance must execute exactly one branch topology."""
    monkeypatch.delenv(ATTENTION_BRANCH_MODE_ENV, raising=False)
    parallel_model = DecoderLanguageModel(_tiny_config("parallel"), rng_seed=25,
                                          dtype="float64")
    legacy_engine = InferenceModel(_tiny_config("legacy"), dtype="float64")
    with pytest.raises(ValueError, match="branch mode mismatch"):
        legacy_engine.set_weights(parallel_model)


def test_resume_contract_records_architecture_migration(tmp_path, monkeypatch):
    """Spec test G: migration is permitted (warn-not-block) and recorded."""
    from mini_llm.train_extended import ExtendedTrainer

    monkeypatch.delenv(ATTENTION_BRANCH_MODE_ENV, raising=False)
    monkeypatch.setenv("MINI_LLM_DUAL_RESOLUTION_FAST_PROJECTIONS", "0")
    documents = [[*range(1, 6), 99], [*range(10, 31), 99]]
    shard = np.array([v for doc in documents for v in doc], dtype=np.uint16)
    shard_path = tmp_path / "tokens.bin"
    shard.tofile(shard_path)

    def trainer(mode, directory):
        config = ModelConfig(
            tokenizer_vocab_size=128, context_length=24, n_layers=2,
            d_model=16, d_ff=32, n_q_heads=4, n_kv_heads=2, d_head=4,
            ffn_mode="dense", n_experts=1, top_k=1, dtype="float64",
            dual_resolution=_tiny_dual(mode),
        )
        return ExtendedTrainer(
            model=DecoderLanguageModel(config, rng_seed=2),
            train_shard_paths=[str(shard_path)], val_shard_paths=[str(shard_path)],
            batch_size=1, seq_length=24, grad_accum_steps=1,
            total_steps=2, warmup_steps=1, val_interval=100, save_interval=100,
            eos_token_id=99, checkpoint_dir=str(directory),
        )

    parallel = trainer("parallel", tmp_path / "ckpt-parallel")
    contract = parallel.training_resume_contract()
    assert contract["attention_branch_mode"] == "parallel"

    # A checkpoint that recorded legacy continued on parallel: permitted, and
    # the diff is recorded as provenance rather than blocking the resume.
    saved_legacy = dict(contract, attention_branch_mode="legacy")
    rows = parallel.validate_training_resume_contract(
        {"training_resume_contract": saved_legacy}
    )
    assert {
        "field": "attention_branch_mode",
        "checkpoint": "legacy",
        "current": "parallel",
    } in rows

    # A pre-fork contract (key absent) resumed on legacy is not drift...
    legacy = trainer("legacy", tmp_path / "ckpt-legacy")
    prefork = {k: v for k, v in legacy.training_resume_contract().items()
               if k != "attention_branch_mode"}
    rows = legacy.validate_training_resume_contract(
        {"training_resume_contract": prefork}
    )
    assert not [row for row in rows if row["field"] == "attention_branch_mode"]

    # ...but the same pre-fork contract resumed on parallel is an auditable
    # architectural change.
    rows = parallel.validate_training_resume_contract(
        {"training_resume_contract": prefork}
    )
    assert [row for row in rows if row["field"] == "attention_branch_mode"] == [
        {"field": "attention_branch_mode", "checkpoint": None, "current": "parallel"}
    ]


def test_architecture_override_never_leaks_into_the_process(tmp_path, monkeypatch):
    """Spec test H: two models, one process, no ambient architecture state."""
    monkeypatch.delenv(ATTENTION_BRANCH_MODE_ENV, raising=False)
    path, _ = _prefork_checkpoint(tmp_path)

    monkeypatch.setenv(ATTENTION_BRANCH_MODE_ENV, "1")
    migrated = read_native_config(path)
    assert migrated.dual_resolution.attention_branch_mode == "parallel"
    model_one = DecoderLanguageModel(migrated, rng_seed=1, dtype="float32")
    assert all(block.attention.parallel_branches for block in model_one.blocks)
    # Loading code must not launder the override into the process environment.
    assert ATTENTION_BRANCH_MODE_ENV in os.environ  # test-set, not code-set
    monkeypatch.delenv(ATTENTION_BRANCH_MODE_ENV)

    preserved = read_native_config(path)
    assert preserved.dual_resolution.attention_branch_mode == "legacy"
    model_two = DecoderLanguageModel(preserved, rng_seed=2, dtype="float32")
    assert all(not block.attention.parallel_branches for block in model_two.blocks)
    # No repository code materialized the variable during the second load.
    assert ATTENTION_BRANCH_MODE_ENV not in os.environ
    # The first model keeps its forked architecture; the second did not
    # inherit it, and neither leaked into a preset built afterwards.
    assert migrated.dual_resolution.attention_branch_mode == "parallel"
    assert ModelConfig.dual_resolution_matrix_pool_sparse_1024_20l(
    ).dual_resolution.attention_branch_mode == "parallel"


def test_torch_backend_matches_native_in_both_modes(monkeypatch):
    """Spec test E (PyTorch side): identical weights, forward + backward parity."""
    torch = pytest.importorskip("torch")
    monkeypatch.delenv(ATTENTION_BRANCH_MODE_ENV, raising=False)
    monkeypatch.setenv("MINI_LLM_DUAL_RESOLUTION_FAST_PROJECTIONS", "0")
    from mini_llm.backends.pytorch.model import TorchDecoderLanguageModel

    for mode in ("legacy", "parallel"):
        config = _tiny_config(mode, dtype="float32")
        native = DecoderLanguageModel(config, rng_seed=13, dtype="float32")
        engine = TorchDecoderLanguageModel(
            config, dtype="float32", device="cpu", activation_checkpointing=False
        ).eval()
        named = dict(engine.canonical_named_parameters())
        native_named = {p.name: p for p in native.parameters()}
        assert set(named) == set(native_named)
        with torch.no_grad():
            for name, parameter in named.items():
                parameter.copy_(torch.as_tensor(np.asarray(native_named[name].data)))

        tokens = np.random.default_rng(23).integers(0, config.vocab_size, size=(1, 9))
        logits_native, cache = native.forward(tokens)
        ids = torch.as_tensor(tokens)
        with torch.no_grad():
            logits_torch = engine(ids)
        np.testing.assert_allclose(
            np.asarray(logits_native), logits_torch.detach().numpy(),
            rtol=5e-4, atol=5e-5,
        )

        upstream = np.random.default_rng(24).normal(
            scale=0.01, size=logits_native.shape
        ).astype(np.float32)
        native.zero_grad()
        native.backward(upstream, cache)
        engine.zero_grad()
        engine(ids).backward(torch.as_tensor(upstream))
        checked = 0
        for name in sorted(named):
            if "attention" not in name:
                continue
            expected = native_named[name].grad
            actual = named[name].grad
            assert expected is not None and actual is not None, name
            np.testing.assert_allclose(
                np.asarray(expected), actual.detach().numpy(),
                rtol=5e-3, atol=1e-6,
            )
            checked += 1
        assert any("global" in name for name in named), "global branch missing"
        assert checked >= 4
