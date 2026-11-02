"""0104F: specialist-only checkpoints bound to one frozen foundation.

The foundation's 348M parameters are never serialized a second time; a
content digest guards against accidentally attaching the residual to a
different foundation checkpoint. Existing native checkpoint serialization
and transactional publishing are reused.
"""
from __future__ import annotations

from dataclasses import asdict
import hashlib
from pathlib import Path

from mini_llm.checkpoint import save_checkpoint, load_checkpoint

from .residual_specialist import ResidualSpecialist, SpecialistConfig


def fingerprint_foundation(checkpoint):
    """Content SHA256 of architecture config and every frozen weight file.

    This is an exact, potentially expensive operation. Call it at checkpoint
    boundaries, not during training updates.
    """
    path = Path(checkpoint).expanduser().resolve()
    config = path / "config.json"
    weights = path / "model_params"
    if not config.is_file() or not weights.is_dir():
        raise FileNotFoundError(f"Incomplete foundation checkpoint: {path}")
    files = [config, *sorted(weights.glob("*.npy"), key=lambda f: f.name)]
    if len(files) < 2:
        raise ValueError("Foundation checkpoint has no parameter files")
    digest = hashlib.sha256()
    for item in files:
        digest.update(item.relative_to(path).as_posix().encode("utf-8"))
        digest.update(b"\\0")
        with item.open("rb") as stream:
            while chunk := stream.read(4 * 1024 * 1024):
                digest.update(chunk)
    return digest.hexdigest()


def save_specialist_checkpoint(path, composition, *, optimizer_state=None,
                               training_state=None):
    """Save trainable residual parameters and the exact foundation identity."""
    frozen_path = composition.foundation.checkpoint_path
    if frozen_path is None:
        raise ValueError("A checkpoint-backed frozen foundation is required")
    frozen_digest = fingerprint_foundation(frozen_path)
    state = dict(training_state or {})
    if "specialist" in state:
        raise ValueError("training_state['specialist'] is reserved for provenance")
    state["specialist"] = {
        "format_version": 1,
        "architecture": asdict(composition.specialist.config),
        "foundation_sha256": frozen_digest,
        "foundation_checkpoint": str(frozen_path),
        "trainable_parameter_names": [p.name for p in composition.parameters()],
    }
    save_checkpoint(path, {p.name: p.data for p in composition.parameters()},
                    optimizer_state=optimizer_state, training_state=state)


def load_specialist_checkpoint(path, foundation, *, verify_foundation=True,
                               rng_seed=0, load_optimizer=True):
    """Restore independently trained specialist; reject mismatched foundations."""
    from .residual_composition import FrozenResidualComposition
    values, optimizer, state = load_checkpoint(path,
                        skip_optimizer=not load_optimizer, skip_training=False)
    info = (state or {}).get("specialist")
    if not isinstance(info, dict) or info.get("format_version") != 1:
        raise ValueError("Not a specialist checkpoint with provenance format 1")
    if verify_foundation:
        if foundation.checkpoint_path is None:
            raise ValueError("Foundation checkpoint path required for verification")
        expected = fingerprint_foundation(foundation.checkpoint_path)
        if expected != info["foundation_sha256"]:
            raise ValueError("Foundation weights/config digest mismatch")
    config = SpecialistConfig(**info["architecture"])
    specialist = ResidualSpecialist(config, rng_seed=rng_seed)
    params = specialist.parameters()
    names = [p.name for p in params]
    if len(names) != len(set(names)) or set(names) != set(values):
        raise ValueError("Specialist parameter schema mismatch")
    if names != info["trainable_parameter_names"]:
        raise ValueError("Specialist parameter order/schema changed")
    for param in params:
        value = values[param.name]
        if tuple(value.shape) != tuple(param.data.shape):
            raise ValueError(f"Specialist parameter shape mismatch: {param.name}")
        param.data[...] = value
    specialist.refresh_compute_buffers()
    return FrozenResidualComposition(foundation, specialist), optimizer, state
