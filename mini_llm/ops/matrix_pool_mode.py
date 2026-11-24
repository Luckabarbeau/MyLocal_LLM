"""Model B attention-branch selection (compatibility view of the config).

The architecture is configured by
``ModelConfig.dual_resolution.attention_branch_mode`` ("legacy" or "parallel"),
which is what gets serialized into ``config.json`` and therefore what a
checkpoint actually records.  ``mini_llm.config`` owns the vocabulary, the
validation and the single resolution helper; this module only exposes it to
operators and keeps the historical environment variable working as an
*override* for pre-fork workflows.

Resolution happens once, when a configuration is loaded or launched, and the
resolved value is written back into the in-memory configuration.  Attention
operators below read the configuration only, so the environment can never
select a graph that the configuration denies, and nothing here writes
``os.environ``.
"""

from ..config import (
    ATTENTION_BRANCH_MODE_ENV,
    ATTENTION_BRANCH_MODES,
    attention_branch_mode_override,
    resolve_attention_branch_mode,
)

__all__ = (
    "ATTENTION_BRANCH_MODES",
    "ATTENTION_BRANCH_MODE_ENV",
    "attention_branch_mode",
    "attention_branch_mode_override",
    "parallel_matrix_pool_branches_enabled",
    "resolve_attention_branch_mode",
)


def attention_branch_mode(config, *, environ=None):
    """Resolved branch mode for a ModelConfig or DualResolutionConfig."""
    return getattr(
        resolve_attention_branch_mode(config, environ=environ),
        "attention_branch_mode",
        "legacy",
    )


def parallel_matrix_pool_branches_enabled(config=None, *, environ=None):
    """Deprecated boolean view of the resolved branch mode.

    With a configuration object this reports the resolved architecture.  Without
    one it can only report an explicit override, falling back to the historical
    environment default for callers that have not migrated to configuration
    driven selection.
    """
    if config is not None:
        return attention_branch_mode(config, environ=environ) == "parallel"
    override = attention_branch_mode_override(environ)
    if override is None:
        return True
    return override == "parallel"
