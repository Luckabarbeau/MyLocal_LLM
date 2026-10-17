"""0080AE: configuration contract for future full-distribution decoder reads.

This release deliberately implements *only* the original expectation path.
An experimental setting must fail during construction rather than silently
run a v1 decoder, produce a misleading checkpoint, or change RNG draw order.
0080AF+ will enable independently verified variants and versioned metadata.
"""

from dataclasses import dataclass
import math


@dataclass(frozen=True)
class SoftInterfaceConfig:
    """Proposed decoder interface, with a strict v1 capability boundary.

    The defaults are intentionally inert. ``read_layer`` identifies the layer
    *after* which the one-read module will be inserted (zero-based).
    ``read_scale`` is beta in the score definition, not a residual gate.
    """

    mode: str = 'expectation'
    source_scope: str = 'own'
    read_heads: int = 4
    read_layer: int = 0
    temperature: float = 1.0
    read_scale: float = 1.0
    boundary_condition: str = 'off'
    probability_embedding_dim: int = 256

    def validate(self, *, model_width, decoder_layers, block_size):
        if self.mode not in ('expectation', 'full_vocab', 'probability_embedding', 'probability_embedding_direct', 'proposal_latents_direct'):
            raise ValueError('unknown decoder soft interface mode')
        if self.source_scope not in ('own', 'all'):
            raise ValueError('vocabulary read source must be own or all')
        if self.boundary_condition not in ('off', 'dec_residual'):
            raise ValueError('boundary condition must be off or dec_residual')
        if int(self.read_heads) != self.read_heads or self.read_heads <= 0:
            raise ValueError('vocabulary read heads must be a positive integer')
        if int(self.read_layer) != self.read_layer or not 0 <= self.read_layer < int(decoder_layers):
            raise ValueError('vocabulary read layer must index an existing decoder layer')
        if not math.isfinite(self.temperature) or self.temperature <= 0:
            raise ValueError('vocabulary proposal temperature must be positive and finite')
        if not math.isfinite(self.read_scale) or self.read_scale < 0:
            raise ValueError('vocabulary query scale must be finite and nonnegative')
        if self.mode == 'full_vocab' and int(model_width) % int(self.read_heads):
            raise ValueError('embedding width must be divisible by read heads')
        if (int(self.probability_embedding_dim) != self.probability_embedding_dim
                or not 1 <= self.probability_embedding_dim <= 4096):
            raise ValueError('probability embedding dimension must be an integer in [1,4096]')
        if self.mode not in ('probability_embedding','probability_embedding_direct') and self.probability_embedding_dim != 256:
            raise ValueError('probability embedding dimension requires probability_embedding mode')
        if self.mode == 'probability_embedding_direct' and int(self.probability_embedding_dim) != int(model_width):
            raise ValueError('direct probability embedding must match main model width')
        if int(block_size) < 1:
            raise ValueError('block size must be positive')
        if self.mode != 'full_vocab' and (
            self.source_scope != 'own' or self.read_heads != 4 or
            self.read_layer != 0 or self.temperature != 1.0 or
            self.read_scale != 1.0
        ):
            raise ValueError('vocabulary read options require full_vocab interface')
        return self

    @property
    def is_v1_baseline(self):
        return self.mode == 'expectation' and self.boundary_condition == 'off'

    def require_implemented(self):
        """Guard both CLI and direct constructors against placeholder training."""
        if not self.is_v1_baseline:
            raise NotImplementedError(
                '0080AE scaffolding only: full_vocab/boundary_condition are not '
                'implemented; wait for the corresponding tested 0080AF+ patch'
            )
