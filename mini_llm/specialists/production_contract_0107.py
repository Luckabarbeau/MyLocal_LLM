"""0107A: explicit production provenance checks for the specialist launch path.

The reusable CPU reference specialist classes deliberately accept tiny Model B
configurations. The *production launcher*, however, must not silently swap the
pretrained stride-16 348M foundation for a legacy non-sparse checkpoint.
"""


def validate_sparse_model_b_foundation(config):
    """Reject a nonproduction foundation before allocating the specialist.

    These are architecture identity checks, NOT a claim that the supplied
    checkpoint contains 7.3B tokens; token exposure must be obtained from a
    completed checkpoint's training manifest/log, not guessed from its shape.
    """
    required = {
        'n_layers': 20, 'd_model': 1024, 'd_ff': 4096,
    }
    for field, expected in required.items():
        actual = getattr(config, field, None)
        if actual != expected:
            raise ValueError(f'expected sparse Model B {field}={expected}, got {actual}')
    # The preset advertises a 65,536-token maximum; train_model.py records
    # the selected 8,192-token training cap in the saved checkpoint config.
    if not 8192 <= int(config.context_length) <= 65536:
        raise ValueError('expected Model B context capacity in [8192, 65536]')
    dual = config.dual_resolution
    expectations = {
        'enabled': True,
        'compression_mode': 'matrix_pool',
        'block_size': 128,
        'local_window': 512,
        'long_local_window': 4096,
        'long_local_heads': 1,
        'summaries_per_block': 4,
        'sparse_retrieval_stride': 16,
        'sparse_retrieval_head': 1,
        'sparse_retrieval_layers': tuple(range(2, 18)),
    }
    for field, expected in expectations.items():
        actual = getattr(dual, field, None)
        if field == 'sparse_retrieval_layers' and actual is not None:
            actual = tuple(actual)
        if actual != expected:
            raise ValueError(f'expected sparse Model B {field}={expected}, got {actual}')


def validate_model_b_specialist(
        spec, foundation_config, *, recompute_attention_probs=None):
    """Require the production specialist to be a truncated Model B block stack."""
    from .residual_specialist import SpecialistConfig
    expected = SpecialistConfig.from_foundation(
        foundation_config, specialist_id=spec.specialist_id,
        n_layers=spec.n_layers,
        recompute_attention_probs=recompute_attention_probs)
    for field in (
        'd_model', 'd_ff', 'n_q_heads', 'n_kv_heads', 'd_head', 'dtype',
        'init_std', 'rms_eps', 'rope_base', 'residual_init_std',
        'dual_resolution',
    ):
        actual = getattr(spec, field)
        wanted = getattr(expected, field)
        if actual != wanted:
            raise ValueError(
                f'specialist Model B geometry mismatch: {field}={actual!r}, '
                f'expected {wanted!r}')


def validate_resume_specialist(spec, *, requested_id, requested_layers):
    """Fail closed if a resumed checkpoint differs from explicitly chosen CLI geometry."""
    if spec.specialist_id != requested_id:
        raise ValueError(
            'resume specialist-id mismatch: checkpoint has '
            f'{spec.specialist_id!r}, requested {requested_id!r}'
        )
    if spec.n_layers != requested_layers:
        raise ValueError(
            'resume specialist-layers mismatch: checkpoint has '
            f'{spec.n_layers}, requested {requested_layers}'
        )
