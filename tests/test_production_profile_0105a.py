from mini_llm.production_profile_0105 import ENVIRONMENT, OPTIONS, audit, parse_cli_options


def test_supplied_production_contract_exact():
    result = audit(dict(ENVIRONMENT), dict(OPTIONS))
    assert result['matches'] and not result['mismatches']
    assert OPTIONS['peak-lr'] == '1.5e-4'
    assert OPTIONS['grad-accum-steps'] == '512'
    assert OPTIONS['gradient-accumulation-weighting'] == 'supervised_tokens'
    assert ENVIRONMENT['MINI_LLM_DUAL_RESOLUTION_SHAPE_BUCKET'] == 'fine'
    assert ENVIRONMENT['MINI_LLM_MATRIX_POOL_FUSED_BACKWARD'] == '1'


def test_reject_mismatch_in_fused_kernel_and_batch():
    env, opts = dict(ENVIRONMENT), dict(OPTIONS)
    env['MINI_LLM_MATRIX_POOL_FUSED_BACKWARD'] = '0'
    opts['grad-accum-steps'] = '256'
    result = audit(env, opts)
    assert not result['matches']
    assert 'MINI_LLM_MATRIX_POOL_FUSED_BACKWARD' in result['mismatches']
    assert '--grad-accum-steps' in result['mismatches']


def test_cli_parser_and_float_alias():
    opts = parse_cli_options(['--peak-lr', '0.00015', '--batch-size=1', '--grad-clip', '1'])
    assert opts['peak-lr'] == '0.00015'
    assert opts['batch-size'] == '1'
    assert opts['grad-clip'] == '1'
    config = dict(OPTIONS, **opts)
    assert audit(dict(ENVIRONMENT), config)['matches']
