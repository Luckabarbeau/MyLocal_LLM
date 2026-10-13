"""0079L observability: no training changes, honest token accounting."""
import csv
import json

import numpy as np
import pytest

from mini_llm.backend import xp, asnumpy
from mini_llm.config import ModelConfig
from mini_llm.model.decoder_lm import DecoderLanguageModel
from mini_llm.predictive_refinement.training_reference import RefinementReferenceConfig
from mini_llm.predictive_refinement.window_training import WindowRefinementTrainer
from mini_llm.predictive_refinement.telemetry import (
    empty_counters, merge_counters, observe_microbatch, rates, source_summary,
)
from mini_llm.predictive_refinement.gradient_diagnostics import (
    weighted_half_statistics, TokenWeightedRefinementGradientMonitor,
)


def ref_model(block=2):
    cfg = ModelConfig(tokenizer_vocab_size=48, context_length=32, n_layers=1,
                      d_model=16, n_q_heads=2, n_kv_heads=1, d_head=8,
                      d_ff=32, n_experts=1, top_k=1, dtype='float64')
    return WindowRefinementTrainer(
        DecoderLanguageModel(cfg, rng_seed=122, dtype='float64'),
        RefinementReferenceConfig(block_size=block, enable_shared_context_training=True),
        rng_seed=123)


def rows():
    tokens = xp.asarray([[4,5,6,7,8,9], [10,11,12,13,0,0]])
    mask = xp.asarray([[1,1,1,1,1,1], [1,1,1,1,0,0]], dtype=xp.bool_)
    return tokens, mask


@pytest.mark.parametrize('group', [1,2])
@pytest.mark.parametrize('gradients', [False,True])
def test_row_loss_is_exact_and_observational(group, gradients):
    ref = ref_model()
    tokens, valid = rows()
    ref.zero_grad()
    a = ref.train_window(tokens, prefix_length=2, valid_token_mask=valid,
                         group_size=group, compute_gradients=gradients)
    grads_a = [asnumpy(p.grad).copy() for p in ref.parameters()]
    ref.zero_grad()
    b = ref.train_window(tokens, prefix_length=2, valid_token_mask=valid,
                         group_size=group, compute_gradients=gradients,
                         record_row_ce=True)
    for key in ('total_loss','commitment_ce','initial_ce'):
        assert float(a[key]) == pytest.approx(float(b[key]), abs=1e-13)
    for ga, p in zip(grads_a, ref.parameters()):
        np.testing.assert_allclose(ga, asnumpy(p.grad), atol=0, rtol=0)
    row_nll = asnumpy(b['row_commitment_ce_sums'])
    assert row_nll.shape == (2,)
    assert row_nll.sum() == pytest.approx(float(b['commitment_ce'])*6, abs=1e-11)


def test_source_ce_and_coverage_are_token_weighted_and_sum_to_global():
    ref = ref_model()
    tokens, valid = rows()
    result = ref.train_window(tokens, prefix_length=2, valid_token_mask=valid,
                              group_size=2, compute_gradients=False, record_row_ce=True)
    doc_rows = [{'source': 'github', 'sampled_document_span_tokens': 15},
                {'source': 'wikipedia', 'sampled_document_span_tokens': 4}]
    rec = observe_microbatch(asnumpy(tokens), asnumpy(valid), 2, result,
                             configured_prefix=2, window_blocks=2, block_size=2,
                             observations=doc_rows)
    assert rec['supervised_tokens'] == 6
    assert rec['target_capacity_tokens'] == 8
    assert rec['valid_source_tokens'] == 10
    assert rec['active_position_evaluations'] == 2*2*(2+1)*2
    assert rec['refinement_full_model_passes'] == 3
    assert rec['active_queries_per_pass_min'] == 8
    assert rates(rec, .5)['target_fill'] == .75
    means = source_summary(rec)
    assert means['github']['supervised_tokens'] == 4
    assert means['wikipedia']['supervised_tokens'] == 2
    weighted = sum(means[k]['commitment_ce']*means[k]['supervised_tokens'] for k in means)
    assert weighted/6 == pytest.approx(float(result['commitment_ce']), abs=1e-11)
    assert means['github']['sampled_document_span_mean'] == 15
    assert means['wikipedia']['target_fill'] == .5
    another = empty_counters()
    merge_counters(another, rec)
    merge_counters(another, rec)
    assert another['supervised_tokens'] == 12
    assert another['active_position_evaluations'] == 48
    assert another['source_records']['github']['loss_nll_sum'] == pytest.approx(
        2 * rec['source_records']['github']['loss_nll_sum'])


def test_source_observation_sampling_has_no_rng_or_dataset_effect(tmp_path):
    from mini_llm.predictive_refinement.real_data import PackedRefinementDocuments
    arr = np.array([4,5,6,1,7,8,1,9,10,11,12,1],dtype=np.uint16)
    for split in ('train','val'):
        arr.tofile(tmp_path / f'{split}_shard_0000.bin')
    dataset = PackedRefinementDocuments(shard_dir=tmp_path,eos_token_id=1)
    rng_a = np.random.default_rng(123)
    rng_b = np.random.default_rng(123)
    base = dataset.sample(rng_a,batch_size=4,prefix_length=3,block_size=2,window_blocks=2)
    full = dataset.sample(rng_b,batch_size=4,prefix_length=3,block_size=2,window_blocks=2,
                          return_observations=True)
    for first, second in zip(base[:3],full[:3]):
        np.testing.assert_array_equal(first,second)
    assert base[3] == full[3] == {'single':4}
    assert rng_a.bit_generator.state == rng_b.bit_generator.state
    assert len(full[4]) == 4
    assert {rec['source'] for rec in full[4]} == {'single'}
    assert all(rec['sampled_document_span_tokens'] >= rec['sampled_fragment_tokens']
               for rec in full[4])


def test_token_weighted_gradient_noise_is_correct_for_unequal_halves():
    # Gradient normalization 1/N is already applied by the training objective.
    g_a = xp.asarray([1.,2.,3.]); g_b = xp.asarray([2.,0.,4.])
    n_a, n_b, total = 2,6,8
    mid = g_a * (n_a / total)
    end = mid + g_b * (n_b / total)
    stats = weighted_half_statistics(mid,end,n_a,n_b,total_valid=total)
    np.testing.assert_allclose(stats['gradient_norm_a'], float(xp.linalg.norm(g_a)))
    np.testing.assert_allclose(stats['gradient_norm_b'], float(xp.linalg.norm(g_b)))
    expected = np.linalg.norm(asnumpy(g_a-g_b))**2 * n_a*n_b / total**2
    assert stats['estimated_gradient_noise'] == pytest.approx(expected)
    with pytest.raises(ValueError):
        weighted_half_statistics(mid,end,0,8,total_valid=8)


def test_gradient_monitor_read_only_and_csv(tmp_path):
    ref = ref_model()
    monitor = TokenWeightedRefinementGradientMonitor(
        ref.parameters(), accum_steps=2, categories='attention,moe,head',
        output=tmp_path/'grad.csv', interval=1)
    assert monitor.begin(0)
    tokens, valid = rows()
    ref.zero_grad()
    # Both microbatches normalized by the same total valid-token denominator.
    a = ref.train_window(tokens, prefix_length=2, valid_token_mask=valid,
                         loss_normalization_tokens=12, group_size=2)
    monitor.capture(1,6)
    b = ref.train_window(tokens, prefix_length=2, valid_token_mask=valid,
                         loss_normalization_tokens=12, group_size=2)
    monitor.capture(2,12)
    before = [asnumpy(p.grad).copy() for p in ref.parameters()]
    monitored_rows = monitor.finish(12)
    monitor.commit(monitored_rows)
    for before_grad, p in zip(before,ref.parameters()):
        np.testing.assert_array_equal(before_grad, asnumpy(p.grad))
    assert len(monitored_rows) >= 2
    assert all(row['valid_supervised_tokens'] == 12 for row in monitored_rows)
    with (tmp_path/'grad.csv').open() as handle:
        assert len(list(csv.DictReader(handle))) == len(monitored_rows)


def test_telemetry_csv_and_jsonl_preserves_original_schema(tmp_path):
    from examples.train_predictive_refinement_text import _source_report, _write_metrics
    from mini_llm.diagnostics.source_metrics import CSV_COLUMNS
    ref = ref_model()
    tokens, valid = rows()
    res = ref.train_window(tokens, prefix_length=2, valid_token_mask=valid,
                           compute_gradients=False, record_row_ce=True)
    obs = [{'source':'a', 'sampled_document_span_tokens':7},
           {'source':'b', 'sampled_document_span_tokens':4}]
    stats = observe_microbatch(asnumpy(tokens),asnumpy(valid),2,res,
                               configured_prefix=2,window_blocks=2,block_size=2,
                               observations=obs)
    csv_path = tmp_path/'source.csv'
    _source_report(stats,step=1,source_csv=csv_path,cumulative=stats)
    with csv_path.open() as handle:
        loaded = list(csv.DictReader(handle))
    assert len(loaded) == 4
    assert tuple(loaded[0]) == CSV_COLUMNS
    assert {'interval','cumulative_since_telemetry_start'} == {r['scope'] for r in loaded}
    path = tmp_path/'metrics.jsonl'
    _write_metrics(path,{'step':1,'gain':float('nan'), 'nested':[float('inf')]})
    assert json.loads(path.read_text()) == {'step':1,'gain':None,'nested':[None]}


def test_real_training_logging_checkpoint_resume_and_gradient_monitor(tmp_path, monkeypatch, capsys):
    """End-to-end 65K tokenizer interface, mixed-length EOS shards, logging."""
    from examples import train_predictive_refinement_text as script
    from mini_llm.predictive_refinement.real_data import sha256
    arr = np.array([4,5,6,1,7,1,8,9,10,11,1], dtype=np.uint16)
    for split in ('train','val'):
        arr.tofile(tmp_path/f'{split}_shard_0000.bin')
    (tmp_path/'tokenizer.json').write_text('{"test":true}')
    class Tokenizer:
        def __len__(self): return 65280
        def encode(self, text): return [4,5]
        def decode(self, ids): return str(ids)
    monkeypatch.setattr(script, '_load_tokenizer', lambda path: (Tokenizer(), 1))
    root = tmp_path/'checkpoint'
    cmd = ['--shard-dir',str(tmp_path),'--steps','2',
           '--checkpoint-dir',str(root), '--batch-size','1', '--grad-accum','2',
           '--prefix-length','2','--block-size','2','--window-blocks','2',
           '--refinement-group-size','2','--layers','1','--d-model','16',
           '--n-q-heads','2','--n-kv-heads','1','--d-ff','32','--experts','1',
           '--log-every','1','--val-every','1','--val-batches','1',
           '--save-every','1','--generate','0',
           '--gradient-noise-monitor','--gradient-noise-interval','1',
           '--gradient-noise-accum-points','2']
    script.main(cmd)
    text = capsys.readouterr().out
    assert 'source_tok/s' in text and 'active_queries/pass=' in text
    assert 'target_fill=' in text and 'Per-source commitment CE' in text
    assert 'Gradient-noise' in text and 'validation source=single' in text
    logs = [json.loads(line) for line in (root/'refinement_metrics.jsonl').read_text().splitlines()]
    assert len([x for x in logs if x['kind']=='training']) == 2
    assert len([x for x in logs if x['kind']=='validation']) == 2
    assert logs[0]['token_accounting']['supervised_tokens'] > 0
    assert logs[0]['throughput']['active_position_evals_per_unique_target'] >= 3
    assert logs[0]['train_position_metrics']['counts']
    assert logs[0]['source_metrics']['single']['commitment_ce'] > 0
    assert 'gradient_family_norms' in logs[0]
    with (root/'refinement_source_metrics.csv').open() as stream:
        source_rows = list(csv.DictReader(stream))
    assert len(source_rows) == 4
    with (root/'refinement_gradient_noise.csv').open() as stream:
        gradient_rows = list(csv.DictReader(stream))
    assert gradient_rows and gradient_rows[0]['valid_supervised_tokens']
    # Resume is still controlled by original checkpoint RNG/provenance.
    script.main(cmd + ['--resume', str(root), '--steps', '3'])
    state = json.loads((root/'training_state.json').read_text())
    assert state['step'] == 3
    telemetry = json.loads((root/'refinement_telemetry.json').read_text())
    assert telemetry['step'] == 3
    assert telemetry['since_step'] == 1
    assert telemetry['counters']['supervised_tokens'] >= logs[0]['token_accounting']['supervised_tokens']


def test_adamw_counterfactual_is_observational_and_correct(tmp_path):
    from mini_llm.optim.adamw import AdamW
    from mini_llm.diagnostics.adamw_update import counterfactual_direction, ADAMW_FIELDS
    ref = ref_model()
    opt = AdamW(ref.parameters(), lr=3e-4)
    monitor = TokenWeightedRefinementGradientMonitor(
        ref.parameters(), accum_steps=2, categories='head',
        output=tmp_path/'noise.csv', interval=1, optimizer=opt,
        adamw_output=tmp_path/'adam.csv', grad_clip=1e9)
    assert monitor.begin(0)
    toks, mask = rows()
    ref.zero_grad()
    ref.train_window(toks, prefix_length=2, valid_token_mask=mask,
                     loss_normalization_tokens=12)
    monitor.capture(1,6)
    ref.train_window(toks, prefix_length=2, valid_token_mask=mask,
                     loss_normalization_tokens=12)
    monitor.capture(2,12)
    before_grad = [asnumpy(p.grad).copy() for p in ref.parameters()]
    before_m = [asnumpy(m).copy() for m in opt.m]
    before_v = [asnumpy(v).copy() for v in opt.v]
    noise = monitor.finish(12)
    monitor.commit(noise)
    assert len(monitor.adamw_rows) == 0
    with (tmp_path/'adam.csv').open() as stream:
        adam_rows = list(csv.DictReader(stream))
    assert len(adam_rows)==1
    assert tuple(adam_rows[0]) == ADAMW_FIELDS
    selected = next(i for i,p in enumerate(ref.parameters()) if p.name == 'output_proj.W')
    gradient = asnumpy(ref.parameters()[selected].grad)
    selector = next(sel for label,p,sel in monitor.selected if p.name == 'output_proj.W')
    direction = counterfactual_direction(
        gradient[selector], before_m[selected][selector], before_v[selected][selector],
        beta1=opt.beta1, beta2=opt.beta2, eps=opt.eps,
        learning_rate=opt.lr, local_step=1,
    )
    # Only the sampled slice is examined: first output-embedding row.
    assert float(adam_rows[0]['adamw_update_norm']) == pytest.approx(float(np.linalg.norm(direction)), rel=1e-5)
    for p, before in zip(ref.parameters(),before_grad):
        np.testing.assert_array_equal(asnumpy(p.grad),before)
    for current, before in zip(opt.m,before_m):
        np.testing.assert_array_equal(asnumpy(current),before)
    for current, before in zip(opt.v,before_v):
        np.testing.assert_array_equal(asnumpy(current),before)


def test_cli_rejects_invalid_gradient_noise_accumulation():
    from examples.train_predictive_refinement_text import parse_args
    with pytest.raises(SystemExit):
        parse_args(['--shard-dir','.', '--grad-accum','1', '--adamw-update-monitor'])


def test_legacy_scalar_csv_writer(tmp_path):
    from mini_llm.predictive_refinement.telemetry import (
        write_scalar_csv, SCALAR_CSV_COLUMNS,
    )
    path = tmp_path/'scalars.csv'
    write_scalar_csv(path, step=10, supervised_tokens=640,
                     train_loss=2.25, learning_rate=1e-4,
                     gradient_norm=.8, steps_per_second=.5,
                     throughput={'supervised_tok_per_second':320.,
                                 'active_queries_per_pass_mean':128.,
                                 'target_fill':.5})
    write_scalar_csv(path, step=10, supervised_tokens=640,
                     val_loss=2.3, first_position_gain=.1)
    with path.open() as f:
        values = list(csv.DictReader(f))
    assert len(values)==2
    assert tuple(values[0]) == SCALAR_CSV_COLUMNS
    assert float(values[0]['train_loss'])==2.25
    assert float(values[0]['supervised_tok_per_second'])==320
    assert values[0]['val_loss']==''
    assert values[1]['train_loss']==''
    assert float(values[1]['first_position_refinement_gain'])==.1
