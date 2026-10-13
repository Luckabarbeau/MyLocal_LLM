"""0080H matched AR vs two-stage CPU benchmark regression tests."""
import json
import numpy as np
import pytest

from examples import compare_block_proposal_cpu_0080h as bench
from examples import prepare_block_proposal_cpu_data as prep
from examples.train_block_proposal_routed_text import _model_config
from tests.test_block_proposal_0080g_cpu import ToyTokenizer


def test_cpu_medium_is_real_6_layer_routed_moe_model():
    args=bench.parse_args(['--mixed-shard-root', '.'])
    config=_model_config(args, 2048)
    assert config.n_layers==6 and config.d_model==192
    assert config.n_experts==4 and config.top_k==2
    assert config.memory_context.integration_mode=='routed_prefix'
    assert config.memory_context.target_length==64
    assert len(config.attention_layers)==6
    assert [h.kind for h in config.attention_layers[0].heads] == [
        "local", "local", "dilated", "global_sparse", "retrieval", "retrieval"]


def test_matched_ar_two_stage_cpu_pipeline(tmp_path, monkeypatch):
    monkeypatch.setattr(prep, 'FastBPETokenizer', ToyTokenizer)
    source=[]
    for name in ('fine_edu', 'cosmopedia_v2'):
        root=tmp_path/name; root.mkdir()
        for i in range(20):
            (root/f'{i:02d}.txt').write_text((f'{name} record {i} and repeated training text.\\n')*12)
        source.extend(['--source', f'{name}={root}'])
    output=tmp_path/'processed'
    prep.prepare(prep.parse_args(['--output',str(output),'--max-documents-per-source','20',
                                  '--val-fraction','.3',*source]))
    monkeypatch.setattr(bench,'_load_tokenizer', lambda path:(ToyTokenizer(),1))
    result=bench.main(['--mixed-shard-root',str(output), '--model-preset','cpu_tiny',
                       '--steps','2','--warmup-steps','1','--block-size','4',
                       '--proposal-width','16','--decoder-width','16',
                       '--decoder-layers','1','--decoder-heads','2',
                       '--output',str(tmp_path/'comparison')])
    ar=result['standard_ar']; tw=result['two_stage']
    assert ar['model_parameters']==tw['model_parameters']
    assert tw['added_parameters']>0 and ar['added_parameters']==0
    assert ar['step_supervised_counts']==tw['step_supervised_counts']
    assert ar['total_measured_unique_tokens']>0
    assert 0 < ar['weighted_supervised_tokens_per_sec'] < float('inf')
    assert 0 < tw['weighted_supervised_tokens_per_sec'] < float('inf')
    assert np.isfinite(ar['validation_final_ce'])
    assert np.isfinite(tw['validation_final_ce'])
    assert np.isfinite(result['comparison']['ar_over_two_stage_tok_s'])
    assert (tmp_path/'comparison'/'summary.json').exists()
    for name in ('two_stage', 'standard_ar'):
        records=[json.loads(line) for line in
                 (tmp_path/'comparison'/f'{name}.jsonl').read_text().splitlines()]
        assert len(records)==2
        assert all(r['grad_norm']>0 and np.isfinite(r['final_ce']) for r in records)
