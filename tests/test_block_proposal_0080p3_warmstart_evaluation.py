"""0080P3: frozen diagnostics reconstruct the same corpus after 0080P2 warm start."""
import json
import shutil

import numpy as np
import pytest

from examples import evaluate_block_proposal_cross_position_0080k as k
from examples import evaluate_block_proposal_ablations_0080j as j
from examples import train_block_proposal_cpu_learning_0080i as train
from tests.test_block_proposal_0080i_learning import prepared, _argv
from tests.test_block_proposal_0080g_cpu import ToyTokenizer


def test_warm_started_k_and_j_validate_current_data_and_preserve_lineage(prepared, tmp_path, monkeypatch):
    source_output = tmp_path / 'original'
    train.main(_argv(prepared, source_output, '--architecture', 'two_stage', '--batch-size', '2'))
    source = source_output / 'checkpoints' / 'two_stage' / 'step_00000002'

    expanded = tmp_path / 'expanded'
    shutil.copytree(prepared, expanded)
    extra = np.array([3, 7, 9, 1], dtype=np.uint16)
    with (expanded / 'fine_edu' / 'train_shard_00000.bin').open('ab') as f:
        f.write(extra.tobytes())
    manifest_path = expanded / 'mixture_manifest.json'
    manifest = json.loads(manifest_path.read_text())
    manifest['sources']['fine_edu']['train_tokens'] += len(extra)
    manifest_path.write_text(json.dumps(manifest))

    output = tmp_path / 'new'
    train.main(_argv(expanded, output, '--architecture', 'two_stage',
                     '--warm-start-checkpoint', str(source), '--learning-rate', '0.0001', '--batch-size', '2'))
    checkpoint = output / 'checkpoints' / 'two_stage' / 'step_00000002'
    state = json.loads((checkpoint / 'training_state.json').read_text())
    assert state['data_provenance']['warm_start']['source_step'] == 2

    for mod in (k, j):
        monkeypatch.setattr(mod, '_load_tokenizer', lambda path: (ToyTokenizer(), 1))
        args = ['--mixed-shard-root', str(expanded), '--training-output', str(output),
                '--checkpoint', str(checkpoint)]
        report = mod.main(args)
        assert report['training_step'] == 2
        assert report['validation_digests'] == state['data_provenance']['validation_digests']
        with pytest.raises(ValueError, match='(validation|tokenizer/data)'):
            mod.main(['--mixed-shard-root', str(prepared), '--training-output', str(output),
                      '--checkpoint', str(checkpoint), '--output', str(tmp_path / 'never.json')])
