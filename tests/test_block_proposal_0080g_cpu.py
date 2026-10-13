"""0080G CPU mixed-corpus preparation + train/val/optimizer checkpoint smoke."""
import json
import numpy as np
import pytest

from examples import prepare_block_proposal_cpu_data as prep
from examples import train_block_proposal_routed_text as cli


class ToyTokenizer:
    eos_token = '<|eos|>'

    def __init__(self, *args, **kwargs):
        self._tokenizer = self

    def token_to_id(self, text):
        return 1 if text == self.eos_token else None

    def __len__(self):
        return 256

    def train(self, texts, vocab_size=None):
        self.training_docs = sum(1 for _ in texts)

    def save(self, path):
        from pathlib import Path
        Path(path).write_text('{"toy":true}')

    def encode(self, text):
        return [3 + ord(c) % 253 for c in text]


def test_prepare_three_sources_and_preflight(tmp_path, monkeypatch, capsys):
    monkeypatch.setattr(prep,'FastBPETokenizer',ToyTokenizer)
    source_args=[]
    for name in ('fine_edu','cosmopedia_v2','python_edu'):
        root=tmp_path/name
        root.mkdir()
        for i in range(36):
            (root/f'{i:03d}.txt').write_text(f'{name} training document {i} and some lines of ordinary words\n'*5)
        source_args.extend(['--source',f'{name}={root}'])
    out=tmp_path/'shards'
    args=prep.parse_args(['--output',str(out),'--vocab-size','512',
                          '--max-documents-per-source','36','--val-fraction','.25',*source_args])
    manifest=prep.prepare(args)
    assert len(manifest['sources'])==3
    for name in manifest['sources']:
        s=manifest['sources'][name]
        assert s['val_tokens']>0 and s['train_tokens']>0
    assert manifest['format_version']=='weighted_packed_mixture_v1'
    monkeypatch.setattr(cli,'_load_tokenizer', lambda path:(ToyTokenizer(),1))
    cli.main(['--mixed-shard-root',str(out),'--model-preset','cpu_tiny',
              '--dtype','float32','--preflight-only'])
    assert '"event": "preflight"' in capsys.readouterr().out
    # Exercise a real optimizer step against the weighted *three-source* mixture.
    cli.main(['--mixed-shard-root',str(out),'--model-preset','cpu_tiny',
              '--dtype','float32','--block-size','4','--proposal-width','16',
              '--decoder-width','16','--decoder-layers','1','--decoder-heads','2',
              '--val-batches','1','--val-every','1','--no-save','--steps','1',
              '--checkpoint-dir',str(tmp_path/'mixed_ckpt')])
    metric=json.loads((tmp_path/'mixed_ckpt'/'metrics.jsonl').read_text().splitlines()[0])
    assert metric['unique_supervised_tokens']>0
    assert np.isfinite(metric['total_loss'])


def test_real_one_optimizer_step_with_compact_routed_cpu(tmp_path, monkeypatch):
    monkeypatch.setattr(cli,'_load_tokenizer', lambda path:(ToyTokenizer(),1))
    data=tmp_path/'shards'
    data.mkdir()
    (data/'tokenizer.json').write_text('{"toy":true}')
    rng=np.random.default_rng(14)
    for split in ('train','val'):
        # Exactly one EOS-delimited document, with enough true history.
        tokens=rng.integers(3,128,size=144,dtype=np.uint16)
        tokens[-1]=1
        tokens.tofile(data/f'{split}_shard_00000.bin')
    # Unmixed mode still uses the same actual document-first routed sampler.
    args=['--shard-dir',str(data),'--model-preset','cpu_tiny',
          '--dtype','float32','--block-size','4','--proposal-width','16',
          '--decoder-width','16','--decoder-layers','1','--decoder-heads','2',
          '--block-group-size','1','--val-batches','1','--val-every','1',
          '--save-every','1','--steps','1','--batch-size','1',
          '--checkpoint-dir',str(tmp_path/'ckpt')]
    cli.main(args)
    metrics=json.loads((tmp_path/'ckpt'/'metrics.jsonl').read_text().splitlines()[0])
    assert metrics['unique_supervised_tokens']>0
    assert np.isfinite(metrics['final_ce'])
    assert np.isfinite(metrics['proposal_ce'])
    assert metrics['grad_norm']>0
    assert (tmp_path/'ckpt'/'step_00000001'/'training_state.json').exists()
