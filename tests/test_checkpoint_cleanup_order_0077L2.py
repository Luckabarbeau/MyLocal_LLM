"""Guard smoke-step saving and the streaming-cache shutdown order."""

import inspect
import sys

import train_model


def test_max_run_steps_is_already_part_of_0077k(monkeypatch):
    monkeypatch.setattr(sys, 'argv', [
        'train_model.py', '--resume-from', 'healthy_checkpoint',
        '--total-steps', '160000', '--max-run-steps', '5',
        '--gradient-accumulation-weighting', 'supervised_tokens',
    ])
    args = train_model.parse_args()
    assert args.max_run_steps == 5
    assert args.total_steps == 160000
    assert args.gradient_accumulation_weighting == 'supervised_tokens'


def test_final_save_precedes_streaming_cache_close():
    source = inspect.getsource(train_model.main)
    save = 'if args.checkpoint_dir and trainer.step % trainer.save_interval != 0:'
    close = 'trainer.streaming_cache.close()'
    assert save in source
    assert source.index(save) < source.index(close)
    assert source.count('trainer.save()') == 1
