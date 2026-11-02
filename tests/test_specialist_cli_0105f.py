"""CLI safety does not import/allocate the 348M foundation without --run."""
import subprocess
import sys
from pathlib import Path


def test_dry_run_cannot_touch_checkpoint(tmp_path):
    base=tmp_path/'foundation'; base.mkdir()
    mixture=tmp_path/'mixed'; mixture.mkdir()
    out=tmp_path/'specialist'
    root=Path(__file__).resolve().parents[1]
    args=[sys.executable,str(root/'scripts/train_0105_specialist.py'),
        '--foundation-checkpoint',str(base),'--mixed-shard-root',str(mixture),
        '--specialist-checkpoint',str(out)]
    result=subprocess.run(args,capture_output=True,text=True,cwd=root)
    assert result.returncode==0, result.stderr
    assert 'DRY RUN' in result.stdout
    assert not out.exists()


def test_cli_does_not_overwrite_existing_specialist(tmp_path):
    base=tmp_path/'foundation'; base.mkdir()
    mixture=tmp_path/'mixed'; mixture.mkdir()
    out=tmp_path/'specialist'; out.mkdir()
    root=Path(__file__).resolve().parents[1]
    result=subprocess.run([sys.executable,str(root/'scripts/train_0105_specialist.py'),
        '--foundation-checkpoint',str(base),'--mixed-shard-root',str(mixture),
        '--specialist-checkpoint',str(out)],capture_output=True,text=True,cwd=root)
    assert result.returncode!=0
    assert 'refusing to replace' in result.stderr
