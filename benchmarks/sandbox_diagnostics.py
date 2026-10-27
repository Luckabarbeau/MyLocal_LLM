"""Non-invasive preflight for generated-code sandbox support.

Use ``python -m benchmarks.sandbox_diagnostics`` on the training host.
Does not execute model-generated code or change host security settings.
"""
from __future__ import annotations
import json
import shutil
import subprocess
import tempfile
from pathlib import Path

from benchmarks.capability_suite import _sandbox


def _sysctl(name):
    target = Path('/proc/sys') / name.replace('.', '/')
    try:
        return target.read_text(encoding='utf-8').strip()
    except OSError:
        return None


def report():
    tools = {name: shutil.which(name) for name in ('bwrap', 'g++', 'python3', 'apparmor_parser')}
    output = {'tools': tools, 'settings': {
        'kernel.apparmor_restrict_unprivileged_userns': _sysctl('kernel.apparmor_restrict_unprivileged_userns'),
        'kernel.unprivileged_userns_clone': _sysctl('kernel.unprivileged_userns_clone'),
        'user.max_user_namespaces': _sysctl('user.max_user_namespaces'),
    }}
    with tempfile.TemporaryDirectory(prefix='mini_llm_sandbox_preflight_') as name:
        output['sandbox_preflight'] = _sandbox(['/usr/bin/python3', '-I', '-S', '-c',
                                               'print("MINI_LLM_SANDBOX_READY")'], Path(name), timeout=4)
    preflight = output['sandbox_preflight']
    output['sandbox_ready'] = (preflight.get('status') == 'finished'
                               and preflight.get('returncode') == 0
                               and 'MINI_LLM_SANDBOX_READY' in preflight.get('stdout', ''))
    if not tools['bwrap']:
        output['suggestion'] = 'Install bubblewrap: sudo apt install bubblewrap. Then rerun this diagnostic.'
    elif 'Failed RTM_NEWADDR' in preflight.get('detail', ''):
        output['suggestion'] = (
            'Bubblewrap is installed but its isolated loopback setup is denied. '
            'On Ubuntu 24.04+, check the packaged bwrap-userns-restrict AppArmor profile; '
            'see benchmarks/README_0096_HARD_CONTEXT_AND_SANDBOX.md. '
            'Avoid disabling AppArmor globally or using --share-net for generated code.')
    elif not output['sandbox_ready']:
        output['suggestion'] = ('Bubblewrap preflight failed or timed out. Inspect the error and '
                                'host security logs; do not bypass the sandbox.')
    else:
        output['suggestion'] = 'Sandbox setup passed. Generated-code execution should now be testable.'
    return output


def main():
    print(json.dumps(report(), indent=2))
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
