"""0093: local, versioned capability probes for base (non-instruction) models.

All tasks are stored in repo JSON; no external corpora are downloaded.
Generated code is NEVER executed or compiled on the host: the optional
Bubblewrap sandbox is required. Unavailable sandbox != failed task.
"""
from __future__ import annotations

import ast
import contextlib
import hashlib
import json
import math
import re
import shutil
from pathlib import Path
import subprocess
import tempfile

import numpy as np

from mini_llm.backend import xp, asnumpy
from benchmarks.live_suite import (
    _generator, _plain_4096_model, _preserve_global_rng,
    _strict_graph_decode_for_generation,
)


def _load(path, family):
    source = Path(path)
    data = json.loads(source.read_text(encoding='utf-8'))
    if data.get('version') != 1 or data.get('family') != family:
        raise ValueError(f'Expected version=1 family={family} in {source}')
    tasks = data.get('tasks')
    if not isinstance(tasks, list) or not tasks:
        raise ValueError(f'No tasks in {source}')
    ids = set()
    for task in tasks:
        if not isinstance(task, dict) or not isinstance(task.get('id'), str) or not task['id']:
            raise ValueError('Each task needs an id')
        if task['id'] in ids:
            raise ValueError(f'Duplicate task id {task["id"]}')
        ids.add(task['id'])
        if not isinstance(task.get('prompt'), str) or not task['prompt']:
            raise ValueError(f'Task {task["id"]} has no prompt')
    return tasks, hashlib.sha256(source.read_bytes()).hexdigest()


def _select_tasks(cases, task_ids):
    if task_ids is None:
        return cases
    requested = set(task_ids)
    known = {case['id'] for case in cases}
    if requested - known:
        raise ValueError(f'Unknown task IDs: {sorted(requested - known)}')
    selected = [case for case in cases if case['id'] in requested]
    if not selected:
        raise ValueError('Task filter selected no test cases')
    return selected


def _token_logprobs(model, tokenizer, prefix, choice, max_context=4096):
    """Score continuation only, not prompt; report tokenizer-boundary fallback.

    Context and candidate are independently tokenized to guarantee that the
    conditional target tokens occupy an unambiguous span. When concatenated
    tokenization has the same prefix, use its tokens instead for exact parity.
    """
    start = [int(v) for v in tokenizer.encode(prefix)]
    target = [int(v) for v in tokenizer.encode(choice)]
    together = [int(v) for v in tokenizer.encode(prefix + choice)]
    if not start or not target:
        raise ValueError('Candidate likelihood needs nonempty prefix and continuation tokens')
    exact = together[:len(start)] == start and len(together) > len(start)
    if exact:
        target = together[len(start):]
    ids = (start + target)[-int(max_context)-1:]
    targets = min(len(target), len(ids) - 1)
    if targets != len(target):
        raise ValueError('Continuation exceeds scoring context; shorten candidate')
    inputs = xp.asarray(np.asarray(ids[:-1], dtype=np.int64)[None, :])
    logits = model.forward(inputs, return_cache=False)
    scores = np.asarray(asnumpy(logits[0, -targets:, :]), dtype=np.float64)
    indices = np.asarray(target, dtype=np.int64)
    z = scores - scores.max(axis=-1, keepdims=True)
    z -= np.log(np.exp(z).sum(axis=-1, keepdims=True))
    logps = z[np.arange(targets), indices]
    return {'sum_logprob': float(logps.sum()), 'mean_logprob': float(logps.mean()),
            'target_tokens': targets, 'tokenization': 'joined' if exact else 'separate_boundary'}


def _generate(generator, prompt, *, max_new_tokens, seed, strategy, temperature, top_k,
              use_kv_cache):
    output, timing = generator.generate(
        prompt=prompt, max_new_tokens=int(max_new_tokens),
        seed=int(seed), strategy=strategy, temperature=float(temperature),
        top_k=int(top_k) if strategy == 'top-k' else None,
        use_kv_cache=bool(use_kv_cache), return_token_ids=True,
    )
    return {'text': output, 'token_ids': timing.get('generated_token_ids', [])}


def run_choices(model, tokenizer, *, tasks, family, scoring='mean', max_context=4096,
                task_ids=None):
    """Deterministic conditional likelihood test for math and logic."""
    if scoring not in ('mean', 'sum'):
        raise ValueError('scoring must be mean or sum')
    cases, fingerprint = _load(tasks, family)
    cases = _select_tasks(cases, task_ids)
    results = []
    with _plain_4096_model(model):
        for case in cases:
            choices = case.get('choices')
            correct = case.get('answer')
            if not isinstance(choices, list) or len(choices) < 2 or len(set(choices)) != len(choices):
                raise ValueError(f'Invalid candidate choices: {case["id"]}')
            if not isinstance(correct, int) or not 0 <= correct < len(choices):
                raise ValueError(f'Invalid correct candidate index: {case["id"]}')
            values = [_token_logprobs(model, tokenizer, case['prompt'], choice,
                                      max_context=max_context) for choice in choices]
            ranking = sorted(range(len(values)), key=lambda i: (-values[i][scoring+'_logprob'], i))
            results.append({'id': case['id'], 'prompt': case['prompt'], 'choices': choices,
                            'category': case.get('category','unspecified'),
                            'difficulty': case.get('difficulty','unspecified'),
                            'scenario_id': case.get('scenario_id'), 'stage': case.get('stage'),
                            'answer': correct, 'prediction': ranking[0],
                            'correct': ranking[0] == correct,
                            'correct_rank': ranking.index(correct)+1, 'scoring': scoring,
                            'scores': values})
    from benchmarks.progression_0097 import summarize_progression
    return {'family': family, 'tasks_sha256': fingerprint, 'scoring': scoring,
            'accuracy': sum(r['correct'] for r in results)/len(results), 'count': len(results),
            'progression': summarize_progression(results), 'cases': results}


def _check_python_syntax(source, function):
    """Separate valid *module* syntax from completing the supplied function.

    The first matching top-level function is the original prompt. A later
    duplicate definition does not count as completing that original body.
    """
    try:
        tree = ast.parse(source)
    except (SyntaxError, ValueError) as exc:
        return {'syntax_valid': False, 'function_found': False,
                'original_function_completed': False,
                'duplicate_function_definitions': False,
                'error': f'{type(exc).__name__}: {exc}'}
    funcs = [node for node in tree.body if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef))]
    matching = [f for f in funcs if f.name == function]
    def executable(f):
        return any(not isinstance(stmt, ast.Pass) and not
                   (isinstance(stmt, ast.Expr) and isinstance(stmt.value, ast.Constant)
                    and isinstance(stmt.value.value, str)) for stmt in f.body)
    completed = bool(matching and executable(matching[0]))
    return {'syntax_valid': True, 'function_found': bool(matching),
            'nontrivial_body': any(executable(f) for f in matching),
            'original_function_completed': completed,
            'duplicate_function_definitions': len(matching) > 1}


def _bubblewrap_available():
    return shutil.which('bwrap') is not None


def _sandbox(command, workspace, *, timeout=5):
    """Execute ONLY inside filesystem-isolated, no-network Bubblewrap.

    No fallback to host subprocess. This is a best-effort local sandbox; for
    hostile model output use a dedicated container/VM with stronger isolation.
    """
    if not _bubblewrap_available():
        return {'status': 'unavailable_no_bwrap'}
    import resource
    def limits():
        resource.setrlimit(resource.RLIMIT_CPU, (max(1, int(timeout)), max(2, int(timeout)+1)))
        resource.setrlimit(resource.RLIMIT_FSIZE, (16 * 1024 * 1024, 16 * 1024 * 1024))
        resource.setrlimit(resource.RLIMIT_NOFILE, (64, 64))
        resource.setrlimit(resource.RLIMIT_AS, (2 * 1024**3, 2 * 1024**3))
        resource.setrlimit(resource.RLIMIT_CORE, (0, 0))
    args = ['bwrap', '--unshare-all', '--die-with-parent', '--new-session',
            '--clearenv', '--setenv', 'PATH', '/usr/bin:/bin',
            '--setenv', 'HOME', '/work', '--setenv', 'TMPDIR', '/work',
            '--tmpfs', '/tmp', '--proc', '/proc', '--dev', '/dev']
    for location in ('/usr', '/bin', '/lib', '/lib64'):
        if Path(location).exists():
            args += ['--ro-bind', location, location]
    args += ['--bind', str(workspace), '/work', '--chdir', '/work', '--'] + list(command)
    try:
        cp = subprocess.run(args, capture_output=True, text=True, timeout=timeout + 2,
                            preexec_fn=limits, cwd=workspace, check=False)
        if cp.returncode != 0 and cp.stderr.lstrip().startswith('bwrap:'):
            hint = ('Ubuntu AppArmor/unprivileged namespace restrictions may block the '
                    'isolated loopback setup; run python -m benchmarks.sandbox_diagnostics. '
                    'Do not switch to --share-net or execute generated code on the host.'
                    if 'Failed RTM_NEWADDR' in cp.stderr else None)
            return {'status': 'sandbox_error', 'detail': cp.stderr[:3000],
                    'diagnostic_hint': hint}
        return {'status': 'finished', 'returncode': cp.returncode,
                'stdout': cp.stdout[:6000], 'stderr': cp.stderr[:3000]}
    except subprocess.TimeoutExpired:
        return {'status': 'timeout'}
    except OSError as exc:
        return {'status': 'sandbox_error', 'detail': str(exc)}


def _python_execution(source, case):
    if not _bubblewrap_available():
        return {'status': 'unavailable_no_bwrap'}
    tests = case.get('tests')
    if not isinstance(tests, list) or not tests:
        raise ValueError(f'Missing python unit tests for {case["id"]}')
    with tempfile.TemporaryDirectory(prefix='mini_llm_pycheck_') as root:
        base = Path(root)
        (base/'candidate.py').write_text(source, encoding='utf-8')
        (base/'tests.json').write_text(json.dumps({'function': case['function'], 'tests': tests}), encoding='utf-8')
        (base/'test_runner.py').write_text('''import json, sys\nwith open('tests.json') as f: spec=json.load(f)\nnamespace={}\nwith open('candidate.py') as f: source=f.read()\nexec(compile(source, 'candidate.py', 'exec'), namespace)\nfn=namespace[spec['function']]\nout=[]\nfor case in spec['tests']:\n    try:\n        actual=fn(*case['args'])\n        out.append({'ok':actual==case['expected']})\n    except BaseException as exc:\n        out.append({'ok':False,'error':type(exc).__name__})\nprint('MINI_LLM_TESTS='+json.dumps(out))\n''', encoding='utf-8')
        run = _sandbox(['/usr/bin/python3', '-I', '-S', 'test_runner.py'], base, timeout=5)
    if run['status'] != 'finished':
        return run
    marker = 'MINI_LLM_TESTS='
    line = next((l[len(marker):] for l in reversed(run['stdout'].splitlines())
                 if l.startswith(marker)), None)
    if line is None:
        return {'status': 'execution_error', 'returncode': run['returncode'],
                'stderr': run['stderr']}
    try:
        outcomes = json.loads(line)
    except ValueError:
        return {'status': 'invalid_runner_output'}
    if not isinstance(outcomes, list) or len(outcomes) != len(tests):
        return {'status': 'invalid_runner_output'}
    return {'status': 'evaluated', 'passed': sum(bool(x.get('ok')) for x in outcomes),
            'total': len(tests), 'all_passed': bool(outcomes) and all(x.get('ok') for x in outcomes)}


def _cpp_execution(source, case, *, run_tests):
    if not _bubblewrap_available():
        return {'status': 'unavailable_no_bwrap'}
    if not shutil.which('g++'):
        return {'status': 'unavailable_no_gxx'}
    with tempfile.TemporaryDirectory(prefix='mini_llm_cppcheck_') as root:
        base = Path(root)
        (base/'candidate.cpp').write_text(source, encoding='utf-8')
        check = _sandbox(['/usr/bin/g++', '-std=c++17', '-fsyntax-only', '-w',
                          'candidate.cpp'], base, timeout=8)
        if check['status'] != 'finished':
            return {'status': check['status'], 'compiles': None,
                    'stderr': check.get('stderr', check.get('detail', ''))}
        if check['returncode']:
            return {'status': 'evaluated', 'compiles': False, 'all_passed': False,
                    'stderr': check.get('stderr', '')}
        if not run_tests:
            return {'status': 'evaluated', 'compiles': True}
        harness = case.get('test_harness')
        if not isinstance(harness, str) or not harness:
            raise ValueError(f'Missing C++ test harness for {case["id"]}')
        (base/'harness.cpp').write_text('#include "candidate.cpp"\n'+harness, encoding='utf-8')
        compile_result = _sandbox(['/usr/bin/g++', '-std=c++17', '-O0', '-w',
                                   'harness.cpp', '-o', '/work/test_bin'], base, timeout=12)
        if compile_result['status'] != 'finished':
            return {'status': compile_result['status'], 'compiles': True,
                    'test_compile_ok': None, 'stderr': compile_result.get('detail', '')}
        if compile_result['returncode']:
            return {'status': 'evaluated', 'compiles': True,
                    'test_compile_ok': False, 'all_passed': False,
                    'stderr': compile_result.get('stderr', '')}
        run = _sandbox(['/work/test_bin'], base, timeout=5)
        return {'status': 'evaluated' if run['status']=='finished' else run['status'],
                'compiles': True, 'test_compile_ok': True,
                'all_passed': run.get('returncode') == 0 if run['status']=='finished' else False}


def run_code(model, tokenizer, *, tasks, language, run_tests=False,
             max_new_tokens=160, seeds=(42,), strategy='greedy',
             temperature=0.8, top_k=40, use_kv_cache=True, task_ids=None,
             require_native_cuda_graph_decode=False,
             prefer_native_cuda_graph_decode=False):
    if language not in ('python', 'cpp'):
        raise ValueError('language must be python or cpp')
    cases, fingerprint = _load(tasks, language)
    cases = _select_tasks(cases, task_ids)
    if ((require_native_cuda_graph_decode or prefer_native_cuda_graph_decode)
            and not use_kv_cache):
        raise ValueError('native CUDA graph decode requires use_kv_cache=true')
    generated = []
    with (_strict_graph_decode_for_generation(
              require_native_cuda_graph_decode, prefer_native_cuda_graph_decode),
          _preserve_global_rng()):
        with _plain_4096_model(model):
            generator = _generator(model, tokenizer)
            for case in cases:
                for seed in seeds:
                    result = _generate(generator, case['prompt'], max_new_tokens=max_new_tokens,
                                       seed=seed, strategy=strategy, temperature=temperature,
                                       top_k=top_k, use_kv_cache=use_kv_cache)
                    source = case['prompt']+result['text']
                    if language == 'python':
                        check = _check_python_syntax(source, case['function'])
                        if run_tests:
                            if check['syntax_valid'] and check['function_found']:
                                check['unit_tests'] = _python_execution(source, case)
                            else:
                                check['unit_tests'] = {'status': 'invalid_source', 'all_passed': False}
                    else:
                        check = _cpp_execution(source, case, run_tests=run_tests)
                    generated.append({'id': case['id'], 'prompt': case['prompt'],
                                      'category': case.get('category','unspecified'),
                                      'difficulty': case.get('difficulty','unspecified'),
                                      'seed': int(seed), 'generated_text': result['text'], 'checks': check})
            del generator
    syntactic = [r['checks'].get('syntax_valid') if language == 'python' else
                 r['checks'].get('compiles') for r in generated]
    checked = [v for v in syntactic if v is not None]
    tests = [r['checks'].get('unit_tests', r['checks']) for r in generated]
    # Invalid syntax/compilation is a definite task failure. A missing sandbox,
    # by contrast, leaves functional correctness unknown; never imply pass@1=0.
    known = [x for x in tests if x.get('status') in ('evaluated', 'invalid_source')
             and 'all_passed' in x]
    complete = run_tests and len(known) == len(generated)
    function_found = [bool(r['checks'].get('function_found')) for r in generated] if language == 'python' else []
    nontrivial = [bool(r['checks'].get('nontrivial_body')) for r in generated] if language == 'python' else []
    originals = [bool(r['checks'].get('original_function_completed')) for r in generated] if language == 'python' else []
    duplicates = [bool(r['checks'].get('duplicate_function_definitions')) for r in generated] if language == 'python' else []
    return {'language': language, 'tasks_sha256': fingerprint, 'count': len(generated),
            'syntax_or_compile_rate': sum(checked)/len(checked) if checked else None,
            'syntax_or_compile_evaluated': len(checked),
            'function_found_rate': sum(function_found)/len(function_found) if function_found else None,
            'nontrivial_body_rate': sum(nontrivial)/len(nontrivial) if nontrivial else None,
            'original_function_completion_rate': sum(originals)/len(originals) if originals else None,
            'duplicate_function_definition_rate': sum(duplicates)/len(duplicates) if duplicates else None,
            'unit_test_pass_rate': sum(bool(x['all_passed']) for x in known)/len(generated)
                                   if complete else None,
            'unit_tests_evaluated': len([x for x in tests if x.get('status') == 'evaluated'
                                         and 'all_passed' in x]),
            'functional_known_count': len(known) if run_tests else 0,
            'functional_unknown_count': (len(generated)-len(known)) if run_tests else len(generated),
            'cases': generated}


def run_terminal(model, tokenizer, *, tasks, scoring='mean', max_context=4096,
                 max_new_tokens=48, seeds=(42,), strategy='greedy', temperature=0.8,
                 top_k=40, use_kv_cache=True, task_ids=None,
                 require_native_cuda_graph_decode=False,
                 prefer_native_cuda_graph_decode=False):
    if scoring not in ('mean', 'sum'):
        raise ValueError('scoring must be mean or sum')
    cases, fingerprint = _load(tasks, 'terminal')
    cases = _select_tasks(cases, task_ids)
    if ((require_native_cuda_graph_decode or prefer_native_cuda_graph_decode)
            and not use_kv_cache):
        raise ValueError('native CUDA graph decode requires use_kv_cache=true')
    results = []
    with (_strict_graph_decode_for_generation(
              require_native_cuda_graph_decode, prefer_native_cuda_graph_decode),
          _preserve_global_rng()):
        with _plain_4096_model(model):
            generator = _generator(model, tokenizer)
            for case in cases:
                choices = case['choices']
                values = [_token_logprobs(model, tokenizer, case['prompt'], choice,
                                          max_context=max_context) for choice in choices]
                selected = max(range(len(values)), key=lambda i: values[i][scoring+'_logprob'])
                samples = []
                for seed in seeds:
                    sample = _generate(generator, case['prompt'], max_new_tokens=max_new_tokens,
                                       seed=seed, strategy=strategy, temperature=temperature,
                                       top_k=top_k, use_kv_cache=use_kv_cache)
                    line = sample['text'].splitlines()[0].strip() if sample['text'].strip() else ''
                    # Regex used as a bounded *static* check, not shell execution.
                    matched = any(re.fullmatch(pat, line) is not None for pat in case['accepted_patterns'])
                    samples.append({'seed': int(seed), 'generated_text': sample['text'],
                                    'first_line': line, 'static_match': matched})
                results.append({'id': case['id'], 'prompt': case['prompt'],
                                'category': case.get('category','unspecified'),
                                'difficulty': case.get('difficulty','unspecified'),
                                'scenario_id': case.get('scenario_id'), 'stage': case.get('stage'),
                                'correct': selected == case['answer'],
                                'chosen': selected, 'answer': case['answer'],
                                'choice_correct': selected == case['answer'],
                                'scores': values, 'generations': samples})
    from benchmarks.progression_0097 import summarize_progression
    return {'tasks_sha256': fingerprint, 'count': len(results),
            'progression': summarize_progression(results),
            'choice_accuracy': sum(r['choice_correct'] for r in results)/len(results),
            'generated_static_match_rate': sum(s['static_match'] for r in results for s in r['generations'])/
                                           sum(len(r['generations']) for r in results),
            'cases': results, 'note': 'Static command matching only; commands are never executed.'}


def run_narrative(model, tokenizer, *, tasks, scoring='mean', max_context=4096,
                  max_new_tokens=128, seeds=(42,), strategy='top-k', temperature=0.8,
                  top_k=40, use_kv_cache=True, task_ids=None,
                  require_native_cuda_graph_decode=False,
                  prefer_native_cuda_graph_decode=False):
    """Score context-dependent factual choices; optionally sample continuations.

    ``max_new_tokens=0`` avoids creating inference/KV-cache state and is the
    intended fast path for large, deterministic narrative challenge sets.
    Individual challenge types are reported separately, with per-choice score
    margins so a ceiling-level accuracy need not mask changing confidence.
    """
    if scoring not in ('mean', 'sum'):
        raise ValueError('scoring must be mean or sum')
    if int(max_new_tokens) < 0:
        raise ValueError('max_new_tokens must be nonnegative')
    if int(max_new_tokens) > 0 and not seeds:
        raise ValueError('At least one seed is required when generation is enabled')
    if (int(max_new_tokens) > 0 and
            (require_native_cuda_graph_decode or prefer_native_cuda_graph_decode) and
            not use_kv_cache):
        raise ValueError('native CUDA graph decode requires use_kv_cache=true')
    cases, fingerprint = _load(tasks, 'narrative')
    cases = _select_tasks(cases, task_ids)
    results = []
    graph_decode = (
        _strict_graph_decode_for_generation(
            require_native_cuda_graph_decode, prefer_native_cuda_graph_decode)
        if int(max_new_tokens) > 0 else contextlib.nullcontext()
    )
    with graph_decode, _preserve_global_rng():
        with _plain_4096_model(model):
            generator = (_generator(model, tokenizer) if int(max_new_tokens) else None)
            for case in cases:
                recall_prompt = case['recall_prompt']
                choices = case['choices']
                answer = case['answer']
                if not isinstance(choices, list) or len(choices) < 2 or len(set(choices)) != len(choices):
                    raise ValueError(f'Invalid narrative choices: {case["id"]}')
                if not isinstance(answer, int) or not 0 <= answer < len(choices):
                    raise ValueError(f'Invalid narrative answer: {case["id"]}')
                values = [_token_logprobs(model, tokenizer, recall_prompt, choice,
                                          max_context=max_context) for choice in choices]
                scores = np.asarray([v[scoring+'_logprob'] for v in values], dtype=np.float64)
                order = sorted(range(len(choices)), key=lambda i: (-scores[i], i))
                chosen = order[0]
                # A cross-candidate softmax is diagnostic, not a calibrated
                # factual confidence probability or a model next-token prob.
                stable = scores - np.max(scores)
                choice_probs = np.exp(stable) / np.exp(stable).sum()
                runner_up = max(scores[j] for j in range(len(scores)) if j != answer)
                samples = []
                if generator is not None:
                    for seed in seeds:
                        text = _generate(generator, case['prompt'], max_new_tokens=max_new_tokens,
                                         seed=seed, strategy=strategy, temperature=temperature,
                                         top_k=top_k, use_kv_cache=use_kv_cache)['text']
                        anchors = [s.lower() for s in case.get('anchors', [])]
                        retention = sum(a in text.lower() for a in anchors)/len(anchors) if anchors else None
                        samples.append({'seed': int(seed), 'generated_text': text,
                                        'anchor_mention_fraction': retention})
                results.append({'id': case['id'], 'challenge_type': case.get('challenge_type', 'simple_recall'),
                                'difficulty': case.get('difficulty', 'unspecified'),
                                'scenario_id': case.get('scenario_id'), 'stage': case.get('stage'),
                                'prompt': case['prompt'], 'recall_prompt': recall_prompt,
                                'choices': choices, 'answer': answer,
                                'recall_correct': chosen == answer,
                                'correct_rank': order.index(answer) + 1,
                                'correct_choice_score_margin': float(scores[answer] - runner_up),
                                'correct_choice_relative_probability': float(choice_probs[answer]),
                                'chosen': chosen, 'scores': values, 'generations': samples})
            del generator
    by_type = {}
    for row in results:
        typ = row['challenge_type']
        by_type.setdefault(typ, []).append(row)
    groups = {}
    for typ, rows in sorted(by_type.items()):
        groups[typ] = {
            'count': len(rows),
            'accuracy': sum(r['recall_correct'] for r in rows)/len(rows),
            'mean_correct_rank': sum(r['correct_rank'] for r in rows)/len(rows),
            'mean_correct_choice_score_margin': sum(r['correct_choice_score_margin'] for r in rows)/len(rows),
            'mean_correct_choice_relative_probability': sum(r['correct_choice_relative_probability'] for r in rows)/len(rows),
        }
    from benchmarks.progression_0097 import summarize_progression
    return {'tasks_sha256': fingerprint, 'count': len(results), 'scoring': scoring,
            'progression': summarize_progression(results, correct_key='recall_correct'),
            'fact_recall_accuracy': sum(r['recall_correct'] for r in results)/len(results),
            'mean_correct_rank': sum(r['correct_rank'] for r in results)/len(results),
            'mean_correct_choice_score_margin': sum(r['correct_choice_score_margin'] for r in results)/len(results),
            'mean_correct_choice_relative_probability': sum(r['correct_choice_relative_probability'] for r in results)/len(results),
            'by_challenge_type': groups,
            'generation_enabled': int(max_new_tokens) > 0,
            'cases': results,
            'warning': ('Multiple-choice likelihood is prompt/length-sensitive; '
                        'relative choice softmax is not calibrated confidence. '
                        'Anchor mentions are a lexical proxy, not a validated coherence score.')}
