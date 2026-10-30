"""0092: Offline reference-analysis HTML and longitudinal metrics CSV."""
import csv
import html
import json
from pathlib import Path

import numpy as np


def summary_metrics(kind, result):
    if kind == 'wikitext':
        return {k: result[k] for k in ('cross_entropy', 'perplexity', 'bits_per_byte', 'scored_tokens')}
    if kind == 'generation':
        summary = result['summary']['overall']
        metrics = {k: summary.get(k) for k in (
            'count', 'mean_repeated_8gram_rate', 'mean_distinct_4',
            'repeat_8gram_onset_fraction',
        )}
        timing = result.get('timing', {})
        metrics.update({
            'decode_tokens_per_second': timing.get('decode_tokens_per_sec'),
            'request_tokens_per_second': timing.get('request_tokens_per_sec'),
            'generation_prefill_seconds': timing.get('prefill_time_s'),
            'native_cuda_graph_generations': timing.get('native_cuda_graph_generations'),
        })
        return metrics
    if kind in ('math', 'logic'):
        return {'task_count': result['count'], 'choice_accuracy': result['accuracy'],
                'ladder_complete_rate': result.get('progression',{}).get('all_stages_passed_rate'),
                'mean_passed_prefix': result.get('progression',{}).get('mean_passed_prefix')}
    if kind in ('python_code', 'cpp_code'):
        return {k: result.get(k) for k in ('count', 'syntax_or_compile_rate',
                 'syntax_or_compile_evaluated', 'unit_test_pass_rate', 'unit_tests_evaluated',
                 'function_found_rate', 'nontrivial_body_rate',
                 'original_function_completion_rate', 'duplicate_function_definition_rate',
                 'functional_known_count',
                 'functional_unknown_count')}
    if kind == 'terminal':
        return {'task_count': result['count'], 'choice_accuracy': result['choice_accuracy'],
                'generated_static_match_rate': result['generated_static_match_rate'],
                'ladder_complete_rate': result.get('progression',{}).get('all_stages_passed_rate'),
                'mean_passed_prefix': result.get('progression',{}).get('mean_passed_prefix')}
    if kind == 'narrative':
        return {'task_count': result['count'], 'fact_recall_accuracy': result['fact_recall_accuracy'],
                'mean_correct_rank': result.get('mean_correct_rank'),
                'mean_correct_choice_score_margin': result.get('mean_correct_choice_score_margin'),
                'mean_correct_choice_relative_probability': result.get('mean_correct_choice_relative_probability'),
                'ladder_complete_rate': result.get('progression',{}).get('all_stages_passed_rate'),
                'mean_passed_prefix': result.get('progression',{}).get('mean_passed_prefix')}
    if kind == 'long_retrieval':
        out = {'task_count': result['count'],
               'long_retrieval_accuracy': sum(x['correct'] for x in result['cases'])/result['count']}
        for d, summary in result['by_distance'].items():
            if d in ('1000','4000','16000'):
                out['retrieval_accuracy_'+d] = summary['accuracy']
                out['retrieval_margin_'+d] = summary['mean_correct_margin']
        return out
    if kind == 'reference_continuation' and 'diagnostics' in result:
        micro = result['diagnostics']['micro']
        all_tokens = micro['all_tokens']; content = micro['content_like_tokens']
        return {
            # Preserve original 0092 macro (mean over reference texts) semantics.
            'reference_count': len(result['cases']),
            'mean_reference_nll': float(np.mean([r['mean_nll'] for r in result['cases']])),
            'mean_top1_accuracy': float(np.mean([r['top1_accuracy'] for r in result['cases']])),
            'mean_top5_accuracy': float(np.mean([r['top5_accuracy'] for r in result['cases']])),
            # New 0095 micro (token-weighted) and content-like lexical subsets.
            'reference_scored_tokens': result['diagnostics']['total_scored_tokens'],
            'reference_micro_nll': all_tokens['mean_nll'],
            'reference_micro_top1_accuracy': all_tokens['top1_accuracy'],
            'reference_micro_top5_accuracy': all_tokens['top5_accuracy'],
            'content_token_count': content['count'],
            'content_mean_nll': content['mean_nll'],
            'content_top1_accuracy': content['top1_accuracy'],
            'content_top5_accuracy': content['top5_accuracy'],
        }

    return {
        'reference_count': len(result['cases']),
        'mean_reference_nll': float(np.mean([r['mean_nll'] for r in result['cases']])),
        'mean_top1_accuracy': float(np.mean([r['top1_accuracy'] for r in result['cases']])),
        'mean_top5_accuracy': float(np.mean([r['top5_accuracy'] for r in result['cases']])),
    }


def reference_html(result):
    """Small dependency-free report with clickable per-token predictions."""
    output = ['''<!doctype html><html lang="en"><head><meta charset="utf-8">
<title>Reference continuation analysis</title><style>
body{font:15px/1.5 system-ui,sans-serif;margin:3rem auto;max-width:1100px;padding:0 1rem;color:#233}
.ref{border:1px solid #ccd;border-radius:12px;padding:1rem;margin:2rem 0}
.token{display:inline-block;margin:1px;padding:2px 4px;border-radius:4px;white-space:pre-wrap;font-family:monospace}
.token details{font-family:system-ui;font-size:12px} summary{cursor:pointer}
pre{white-space:pre-wrap;background:#f3f5f8;padding:1rem;border-radius:8px;overflow-wrap:anywhere}
small{color:#556}
</style></head><body><h1>Reference continuation analysis</h1>
<p>Colors reflect the probability of the reference token, not correctness.</p>''']
    if 'diagnostics' in result:
        output.append('<h2>Category-level reference prediction</h2>')
        output.append('<p><small>Content-like tokens are a heuristic subset of subword pieces, not validated semantic facts.</small></p>')
        output.append('<table border="1" cellpadding="7" cellspacing="0"><tr><th>Domain</th><th>Tokens</th><th>Top-1</th><th>Top-5</th><th>NLL</th><th>Content-like top-1</th><th>Content-like top-5</th><th>Content-like NLL</th></tr>')
        for category, stats in result['diagnostics']['by_category'].items():
            allm = stats['all_tokens']; content = stats['content_like_tokens']
            def percentage(v): return f'{v:.1%}' if v is not None else 'n/a'
            def decimal(v): return f'{v:.3f}' if v is not None else 'n/a'
            output.append('<tr><td>'+html.escape(str(category))+'</td><td>'+str(allm['count'])+'</td><td>'+percentage(allm['top1_accuracy'])+'</td><td>'+percentage(allm['top5_accuracy'])+'</td><td>'+decimal(allm['mean_nll'])+'</td><td>'+percentage(content['top1_accuracy'])+'</td><td>'+percentage(content['top5_accuracy'])+'</td><td>'+decimal(content['mean_nll'])+'</td></tr>')
        output.append('</table>')
    for case in result['cases']:
        output.append('<section class="ref"> <h2>' + html.escape(case['id']) + '</h2>')
        output.append(f'<small>NLL {case["mean_nll"]:.3f}; top-1 {case["top1_accuracy"]:.1%}; top-5 {case["top5_accuracy"]:.1%}</small>')
        content = case.get('diagnostics',{}).get('content_like_tokens',{})
        if content.get('count'):
            output.append(f'<p><small>Content-like tokens: {content["count"]} · top-1 {content["top1_accuracy"]:.1%} · top-5 {content["top5_accuracy"]:.1%} · mean NLL {content["mean_nll"]:.3f}</small></p>')
        surprises = case.get('diagnostics',{}).get('most_surprising',[])
        if surprises:
            output.append('<details><summary>Eight most surprising reference tokens</summary><pre>'+
                          html.escape(json.dumps(surprises, ensure_ascii=False, indent=2))+'</pre></details>')
        output.append('<p>')
        for token in case['token_predictions']:
            prob = token['actual_probability']
            color = '#d6f5df' if prob >= .3 else '#fff1bf' if prob >= .05 else '#f9d7d5'
            label = html.escape(token['actual_text']) or '∅'
            suggestions = ', '.join(f"{html.escape(str(t['text']))}: {t['probability']:.3g}" for t in token['top'])
            title = html.escape(f"P={prob:.4g}; rank={token['actual_rank']}; top: {suggestions}", quote=True)
            output.append(f'<span class="token" style="background:{color}" title="{title}">{label}</span>')
        output.append('</p><h3>Free continuations from cut points</h3>')
        for sample in case['cutoff_generations']:
            output.append(f'<details><summary>Cut {sample["token_cutoff"]} · seed {sample["seed"]} · {html.escape(sample["strategy"])} · {html.escape(sample["prefix_contract"])}</summary>')
            if 'matched_reference_prefix_tokens' in sample:
                output.append('<p><small>Exact matching prefix tokens: '+str(sample['matched_reference_prefix_tokens'])+
                              '; reference next-token rank: '+str(sample['reference_next_token_rank'])+
                              '; next-token probability: '+f'{sample["reference_next_token_probability"]:.3g}'+
                              '; repeated 8-gram rate: '+
                              str(sample.get('generation_repetition',{}).get('repeated_8gram_rate'))+
                              ' (reference overlap is not a correctness score)</small></p>')
            output.append('<h4>Prefix</h4><pre>'+html.escape(sample['prefix'])+'</pre>')
            output.append('<h4>Model generation</h4><pre>'+html.escape(sample['generated_text'])+'</pre>')
            output.append('<h4>Reference suffix (one possible continuation)</h4><pre>'+html.escape(sample['reference_suffix'])+'</pre></details>')
        output.append('</section>')
    output.append('</body></html>')
    return '\n'.join(output)


def rebuild_summary_csv(root, config_hash):
    columns = ('benchmark_id', 'benchmark_type', 'optimizer_step', 'supervised_tokens_total',
               'cross_entropy', 'perplexity', 'bits_per_byte', 'scored_tokens',
               'mean_repeated_8gram_rate', 'mean_distinct_4', 'repeat_8gram_onset_fraction',
               'mean_reference_nll', 'mean_top1_accuracy', 'mean_top5_accuracy',
               'reference_count', 'reference_scored_tokens',
               'reference_micro_nll', 'reference_micro_top1_accuracy',
               'reference_micro_top5_accuracy',
               'content_token_count', 'content_mean_nll',
               'content_top1_accuracy', 'content_top5_accuracy',
               'checkpoint_id', 'task_count', 'count', 'choice_accuracy',
               'syntax_or_compile_rate', 'syntax_or_compile_evaluated',
               'unit_test_pass_rate', 'unit_tests_evaluated',
               'function_found_rate', 'nontrivial_body_rate',
               'original_function_completion_rate', 'duplicate_function_definition_rate',
               'functional_known_count',
               'functional_unknown_count',
               'generated_static_match_rate', 'fact_recall_accuracy',
               'mean_correct_rank', 'mean_correct_choice_score_margin',
               'mean_correct_choice_relative_probability',
               'ladder_complete_rate', 'mean_passed_prefix',
               'long_retrieval_accuracy',
               'retrieval_accuracy_1000', 'retrieval_accuracy_4000', 'retrieval_accuracy_16000',
               'retrieval_margin_1000', 'retrieval_margin_4000', 'retrieval_margin_16000')
    collected = []
    for path in Path(root).glob('step_*/**/result.json'):
        try:
            record = json.loads(path.read_text(encoding='utf-8'))
        except (OSError, ValueError):
            continue
        if record.get('suite_hash') == config_hash and record.get('status') == 'complete':
            row = {k: record.get(k, '') for k in columns}
            row.update({k: v for k, v in record.get('summary_metrics', {}).items() if k in columns})
            collected.append(row)
    collected.sort(key=lambda x: (int(x['optimizer_step']), str(x['benchmark_id'])))
    dest = Path(root) / 'metrics.csv'
    dest.parent.mkdir(parents=True, exist_ok=True)
    temp = dest.with_name(dest.name + '.tmp')
    with temp.open('w', encoding='utf-8', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=columns)
        writer.writeheader()
        writer.writerows(collected)
    temp.replace(dest)


def narrative_detail_csv(result, dest):
    """Write comparable per-subtype and per-item context-tracking metrics."""
    dest = Path(dest)
    group_columns = ('challenge_type', 'count', 'accuracy', 'mean_correct_rank',
                     'mean_correct_choice_score_margin',
                     'mean_correct_choice_relative_probability',
               'ladder_complete_rate', 'mean_passed_prefix',
               'long_retrieval_accuracy',
               'retrieval_accuracy_1000', 'retrieval_accuracy_4000', 'retrieval_accuracy_16000',
               'retrieval_margin_1000', 'retrieval_margin_4000', 'retrieval_margin_16000')
    with (dest/'narrative_by_type.csv').open('w', encoding='utf-8', newline='') as stream:
        writer = csv.DictWriter(stream, fieldnames=group_columns)
        writer.writeheader()
        for kind, stats in sorted(result.get('by_challenge_type', {}).items()):
            writer.writerow({'challenge_type': kind, **stats})
    case_columns = ('id', 'challenge_type', 'difficulty', 'recall_correct',
                    'correct_rank', 'correct_choice_score_margin',
                    'correct_choice_relative_probability')
    with (dest/'narrative_per_case.csv').open('w', encoding='utf-8', newline='') as stream:
        writer = csv.DictWriter(stream, fieldnames=case_columns)
        writer.writeheader()
        for item in result['cases']:
            writer.writerow({k: item.get(k) for k in case_columns})


def capability_html(kind, result):
    """Lightweight offline inspection report; generated text is untrusted HTML."""
    esc = html.escape
    out = ['<!doctype html><html lang="en"><head><meta charset="utf-8">',
           '<title>0093 capability benchmark</title><style>',
           'body{font:15px/1.5 system-ui,sans-serif;margin:2rem auto;max-width:1050px;padding:0 1rem}',
           'article{border:1px solid #cbd5e1;padding:1rem;margin:1rem 0;border-radius:10px}',
           'pre{white-space:pre-wrap;overflow-wrap:anywhere;background:#f5f6f9;padding:1rem}',
           '</style></head><body><h1>'+esc(kind)+'</h1>']
    if result.get('progression'):
        progression=result['progression']
        out.append('<h2>Progressive difficulty and first failed stage</h2>')
        out.append('<pre>'+esc(json.dumps({'by_difficulty':progression['by_difficulty'],
                          'by_category':progression['by_category'],
                          'ladders':progression['ladders']}, indent=2, ensure_ascii=False))+'</pre>')
    if kind == 'narrative' and result.get('by_challenge_type'):
        out.append('<h2>Context challenge subtypes</h2><pre>'+esc(json.dumps(result['by_challenge_type'], indent=2, ensure_ascii=False))+'</pre>')
    for case in result.get('cases', []):
        out.append('<article><h2>'+esc(str(case['id']))+'</h2>')
        if case.get('prompt') is not None:
            out.append('<h3>Prompt</h3><pre>'+esc(str(case['prompt']))+'</pre>')
        if case.get('scores') is not None:
            out.append('<h3>Candidate scores</h3><pre>'+esc(json.dumps(case['scores'],
                             indent=2, ensure_ascii=False))+'</pre>')
        for k in ('checks','correct','choice_correct','recall_correct','correct_rank'):
            if k in case:
                out.append('<p><b>'+esc(k)+'</b>: '+esc(str(case[k]))+'</p>')
        gen_list = case.get('generations')
        if gen_list is None and 'generated_text' in case:
            gen_list = [case]
        for generation in gen_list or []:
            out.append('<h3>Generated text (seed '+esc(str(generation.get('seed'))) +')</h3><pre>'+
                       esc(str(generation.get('generated_text','')) )+'</pre>')
        out.append('</article>')
    out.append('</body></html>')
    return '\n'.join(out)


def reference_per_case_csv(result, destination):
    """Write compact comparable rows; full token diagnostics remain in result.json."""
    columns=('reference_id','category','scored_tokens','mean_nll','top1_accuracy',
             'top5_accuracy','top10_accuracy','content_tokens',
             'content_mean_nll','content_top1_accuracy','content_top5_accuracy')
    destination=Path(destination)
    with destination.open('w',encoding='utf-8',newline='') as f:
        writer=csv.DictWriter(f,fieldnames=columns)
        writer.writeheader()
        for case in result['cases']:
            content=case.get('diagnostics',{}).get('content_like_tokens',{})
            writer.writerow({
                'reference_id':case['id'],'category':case.get('category'),
                'scored_tokens':len(case['token_predictions']),
                'mean_nll':case['mean_nll'],
                'top1_accuracy':case['top1_accuracy'],
                'top5_accuracy':case['top5_accuracy'],
                'top10_accuracy':case['top10_accuracy'],
                'content_tokens':content.get('count'),
                'content_mean_nll':content.get('mean_nll'),
                'content_top1_accuracy':content.get('top1_accuracy'),
                'content_top5_accuracy':content.get('top5_accuracy'),
            })


def long_retrieval_html(result):
    """Show actual *token distance* and score margins, never false full-context claims."""
    out=['<!doctype html><html><meta charset="utf-8"><title>Long retrieval</title>',
         '<style>body{font:15px system-ui;max-width:980px;margin:auto;padding:2em}table{border-collapse:collapse}td,th{border:1px solid #bbb;padding:.5em}pre{white-space:pre-wrap}</style>',
         '<h1>Distance-controlled factual retrieval</h1><p>'+html.escape(result['warning'])+'</p>',
         '<pre>'+html.escape(json.dumps(result['by_distance'],indent=2))+'</pre>',
         '<table><tr><th>Scenario</th><th>Actual distance (tokens)</th><th>Long memory</th><th>Answer</th><th>Prediction</th><th>Margin</th><th>Correct</th></tr>']
    for r in result['cases']:
        cols=(r['id'],str(r['measured_distance_tokens']),str(r['history_path_active']),r['answer'],
              r['prediction'],f"{r['correct_margin']:.3f}",str(r['correct']))
        out.append('<tr>'+''.join('<td>'+html.escape(str(c))+'</td>' for c in cols)+'</tr>')
    out.append('</table></html>')
    return '\n'.join(out)

def long_retrieval_case_csv(result, destination):
    keys=['id','category','requested_distance_tokens','measured_distance_tokens','prompt_tokens',
          'history_path_active','fact_in_historical_store','answer','prediction','correct','correct_rank',
          'full_vocab_rank','correct_probability','correct_margin']
    with Path(destination).open('w',encoding='utf-8',newline='') as f:
        w=csv.DictWriter(f,fieldnames=keys);w.writeheader()
        for row in result['cases']:
            w.writerow({k:row.get(k) for k in keys})


def progression_detail_csv(result, destination):
    """Stable, compact stage and difficulty summaries for comparing checkpoints."""
    progression = result.get('progression', {})
    if not progression:
        return
    root = Path(destination)
    with (root/'progression_by_scenario.csv').open('w', encoding='utf-8', newline='') as f:
        columns=('scenario_id','first_failed_stage','passed_prefix_length',
                 'all_passed','highest_stage_tested','stage_results')
        w=csv.DictWriter(f,fieldnames=columns);w.writeheader()
        for ladder in progression.get('ladders',[]):
            w.writerow({**{k:ladder.get(k) for k in columns if k!='stage_results'},
                        'stage_results':json.dumps(ladder['stages'],ensure_ascii=False)})
    with (root/'progression_by_difficulty.csv').open('w', encoding='utf-8', newline='') as f:
        w=csv.DictWriter(f,fieldnames=('difficulty','count','accuracy'));w.writeheader()
        for difficulty,metrics in progression.get('by_difficulty',{}).items():
            w.writerow({'difficulty':difficulty,**metrics})
