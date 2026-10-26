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
        return {k: summary.get(k) for k in (
            'count', 'mean_repeated_8gram_rate', 'mean_distinct_4',
            'repeat_8gram_onset_fraction',
        )}
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
    for case in result['cases']:
        output.append('<section class="ref"><h2>' + html.escape(case['id']) + '</h2>')
        output.append(f'<small>NLL {case["mean_nll"]:.3f}; top-1 {case["top1_accuracy"]:.1%}; top-5 {case["top5_accuracy"]:.1%}</small>')
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
               'mean_reference_nll', 'mean_top1_accuracy', 'mean_top5_accuracy')
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
