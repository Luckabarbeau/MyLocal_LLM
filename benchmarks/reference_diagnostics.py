"""0095: distribution diagnostics with explicit nonsemantic-token caveats.

All metrics operate on teacher-forced *reference* tokens. A reference is only
one acceptable text, not a unique ground truth for natural language. The
'content' heuristic identifies likely informative token pieces, not facts.
"""
from __future__ import annotations

from collections import defaultdict
import math
import re

STOPWORDS = frozenset('a an the and or to of in on at by as is are was were be been being for from with it this that these those i you he she they we his her their our not do does did if then than while when which who what where how can could should would will have has had one two into about over under after before also very only so but there here'.split())


def token_kind(decoded, *, category=None):
    """Heuristic grouping for subword pieces; *not* semantic correctness."""
    raw = str(decoded)
    compact = raw.strip().replace('Ġ', '').replace('▁', '')
    if not compact:
        return 'whitespace'
    if re.fullmatch(r'[0-9]+(?:\.[0-9]+)?', compact):
        return 'number'
    if re.fullmatch(r'[^\w\s]+', compact, flags=re.UNICODE):
        return 'operator' if category in ('python', 'cpp', 'shell') else 'punctuation'
    if compact.lower() in STOPWORDS:
        return 'stopword'
    return 'content'


def summarize_rows(rows, *, category=None):
    """Return weighted per-token statistics plus rough lexical-token groupings."""
    if not rows:
        return {'token_count': 0}
    groups = defaultdict(list)
    for r in rows:
        kind = r.get('token_kind') or token_kind(r.get('actual_text', ''), category=category)
        groups[kind].append(r)
    content = [r for r in rows if (r.get('token_kind') or token_kind(r.get('actual_text',''),category=category)) in ('content','number','operator')]
    def metrics(items):
        if not items:
            return {'count':0,'mean_nll': None,'top1_accuracy': None,'top5_accuracy':None,'top10_accuracy':None,'mean_target_probability':None,'mean_entropy_nats':None,'mean_rank':None}
        size = len(items)
        return {'count':size,
                'mean_nll':sum(r['nll'] for r in items)/size,
                'top1_accuracy':sum(r['actual_rank']==1 for r in items)/size,
                'top5_accuracy':sum(r['actual_rank']<=5 for r in items)/size,
                'top10_accuracy':sum(r['actual_rank']<=10 for r in items)/size,
                'mean_target_probability':sum(r['actual_probability'] for r in items)/size,
                'mean_entropy_nats':sum(r['entropy_nats'] for r in items)/size,
                'mean_rank':sum(r['actual_rank'] for r in items)/size}
    out={'all_tokens':metrics(rows),'content_like_tokens':metrics(content),
         'by_token_kind':{key:metrics(items) for key,items in sorted(groups.items())},
         'position_quartiles':{}}
    for q in range(4):
        start=(q*len(rows))//4
        end=((q+1)*len(rows))//4
        out['position_quartiles'][f'{q*25}-{(q+1)*25}%']=metrics(rows[start:end])
    surprising=sorted(rows,key=lambda r:r['nll'],reverse=True)[:min(8,len(rows))]
    out['most_surprising']=[{'position':r['position'], 'token':r.get('actual_text',''),
                             'rank':r['actual_rank'],'probability':r['actual_probability'],
                             'nll':r['nll'],
                             'top1':r['top'][0]['text'] if r.get('top') else None}
                            for r in surprising]
    out['warning']='Subword lexical classes are heuristic and reference likelihood is not equivalent to factual correctness.'
    return out


def summarize_cases(cases):
    by_category=defaultdict(list)
    all_rows=[]
    for c in cases:
        cat=c.get('category') or 'unknown'
        by_category[cat].extend(c['token_predictions'])
        all_rows.extend(c['token_predictions'])
    return {'total_references':len(cases), 'total_scored_tokens':len(all_rows),
            'micro':summarize_rows(all_rows),
            'by_category':{cat:summarize_rows(rows,category=cat) for cat,rows in sorted(by_category.items())},
            'note':'Micro averages weight each reference token equally; inspect per-domain distributions for improvements.'}
