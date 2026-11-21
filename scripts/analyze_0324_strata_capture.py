"""Offline, diagnostic-only session analysis of a bounded native engine capture.

Only contiguous captures are accepted. Line numbers are 1-based and inclusive
in the decoded original log (head + tail, with byte coverage verified first).
Reported rates are preserved, never recomputed or promoted across sessions.
"""
import argparse
from datetime import datetime
import hashlib
import json
from pathlib import Path
import re


MAX_INPUT_BYTES = 2 * 1024**2
TIMESTAMP = r'[0-9]{4}-[0-9]{2}-[0-9]{2} [0-9]{2}:[0-9]{2}:[0-9]{2}'
MARKER = re.compile(r'\[strata\] (' + TIMESTAMP + r') engine started: (.+)')
NUMBER = r'[0-9]+(?:\.[0-9]+)?'
GENERATION = re.compile(
    r'strata serve: prompt (?P<prompt_tokens>[0-9]+) tokens = '
    r'(?P<reused_tokens>[0-9]+) reused \+ (?P<read_tokens>[0-9]+) read in '
    r'(?P<prompt_elapsed_ms>[0-9]+) ms \((?P<prompt_reported_tok_s>' + NUMBER + r') tok/s\), '
    r'(?P<generated_tokens>[0-9]+) generated in (?P<generation_elapsed_ms>[0-9]+) ms '
    r'\((?P<generation_reported_tok_s>' + NUMBER + r') tok/s\), '
    r'drafts accepted (?P<drafts_accepted>[0-9]+) of (?P<drafts_proposed>[0-9]+)'
    r'(?:, (?P<checkpoints>[0-9]+) checkpoints)?'
)
TAIL_SKIP = re.compile(
    r'strata serve: route tail skip: ([0-9]+) missed experts skipped '
    r'\(([0-9]+) entries\) since the start'
)
EVIDENCE_PATTERNS = {
    'arena_residency': re.compile(r'arena|resident RAM mode|FileExpertSource:|experts via mmap|file tier', re.I),
    'memory_guard': re.compile(r'memory guard:'),
    'cache': re.compile(r'expert cache|pre-filled|token graph hit path|R4 hit path|PROFILE,|lend region', re.I),
    'tail_skip': re.compile(r'route.tail.skip', re.I),
    'adaptive': re.compile(r'adaptive', re.I),
    'pool': re.compile(r'CPU expert kernels|CPU pool|pool workers|expert-pool|CPU workers|aux cpus'),
    'native': re.compile(r'native pack:|native projection|native Q5_K|PLE on,|PLE direct read|GPU 0:'),
    'pcie': re.compile(r'PCIe|host->device', re.I),
    'draft': re.compile(r'suffix drafts:|lookup-max-t|MTP windows'),
}


def validate_timestamp(value):
    if not isinstance(value, str) or not re.fullmatch(TIMESTAMP, value):
        raise ValueError('current-start must use YYYY-MM-DD HH:MM:SS')
    datetime.strptime(value, '%Y-%m-%d %H:%M:%S')
    return value


def evidence(line_number, text):
    # Keep unusually large argv/evidence lines from becoming a raw-log dump.
    return dict(line=line_number, text=text[:4096], text_truncated=len(text) > 4096)


def analyze_capture(engine_capture, current_start):
    """Read one <=2 MiB JSON capture; return a deterministic, session-scoped report."""
    validate_timestamp(current_start)
    path = Path(engine_capture)
    with path.open('rb') as stream:
        raw = stream.read(MAX_INPUT_BYTES + 1)
    if len(raw) > MAX_INPUT_BYTES:
        raise ValueError('engine capture exceeds 2 MiB input bound')
    capture = json.loads(raw.decode('utf-8'))
    if not isinstance(capture, dict):
        raise ValueError('engine capture must be a JSON object')
    if capture.get('truncated') is not False:
        raise ValueError('truncated or unverified head/tail capture; refusing to stitch gaps')
    head, tail = capture.get('head'), capture.get('tail')
    size = capture.get('size_bytes')
    if not isinstance(head, str) or not isinstance(tail, str):
        raise ValueError('head and tail must both be strings')
    if type(size) is not int or size < 0:
        raise ValueError('size_bytes must be a nonnegative integer')
    if len(head.encode('utf-8')) + len(tail.encode('utf-8')) != size:
        raise ValueError('head/tail byte coverage does not match size_bytes; refusing to stitch gaps')
    source_hash = hashlib.sha256(raw).hexdigest()
    lines = (head + tail).splitlines()
    markers = [(index, match) for index, text in enumerate(lines)
               if (match := MARKER.fullmatch(text))]
    current_indices = [i for i, (_, match) in enumerate(markers)
                       if match.group(1) == current_start]
    if len(current_indices) != 1:
        raise ValueError(f'expected exactly one matching current marker; found {len(current_indices)}')
    current_index = current_indices[0]
    sessions = []
    for index, (start, marker) in enumerate(markers):
        end = markers[index + 1][0] if index + 1 < len(markers) else len(lines)
        scope = 'historical' if index < current_index else 'current' if index == current_index else 'unexpected'
        session = dict(
            session_index=index, scope=scope, start_timestamp=marker.group(1),
            source_sha256=source_hash, line_span=dict(start=start + 1, end=end),
            evidence={'native_argv': [evidence(start + 1, lines[start])]},
            generation_stats=[], cumulative_tail_skip=[], unparsed_generation_lines=[],
        )
        for offset in range(start + 1, end):
            text = lines[offset]
            line_number = offset + 1
            match = GENERATION.fullmatch(text)
            if match:
                values = {key: (float(value) if key.endswith('_tok_s') else int(value))
                          for key, value in match.groupdict().items() if value is not None}
                session['generation_stats'].append(dict(
                    line=line_number, scope='request', **values))
            elif text.startswith('strata serve: prompt ') and ' generated in ' in text:
                session['unparsed_generation_lines'].append(evidence(line_number, text))
            tail_match = TAIL_SKIP.fullmatch(text)
            if tail_match:
                session['cumulative_tail_skip'].append(dict(
                    line=line_number, scope='session_cumulative_since_start',
                    missed_experts_skipped=int(tail_match.group(1)), entries=int(tail_match.group(2))))
            for category, pattern in EVIDENCE_PATTERNS.items():
                if pattern.search(text):
                    session['evidence'].setdefault(category, []).append(evidence(line_number, text))
        session['completed_generation_count'] = len(session['generation_stats'])
        sessions.append(session)
    current = sessions[current_index]
    current_stats = current['generation_stats']
    return dict(
        schema='strata.capture.sessions/1',
        source=dict(capture_path=str(path), sha256=source_hash, capture_bytes=len(raw),
                    log_path=capture.get('path'), log_size_bytes=size, line_count=len(lines),
                    line_number_basis='1-based inclusive decoded contiguous head+tail'),
        current_start=current_start,
        classification_basis='marker order relative to the unique matching current marker',
        excluded_unscoped_prefix=dict(line_count=markers[0][0],
                                     line_span=dict(start=1, end=markers[0][0]) if markers[0][0] else None),
        sessions=sessions,
        current_summary=dict(
            session_index=current_index, completed_generation_count=len(current_stats),
            fresh_throughput_available=bool(current_stats),
            reported_generation_rates_tok_s=[row['generation_reported_tok_s'] for row in current_stats],
            conclusion=('Current session has completed generation summaries; rates are reported, not recomputed.'
                        if current_stats else 'No completed generation summary in current session; no fresh throughput.'),
        ),
        limitations=[
            'Offline log evidence only; no model, telemetry, checkpoint or configuration equivalence validation.',
            'Earlier sessions are historical; later sessions are unexpected; unscoped prefix is excluded.',
            'Generation summaries describe recorded requests, not necessarily successful uncancelled answers.',
            'Unrecognized generation summaries are retained separately, not interpreted as measured rates.',
            'Tail-skip counts are cumulative, not request counts; no tail fractions or throughput conversions.',
            'Allocation/startup evidence does not establish completed residency or server readiness.',
            'Reported rates are not recomputed, aggregated, or selected by historical maximum.',
            'Byte coverage is verified; captures with redaction-altered byte lengths are conservatively rejected.',
        ],
    )


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--current-start', required=True, help='Exact current engine timestamp: YYYY-MM-DD HH:MM:SS')
    parser.add_argument('--engine-capture', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True, help='Exclusively create a new JSON report')
    args = parser.parse_args(argv)
    try:
        report = analyze_capture(args.engine_capture, args.current_start)
        payload = json.dumps(report, indent=2, sort_keys=True, allow_nan=False) + '\n'
        with args.output.open('x', encoding='utf-8') as stream:
            stream.write(payload)
    except (OSError, ValueError, RecursionError) as exc:
        parser.error(str(exc))
    summary = report['current_summary']
    print(f"Analyzed {len(report['sessions'])} sessions; "
          f"current completed generations: {summary['completed_generation_count']}; "
          f"fresh throughput available: {summary['fresh_throughput_available']}")
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
