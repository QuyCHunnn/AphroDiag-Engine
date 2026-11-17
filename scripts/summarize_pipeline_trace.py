"""Validate and summarize nested wall spans. Corrupt/incomplete traces fail gates."""
import argparse
import json
import math
from pathlib import Path


def merged(values):
    result = []
    for left, right in sorted(values):
        if result and left <= result[-1][1]:
            result[-1] = (result[-1][0], max(right, result[-1][1]))
        else:
            result.append((left, right))
    return result


def summarize(path):
    starts, ends, groups, intervals = {}, set(), {}, {}
    for number, line in enumerate(path.read_text().splitlines(), 1):
        try:
            event = json.loads(line)
            identity, phase, kind = event['span'], event['phase'], event['event']
            timestamp = event['monotonic_seconds']
            if (not isinstance(identity, str) or not identity or not isinstance(phase, str)
                    or not phase or type(timestamp) not in (int, float) or not math.isfinite(timestamp)):
                raise ValueError('invalid span identity, phase or timestamp')
            if kind == 'begin':
                if identity in starts:
                    raise ValueError('duplicate begin')
                starts[identity] = event
            elif kind == 'end':
                if identity not in starts or identity in ends:
                    raise ValueError('orphan or duplicate end')
                begin = starts[identity]
                seconds = event['seconds']
                if (begin['phase'] != phase or timestamp < begin['monotonic_seconds']
                        or type(seconds) not in (int, float) or not math.isfinite(seconds)
                        or seconds < 0 or event['status'] not in {'passed', 'failed'}):
                    raise ValueError('invalid end')
                ends.add(identity)
                groups.setdefault(phase, []).append(event)
                intervals.setdefault(phase, []).append((begin['monotonic_seconds'], timestamp))
            else:
                raise ValueError('unknown event')
        except (ValueError, KeyError, TypeError) as exc:
            raise ValueError(f'Invalid trace event at line {number}: {exc}') from exc
    if not starts or len(starts) != len(ends):
        raise ValueError('Empty trace or unclosed spans')
    rows = {phase: {'calls': len(values), 'inclusive_span_seconds': sum(v['seconds'] for v in values),
                    'union_wall_seconds': sum(b-a for a, b in merged(intervals[phase])),
                    'failed': sum(v['status'] != 'passed' for v in values)} for phase, values in groups.items()}
    a = merged(intervals.get('asr.generate', []))
    b = merged(intervals.get('tts.infer', []))
    overlap = sum(max(0, min(x2, y2)-max(x1, y1)) for x1, x2 in a for y1, y2 in b)
    return {'status': 'passed', 'phases': rows, 'asr_tts_infer_overlap_seconds': overlap,
            'unclosed_spans': 0, 'interpretation': 'wall intervals, not exclusive compute; do not sum nested phases'}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('run', type=Path)
    args = parser.parse_args()
    try:
        result = summarize(args.run/'spans.jsonl')
    except (OSError, ValueError) as exc:
        result = {'status': 'failed', 'error': str(exc)}
    (args.run/'trace-summary.json').write_text(json.dumps(result, indent=2)+'\n')
    print(json.dumps(result, indent=2))
    return 0 if result['status'] == 'passed' else 1


if __name__ == '__main__':
    raise SystemExit(main())
