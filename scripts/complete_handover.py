"""Run a reproducible VI-SRT voiceover twice and record technical acceptance evidence.
Usage: .venv/bin/python scripts/complete_handover.py VIDEO VI_SRT OUTPUT
The first measured run may reuse existing cache; it is never called a cold run.
"""
import hashlib
import json
from pathlib import Path
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parents[1]


def digest(path):
    h = hashlib.sha256()
    with path.open('rb') as f:
        for block in iter(lambda: f.read(1024 * 1024), b''):
            h.update(block)
    return h.hexdigest()


def main():
    if len(sys.argv) != 4:
        raise SystemExit(__doc__)
    video, srt, out = [Path(p).resolve() for p in sys.argv[1:]]
    out.mkdir(parents=True, exist_ok=True)
    evidence = out / 'evidence'
    evidence.mkdir(exist_ok=True)
    cache = ROOT / '.jellydub-cache'
    command = [str(ROOT / '.venv/bin/jellydub'), 'dub', str(video), '--srt', str(srt),
               '--source-lang', 'vi', '--voice', 'vi-VN-NamMinhNeural', '--mix', 'voiceover',
               '--mkv', '--mp4', '--audio', '--out-dir', str(out), '--cache-dir', str(cache)]
    before = {str(p): digest(p) for p in (video, srt)}
    report = {'command': command, 'source_sha256_before': before, 'runs': [],
              'cold_run_measured': False, 'human_listening': 'pending', 'vlc_acceptance': 'pending'}
    def snapshot():
        return {str(p.relative_to(cache)): [p.stat().st_size, p.stat().st_mtime_ns]
                for p in cache.rglob('*') if p.is_file()}
    for number in (1, 2):
        old = snapshot()
        started = time.perf_counter()
        with (evidence / f'run-{number}.log').open('w') as log:
            result = subprocess.run(command, stdout=log, stderr=subprocess.STDOUT)
        new = snapshot()
        report['runs'].append({'number': number, 'seconds': round(time.perf_counter()-started, 3),
                              'returncode': result.returncode,
                              'cache_files_created': len(new.keys()-old.keys()),
                              'cache_files_modified': sum(new[k] != old[k] for k in new.keys() & old.keys())})
        (evidence / 'verification.json').write_text(json.dumps(report, ensure_ascii=False, indent=2))
        if result.returncode:
            raise SystemExit(f'Run {number} failed; see evidence/run-{number}.log')
    report['source_sha256_after'] = {str(p): digest(p) for p in (video, srt)}
    report['source_unchanged'] = before == report['source_sha256_after']
    report['outputs'] = []
    for p in sorted(out.iterdir()):
        if p.suffix not in {'.mkv', '.mp4', '.mp3', '.srt'}:
            continue
        entry = {'path': p.name, 'bytes': p.stat().st_size, 'sha256': digest(p)}
        if p.suffix != '.srt':
            data = subprocess.check_output(['ffprobe', '-v', 'error', '-show_streams', '-show_format', '-of', 'json', str(p)])
            entry['probe'] = json.loads(data)
            # Decode every audio and video track, including the original audio retained in MKV.
            result = subprocess.run(['ffmpeg', '-v', 'error', '-i', str(p), '-map', '0:v?', '-map', '0:a?', '-f', 'null', '-'], capture_output=True, text=True)
            entry['decode_returncode'] = result.returncode
            entry['decode_errors'] = result.stderr
        report['outputs'].append(entry)
    report['technical_pass'] = report['source_unchanged'] and all(x.get('decode_returncode', 0) == 0 for x in report['outputs'])
    (evidence / 'verification.json').write_text(json.dumps(report, ensure_ascii=False, indent=2))
    print(json.dumps({k:v for k,v in report.items() if k not in {'outputs', 'command'}}, ensure_ascii=False, indent=2))
    if not report['technical_pass']:
        raise SystemExit(1)


if __name__ == '__main__':
    main()
