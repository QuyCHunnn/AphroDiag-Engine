"""Rebuild the reviewed Delta Force delivery using cached or new speech as needed."""
from pathlib import Path
import hashlib
import json
import subprocess
import sys
import time

root = Path(__file__).resolve().parents[1]
base = root / 'artifacts/delta-force'
source = next((root / 'videos').glob('*W39avTeRKjs*.mp4'))
verification = json.loads((base / 'verification.json').read_text())
if hashlib.sha256(source.read_bytes()).hexdigest() != verification['source_sha256']:
    raise SystemExit('Source differs from the reviewed video; regenerate/review the transcript and speaker map first.')
command = [str(root / '.venv/bin/jellydub'), 'dub', str(source),
           '--srt', str(base / 'delta-force.reviewed.vi.srt'), '--source-lang', 'vi',
           '--config', str(base / 'settings.toml'), '--cache-dir', str(base / 'cache'),
           '--out-dir', str(root / 'delivery/delta-force')]
start = time.perf_counter()
result = subprocess.run(command, cwd=root)
(base / 'rebuild.json').write_text(json.dumps({
    'command': command, 'returncode': result.returncode,
    'wall_seconds': time.perf_counter() - start,
    'mode': 'reviewed subtitle rebuild; existing caches retained',
}, indent=2))
sys.exit(result.returncode)
