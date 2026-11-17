"""One setup command for the repository CLI (no app packaging).
Run from the checkout: uv run python scripts/setup_engine.py
"""
import json
from pathlib import Path
import platform
import shutil
import subprocess
import sys


def main():
    root = Path(__file__).resolve().parents[1]
    uv = shutil.which('uv')
    if uv is None:
        raise SystemExit('Install uv first: https://docs.astral.sh/uv/getting-started/installation/')
    missing = [name for name in ('ffmpeg','ffprobe') if shutil.which(name) is None]
    if missing:
        raise SystemExit('Missing '+', '.join(missing)+'. Ubuntu/WSL: sudo apt install ffmpeg. Other OS: https://ffmpeg.org/download.html')
    profiles = ['translate','asr','vieneu']
    if sys.platform in {'linux','win32'} and platform.machine().lower() in {'x86_64','amd64'}:
        profiles.append('intel')
    command = [uv,'sync','--locked','--inexact']
    for profile in profiles:
        command += ['--extra',profile]
    print('Installing CLI runtime: '+', '.join(profiles),flush=True)
    subprocess.run(command,cwd=root,check=True)
    subprocess.run([uv,'run','--no-sync','jellydub','doctor'],cwd=root,check=True)
    (root/'.venv/jellydub-setup.json').write_text(json.dumps({'platform':platform.platform(),'profiles':profiles},indent=2))
    print('\nReady. Run: uv run --no-sync jellydub auto "video.mp4" --out-dir output')


if __name__=='__main__':
    main()
