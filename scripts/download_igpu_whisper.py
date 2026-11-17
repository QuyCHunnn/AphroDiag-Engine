"""Download public, pre-exported OpenVINO Whisper weights; no remote model code."""
import os
import sys
from pathlib import Path
os.environ.setdefault('HF_HUB_DISABLE_XET','1')
from huggingface_hub import snapshot_download
root=Path(__file__).resolve().parents[1]
name=sys.argv[1] if len(sys.argv)>1 else 'tiny'
if name not in {'tiny','small'}: raise SystemExit('Expected tiny or small')
path=snapshot_download(f'OpenVINO/whisper-{name}-int8-ov',
    cache_dir=str(root/'artifacts/igpu-setup/model-cache'),
    allow_patterns=['*.json','*.xml','*.bin','*.txt'])
(root/f'artifacts/igpu-setup/whisper-{name}-model-path.txt').write_text(path+'\n')
if name=='tiny': (root/'artifacts/igpu-setup/whisper-model-path.txt').write_text(path+'\n')
print(path)
