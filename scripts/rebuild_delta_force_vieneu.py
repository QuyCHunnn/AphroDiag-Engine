"""Rebuild Delta Force with local Vietnamese male/female VieNeu presets."""
from pathlib import Path
import hashlib,json,os,subprocess,sys,time
root=Path(__file__).resolve().parents[1];base=root/'artifacts/delta-force-vieneu'
source=next((root/'videos').glob('*W39avTeRKjs*.mp4'))
expected=json.loads((root/'artifacts/delta-force/verification.json').read_text())['source_sha256']
if hashlib.sha256(source.read_bytes()).hexdigest()!=expected:raise SystemExit('Source differs from reviewed video')
command=[str(root/'.venv/bin/jellydub'),'dub',str(source),'--srt',str(root/'artifacts/delta-force/delta-force.reviewed.vi.srt'),'--source-lang','vi','--config',str(base/'settings.toml'),'--cache-dir',str(base/'cache'),'--out-dir',str(root/'delivery/delta-force-vieneu')]
environment={**os.environ,'HF_HUB_OFFLINE':'1'};start=time.perf_counter();result=subprocess.run(command,cwd=root,env=environment)
(base/'run.json').write_text(json.dumps(dict(command=command,returncode=result.returncode,wall_seconds=time.perf_counter()-start,offline=True,backend='VieNeu v3 Turbo ONNX CPU',male='Hải Đăng',female='Trúc Ly'),ensure_ascii=False,indent=2));sys.exit(result.returncode)
