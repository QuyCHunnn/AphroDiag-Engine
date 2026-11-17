"""Measured 180-second auto pipeline benchmark, fresh job cache then warm replay."""
from pathlib import Path
import functools, hashlib, json, subprocess, time
from jellydub import pipeline
from jellydub.cli import app
import sys
root=Path(__file__).resolve().parents[1]
base=root/'artifacts/optimization-checkpoints/benchmark-180s-en'
base.mkdir(parents=True,exist_ok=True)
if len(sys.argv)>1 and sys.argv[1]=='worker':
 label_case=sys.argv[2]
 stages=[]
 for name in ('transcribe_video','_translated','synthesize_cues','align','render_timeline','extract_audio','mix','mux_mkv'):
  original=getattr(pipeline,name)
  def wrap(fn,label):
   @functools.wraps(fn)
   def measured(*args,**kwargs):
    start=time.perf_counter()
    try:return fn(*args,**kwargs)
    finally:
     stages.append({'stage':label,'seconds':time.perf_counter()-start})
     (base/(label_case+'-stages.json')).write_text(json.dumps(stages,indent=2))
   return measured
  setattr(pipeline,name,wrap(original,name))
 args=sys.argv[3:];sys.argv=[sys.argv[0],*args];app();raise SystemExit
source=next((root/'videos').glob('*[[]n0LY3XcZaCs[]].mp4'))
video=base/'input'/'sample.mp4';video.parent.mkdir(exist_ok=True)
if not video.exists():subprocess.run(['ffmpeg','-v','error','-y','-i',str(source),'-t','180','-map','0:v:0','-map','0:a:0','-c','copy',str(video)],check=True)
model=(root/'artifacts/igpu-setup/whisper-small-model-path.txt').read_text().strip()
config=base/'settings.toml';config.write_text('[jellydub]\ndownload=false\nmix="voiceover"\nvoice="vi-VN-NamMinhNeural"\nasr_model='+json.dumps(model)+'\n')
from jellydub.subtitles import parse_srt,write_srt
cues=parse_srt(root/'dubbed/vendor-quotations/vendor-quotations.en.srt')
cues=[c for c in cues if c.start < 180]
for c in cues:c.end=min(c.end,180)
srt=base/'input'/'sample.en.srt';write_srt(cues,srt)
original_hash=hashlib.sha256(video.read_bytes()).hexdigest()
records=[]
for case in ('cold','warm'):
 args=[str(root/'.venv/bin/python'),str(Path(__file__).resolve()),'worker',case,'auto',str(video),'--srt',str(srt),'--config',str(config),'--cache-dir',str(base/'cache'),'--out-dir',str(base/'output')]
 start=time.perf_counter()
 with (base/(case+'.log')).open('w') as log:r=subprocess.run(args,stdout=log,stderr=subprocess.STDOUT,timeout=2400)
 record={'case':case,'returncode':r.returncode,'wall_seconds':time.perf_counter()-start,'source_unchanged':hashlib.sha256(video.read_bytes()).hexdigest()==original_hash,'stages':json.loads((base/(case+'-stages.json')).read_text()) if (base/(case+'-stages.json')).exists() else []}
 records.append(record);(base/'results.json').write_text(json.dumps(records,indent=2));print(json.dumps(record),flush=True)
 if r.returncode:raise SystemExit(r.returncode)
mkv=next((base/'output').glob('*.mkv'))
r=subprocess.run(['ffmpeg','-v','error','-i',str(mkv),'-map','0:v?','-map','0:a?','-f','null','-'],capture_output=True,text=True)
(base/'validation.json').write_text(json.dumps({'decode_exit_code':r.returncode,'decode_errors':r.stderr,'files':[{'name':p.name,'bytes':p.stat().st_size} for p in (base/'output').iterdir()]},indent=2))
print('Decode:',r.returncode,flush=True)
