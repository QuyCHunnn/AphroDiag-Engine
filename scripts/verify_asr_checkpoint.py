"""Exercise integrated GPU ASR and a controlled GPU-error/real-CPU fallback."""
from pathlib import Path
from unittest.mock import patch
import json,sys,time
from jellydub.transcribe import transcribe_video
from jellydub import asr_openvino
root=Path(__file__).resolve().parents[1];out=root/'artifacts/optimization-checkpoints/cp3';out.mkdir(parents=True,exist_ok=True)
model=(root/'artifacts/igpu-setup/whisper-small-model-path.txt').read_text().strip()
video=root/'artifacts/performance-audit/asr/sample-20s.wav'
reports=[]
for mode in ['gpu','injected-gpu-error']:
 details={};messages=[];original=asr_openvino.transcribe
 def injected(audio,**kwargs):
  if kwargs['device']=='GPU':raise RuntimeError('Injected GPU failure for fallback acceptance')
  return original(audio,**kwargs)
 start=time.perf_counter()
 with patch.object(asr_openvino,'transcribe',injected if mode!='gpu' else original):
  cues,language=transcribe_video(video,model_name=model,device='auto',backend='auto',cache_dir=out/'models',beam_size=5,report=lambda s:(messages.append(s),print(s,flush=True)),details=details)
 item={'mode':mode,'wall_seconds':time.perf_counter()-start,'language':language,'details':details,'messages':messages,'cues':[{'index':c.index,'start':c.start,'end':c.end,'text':c.text} for c in cues]}
 reports.append(item);(out/'integrated-asr.json').write_text(json.dumps(reports,indent=2))
 assert language=='en' and cues and details['beam_size']==5
 if mode=='gpu':assert details['device']=='gpu' and not details['fallback']
 else:assert details['device']=='cpu' and details['fallback']
print('Integrated GPU and real CPU fallback passed')
