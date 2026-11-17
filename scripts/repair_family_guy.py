"""Apply the reviewed mixed-speaker correction, reusing the measured TTS cache."""
from pathlib import Path
import functools,hashlib,json,os,resource,subprocess,time
from jellydub import pipeline
from jellydub.config import Config
from jellydub.models import Cue
from jellydub.subtitles import write_srt
from jellydub.tts.vieneu import VieNeuTTS
root=Path(__file__).resolve().parents[1];base=root/'artifacts/family-guy-vieneu';work=base/'measured-01';ledger=json.loads((base/'speaker-ledger.json').read_text());source=root/'videos/Family Guy - Peter falls down the stairs.mp4'
assert hashlib.sha256(source.read_bytes()).hexdigest()==ledger['source_sha256']
os.environ['HF_HUB_OFFLINE']='1';record={'stages':[],'new_tts_calls':0,'purpose':'Correct mixed Peter/Lois utterance; preserve first-pass benchmark separately'};start=time.perf_counter()
cues=[Cue(i,e['start'],e['end'],e['vietnamese'],e['speaker']) for i,e in enumerate(ledger['utterances'],1)];subtitle=base/'reviewed.vi.srt';write_srt(cues,subtitle)
original_speak=VieNeuTTS.synthesize
def speak(self,*a,**kw):
 record['new_tts_calls']+=1;return original_speak(self,*a,**kw)
VieNeuTTS.synthesize=speak
for name in ('synthesize_cues','align','render_timeline','extract_audio','mix','mux_mkv','_checked_audio'):
 original=getattr(pipeline,name)
 def wrap(fn,label):
  @functools.wraps(fn)
  def call(*a,**kw):
   t=time.perf_counter()
   try:return fn(*a,**kw)
   finally:
    record['stages'].append({'name':label,'seconds':time.perf_counter()-t});(base/'repair-report.json').write_text(json.dumps(record,indent=2))
  return call
 setattr(pipeline,name,wrap(original,name))
config=Config(source_lang='vi',tts='vieneu',tts_workers=1,voice=ledger['voices']['PETER'],speaker_voices=ledger['voices'],device='cpu',mix='voiceover',max_tempo=1.6,cache_dir=work,out_dir=root/'delivery/family-guy-vieneu',download=False)
outputs=pipeline.dub(source,subtitle,config,skip_timing_check=True,report=lambda s:print(s,flush=True));record['time_to_files_seconds']=time.perf_counter()-start
r=subprocess.run(['ffmpeg','-v','error','-i',str(outputs['mkv']),'-map','0:v?','-map','0:a?','-f','null','-'],capture_output=True,text=True)
record.update(total_seconds=time.perf_counter()-start,decode_returncode=r.returncode,decode_errors=r.stderr,source_unchanged=hashlib.sha256(source.read_bytes()).hexdigest()==ledger['source_sha256'],outputs={k:str(v) for k,v in outputs.items()});(base/'repair-report.json').write_text(json.dumps(record,indent=2));assert r.returncode==0;print(record,flush=True)
