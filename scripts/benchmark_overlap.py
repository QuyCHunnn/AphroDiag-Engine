"""Sequential/overlap real CLI pipeline with isolated empty processing caches."""
from pathlib import Path
from dataclasses import replace
import json,time,subprocess,sys,argparse
from jellydub.config import Config
from jellydub.pipeline import auto
from jellydub.cache import file_digest
root=Path(__file__).resolve().parents[1]
parser=argparse.ArgumentParser();parser.add_argument('--mode',choices=['sequential','overlap']);parser.add_argument('--run',default='overlap-ab');args=parser.parse_args()
base=root/'artifacts/checkpoints-2026-11-17'/args.run
base.mkdir(parents=True,exist_ok=True)
source=root/'videos/Family Guy - Peter falls down the stairs.mp4'
model=(root/'artifacts/igpu-setup/whisper-small-model-path.txt').read_text().strip()
for mode in ((args.mode,) if args.mode else ('sequential','overlap')):
 if mode=='overlap' and args.mode is None:
  subprocess.run([sys.executable,__file__,'--mode','overlap','--run',args.run],check=True);break
 work=base/mode
 if work.exists():raise SystemExit('Preserve previous run: '+str(work))
 work.mkdir(); messages=[]
 config=Config(source_lang='en',asr_backend='genai',asr_python=str(root/'.venv-native-asr-next/bin/python'),asr_model=model,asr_device='gpu',tts='vieneu',tts_device='cpu',translate_device='cpu',tts_precision='int8',pipeline_mode=mode,cache_dir=work/'cache',out_dir=work/'output',download=False,mix='voiceover',max_tempo=1.6)
 started=time.perf_counter(); record={'mode':mode,'status':'running','processing_cache_initially_empty':True,'prepared_ledger':False}
 def report(s):
  messages.append({'seconds':time.perf_counter()-started,'message':s});print(mode,s,flush=True)
 try:
  outputs=auto(source,None,config,report=report)
  record.update(status='passed',seconds=time.perf_counter()-started,outputs={k:str(v) for k,v in outputs.items()})
  check=subprocess.run(['ffmpeg','-v','error','-i',str(outputs['mkv']),'-map','0:v?','-map','0:a?','-f','null','-'],capture_output=True,text=True)
  record.update(decode_returncode=check.returncode,decode_errors=check.stderr,subtitle_sha256=file_digest(outputs['srt']))
 except BaseException as exc:
  record.update(status='failed',seconds=time.perf_counter()-started,error=str(exc));raise
 finally:
  record['messages']=messages;(work/'report.json').write_text(json.dumps(record,ensure_ascii=False,indent=2))
