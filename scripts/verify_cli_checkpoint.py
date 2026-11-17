"""Real short-video CLI acceptance: VI SRT, EN SRT, and no SRT/auto ASR.
Uses the public sample. Existing reviewed NamMinh speech seeds the VI case;
new uncached speech, if required, is synthesized through actual Edge-TTS.
"""
from pathlib import Path
import hashlib,json,shutil,subprocess,time
from jellydub.cache import cache_key
from jellydub.subtitles import parse_srt,write_srt
from jellydub.tts.edge import EdgeTTS
root=Path(__file__).resolve().parents[1]
base=root/'artifacts/optimization-checkpoints/cp4';base.mkdir(parents=True,exist_ok=True)
source=next((root/'videos').glob('*[[]n0LY3XcZaCs[]].mp4'))
model=(root/'artifacts/igpu-setup/whisper-small-model-path.txt').read_text().strip()
cache=base/'cache';(cache/'tts').mkdir(parents=True,exist_ok=True)
config=base/'settings.toml'
config.write_text('[jellydub]\ndownload=false\nmix="voiceover"\nvoice="vi-VN-NamMinhNeural"\ntts_workers=1\nasr_model='+json.dumps(model)+'\n')
inputs=base/'inputs';inputs.mkdir(exist_ok=True)
clip=base/'clip.mp4'
if not clip.exists():
 subprocess.run(['ffmpeg','-v','error','-y','-ss','13.48','-i',str(source),'-t','8.68','-vf','scale=640:360','-c:v','libx264','-preset','ultrafast','-crf','28','-c:a','aac',str(clip)],check=True)
reports=[];voice=EdgeTTS(voice='vi-VN-NamMinhNeural')
for case in ('vi','en','asr'):
 folder=inputs/case;folder.mkdir(exist_ok=True)
 video=folder/'sample.mp4';shutil.copy2(clip,video)
 args=[str(root/'.venv/bin/jellydub'),'auto',str(video),'--out-dir',str(base/case/'output'),'--cache-dir',str(cache),'--config',str(config)]
 if case!='asr':
  text=root/f'dubbed/vendor-quotations/vendor-quotations.{case}.srt'
  cues=parse_srt(text)[1:3]
  for i,c in enumerate(cues):c.index=i+1;c.start-=13.48;c.end-=13.48
  srt=folder/f'sample.{case}.srt';write_srt(cues,srt)
  args+=['--srt',str(srt)]
  if case=='vi':
   for cue in cues:
    name=cache_key(voice.cache_id(),cue.text,cue.speaker)+'.wav'
    matches=list((root/'.jellydub-cache').glob('Compare*/tts/'+name))
    if not matches:raise RuntimeError('Missing exact reviewed VI speech')
    shutil.copy2(matches[0],cache/'tts'/name)
 before=hashlib.sha256(video.read_bytes()).hexdigest();speech_before=len(list((cache/'tts').glob('*.wav')))
 start=time.perf_counter()
 with (base/f'{case}.log').open('w') as log:
  result=subprocess.run(args,stdout=log,stderr=subprocess.STDOUT,timeout=300)
 record={'case':case,'command':args,'returncode':result.returncode,'wall_seconds':time.perf_counter()-start,'new_speech_cache_files':len(list((cache/'tts').glob('*.wav')))-speech_before,'source_unchanged':hashlib.sha256(video.read_bytes()).hexdigest()==before}
 if result.returncode==0:
  outputs=list((base/case/'output').iterdir());record['only_mkv_srt']={p.suffix for p in outputs}=={'.mkv','.srt'}
  mkv=next(p for p in outputs if p.suffix=='.mkv')
  decoded=subprocess.run(['ffmpeg','-v','error','-i',str(mkv),'-map','0:v?','-map','0:a?','-f','null','-'],capture_output=True,text=True)
  record['decode_exit_code']=decoded.returncode;record['decode_errors']=decoded.stderr
  video_hashes=[subprocess.check_output(['ffmpeg','-v','error','-i',str(path),'-map','0:v:0','-c','copy','-f','hash','-hash','sha256','-'],text=True).strip() for path in [video,mkv]]
  record['video_stream_copy']=len(set(video_hashes))==1
  tracks=json.loads(subprocess.check_output(['ffprobe','-v','error','-show_streams','-of','json',str(mkv)]))['streams']
  record['vietnamese_defaults']=all(any(t['codec_type']==kind and t.get('tags',{}).get('language')=='vie' and t['disposition']['default']==1 for t in tracks) for kind in ['audio','subtitle'])
  qc=next(json.loads(p.read_text()) for p in (cache/'reports').glob('*.json') if json.loads(p.read_text()).get('video')==str(video))
  record['qc']=qc
 reports.append(record);(base/'verification.json').write_text(json.dumps(reports,indent=2))
 print(case,record['returncode'],round(record['wall_seconds'],3),'new speech',record['new_speech_cache_files'],flush=True)
 if result.returncode:raise SystemExit(f'{case} failed; see {base/case}.log')
 assert record['source_unchanged'] and record['video_stream_copy'] and record['only_mkv_srt'] and record['vietnamese_defaults'] and record['decode_exit_code']==0
 if case=='asr':assert record['qc']['subtitle_source']=='asr'
print('CLI acceptance passed for VI, EN and auto ASR')
