"""Measure a complete fresh-cache Family Guy pipeline with a reviewed speaker ledger.
Models are already downloaded. The ledger is source-specific human preparation;
its preparation time is deliberately not called automated pipeline time.
"""
from pathlib import Path
from contextlib import contextmanager
from dataclasses import asdict, replace
import argparse,hashlib,json,os,resource,subprocess,time
from jellydub.asr_openvino import load_audio
from jellydub.transcribe import transcribe_video
from jellydub.subtitles import write_srt
from jellydub.models import Cue,MixMode
from jellydub.ffmpeg import probe,duration
from jellydub.translate import OpusMTTranslator,translate_cues
from jellydub.tts.vieneu import VieNeuTTS
from jellydub.tts.synthesize import synthesize_cues, _has_speakable_content
from jellydub.timing import align,render_timeline
from jellydub.mix import extract_audio,mix
from jellydub.export import mux_mkv

root=Path(__file__).resolve().parents[1]
parser=argparse.ArgumentParser();parser.add_argument('--run',default='measured-01');parser.add_argument('--out-dir',type=Path)
parser.add_argument('--asr-backend',choices=['openvino','genai'],default='openvino');parser.add_argument('--asr-python');args=parser.parse_args()
base=root/'artifacts/family-guy-vieneu';work=base/args.run
if work.exists():raise SystemExit('Use a new --run name for a genuinely fresh processing cache')
ledger=json.loads((base/'speaker-ledger.json').read_text());source=root/'videos/Family Guy - Peter falls down the stairs.mp4'
if hashlib.sha256(source.read_bytes()).hexdigest()!=ledger['source_sha256']:raise SystemExit('Reviewed ledger belongs to a different source')
work.mkdir(parents=True);output=args.out_dir or root/'delivery/family-guy-vieneu';output.mkdir(parents=True,exist_ok=True)
os.environ['HF_HUB_OFFLINE']='1'
record={'source':str(source),'asr_backend':args.asr_backend,'cache_initially_empty':True,'model_downloads_included':False,'speaker_preparation_included':False,'voices':ledger['voices'],'stages':[],'run_status':'running'}
report=work/'report.json';started=time.perf_counter()
def save():report.write_text(json.dumps(record,ensure_ascii=False,indent=2))
@contextmanager
def stage(name,device):
 t=time.perf_counter();r0=resource.getrusage(resource.RUSAGE_SELF);c0=resource.getrusage(resource.RUSAGE_CHILDREN);item={'name':name,'execution':device,'status':'running'}
 print('START',name,device,flush=True)
 try:
  yield item
  item['status']='passed'
 except BaseException as e:
  item.update(status='failed',error=str(e));raise
 finally:
  r1=resource.getrusage(resource.RUSAGE_SELF);c1=resource.getrusage(resource.RUSAGE_CHILDREN)
  item.update(wall_seconds=time.perf_counter()-t,python_process_cpu_seconds=r1.ru_utime+r1.ru_stime-r0.ru_utime-r0.ru_stime,child_process_cpu_seconds=c1.ru_utime+c1.ru_stime-c0.ru_utime-c0.ru_stime,process_max_rss_kib=r1.ru_maxrss)
  record['stages'].append(item);record['elapsed_seconds']=time.perf_counter()-started;save();print('END',name,round(item['wall_seconds'],3),flush=True)
try:
 with stage('01_probe_input','CPU / ffprobe') as s:
  metadata=probe(source);total=float(metadata['format']['duration']);record['video_seconds']=total;record['source_sha256']=hashlib.sha256(source.read_bytes()).hexdigest()
 with stage('02_transcribe',f'Intel GPU / {args.asr_backend}; audio decode + VAD on CPU') as s:
  details={};model=(root/'artifacts/igpu-setup/whisper-small-model-path.txt').read_text().strip()
  cues,language=transcribe_video(source,model_name=model,device='gpu',backend=args.asr_backend,cache_dir=work/'asr-models',language='en',beam_size=5,details=details,python_executable=args.asr_python,report=lambda text:print(text,flush=True))
  s.update(asr_execution=details,language=language,cue_count=len(cues));write_srt(cues,work/'raw.en.srt')
 with stage('03_assign_speakers','CPU / reviewed source-specific ledger') as s:
  indexed={c.index:c for c in cues};assigned=[]
  for entry in ledger['utterances']:
   original=indexed[entry['asr_index']]
   if original.text.strip()!=entry['asr_text'].strip():raise RuntimeError('ASR text changed; review speaker ledger before assigning voices')
   assigned.append(Cue(len(assigned)+1,entry['start'],entry['end'],entry['english'],entry['speaker']))
  cues=assigned;write_srt(cues,work/'speakers.en.srt');s['cue_count']=len(cues)
 with stage('04_translate','CPU / PyTorch OPUS EN-VI') as s:
  translator=OpusMTTranslator(device='cpu')
  try:
   translated=translate_cues(cues,translator,work/'translations',on_progress=lambda n,total:print(f'Translation {n}/{total}',flush=True))
   s['actual_device']=str(translator._model.device);write_srt(translated,work/'automatic.vi.srt')
  finally:translator.close()
 with stage('05_apply_reviewed_translation','CPU / prepared review overrides') as s:
  reviews=[];reviewed=[]
  for cue,entry in zip(translated,ledger['utterances']):
   value=entry.get('vietnamese') or cue.text
   reviewed.append(replace(cue,text=value));reviews.append({'index':cue.index,'speaker':cue.speaker,'auto':cue.text,'final':value})
  cues=reviewed;write_srt(cues,work/'final.vi.srt');(work/'translation-review.json').write_text(json.dumps(reviews,ensure_ascii=False,indent=2));s['override_count']=sum(c['auto']!=c['final'] for c in reviews)
 with stage('06_synthesize_all_voices','CPU / VieNeu v3 Turbo ONNX FP32') as s:
  provider=VieNeuTTS(voice=ledger['voices']['PETER'],speaker_voices=ledger['voices'],device='cpu',cache_dir=work/'voices')
  try:
   results,failures=synthesize_cues(cues,provider,work/'tts',workers=1,on_progress=lambda n,total:print(f'TTS {n}/{total}',flush=True))
   if failures:raise RuntimeError(str([(c.index,error) for c,error in failures]))
   model=provider._model;s['backend']=model.backend if model else 'cache';s['providers']=model.engine.sess_pre.get_providers() if model else [];s['generated_clips']=len(results);s['execution']=provider.execution
  finally:provider.close()
 with stage('07_align_speech','CPU / FFmpeg') as s:
  clips=align(cues,results,work/'aligned',max_tempo=1.6);s['clips']=[asdict(c)|{'path':str(c.path)} for c in clips]
  expected_spoken=sum(_has_speakable_content(c.text) for c in cues)
  if len(clips)!=expected_spoken:raise RuntimeError('Missing speech clip')
 with stage('08_render_timeline','CPU / PCM'):
  render_timeline(clips,total,work/'timeline.wav')
 with stage('09_extract_original_audio','CPU / FFmpeg'):
  extract_audio(source,work/'original.wav')
 with stage('10_mix_voiceover','CPU / FFmpeg'):
  mix(MixMode.VOICEOVER,work/'timeline.wav',work/'mixed.wav',original_wav=work/'original.wav')
 with stage('11_export_mkv_srt','CPU / FFmpeg; video stream copy'):
  srt=output/'Family Guy - Peter falls down the stairs.vi.srt';write_srt(cues,srt);mkv=output/'Family Guy - Peter falls down the stairs.AI-Dub.vi.mkv';mux_mkv(source,work/'mixed.wav',srt,mkv)
 record['time_to_files_seconds']=time.perf_counter()-started
 with stage('12_validate_output','CPU / ffprobe + FFmpeg full decode') as s:
  decoded=subprocess.run(['ffmpeg','-v','error','-i',str(mkv),'-map','0:v?','-map','0:a?','-f','null','-'],capture_output=True,text=True);s.update(decode_returncode=decoded.returncode,decode_errors=decoded.stderr)
  if decoded.returncode:raise RuntimeError('Output decode failed')
  hashes=[subprocess.check_output(['ffmpeg','-v','error','-i',str(p),'-map','0:v:0','-c','copy','-f','hash','-hash','sha256','-'],text=True).strip() for p in [source,mkv]];s['video_stream_copy']=len(set(hashes))==1
  if not s['video_stream_copy']:raise RuntimeError('Video stream differs')
  s['streams']=probe(mkv)['streams'];s['source_unchanged']=hashlib.sha256(source.read_bytes()).hexdigest()==record['source_sha256'];s['output_files']=[{'path':str(p),'bytes':p.stat().st_size} for p in [mkv,srt]]
 record['run_status']='completed';record['total_seconds_including_validation']=time.perf_counter()-started;save()
 print(json.dumps({'total_seconds':record['total_seconds_including_validation'],'report':str(report)},indent=2),flush=True)
except BaseException:
 record['run_status']='failed';record['total_seconds_including_validation']=time.perf_counter()-started;save();raise
