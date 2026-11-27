"""Run the real public sample twice, with exact cached speech and no online TTS.
Usage: .venv/bin/python scripts/verify_optimization_checkpoint.py cp1
"""
from collections import Counter
from pathlib import Path
import hashlib,json,resource,shutil,subprocess,sys,time
from jellydub import pipeline as p
from jellydub.config import Config
from jellydub.cache import cache_key
from jellydub.subtitles import parse_srt
from jellydub.tts.edge import EdgeTTS
root=Path(__file__).resolve().parents[1]
base=root/'artifacts/optimization-checkpoints'/sys.argv[1];base.mkdir(parents=True,exist_ok=True)
video=next((root/'videos').glob('*[[]n0LY3XcZaCs[]].mp4'));srt=root/'dubbed/vendor-quotations/vendor-quotations.vi.srt'
config=Config(source_lang='vi',voice='vi-VN-NamMinhNeural',mix='voiceover',out_dir=base/'output',cache_dir=base/'cache',download=False)
config.validate();selected,_=p._resolve_subtitle(video,srt,config,lambda message:None)
work=p._work_dir(video,selected,config)
if (work/'timeline.wav').exists():raise SystemExit('Use a new checkpoint label for cold measurement')
(work/'tts').mkdir(exist_ok=True)
provider=EdgeTTS(voice=config.voice)
for cue in parse_srt(selected):
 name=cache_key(provider.cache_id(),cue.text,cue.speaker)+'.wav'
 matches=list((root/'.jellydub-cache').glob('Compare*/tts/'+name))
 if not matches:raise SystemExit(f'Missing speech {cue.index}')
 shutil.copy2(matches[0],work/'tts'/name)
def forbidden(*a,**kw):raise RuntimeError('No online TTS allowed in checkpoint media verification')
EdgeTTS.synthesize=forbidden
# CP2 can import exact legacy speech into its shared cache using the normal path.
original_run=subprocess.run;counts=Counter()
def measured(*a,**kw):
 argv=a[0] if a else kw.get('args',[])
 if isinstance(argv,(list,tuple)) and argv:counts[Path(str(argv[0])).name]+=1
 return original_run(*a,**kw)
subprocess.run=measured
fingerprint=lambda path:hashlib.sha256(path.read_bytes()).hexdigest()
before={str(path):fingerprint(path) for path in (video,srt)}
report={'scope':'VI SRT + cached Edge speech, cold-media and repeat; no live ASR/translation/TTS','runs':[]}
for i in ((1,2,3) if sys.argv[1].startswith("cp2") else (1,2)):
 if i==3:config.out_dir=base/"output-moved"
 counts.clear();cpu=resource.getrusage(resource.RUSAGE_CHILDREN);prior=cpu.ru_utime+cpu.ru_stime;start=time.perf_counter()
 with (base/f'run-{i}.log').open('w') as log:
  outputs=p.dub(video,srt,config,report=lambda message:(log.write(message+'\n'),log.flush()))
 cpu=resource.getrusage(resource.RUSAGE_CHILDREN)
 report['runs'].append({'run':i,'wall_seconds':time.perf_counter()-start,'child_cpu_seconds':cpu.ru_utime+cpu.ru_stime-prior,'commands':dict(counts)})
 (base/'verification.json').write_text(json.dumps(report,indent=2));print(report['runs'][-1],flush=True)
report['checks']={'output_only_mkv_srt':{path.suffix for path in config.out_dir.iterdir()}=={'.mkv','.srt'},'source_unchanged':all(fingerprint(Path(path))==value for path,value in before.items()),'no_mp3_intermediate':not list(work.glob('*.mp3'))}
for path in (video,outputs['mkv']):
 value=subprocess.check_output(['ffmpeg','-v','error','-i',str(path),'-map','0:v:0','-c','copy','-f','hash','-hash','sha256','-'],text=True).strip()
 report.setdefault('video_hashes',[]).append(value)
report['checks']['video_stream_copy']=len(set(report['video_hashes']))==1
probe=json.loads(subprocess.check_output(['ffprobe','-v','error','-show_streams','-of','json',str(outputs['mkv'])]))
report['streams']=probe['streams']
for kind in ('audio','subtitle'):report['checks']['vietnamese_'+kind]=any(s['codec_type']==kind and s.get('tags',{}).get('language')=='vie' for s in probe['streams'])
decoded=subprocess.run(['ffmpeg','-v','error','-i',str(outputs['mkv']),'-map','0:v?','-map','0:a?','-f','null','-'],capture_output=True,text=True)
report['checks']['full_decode']=decoded.returncode==0;report['decode_errors']=decoded.stderr
report['passed']=all(report['checks'].values());(base/'verification.json').write_text(json.dumps(report,indent=2))
print(json.dumps(report['checks'],indent=2))
if not report['passed']:raise SystemExit(1)
