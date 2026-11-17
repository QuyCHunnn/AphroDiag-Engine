"""End-to-end SDK TTS: serial CPU, concurrent CPU, CPU/GPU acoustic batching.
Global prefill/decode and codec remain CPU. No change to CLI defaults.
"""
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor
import argparse,os,json,time,hashlib,resource
os.environ['HF_HUB_OFFLINE']='1'
import numpy as np
from threadpoolctl import threadpool_limits
from jellydub.tts.vieneu import VieNeuTTS
from jellydub.tts.batched_frame_graph import BatchedFrameRuntime
from jellydub.tts.gpu_batch_runtime import AcousticBatcher,clone_model_for_request
from jellydub.subtitles import parse_srt
root=Path(__file__).resolve().parents[1]
p=argparse.ArgumentParser();p.add_argument('--run',default='tts-gpu-batch-full');p.add_argument('--cohort',choices=['mixed','matched'],default='mixed');p.add_argument('--batch-wait',type=float,default=.012);args=p.parse_args()
out=root/'artifacts/next-optimization'/args.run
if out.exists():raise SystemExit('Preserve prior results')
out.mkdir(parents=True)
voices=json.loads((root/'artifacts/family-guy-vieneu/speaker-ledger.json').read_text())['voices']
cues=parse_srt(root/'artifacts/family-guy-vieneu/reviewed.vi.srt')
selected=[next(c for c in cues if c.speaker==speaker) for speaker in ['PETER','LOIS','STEWIE','BRIAN']]
if args.cohort=='matched':selected=[next(c for c in cues if c.index==i) for i in [2,10,16,4]]
provider=VieNeuTTS(device='cpu',blas_threads=2)
record={'status':'running','scope':'full SDK TTS; acoustic graph only on GPU in GPU mode; no processing cache','cohort':args.cohort,'cue_indices':[c.index for c in selected],'batch_wait_seconds':args.batch_wait,'watermark_enabled':False,'runs':[],'setups':[]}
def save():(out/'report.json').write_text(json.dumps(record,ensure_ascii=False,indent=2))
try:
 model=provider._load()
 with threadpool_limits(limits=2,user_api='blas'):
  np.random.seed(123);model.infer('Xin chào.',voice=voices['PETER'],apply_watermark=False)
  for mode in ['cpu-serial','cpu-parallel','cpu-batch4','gpu-batch4']:
   runtime=batcher=None
   if 'batch4' in mode:
    started=time.perf_counter();runtime=BatchedFrameRuntime(model.engine,4,'GPU' if mode.startswith('gpu') else 'CPU');batcher=AcousticBatcher(runtime,max_wait_seconds=args.batch_wait)
    record['setups'].append({'mode':mode,'seconds':time.perf_counter()-started,'devices':runtime.devices});save()
   def one(cue):
    seed=int.from_bytes(hashlib.sha256((str(cue.speaker)+'|'+cue.text).encode()).digest()[:4],'little')
    instance=clone_model_for_request(model,seed,batcher)
    started=time.perf_counter();audio=instance.infer(cue.text,voice=voices[cue.speaker],apply_watermark=False)
    elapsed=time.perf_counter()-started;audio=np.asarray(audio,np.float32)
    if not audio.size or not np.isfinite(audio).all():raise RuntimeError('Invalid waveform')
    return audio,{'cue':cue.index,'speaker':cue.speaker,'seconds':elapsed,'samples':audio.size,'sha256':hashlib.sha256(audio.tobytes()).hexdigest()}
   try:
    for trial in [1,2]:
     started=time.perf_counter()
     if mode=='cpu-serial':results=[one(c) for c in selected]
     else:
      with ThreadPoolExecutor(max_workers=4) as pool:results=list(pool.map(one,selected))
     elapsed=time.perf_counter()-started
     rows=[]
     for (audio,row),cue in zip(results,selected):
      model.save(audio,str(out/f'{mode}-{trial}-{cue.index}-{cue.speaker}.wav'));rows.append(row)
     entry={'mode':mode,'trial':trial,'seconds':elapsed,'outputs':rows}
     if runtime:entry['batch_histogram']=dict(runtime.histogram)
     record['runs'].append(entry);save();print(mode,trial,round(elapsed,3),flush=True)
   finally:
    if batcher:batcher.close()
  reference=record['runs'][0]['outputs']
  for run in record['runs']:
   run['same_waveforms_as_serial']=all(a['sha256']==b['sha256'] for a,b in zip(reference,run['outputs']))
  record.update(status='measured',process_max_rss_kib=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,quality_status='pending listening if waveform differs');save()
except BaseException as exc:
 record.update(status='failed',error=str(exc));save();raise
finally:provider.close()
