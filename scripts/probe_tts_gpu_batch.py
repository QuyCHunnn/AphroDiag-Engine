"""True batch acoustic throughput and row-isolation probe; not full TTS throughput."""
import os,time,json,hashlib
from pathlib import Path
os.environ['HF_HUB_OFFLINE']='1'
import numpy as np
from threadpoolctl import threadpool_limits
from jellydub.tts.vieneu import VieNeuTTS
from jellydub.tts.batched_frame_graph import BatchedFrameRuntime
root=Path(__file__).resolve().parents[1];out=root/'artifacts/next-optimization/tts-gpu-batch-probe'
if out.exists():raise SystemExit('Preserve prior results')
out.mkdir(parents=True)
record={'status':'running','scope':'single acoustic frame only; global decoder and codec excluded','runs':[]}
def save():(out/'report.json').write_text(json.dumps(record,indent=2))
provider=VieNeuTTS(device='cpu',blas_threads=2)
try:
 model=provider._load();engine=model.engine;original=engine._acoustic_frame;samples=[]
 def observed(h,temperature,top_k,top_p,rep_pen,hist):
  if len(samples)<8:
   history=np.zeros(engine.audio_emb.shape[:2],bool)
   if hist is not None:
    for i in range(engine.n_vq):history[i,list(hist[i])]=True
   rng=np.random.RandomState();rng.set_state(np.random.get_state())
   uniforms=rng.random(engine.n_vq).astype(np.float32)
   tokens=np.stack([h[0].astype(np.float32),engine.text_emb[engine.sgs].astype(np.float32)])[None]
   start=time.perf_counter();codes,eos=original(h,temperature,top_k,top_p,rep_pen,hist)
   samples.append({'tokens':tokens[0].copy(),'history':history,'uniforms':uniforms,'codes':codes,'eos':eos,'seconds':time.perf_counter()-start})
   return codes,eos
  return original(h,temperature,top_k,top_p,rep_pen,hist)
 engine._acoustic_frame=observed
 with threadpool_limits(limits=2,user_api='blas'):
  np.random.seed(345);model.infer('Xin chào, tôi đang kiểm tra tốc độ xử lý.',voice='Quốc Tuấn')
  engine._acoustic_frame=original
  record['sdk_cpu_frame_seconds']=[s['seconds'] for s in samples]
  reference=None
  for device,batch in [('CPU',1),('CPU',2),('GPU',1),('GPU',2),('GPU',4)]:
   start=time.perf_counter();runtime=BatchedFrameRuntime(engine,batch,device);setup=time.perf_counter()-start
   timings=[];outputs=[]
   for trial in range(3):
    started=time.perf_counter();current=[]
    for offset in range(0,len(samples),batch):
     rows=samples[offset:offset+batch]
     codes,eos=runtime.run([s['tokens'] for s in rows],[s['history'] for s in rows],[s['uniforms'] for s in rows])
     current.extend([{'codes':c.tolist(),'eos':bool(e)} for c,e in zip(codes,eos)])
    elapsed=time.perf_counter()-started
    if trial:timings.append(elapsed)
    outputs=current
   if reference is None:reference=outputs
   row={'device':device,'batch':batch,'setup_seconds':setup,'seconds_for_eight_frames':timings,'execution_devices':runtime.devices,'matches_graph_b1':outputs==reference,'matches_sdk_samples':all(o['codes']==s['codes'] and o['eos']==s['eos'] for o,s in zip(outputs,samples)),'actual_row_batch_histogram':runtime.histogram}
   record['runs'].append(row);save();print(json.dumps(row),flush=True)
   del runtime
 record['status']='measured';save()
except BaseException as exc:
 record.update(status='failed',error=str(exc));save();raise
finally:provider.close()
