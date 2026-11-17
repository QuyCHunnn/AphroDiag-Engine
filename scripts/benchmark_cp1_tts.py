"""Fresh-cache TTS A/B for the same reviewed 44 Family Guy utterances.
Run baseline (8 BLAS threads) and optimized (2) sequentially, never concurrently.
Fixed per-cue seeds allow bit-for-bit comparison of normalized waveform files.
"""
from pathlib import Path
import argparse,hashlib,json,os,time
import numpy as np
from threadpoolctl import threadpool_info
from jellydub.subtitles import parse_srt
from jellydub.tts.vieneu import VieNeuTTS
from jellydub.tts.synthesize import synthesize_cues

root=Path(__file__).resolve().parents[1]
parser=argparse.ArgumentParser();parser.add_argument('variant',choices=['baseline8','optimized2']);parser.add_argument('--out-dir', type=Path, default=root/'artifacts/cp1-performance');args=parser.parse_args()
base=args.out_dir;work=base/args.variant
if work.exists():raise SystemExit('Benchmark directory already exists; use preserved results instead of measuring cache hits.')
work.mkdir(parents=True);os.environ['HF_HUB_OFFLINE']='1'
ledger=json.loads((root/'artifacts/family-guy-vieneu/speaker-ledger.json').read_text());cues=parse_srt(root/'artifacts/family-guy-vieneu/reviewed.vi.srt')
provider=VieNeuTTS(voice=ledger['voices']['PETER'],speaker_voices=ledger['voices'],device='cpu',blas_threads=8 if args.variant=='baseline8' else 2)
record={'variant':args.variant,'blas_threads':provider.blas_threads,'processing_cache_initially_empty':True,'model_downloads_included':False,'dataset_subtitle_cues':len(cues),'calls':[],'status':'running'}
report=work/'report.json';original=provider.synthesize;load=provider._load;started=time.perf_counter()
def save():report.write_text(json.dumps(record,ensure_ascii=False,indent=2))
def timed_load():
 initial=provider._model is None;t=time.perf_counter();model=load()
 if initial:
  record['model_load_seconds']=time.perf_counter()-t;record['onnx_threads']=model.engine.ort_intra_op_threads;record['providers']=model.engine.sess_pre.get_providers()
  infer=model.infer
  def observed_infer(*a,**kw):
   record['actual_blas_threads']=sorted({p['num_threads'] for p in threadpool_info() if p['user_api']=='blas'})
   return infer(*a,**kw)
  model.infer=observed_infer
 return model
provider._load=timed_load
def seeded(text,path,speaker=None):
 seed=int.from_bytes(hashlib.sha256((str(speaker)+'|'+text).encode()).digest()[:4],'little');np.random.seed(seed);t=time.perf_counter();original(text,path,speaker=speaker)
 record['calls'].append({'speaker':speaker,'text':text,'seed':seed,'seconds':time.perf_counter()-t});save()
provider.synthesize=seeded
try:
 results,failures=synthesize_cues(cues,provider,work/'speech',workers=1,on_progress=lambda n,total:print(f'{args.variant} {n}/{total}',flush=True))
 record['wall_seconds']=time.perf_counter()-started
 if failures:raise RuntimeError(str([(c.index,error) for c,error in failures]))
 record['clips']=[{'cue_index':r.cue_index,'duration':r.duration,'sha256':hashlib.sha256(r.path.read_bytes()).hexdigest(),'path':str(r.path)} for r in results]
 record['speech_count']=len(results);record['status']='passed';save();assert len(results)==len(record['calls'])==44
finally:
 provider.close();record['blas_threads_after_close']=sorted({p['num_threads'] for p in threadpool_info() if p['user_api']=='blas'});save()
print(json.dumps({k:v for k,v in record.items() if k not in ['clips','calls']},indent=2),flush=True)
