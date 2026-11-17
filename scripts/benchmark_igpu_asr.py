"""Same public Whisper Tiny INT8 model and audio on CPU/GPU; isolated from engine.
Use --device CPU or GPU. No AUTO fallback. Run benchmarks sequentially after
profile_engine.py so CPU contention does not invalidate the comparison.
"""
import argparse
import json
from pathlib import Path
import resource
import subprocess
import time
import wave
import numpy as np
import openvino as ov
import openvino_genai as genai

root=Path(__file__).resolve().parents[1]
out=root/'artifacts/performance-audit/asr'
out.mkdir(parents=True,exist_ok=True)
parser=argparse.ArgumentParser()
parser.add_argument('--device',required=True,choices=['CPU','GPU'])
parser.add_argument('--prepare-only',action='store_true')
parser.add_argument('--model',choices=['tiny','small'],default='tiny')
parser.add_argument('--beams',type=int,default=1)
parser.add_argument('--run-name',default=None)
parser.add_argument('--max-samples',type=int,default=3)
parser.add_argument('--disable-sdpa',action='store_true')
parser.add_argument('--nested-properties',action='store_true')
parser.add_argument('--precision',choices=['f32','f16'],default=None)
args=parser.parse_args()
if args.model!='tiny':
    out=out/(args.model if args.beams==5 else f'{args.model}-beam{args.beams}')
    out.mkdir(parents=True,exist_ok=True)
if args.run_name:
    out=out/args.run_name
    out.mkdir(parents=True,exist_ok=True)
video=next((root/'videos').glob('*[[]n0LY3XcZaCs[]].mp4'))
samples=[]
for start,length in [(4,20),(65,40),(130,60)]:
    path=out/f'sample-{length}s.wav'
    if not path.exists():
        subprocess.run(['ffmpeg','-v','error','-nostdin','-y','-ss',str(start),'-i',str(video),'-t',str(length),'-map','0:a:0','-ar','16000','-ac','1','-c:a','pcm_s16le',str(path)],check=True)
    samples.append({'path':str(path),'source_start_seconds':start,'duration_seconds':length})
(out/'samples.json').write_text(json.dumps(samples,indent=2))
if args.prepare_only:
    print('Prepared 3 samples');raise SystemExit(0)
model=Path((root/f'artifacts/igpu-setup/whisper-{args.model}-model-path.txt').read_text().strip())
core=ov.Core()
if args.device not in core.available_devices:
    raise SystemExit(f'{args.device} unavailable: {core.available_devices}')
report={'device':args.device,'device_name':core.get_property(args.device,'FULL_DEVICE_NAME'),
        'model_path':str(model),'openvino_version':ov.__version__,'genai_version':genai.__version__,
        'settings':{'language':'<|en|>','task':'transcribe','return_timestamps':True,'num_beams':args.beams},
        'scope':'Native GenAI experiment; main Intel CLI currently uses Optimum generation','samples':[]}
properties={'GPU_ENABLE_SDPA_OPTIMIZATION':False} if args.disable_sdpa and args.device=='GPU' else {}
if properties and args.nested_properties:properties={'DEVICE_PROPERTIES':{'GPU':properties}}
if args.precision:properties['INFERENCE_PRECISION_HINT']=args.precision
report['properties']=properties
t=time.perf_counter();pipe=genai.WhisperPipeline(str(model),args.device,**properties)
report['model_load_compile_seconds']=time.perf_counter()-t
report_path=out/f'{args.device.lower()}.json'
report_path.write_text(json.dumps(report,indent=2))
for sample in samples[:args.max_samples]:
    with wave.open(sample['path']) as w:
        data=np.frombuffer(w.readframes(w.getnframes()),dtype=np.int16).astype(np.float32)/32768.
    trials=[]
    for trial in range(2):
        cpu=time.process_time();start=time.perf_counter()
        try:
            result=pipe.generate(data,**report['settings'])
        except Exception as exc:
            report['failure']={'sample':sample,'trial':trial+1,'type':type(exc).__name__,'message':str(exc)}
            report_path.write_text(json.dumps(report,indent=2))
            raise
        elapsed=time.perf_counter()-start;cpu=time.process_time()-cpu
        text=' '.join(result.texts) if hasattr(result,'texts') else str(result)
        chunks=[{'start':c.start_ts,'end':c.end_ts,'text':c.text} for c in (result.chunks or [])] if hasattr(result,'chunks') else []
        trials.append({'trial':trial+1,'wall_seconds':elapsed,'process_cpu_seconds':cpu,
                       'real_time_factor':elapsed/sample['duration_seconds'],'text':text,'chunks':chunks})
        print(args.device,sample['duration_seconds'],'sec sample','trial',trial+1,round(elapsed,3),'seconds',flush=True)
        report['in_progress']={'sample':sample,'trials':trials}
        report_path.write_text(json.dumps(report,indent=2))
    report['samples'].append({**sample,'trials':trials})
    report.pop('in_progress',None)
    report_path.write_text(json.dumps(report,indent=2))
print('Saved',report_path)
