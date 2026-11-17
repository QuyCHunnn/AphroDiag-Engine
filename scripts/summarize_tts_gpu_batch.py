"""Aggregate true acoustic batching probes; never infer full-video throughput."""
import json,statistics
from pathlib import Path
root=Path(__file__).resolve().parents[1];base=root/'artifacts/next-optimization'
cohorts=[]
for name in ['tts-gpu-batch-full','tts-gpu-batch-matched']:
 r=json.loads((base/name/'report.json').read_text())
 if r['status']!='measured':raise SystemExit('Incomplete run: '+name)
 rows=[]
 for mode in ['cpu-serial','cpu-parallel','cpu-batch4','gpu-batch4']:
  trials=[t for t in r['runs'] if t['mode']==mode]
  entry={'mode':mode,'median_seconds':statistics.median(t['seconds'] for t in trials),'trials_seconds':[t['seconds'] for t in trials],'same_audio_as_seeded_serial':all(t['same_waveforms_as_serial'] for t in trials)}
  if 'batch_histogram' in trials[-1]:
   histogram=trials[-1]['batch_histogram'];calls=sum(histogram.values());entry.update(actual_active_rows_histogram=histogram,mean_active_rows=sum(int(k)*v for k,v in histogram.items())/calls,compiled_capacity=4)
  rows.append(entry)
 timings={r['mode']:r['median_seconds'] for r in rows}
 cohorts.append({'cohort':r.get('cohort','mixed'),'rows':rows,'gpu_speedup_vs_serial':timings['cpu-serial']/timings['gpu-batch4'],'gpu_speedup_vs_cpu_parallel':timings['cpu-parallel']/timings['gpu-batch4'],'setups':r['setups']})
report={'scope':'four-cue full SDK synthesis per cohort; only acoustic graph on GPU; global prefill/decode/codec CPU; no waveform cache; model warm; setup excluded','cohorts':cohorts,'gpu_default_adopted':False,'reason':'No consistent advantage over CPU alternatives; batched graph waveforms differ and listening is unvalidated','limits':['two trials per mode, fixed mode order, not balanced multi-process A/B','short cohorts, no full-video or 44-cue validation','same RNG seeds do not guarantee numerically identical sampling across batch sizes','watermark disabled in all modes','capacity four is not sustained active batch four']}
(base/'tts-gpu-batch-summary.json').write_text(json.dumps(report,indent=2));print(json.dumps(report,indent=2))
