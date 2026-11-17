"""Run a fresh CLI job, recording elapsed messages and sampled Linux process-tree RSS.
Example: .venv/bin/python scripts/trace_pipeline_job.py --run baseline1 --clip tutorial180
"""
import argparse,json,os,subprocess,threading,time,sys,signal
from pathlib import Path
root=Path(__file__).resolve().parents[1]
p=argparse.ArgumentParser();p.add_argument('--run',required=True);p.add_argument('--clip',required=True);p.add_argument('--mode',choices=['sequential','overlap'],default='overlap');args=p.parse_args()
manifest=json.loads((root/'artifacts/next-optimization/corpus/manifest.json').read_text())
clip=next(c for c in manifest['clips'] if c['id']==args.clip)
work=root/'artifacts/next-optimization/baselines'/args.run
if work.exists():raise SystemExit('Run directory already exists; preserve measurements')
work.mkdir(parents=True)
model=(root/'artifacts/igpu-setup/whisper-small-model-path.txt').read_text().strip()
command=[str(root/'.venv/bin/jellydub'),'auto',clip['clip'],'--source-lang','en','--asr-backend','genai','--asr-python',str(root/'.venv-native-asr-next/bin/python'),'--asr-model',model,'--asr-device','gpu','--tts','vieneu','--tts-device','cpu','--translate-device','cpu','--tts-precision','int8','--pipeline-mode',args.mode,'--cache-dir',str(work/'cache'),'--out-dir',str(work/'output')]
# Explicit config prevents online subtitle search, preserves native beam=5.
config=work/'config.toml';config.write_text('[jellydub]\ndownload=false\nmax_tempo=1.6\nmix="voiceover"\n')
command+=['--config',str(config)]
started=time.perf_counter();events=[];samples=[];sample_errors=[];pipeline_error=None;code=1;stop=threading.Event()
(work/'report.json').write_text(json.dumps({'status':'running','command':command}))
try:
 process=subprocess.Popen(command,stdout=subprocess.PIPE,stderr=subprocess.STDOUT,text=True,start_new_session=True,env={**os.environ,'HF_HUB_OFFLINE':'1','JELLYDUB_TRACE_FILE':str(work/'spans.jsonl')})
except Exception as exc:
 (work/'report.json').write_text(json.dumps({'status':'failed','error':str(exc),'command':command}))
 raise SystemExit(1)
def sample():
 while not stop.is_set():
  table={}
  for path in Path('/proc').glob('[0-9]*/status'):
   try:
    data={line.split(':',1)[0]:line.split(':',1)[1].strip() for line in path.read_text().splitlines() if ':' in line}
    table[int(path.parent.name)]=(int(data['PPid']),int(data.get('VmRSS','0 kB').split()[0]))
   except (OSError,ValueError,KeyError):continue
  descendants={process.pid}
  while True:
   new={pid for pid,(parent,rss) in table.items() if parent in descendants}
   if new<=descendants:break
   descendants|=new
  samples.append({'seconds':time.perf_counter()-started,'rss_sum_kib':sum(table.get(pid,(0,0))[1] for pid in descendants),'processes':len(descendants)})
  stop.wait(.5)
def checked_sample():
 try:sample()
 except Exception as exc:sample_errors.append(str(exc))
thread=threading.Thread(target=checked_sample,daemon=True);thread.start()
try:
 with (work/'messages.jsonl').open('w') as log:
  for line in process.stdout:
   event={'seconds':time.perf_counter()-started,'message':line.rstrip()};events.append(event);log.write(json.dumps(event,ensure_ascii=False)+'\n');log.flush();print(line,end='',flush=True)
 code=process.wait()
except Exception as exc:
 pipeline_error=str(exc)
finally:
 if process.poll() is None:
  os.killpg(process.pid,signal.SIGTERM)
  try:process.wait(timeout=10)
  except subprocess.TimeoutExpired:os.killpg(process.pid,signal.SIGKILL);process.wait()
 stop.set();thread.join();process.stdout.close()
report={'status':'passed' if code==0 and samples and not sample_errors and pipeline_error is None else 'failed','returncode':code,'seconds':time.perf_counter()-started,'mode':'cold_process_empty_processing_cache','clip':clip,'command':command,'peak_sampled_process_tree_rss_kib':max((s['rss_sum_kib'] for s in samples),default=None),'rss_samples':samples,'sampling_errors':sample_errors,'pipeline_error':pipeline_error,'sampling_status':'passed' if samples and not sample_errors else 'failed','measurement_limits':['RSS sum includes shared pages in multiple processes; not unique physical RAM','GPU utilization not sampled','Worker-ready empty-cache mode not implemented yet','Spans are nested wall time, not additive exclusive compute; import/VAD/export subphases still require attribution'],'critical_path_trace_complete':False,'spans_file':str(work/'spans.jsonl')}
(work/'report.json').write_text(json.dumps(report,ensure_ascii=False,indent=2))
sys.exit(code if code else (0 if report['status']=='passed' else 1))
