"""ASR precision A/B; same Small/beam5/VAD/guard, no change to CLI defaults."""
from pathlib import Path
from dataclasses import asdict
import argparse,json,time,os,re
from difflib import SequenceMatcher
os.environ['HF_HUB_OFFLINE']='1'
from jellydub.asr_openvino import load_audio
from jellydub.asr_native import transcribe
root=Path(__file__).resolve().parents[1]
p=argparse.ArgumentParser();p.add_argument('--clip',default='family-regression');args=p.parse_args()
manifest=json.loads((root/'artifacts/next-optimization/corpus/manifest.json').read_text());clip=next(c for c in manifest['clips'] if c['id']==args.clip)
base=root/'artifacts/next-optimization/native-precision'/args.clip
if base.exists():raise SystemExit('Preserve prior benchmark evidence')
base.mkdir(parents=True);audio=load_audio(Path(clip['clip']))
model=(root/'artifacts/igpu-setup/whisper-small-model-path.txt').read_text().strip();runs=[]
for precision in ('f32','f16'):
 details={};started=time.perf_counter()
 try:
  cues,language=transcribe(audio,model_name=model,device='GPU',cache_dir=base/precision,language='en',beam_size=5,python_executable=str(root/'.venv-native-asr-next/bin/python'),report=lambda s:print(precision,s,flush=True),details=details,precision=precision)
  runs.append({'precision':precision,'status':'passed','seconds':time.perf_counter()-started,'cues':[asdict(c) for c in cues],'details':details})
 except Exception as exc:
  runs.append({'precision':precision,'status':'failed','seconds':time.perf_counter()-started,'error':str(exc)})
 (base/'report.json').write_text(json.dumps({'clip':clip,'runs':runs},ensure_ascii=False,indent=2))
if all(r['status']=='passed' for r in runs):
 a,b=runs;different=[{'index':x['index'],'f32':x['text'],'f16':y['text']} for x,y in zip(a['cues'],b['cues']) if x['text']!=y['text']]
 result={'runs':runs,'clip':clip,'same_cue_count':len(a['cues'])==len(b['cues']),'same_full_cues':a['cues']==b['cues'],'different_text_cues':different,'speedup':a['seconds']/b['seconds'],'quality_status':'not listening validated'}
 words=lambda run:re.findall(r'\w+',' '.join(c['text'] for c in run['cues']).casefold())
 result.update(normalized_words_identical=words(a)==words(b),word_sequence_agreement=SequenceMatcher(None,words(a),words(b),autojunk=False).ratio(),cue_counts=[len(a['cues']),len(b['cues'])],comparison_note='different_text_cues is positional, not aligned when segmentation differs')
 (base/'report.json').write_text(json.dumps(result,ensure_ascii=False,indent=2));print(json.dumps({k:v for k,v in result.items() if k not in {'runs','clip'}},indent=2))
