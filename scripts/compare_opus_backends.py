"""Summarize completed OPUS backend runs without claiming semantic acceptance."""
import argparse,json,statistics
from pathlib import Path
p=argparse.ArgumentParser();p.add_argument('directory',type=Path);args=p.parse_args()
reports={v:json.loads((args.directory/v/'report.json').read_text()) for v in ['torch-cpu','ov-cpu','ov-gpu']}
if any(r['status']!='passed' for r in reports.values()):raise SystemExit('All runs must pass')
reference=reports['torch-cpu']['trials'][0]['translations']
rows=[]
for name,r in reports.items():
 rows.append({'variant':name,'load_compile_seconds':r['load_compile_seconds'],'inference_median_seconds':statistics.median(t['seconds'] for t in r['trials']),'trials':[t['seconds'] for t in r['trials']],'different_cues_per_trial':[[x['index'] for x,y in zip(reference,t['translations']) if x!=y] for t in r['trials']],'cue_count_matches':all(len(t['translations'])==len(reference) for t in r['trials']),'execution':r['execution']})
result={'rows':rows,'comparison':'same source cues and generation policy; one process per backend, two fresh-processing-cache trials; no blinded quality evaluation','quality_status':'translation agreement with existing backend only; repeated-output corrections still require review','end_to_end_verified':False}
(args.directory/'comparison.json').write_text(json.dumps(result,indent=2));print(json.dumps(result,indent=2))
