"""Verify technical artifacts and unchanged source after the performance audit."""
from pathlib import Path
import hashlib,json,subprocess
root=Path(__file__).resolve().parents[1];base=root/'artifacts/performance-audit/run-20261116'
profile=json.loads((base/'profile.json').read_text())
video=Path(profile['video'])
sources=json.loads((base/'baseline-source-hashes.json').read_text())
report={'source_files_unchanged':all(hashlib.sha256((root/p).read_bytes()).hexdigest()==h for p,h in sources.items()),'audio_ab':[]}
for mode in ('mp3_then_aac','direct_wav_to_aac'):
 path=base/'audio-ab'/f'{mode}.mkv'
 proc=subprocess.run(['ffmpeg','-v','error','-i',str(path),'-map','0:a','-vn','-f','null','-'],capture_output=True,text=True)
 report['audio_ab'].append({'mode':mode,'audio_decode_exit_code':proc.returncode,'errors':proc.stderr})
paths=[video,*[base/'audio-ab'/f'{m}.mkv' for m in ('mp3_then_aac','direct_wav_to_aac')]]
hashes=[]
for path in paths:
 hashes.append(subprocess.check_output(['ffmpeg','-v','error','-i',str(path),'-map','0:v:0','-c','copy','-f','hash','-hash','sha256','-'],text=True).strip())
report['source_video_stream_hash']=hashes[0]
report['copied_video_streams_match_source']=len(set(hashes))==1
report['passed']=report['source_files_unchanged'] and report['copied_video_streams_match_source'] and all(x['audio_decode_exit_code']==0 for x in report['audio_ab'])
(base/'verification.json').write_text(json.dumps(report,indent=2));print(json.dumps(report,indent=2))
if not report['passed']:raise SystemExit(1)
