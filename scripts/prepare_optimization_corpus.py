"""Prepare duration-checked local benchmark clips without padding/looping sources."""
from pathlib import Path
import json,subprocess
from jellydub.cache import file_digest
from jellydub.ffmpeg import duration,probe
root=Path(__file__).resolve().parents[1];out=root/'artifacts/next-optimization/corpus'
out.mkdir(parents=True,exist_ok=True)
specs=[('tutorial180','Compare vendor',180,'tutorial_single_speaker'),('effects180','YTSave',180,'game_voices_effects'),('family-regression','Family Guy',None,'dialogue_multi_role_regression')]
records=[]
for name,prefix,seconds,category in specs:
 source=next(p for p in (root/'videos').glob('*.mp4') if p.name.startswith(prefix))
 target=out/(name+'.mkv')
 if not target.exists():
  args=['ffmpeg','-nostdin','-v','error','-i',str(source)]
  if seconds:args+=['-t',str(seconds)]
  subprocess.run(args+['-map','0:v:0','-map','0:a:0','-c','copy',str(target)],check=True)
 actual=duration(target)
 records.append({'id':name,'category':category,'source':str(source),'source_sha256':file_digest(source),'clip':str(target),'clip_sha256':file_digest(target),'duration_seconds':actual,'requested_seconds':seconds,'duration_within_0_1s':seconds is None or abs(actual-seconds)<=.1,'reference_status':'not_listening_validated','automatic_speaker_labels':False})
manifest={'clips':records,'missing_acceptance_cases':['180-second multi-role dialogue with listening-validated ground truth'],'limitations':['Family Guy is 156.665 seconds, never padded or called 180s','Game source has different speech density; not interchangeable with tutorial','No quality acceptance from media hashes']}
(out/'manifest.json').write_text(json.dumps(manifest,indent=2));print(json.dumps(manifest,indent=2))
