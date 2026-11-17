"""Measure valid PCM WAV duration using existing ffprobe helper vs WAV headers.
This measures metadata lookup only; it does not replace full corruption checks.
"""
import json
from pathlib import Path
import time
import wave
from jellydub.ffmpeg import duration
root=Path(__file__).resolve().parents[1]
out=root/'artifacts/performance-audit/run-20261116'
folder=next(p for p in (out/'cache').glob('*/tts') if len(list(p.glob('*.wav')))==38)
files=sorted(folder.glob('*.wav'))
rows=[]
for trial in range(3):
    start=time.perf_counter();a=[duration(p) for p in files];probe_seconds=time.perf_counter()-start
    start=time.perf_counter();b=[]
    for p in files:
        with wave.open(str(p)) as w:b.append(w.getnframes()/w.getframerate())
    header_seconds=time.perf_counter()-start
    rows.append({'trial':trial+1,'file_count':len(files),'ffprobe_seconds':probe_seconds,
                 'wav_header_seconds':header_seconds,'max_duration_difference_seconds':max(abs(x-y) for x,y in zip(a,b))})
(out/'wav-metadata-ab.json').write_text(json.dumps(rows,indent=2));print(json.dumps(rows,indent=2))
