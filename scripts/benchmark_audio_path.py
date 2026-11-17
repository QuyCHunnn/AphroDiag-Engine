"""A/B actual current MP3->AAC path versus direct WAV->AAC mux, three pairs."""
from pathlib import Path
import json
import resource
import time
from jellydub.export import export_mp3,mux_mkv
from jellydub.ffmpeg import probe
root=Path(__file__).resolve().parents[1]
base=root/'artifacts/performance-audit/run-20261116'
out=base/'audio-ab';out.mkdir(exist_ok=True)
video=next((root/'videos').glob('*[[]n0LY3XcZaCs[]].mp4'))
srt=next((base/'both').glob('*.vi.srt'))
mixed=next((base/'cache').glob('*/mixed-voiceover.wav'))
results=[]
for trial in range(3):
    # Alternate order to reduce systematic warmup bias.
    for mode in (['mp3_then_aac','direct_wav_to_aac'] if trial%2==0 else ['direct_wav_to_aac','mp3_then_aac']):
        target=out/f'{mode}.mkv'
        child=resource.getrusage(resource.RUSAGE_CHILDREN);before=child.ru_utime+child.ru_stime
        start=time.perf_counter()
        if mode=='mp3_then_aac':
            mp3=out/'intermediate.mp3';export_mp3(mixed,mp3);mux_mkv(video,mp3,srt,target)
        else:
            mux_mkv(video,mixed,srt,target)
        elapsed=time.perf_counter()-start
        child=resource.getrusage(resource.RUSAGE_CHILDREN)
        item={'mode':mode,'trial':trial+1,'wall_seconds':elapsed,'child_cpu_seconds':child.ru_utime+child.ru_stime-before,
              'bytes':target.stat().st_size,'probe':probe(target)}
        results.append(item)
        (out/'results.json').write_text(json.dumps(results,indent=2))
        print(mode,trial+1,round(elapsed,3),'seconds',flush=True)
