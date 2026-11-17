"""Profile real pipeline stages without changing engine source or calling online TTS.
Uses the public 211-second sample and matching cached Edge speech. Reports cold
media processing (speech already cached), warm repeat, and MKV-only comparison.
"""
from collections import Counter
from dataclasses import replace
import functools
import json
from pathlib import Path
import resource
import shutil
import subprocess
import time
import sys

from jellydub import pipeline as p
from jellydub.config import Config
from jellydub.cache import cache_key
from jellydub.subtitles import parse_srt
from jellydub.tts.edge import EdgeTTS

ROOT = Path(__file__).resolve().parents[1]
OUT = Path(sys.argv[1]).resolve() if len(sys.argv) > 1 else ROOT / 'artifacts/performance-audit'
OUT.mkdir(parents=True, exist_ok=True)
VIDEO = next((ROOT / 'videos').glob('*[[]n0LY3XcZaCs[]].mp4'))
SRT = ROOT / 'dubbed/vendor-quotations/vendor-quotations.vi.srt'
CACHE = OUT / 'cache'
rows = []
active = 'outside_stages'
commands = Counter()
original_run = subprocess.run


def measured_run(*args, **kwargs):
    argv = args[0] if args else kwargs.get('args', [])
    if isinstance(argv, (list, tuple)) and argv:
        commands[(active, Path(str(argv[0])).name)] += 1
    return original_run(*args, **kwargs)


def usage():
    own = resource.getrusage(resource.RUSAGE_SELF)
    child = resource.getrusage(resource.RUSAGE_CHILDREN)
    return own.ru_utime + own.ru_stime, child.ru_utime + child.ru_stime


def instrument(name):
    original = getattr(p, name)
    @functools.wraps(original)
    def wrapped(*args, **kwargs):
        global active
        previous = active
        active = name
        before = usage()
        start = time.perf_counter()
        try:
            return original(*args, **kwargs)
        finally:
            after = usage()
            rows.append({'stage': name, 'wall_seconds': time.perf_counter()-start,
                         'python_cpu_seconds': after[0]-before[0],
                         'child_cpu_seconds': after[1]-before[1]})
            active = previous
            if name in {'export_mp4', 'mux_mkv', 'align', 'mix', 'synthesize_cues'}:
                print(name, round(rows[-1]['wall_seconds'],3), 'seconds', flush=True)
    setattr(p, name, wrapped)


for name in ('_resolve_subtitle','_translated','synthesize_cues','align','render_timeline',
             'extract_audio','mix','check_audio','export_mp3','mux_mkv','export_mp4','_copy_output'):
    instrument(name)
subprocess.run = measured_run

def no_online_synthesis(*args, **kwargs):
    raise RuntimeError('Audit forbids online TTS; expected a matching cached cue')

EdgeTTS.synthesize = no_online_synthesis


def prepare(config):
    # Resolve using the actual pipeline, then seed ONLY exact matching speech keys.
    selected, _ = p._resolve_subtitle(VIDEO, SRT, config, lambda message: None)
    work = p._work_dir(VIDEO, selected, config)
    if any(work.glob('export.*')) or (work/'timeline.wav').exists():
        raise RuntimeError('Use a fresh audit directory for a cold-media measurement')
    (work/'tts').mkdir(parents=True, exist_ok=True)
    provider = EdgeTTS(voice=config.voice)
    cues = parse_srt(selected)
    for cue in cues:
        filename = cache_key(provider.cache_id(), cue.text, cue.speaker) + '.wav'
        candidates = list((ROOT/'.jellydub-cache').glob('Compare*/tts/'+filename))
        if not candidates:
            raise RuntimeError(f'Missing cached speech for cue {cue.index}; refusing network synthesis')
        shutil.copy2(candidates[0], work/'tts'/filename)
    return len(cues)


report = {'scope': 'Real VI-SRT pipeline; speech cached; ASR/translation/online TTS not benchmarked',
          'video': str(VIDEO), 'cpu_count': __import__('os').cpu_count(), 'runs': []}
config = Config(source_lang='vi',voice='vi-VN-NamMinhNeural',mix='voiceover',mkv=True,mp4=True,
                out_dir=OUT/'both',cache_dir=CACHE,download=False)
config.validate()
prepare(config)
only = replace(config,mp4=False,out_dir=OUT/'mkv-only')
prepare(only)
for label, settings in [('both-cold-media',config),('both-warm',config),('mkv-only-cold-media',only)]:
    rows.clear(); commands.clear()
    before = usage(); start = time.perf_counter()
    with (OUT/f'{label}.log').open('w') as log:
        outputs = p.dub(VIDEO,SRT,settings,report=lambda text: (log.write(text+'\n'),log.flush()))
    after = usage(); elapsed = time.perf_counter()-start
    result = {'label':label,'wall_seconds':elapsed,'python_cpu_seconds':after[0]-before[0],
              'child_cpu_seconds':after[1]-before[1],'stages':list(rows),
              'commands':[{'stage':k[0],'tool':k[1],'count':v} for k,v in commands.items()],
              'outputs':{k:{'bytes':v.stat().st_size,'path':str(v)} for k,v in outputs.items()}}
    report['runs'].append(result)
    (OUT/'profile.json').write_text(json.dumps(report,indent=2))
    print(label,round(elapsed,3),'seconds',flush=True)
print('Profile saved:',OUT/'profile.json',flush=True)
