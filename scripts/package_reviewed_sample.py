"""Wrap the completed sample captions without changing words/timing or narration.
Run after complete_handover.py; packaging is measured separately from engine cache.
"""
from pathlib import Path
import json
import hashlib
import sys
import subprocess
import textwrap
import time
import srt
from jellydub.export import mux_mkv, export_mp4

root = Path(__file__).resolve().parents[1]
folder = root / 'delivery/vendor-quotations'
video = next((root / 'videos').glob('*[[]n0LY3XcZaCs[]].mp4'))
source = folder / 'inputs/sample.vi.srt'
items = list(srt.parse(source.read_text()))
for cue in items:
    cue.content = '\n'.join(textwrap.wrap(' '.join(cue.content.split()), width=42, break_long_words=False, break_on_hyphens=False))
caption = folder / f'{video.stem}.vi.srt'
caption.write_text(srt.compose(items), encoding='utf-8')
audio = folder / f'{video.stem}.AI-Dub.vi.mp3'
previous = json.loads((folder/'evidence/packaging.json').read_text()) if '--verify-only' in sys.argv else None
started = time.perf_counter()
if '--verify-only' not in sys.argv:
    mux_mkv(video, audio, caption, folder / f'{video.stem}.AI-Dub.vi.mkv')
    export_mp4(video, audio, caption, folder / f'{video.stem}.AI-Dub.vi.mp4')
    packaging_seconds = round(time.perf_counter()-started, 3)
else:
    packaging_seconds = json.loads((folder/'evidence/packaging.json').read_text())['packaging_seconds']
report = {'packaging_seconds': packaging_seconds,
          'caption_words_and_timing_unchanged': True,
          'cue_count': len(items), 'max_line_characters': max(len(line) for c in items for line in c.content.splitlines()),
          'max_lines_per_cue': max(len(c.content.splitlines()) for c in items), 'outputs': []}
for extension in ('mkv', 'mp4'):
    path = folder / f'{video.stem}.AI-Dub.vi.{extension}'
    old = next((x for x in previous['outputs'] if x['path'] == path.name), None) if previous else None
    if old and old['sha256'] == hashlib.sha256(path.read_bytes()).hexdigest():
        result = subprocess.CompletedProcess([], old['decode_returncode'], stderr=old['decode_errors'])
    else:
        result = subprocess.run(['ffmpeg','-v','error','-i',str(path),'-map','0:v?','-map','0:a?','-f','null','-'],capture_output=True,text=True)
    report['outputs'].append({'path': path.name, 'bytes':path.stat().st_size,
                             'sha256': hashlib.sha256(path.read_bytes()).hexdigest(),
                             'probe': json.loads(subprocess.check_output(['ffprobe','-v','error','-show_streams','-show_format','-of','json',str(path)])),
                             'decode_returncode':result.returncode,'decode_errors':result.stderr})
mkv = next(x for x in report['outputs'] if x['path'].endswith('.mkv'))
mp4 = next(x for x in report['outputs'] if x['path'].endswith('.mp4'))
original = json.loads(subprocess.check_output(['ffprobe','-v','error','-show_streams','-of','json',str(video)]))['streams']
ms = mkv['probe']['streams']
ps = mp4['probe']['streams']
count = lambda streams, kind: sum(x['codec_type'] == kind for x in streams)
extracted = subprocess.check_output(['ffmpeg','-v','error','-i',str(folder / mkv['path']),'-map','0:s:0','-f','srt','-']).decode('utf-8')
embedded = list(srt.parse(extracted))
# Matroska may shift every stream to avoid negative encoder timestamps.
# Compare relative video timing; allow a bounded uniform AAC mux offset (25 ms).
video_start = lambda streams: float(next(x for x in streams if x['codec_type']=='video').get('start_time', 0))
shift = video_start(ms) - video_start(original)
report['mkv_container_shift_seconds'] = shift
subtitle_offset = (embedded[0].start - items[0].start).total_seconds() - shift
report['mkv_subtitle_offset_relative_to_video_seconds'] = subtitle_offset
report['subtitle_mux_tolerance_seconds'] = .025
report['decode_evidence_reused_by_sha256'] = previous is not None
report['checks'] = {
    'mkv_audio_tracks': count(ms, 'audio') == count(original, 'audio') + 1,
    'mkv_subtitle_tracks': count(ms, 'subtitle') == count(original, 'subtitle') + 1,
    'mkv_vietnamese_audio': any(x['codec_type']=='audio' and x.get('tags',{}).get('language')=='vie' for x in ms),
    'mkv_vietnamese_subtitle': any(x['codec_type']=='subtitle' and x.get('tags',{}).get('language')=='vie' for x in ms),
    'embedded_subtitle_matches': len(embedded) == len(items) and abs(subtitle_offset) <= .025 and all(
        x.content == y.content and abs((x.start-y.start).total_seconds()-shift-subtitle_offset) <= .001
        and abs((x.end-y.end).total_seconds()-shift-subtitle_offset) <= .001 for x,y in zip(embedded,items)),
    'mp4_single_vietnamese_audio': count(ps,'audio') == 1 and all(x.get('tags',{}).get('language')=='vie' and x['codec_name']=='aac' for x in ps if x['codec_type']=='audio'),
    'mp4_h264': any(x['codec_type']=='video' and x['codec_name']=='h264' for x in ps),
    'mp4_no_selectable_subtitles': count(ps,'subtitle') == 0,
}
# Representative cue timestamps, plus the longest subtitle block.
for label, position in [('start',8),('middle',110),('end',200),('long-caption',201)]:
    path = folder / 'evidence' / f'{label}.jpg'
    subprocess.run(['ffmpeg','-v','error','-y','-ss',str(position),'-i',str(folder / f'{video.stem}.AI-Dub.vi.mp4'),'-frames:v','1',str(path)],check=True)
(folder/'evidence/packaging.json').write_text(json.dumps(report,ensure_ascii=False,indent=2))
print(json.dumps({k:v for k,v in report.items() if k != 'outputs'},ensure_ascii=False,indent=2))
if not all(report['checks'].values()) or any(x['decode_returncode'] for x in report['outputs']):
    raise SystemExit(1)
