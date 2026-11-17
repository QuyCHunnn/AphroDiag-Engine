"""Read-only snapshot of FFmpeg process commands, resident memory and threads."""
import json
from pathlib import Path
import subprocess
rows=[]
for entry in Path('/proc').iterdir():
    if not entry.name.isdigit():continue
    try:
        if (entry/'comm').read_text().strip()!='ffmpeg':continue
        status=dict(line.split(':',1) for line in (entry/'status').read_text().splitlines() if ':' in line)
        rows.append({'pid':int(entry.name),'command':(entry/'cmdline').read_bytes().decode().split('\0')[:-1],
                     'status':{k:status[k].strip() for k in ['PPid','VmRSS','VmHWM','Threads'] if k in status}})
    except (OSError,UnicodeError):pass
out=Path(__file__).resolve().parents[1]/'artifacts/performance-audit/run-20261116'
(out/'ffmpeg-process-snapshot.json').write_text(json.dumps(rows,indent=2))
print(json.dumps(rows,indent=2))
