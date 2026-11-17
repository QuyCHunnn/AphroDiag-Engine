"""Cold-process, empty-processing-cache Family Guy benchmark; key never saved."""
import argparse
import getpass
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import time

root = Path(__file__).resolve().parents[1]
parser = argparse.ArgumentParser()
parser.add_argument('--run', required=True)
args = parser.parse_args()
work = root/'artifacts/gemini-family-guy'/args.run
if work.exists():
    raise SystemExit('Run already exists; choose a new run name')
work.mkdir(parents=True)
source = root/'videos/Family Guy - Peter falls down the stairs.mp4'
input_dir = work/'input'
input_dir.mkdir()
video = input_dir/source.name
shutil.copy2(source, video)
assert list(input_dir.iterdir()) == [video]
model = (root/'artifacts/igpu-setup/whisper-small-model-path.txt').read_text().strip()
config = work/'settings.toml'
config.write_text((root/'jellydub.gemini.toml').read_text()+'\ndownload = false\n')
command = [str(root/'.venv/bin/jellydub'), 'auto', str(video), '--config', str(config),
    '--source-lang', 'en', '--asr-backend', 'genai', '--asr-model', model,
    '--asr-python', str(root/'.venv-native-asr-next/bin/python'), '--asr-device', 'gpu',
    '--asr-beam-size', '5', '--translate-device', 'cpu', '--pipeline-mode', 'overlap',
    '--cache-dir', str(work/'cache'), '--out-dir', str(work/'output')]
env = dict(os.environ)
env['GEMINI_API_KEY'] = getpass.getpass('Gemini API key (not saved): ')
env['HF_HUB_OFFLINE'] = '1'
env['JELLYDUB_TRACE_FILE'] = str(work/'spans.jsonl')
record = {'command': command, 'input': str(source), 'mode': 'cold_process_empty_processing_cache',
    'existing_srt_used': False, 'processing_cache_initially_empty': not (work/'cache').exists(),
    'weights_preinstalled': True, 'debugging_time_included': False, 'status': 'running'}
(work/'report.json').write_text(json.dumps(record, indent=2)+'\n')
started = time.perf_counter()
process = subprocess.Popen(command, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, env=env)
env.pop('GEMINI_API_KEY', None)
try:
    with (work/'messages.jsonl').open('w') as log:
        for line in process.stdout:
            seconds = time.perf_counter()-started
            log.write(json.dumps({'seconds': seconds, 'message': line.rstrip()}, ensure_ascii=False)+'\n')
            log.flush()
            print(f'[{seconds:8.2f}s] {line}', end='', flush=True)
    code = process.wait()
finally:
    if process.poll() is None:
        process.terminate()
        process.wait()
elapsed = time.perf_counter()-started
record.update(status='passed' if code == 0 else 'failed', returncode=code, pipeline_seconds=elapsed)
(work/'report.json').write_text(json.dumps(record, indent=2)+'\n')
print(json.dumps({'status': record['status'], 'pipeline_seconds': elapsed, 'report': str(work/'report.json')}), flush=True)
# Extra full decode is deliberately outside the measured pipeline interval.
if code == 0:
    outputs = list((work/'output').glob('*.mkv'))
    qa = subprocess.run(['ffmpeg','-v','error','-nostdin','-i',str(outputs[0]),'-map','0:v?','-map','0:a?','-f','null','-'], capture_output=True,text=True)
    record.update(extra_decode_returncode=qa.returncode,extra_decode_errors=qa.stderr)
    (work/'report.json').write_text(json.dumps(record, indent=2)+'\n')
sys.exit(code)
