"""Real FFmpeg export/cache checks; ASR, translation and speech use offline fixtures."""
import hashlib
import json
import subprocess

from typer.testing import CliRunner

from jellydub import pipeline
from jellydub.cli import app
from jellydub.ffmpeg import probe, run_ffmpeg
from jellydub.models import Cue
from jellydub.tts.base import TTSProvider


def test_video_only_auto_exports_and_resumes_without_repeating_models(tmp_path, monkeypatch):
    video = tmp_path / 'English video [1].mkv'
    run_ffmpeg(['-f', 'lavfi', '-i', 'color=c=black:s=160x90:r=10:d=3',
                '-f', 'lavfi', '-i', 'sine=frequency=300:duration=3',
                '-c:v', 'ffv1', '-c:a', 'pcm_s16le', str(video)])
    original = hashlib.sha256(video.read_bytes()).digest()
    calls = []

    def transcribe(*args, **kwargs):
        assert kwargs['language'] is None
        calls.append('asr')
        return [Cue(1, .2, 1.5, 'Hello, welcome.')], 'en'

    class Translator:
        def cache_id(self):
            return 'auto-media-translator'

        def translate_batch(self, texts):
            calls.append('translation')
            return ['Xin chào, chào mừng bạn.' for text in texts]

        def close(self):
            pass

    class Speech(TTSProvider):
        def cache_id(self):
            return 'auto-media-speech'

        def synthesize(self, text, out_path, speaker=None):
            calls.append('tts')
            run_ffmpeg(['-f', 'lavfi', '-i', 'sine=frequency=800:duration=0.8',
                        '-ar', '48000', '-ac', '1', str(out_path)])

    monkeypatch.setattr(pipeline, 'transcribe_video', transcribe)
    monkeypatch.setattr(pipeline, 'get_translator', lambda **kwargs: Translator())
    monkeypatch.setattr(pipeline, 'get_provider', lambda *a, **kw: Speech())
    def forbid_search(*args, **kwargs):
        raise AssertionError('auto must not search for local subtitles')
    monkeypatch.setattr(pipeline, 'find_local_subtitles', forbid_search)
    output, cache = tmp_path / 'output', tmp_path / 'cache'
    args = ['auto', str(video), '--out-dir', str(output), '--cache-dir', str(cache)]
    runner = CliRunner()
    for _ in range(2):
        result = runner.invoke(app, args)
        assert result.exit_code == 0, result.output
    assert calls == ['asr', 'translation', 'tts']
    assert hashlib.sha256(video.read_bytes()).digest() == original
    assert {p.suffix for p in output.iterdir()} == {'.mkv', '.srt'}
    mkv = next(output.glob('*.mkv'))
    streams = probe(mkv)['streams']
    assert any(s['codec_type'] == 'audio' and s.get('tags', {}).get('language') == 'vie'
               and s['disposition']['default'] for s in streams)
    assert any(s['codec_type'] == 'subtitle' and s.get('tags', {}).get('language') == 'vie'
               for s in streams)
    report = json.loads(next((cache / 'reports').glob('*.json')).read_text())
    assert report['status'] == 'completed' and report['translation'] == 'performed'
    subprocess.run(['ffmpeg', '-v', 'error', '-i', str(mkv), '-f', 'null', '-'],
                   check=True, capture_output=True)
