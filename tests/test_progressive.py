"""Context, cache reuse and process cancellation for speculative prefetch."""
from dataclasses import asdict
import json
from pathlib import Path
import subprocess
import sys

import pytest

from jellydub.config import Config
from jellydub.models import Cue, JellyDubError
from jellydub import progressive_worker as worker
from jellydub.progressive import ProgressivePrefetch
from jellydub.translate import translate_cues
from jellydub.tts.base import TTSProvider
from jellydub.tts.synthesize import synthesize_cues
from jellydub.ffmpeg import run_ffmpeg


def test_prefetch_preserves_cross_region_translation_context_and_reuses_cache(tmp_path, monkeypatch):
    calls, speech, closed = [], [], []
    class Translator:
        def cache_id(self): return "progressive-context-test"
        def translate_batch(self, texts, **kwargs):
            calls.extend(texts)
            return [{"This is a test.": "Đây là thử nghiệm.", "Bye!": "Tạm biệt!"}[text] for text in texts]
        def close(self): closed.append("translator")
    class Tone(TTSProvider):
        def cache_id(self): return "progressive-speech-test"
        def synthesize(self, text, out_path, speaker=None):
            speech.append(text)
            run_ffmpeg(["-f", "lavfi", "-i", "sine=duration=0.2", "-ac", "1", "-ar", "48000", str(out_path)])
        def close(self): closed.append("tts")
    translator, provider = Translator(), Tone()
    monkeypatch.setattr(worker, "get_translator", lambda **kw: translator)
    monkeypatch.setattr(worker, "get_provider", lambda *a, **kw: provider)
    cues = [Cue(1, 0, 1, "This is"), Cue(2, 1.2, 2, "a test."), Cue(3, 4, 5, "Bye!")]
    config = Config(source_lang="en", cache_dir=tmp_path)
    stream = [json.dumps(asdict(config), default=str), json.dumps([asdict(cues[0])]),
              json.dumps([asdict(c) for c in cues[1:]])]
    worker.main(iter(stream))
    assert calls == ["This is a test.", "Bye!"]
    assert closed == ["translator", "tts"]
    translated = translate_cues(cues, translator, tmp_path/"translations")
    results, failures = synthesize_cues(translated, provider, tmp_path/"tts")
    assert not failures and len(results) == 3
    assert calls == ["This is a test.", "Bye!"] and len(speech) == 3
    assert [c.text for c in translated] == ["Đây là", "thử nghiệm.", "Tạm biệt!"]


@pytest.mark.parametrize("gap,speaker", [(2, None), (.2, "OTHER")])
def test_prefetch_does_not_merge_separate_dialogue(gap, speaker):
    a = Cue(1, 0, 1, "unfinished")
    b = Cue(2, 1+gap, 3+gap, "Next!", speaker)
    ready, pending = worker.closed_prefix([a, b])
    assert ready == [a, b] and not pending


def test_final_unfinished_cue_is_flushed():
    cue = Cue(1, 0, 1, "unfinished")
    assert worker.closed_prefix([cue]) == ([], [cue])
    assert worker.closed_prefix([cue], final=True) == ([cue], [])


def test_prefetch_process_protocol_and_cleanup(tmp_path, monkeypatch):
    from jellydub import progressive
    original = subprocess.Popen
    def start(argv, **kwargs):
        return original([sys.executable, "-c", "import sys; rows=list(sys.stdin); print('received',len(rows)-1)"], **kwargs)
    monkeypatch.setattr(progressive.subprocess, "Popen", start)
    messages = []
    prefetch = ProgressivePrefetch(Config(cache_dir=tmp_path), messages.append)
    folder = prefetch.directory.name
    prefetch.submit([Cue(1, 0, 1, "Hello")]); prefetch.finish(); prefetch.close()
    assert prefetch.process.returncode == 0 and messages == ["received 1"]
    assert not Path(folder).exists()


def test_prefetch_cancellation_terminates_its_worker(tmp_path, monkeypatch):
    from jellydub import progressive
    original = subprocess.Popen
    monkeypatch.setattr(progressive.subprocess, "Popen", lambda argv, **kw:
                        original([sys.executable, "-c", "import time; time.sleep(120)"], **kw))
    prefetch = ProgressivePrefetch(Config(cache_dir=tmp_path), lambda s: None)
    prefetch.close()
    assert prefetch.process.poll() is not None


@pytest.mark.parametrize("values", [{"source_lang": "auto"}, {"source_lang": "en", "translator": "gemini"}])
def test_overlap_rejects_unresolved_language_or_unsupported_context_model(values):
    with pytest.raises(JellyDubError, match="overlap requires"):
        Config(pipeline_mode="overlap", **values).validate()


def test_ffmpeg_does_not_consume_worker_protocol_stdin(tmp_path):
    code = """
import sys,json
from jellydub.ffmpeg import run_ffmpeg
print('ready',flush=True)
run_ffmpeg(['-f','lavfi','-i','sine=duration=0.4','-ar','48000','-ac','1',sys.argv[1]])
print(json.loads(sys.stdin.readline())['message'],flush=True)
"""
    process = subprocess.Popen([sys.executable, '-c', code, str(tmp_path/'tone.wav')],
                               stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
    assert process.stdout.readline().strip() == 'ready'
    stdout, stderr = process.communicate(json.dumps({'message': 'protocol intact'})+'\n', timeout=20)
    assert process.returncode == 0, stderr
    assert stdout.strip() == 'protocol intact'
