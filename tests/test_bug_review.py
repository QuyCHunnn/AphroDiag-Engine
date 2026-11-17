"""Regressions for the screenshot review; no cloud calls or model downloads."""
from concurrent.futures import ThreadPoolExecutor
from dataclasses import asdict, replace
import importlib.util
import json
from pathlib import Path
import subprocess
import sys
import threading
from types import SimpleNamespace

import pytest

from jellydub.cache import atomic_write_bytes, cache_key
from jellydub.config import Config
from jellydub.models import Cue
from jellydub import progressive_worker as worker
from jellydub.progressive import ProgressivePrefetch
from jellydub.translate import OpusMTTranslator, translate_cues
from jellydub.tts.base import TTSProvider
from jellydub.tts.vieneu import VieNeuTTS
from jellydub.tts.synthesize import synthesize_cues
from jellydub.ffmpeg import run_ffmpeg


def test_opus_prefetch_uses_final_strict_cache_without_loading_model(tmp_path, monkeypatch):
    calls = []
    translator = OpusMTTranslator(device='cpu')
    def translate(texts, **kwargs):
        calls.extend(texts)
        return ['Việt '+text for text in texts]
    monkeypatch.setattr(translator, 'translate_batch', translate)
    monkeypatch.setattr(worker, 'get_translator', lambda **kw: translator)
    class Provider(TTSProvider):
        def cache_id(self): return 'test'
        def synthesize(self, *a, **kw): pytest.fail('synthesis mocked below')
    monkeypatch.setattr(worker, 'get_provider', lambda *a, **kw: Provider())
    monkeypatch.setattr(worker, 'synthesize_cues', lambda *a, **kw: ([], []))
    cues = [Cue(1, 0, 1, 'First.'), Cue(2, 2, 3, 'Second.'), Cue(3, 4, 5, 'Third.')]
    config = Config(source_lang='en', cache_dir=tmp_path)
    worker.main(iter([json.dumps(asdict(config), default=str),
                      json.dumps([asdict(cues[0])]), json.dumps([asdict(c) for c in cues[1:]])]))
    assert len(calls) == 3
    final = translate_cues(cues, translator, tmp_path/'translations', strict_context_cache=True)
    assert len(calls) == 3
    assert [c.text for c in final] == ['Việt '+c.text for c in cues]
    changed = [cues[0], replace(cues[1], text='Edited.'), cues[2]]
    translate_cues(changed, translator, tmp_path/'translations', strict_context_cache=True)
    assert calls == ['First.', 'Second.', 'Third.', 'Edited.']
    assert translator._model is None


def test_worker_timeout_kills_and_falls_back(tmp_path, monkeypatch):
    from jellydub import progressive
    original = subprocess.Popen
    monkeypatch.setattr(progressive.subprocess, 'Popen', lambda argv, **kw:
                        original([sys.executable, '-c', 'import time; time.sleep(120)'], **kw))
    messages = []
    prefetch = ProgressivePrefetch(Config(cache_dir=tmp_path), messages.append)
    prefetch.finish(timeout=.05)
    assert prefetch.failed and prefetch.process.poll() is not None
    assert any('timed out' in line for line in messages)
    assert any('standard cached pipeline' in line for line in messages)
    prefetch.close()


def test_atomic_parallel_writers_use_distinct_temporary_files(tmp_path, monkeypatch):
    import jellydub.cache as cache
    original = cache.os.replace
    barrier = threading.Barrier(2)
    sources = []
    def replace_file(source, target):
        sources.append(source)
        barrier.wait(timeout=5)
        original(source, target)
    monkeypatch.setattr(cache.os, 'replace', replace_file)
    target = tmp_path/'cache.json'
    with ThreadPoolExecutor(2) as pool:
        list(pool.map(lambda data: atomic_write_bytes(target, data), [b'first'*1000, b'second'*1000]))
    assert len(set(sources)) == 2
    assert target.read_bytes() in [b'first'*1000, b'second'*1000]
    assert list(tmp_path.iterdir()) == [target]


def test_default_vieneu_threads_not_forwarded_to_sdk(monkeypatch):
    calls = []
    def sdk(mode, device):
        calls.append((mode, device))
        return SimpleNamespace()
    monkeypatch.setitem(sys.modules, 'vieneu', SimpleNamespace(Vieneu=sdk))
    VieNeuTTS()._load()
    assert calls == [('v3turbo', 'cpu')]


def test_legacy_import_does_not_write_metadata_to_old_cache(tmp_path, monkeypatch):
    provider = VieNeuTTS()
    cue = Cue(1, 0, 1, 'Hello.')
    old = tmp_path/'old'/'tts'
    old.mkdir(parents=True)
    source = old/(cache_key(provider.cache_id(), cue.text, cue.speaker)+'.wav')
    run_ffmpeg(['-f', 'lavfi', '-i', 'sine=duration=0.2', '-ar', '48000', '-ac', '1', str(source)])
    monkeypatch.setattr(provider, 'synthesize', lambda *a, **kw: pytest.fail('expected import'))
    result, failures = synthesize_cues([cue], provider, tmp_path/'tts')
    assert result and not failures
    assert not source.with_suffix('.json').exists()
    assert result[0].path.with_suffix('.json').exists()


def load_summary():
    path = Path(__file__).resolve().parents[1]/'scripts/summarize_pipeline_trace.py'
    spec = importlib.util.spec_from_file_location('trace_summary_test', path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@pytest.mark.parametrize('rows', ['not json', '{}', '',
    '{"event":"end","span":"x","phase":"a","monotonic_seconds":1,"seconds":1,"status":"passed"}',
    '{"event":"begin","span":"x","phase":"a","monotonic_seconds":1}'])
def test_trace_rejects_corrupt_or_incomplete_data(tmp_path, rows):
    trace = tmp_path/'spans.jsonl'
    trace.write_text(rows)
    with pytest.raises(ValueError):
        load_summary().summarize(trace)


def test_valid_trace_and_corrupt_cli_failure(tmp_path):
    trace = tmp_path/'spans.jsonl'
    trace.write_text('\n'.join(json.dumps(e) for e in [
        dict(event='begin', span='x', phase='asr.generate', monotonic_seconds=1),
        dict(event='end', span='x', phase='asr.generate', monotonic_seconds=2, seconds=1, status='passed')]))
    assert load_summary().summarize(trace)['phases']['asr.generate']['calls'] == 1
    trace.write_text('broken')
    result = subprocess.run([sys.executable, 'scripts/summarize_pipeline_trace.py', str(tmp_path)], capture_output=True)
    assert result.returncode == 1
    assert json.loads((tmp_path/'trace-summary.json').read_text())['status'] == 'failed'


@pytest.mark.parametrize('sampling', ['empty', 'error'])
def test_pipeline_trace_records_sampling_failure(tmp_path, monkeypatch, sampling):
    import io
    import runpy
    import shutil
    root = Path(__file__).resolve().parents[1]
    (tmp_path/'scripts').mkdir()
    script = tmp_path/'scripts/trace_pipeline_job.py'
    shutil.copy2(root/'scripts/trace_pipeline_job.py', script)
    corpus = tmp_path/'artifacts/next-optimization/corpus'
    corpus.mkdir(parents=True)
    (corpus/'manifest.json').write_text(json.dumps({'clips': [{'id': 'test', 'clip': 'dummy.mp4'}]}))
    model = tmp_path/'artifacts/igpu-setup'
    model.mkdir()
    (model/'whisper-small-model-path.txt').write_text('model')
    class Process:
        pid = 999999
        stdout = io.StringIO('done\n')
        def wait(self, **kw): return 0
        def poll(self): return 0
    monkeypatch.setattr(subprocess, 'Popen', lambda *a, **kw: Process())
    class Thread:
        def __init__(self, target, **kw): self.target = target
        def start(self):
            if sampling == 'error': self.target()
        def join(self): pass
    monkeypatch.setattr(threading, 'Thread', Thread)
    original = Path.glob
    def glob(path, pattern):
        if path == Path('/proc'):
            raise RuntimeError('sampler failed')
        return original(path, pattern)
    monkeypatch.setattr(Path, 'glob', glob)
    monkeypatch.setattr(sys, 'argv', [str(script), '--run', 'test', '--clip', 'test'])
    with pytest.raises(SystemExit) as exc:
        runpy.run_path(str(script), run_name='__main__')
    assert exc.value.code == 1
    report = json.loads((tmp_path/'artifacts/next-optimization/baselines/test/report.json').read_text())
    assert report['status'] == 'failed' and report['sampling_status'] == 'failed'
    assert report['peak_sampled_process_tree_rss_kib'] is None
    assert bool(report['sampling_errors']) == (sampling == 'error')
