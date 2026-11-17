from dataclasses import replace
import os
from pathlib import Path

import pytest

from jellydub import pipeline as p
from jellydub.config import Config
from jellydub.ffmpeg import duration, run_ffmpeg
from jellydub.models import Cue, JellyDubError
from jellydub.tts.base import TTSProvider
from jellydub.tts.synthesize import synthesize_cues


def mutate_same_size_and_mtime(path):
    stat = path.stat()
    data = bytearray(path.read_bytes())
    data[-1] ^= 1
    path.write_bytes(data)
    os.utime(path, ns=(stat.st_atime_ns, stat.st_mtime_ns))


def test_output_settings_reuse_work_but_speech_changes_invalidate(tmp_path):
    video, srt = tmp_path/'video.mp4', tmp_path/'video.vi.srt'
    video.write_bytes(b'video')
    srt.write_text('1\n00:00:00,000 --> 00:00:01,000\nXin chào\n')
    c = Config(cache_dir=tmp_path/'cache')
    work = p._work_dir(video, srt, c)
    for changed in (replace(c, out_dir=tmp_path/'other'), replace(c, mp4=True),
                    replace(c, audio=True), replace(c, subtitle_font='Arial')):
        assert p._work_dir(video, srt, changed) == work
    assert p._work_dir(video, srt, replace(c, voice='different')) != work
    srt.write_text(srt.read_text().replace('Xin chào', 'Tạm biệt'))
    assert p._work_dir(video, srt, c) != work


def test_pcm_metadata_without_probe_and_reject_truncation(tmp_path, monkeypatch):
    path = tmp_path/'tone.wav'
    run_ffmpeg(['-f','lavfi','-i','sine=duration=0.2','-ar','48000','-ac','1',str(path)])
    monkeypatch.setattr('jellydub.ffmpeg.probe', lambda path: pytest.fail('PCM duration must not spawn ffprobe'))
    assert duration(path) == pytest.approx(.2)
    path.write_bytes(path.read_bytes()[:-20])
    with pytest.raises(JellyDubError, match='Truncated'):
        duration(path)


def test_stage_invalidates_on_input_and_output_corruption(tmp_path):
    source, target = tmp_path/'input', tmp_path/'output'
    source.write_bytes(b'hello')
    calls = []
    def build(path):
        calls.append(True)
        path.write_bytes(source.read_bytes())
    p._stage(target, build, inputs=(source,))
    p._stage(target, build, inputs=(source,))
    assert len(calls) == 1
    mutate_same_size_and_mtime(target)
    p._stage(target, build, inputs=(source,))
    assert target.read_bytes() == b'hello' and len(calls) == 2
    mutate_same_size_and_mtime(source)
    p._stage(target, build, inputs=(source,))
    assert target.read_bytes() == source.read_bytes() and len(calls) == 3


def test_qc_reuse_requires_same_content_and_expected_duration(tmp_path, monkeypatch):
    path = tmp_path/'mixed.wav';path.write_bytes(b'audio')
    calls = []
    monkeypatch.setattr(p, 'check_audio', lambda *a: calls.append(a) or [])
    assert p._checked_audio(path, 1, tmp_path) == []
    assert p._checked_audio(path, 1, tmp_path) == []
    assert len(calls) == 1
    mutate_same_size_and_mtime(path)
    p._checked_audio(path, 1, tmp_path)
    p._checked_audio(path, 2, tmp_path)
    assert len(calls) == 3
    monkeypatch.setattr(p, 'check_audio', lambda *a: ['silent'])
    assert p._checked_audio(path, 3, tmp_path) == ['silent']
    assert p._checked_audio(path, 3, tmp_path) == ['silent']


class CountingSpeech(TTSProvider):
    def __init__(self): self.calls = []
    def cache_id(self): return 'test-voice'
    def synthesize(self, text, out_path, speaker=None):
        self.calls.append(text)
        run_ffmpeg(['-f','lavfi','-i','sine=duration=0.15',str(out_path)])


def test_shared_speech_reuses_other_cues_and_repairs_payload_corruption(tmp_path, monkeypatch):
    monkeypatch.setattr('jellydub.tts.synthesize.time.sleep', lambda _: None)
    provider = CountingSpeech();cache = tmp_path/'tts'
    cues = [Cue(1,0,1,'one'), Cue(2,1,2,'two')]
    first, failures = synthesize_cues(cues,provider,cache)
    assert not failures and provider.calls == ['one','two']
    original = first[0].path.read_bytes()
    cues[1].text = 'changed'
    second, failures = synthesize_cues(cues,provider,cache)
    assert not failures and provider.calls == ['one','two','changed']
    assert second[0].path == first[0].path and second[0].path.read_bytes() == original
    mutate_same_size_and_mtime(second[0].path)
    repaired, failures = synthesize_cues(cues,provider,cache)
    assert not failures and provider.calls[-1] == 'one' and len(provider.calls) == 4
    assert repaired[0].path.read_bytes() == original
