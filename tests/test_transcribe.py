import sys
from types import ModuleType, SimpleNamespace

import pytest

from jellydub.models import JellyDubError
from jellydub.transcribe import transcribe_video


def test_transcribe_video_converts_segments_and_detects_language(monkeypatch, tmp_path):
    calls = {}

    class Model:
        def __init__(self, model_name, **options):
            calls['model'] = model_name
            calls['options'] = options

        def transcribe(self, video, **options):
            calls['video'] = video
            calls['transcribe_options'] = options
            segments = [
                SimpleNamespace(start=0.1, end=1.2, text=' Hello '),
                SimpleNamespace(start=1.2, end=1.2, text='invalid duration'),
                SimpleNamespace(start=2.0, end=3.0, text='   '),
                SimpleNamespace(start=2.0, end=3.1, text='World'),
            ]
            return iter(segments), SimpleNamespace(language='en')

    monkeypatch.setitem(sys.modules, 'faster_whisper', ModuleType('faster_whisper'))
    sys.modules['faster_whisper'].WhisperModel = Model
    cues, language = transcribe_video(
        tmp_path / 'movie.mp4',
        model_name='small',
        device='cpu',
        cache_dir=tmp_path / 'models',
        language='en',
    )

    assert language == 'en'
    assert [(cue.index, cue.start, cue.end, cue.text) for cue in cues] == [
        (1, 0.1, 1.2, 'Hello'),
        (2, 2.0, 3.1, 'World'),
    ]
    assert calls['options'] == {
        'device': 'cpu',
        'compute_type': 'int8',
        'download_root': str(tmp_path / 'models'),
    }
    assert calls['transcribe_options'] == {
        'language': 'en',
        'beam_size': 5,
        'vad_filter': True,
    }


def test_transcribe_video_rejects_unsupported_language(monkeypatch, tmp_path):
    class Model:
        def __init__(self, *args, **kwargs):
            pass

        def transcribe(self, *args, **kwargs):
            return iter([SimpleNamespace(start=0, end=1, text='Bonjour')]), SimpleNamespace(language='fr')

    module = ModuleType('faster_whisper')
    module.WhisperModel = Model
    monkeypatch.setitem(sys.modules, 'faster_whisper', module)
    with pytest.raises(JellyDubError, match='unsupported language'):
        transcribe_video(tmp_path / 'movie.mp4', model_name='small', device='cpu',
                         cache_dir=tmp_path / 'models')


def test_transcribe_video_reports_missing_optional_dependency(monkeypatch, tmp_path):
    monkeypatch.setitem(sys.modules, 'faster_whisper', None)
    with pytest.raises(JellyDubError, match='uv sync --extra asr'):
        transcribe_video(tmp_path / 'movie.mp4', model_name='small', device='cpu', backend='faster-whisper',
                         cache_dir=tmp_path / 'models')
