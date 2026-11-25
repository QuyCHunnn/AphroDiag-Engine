import base64
import io
import json
import math
import struct
import urllib.error
import wave

import pytest

from jellydub.config import Config, load_config
from jellydub.models import JellyDubError
from jellydub.pipeline import tts_options
from jellydub.tts import get_provider
from jellydub.tts.base import TTSError
from jellydub.tts.gemini import GeminiTTS, _NoRedirect


def wav_bytes():
    output = io.BytesIO()
    with wave.open(output, 'wb') as audio:
        audio.setnchannels(1)
        audio.setsampwidth(2)
        audio.setframerate(24000)
        audio.writeframes(b''.join(struct.pack('<h', int(8000*math.sin(2*math.pi*440*i/24000))) for i in range(4800)))
    return output.getvalue()


def response(monkeypatch, payload, capture=None):
    def open_request(request, timeout):
        if capture is not None:
            capture.append(request)
        return io.BytesIO(json.dumps(payload).encode())
    monkeypatch.setattr('jellydub.tts.gemini._open', open_request)
    monkeypatch.setenv('GEMINI_API_KEY', 'test-secret')


def modern(data, mime='audio/wav'):
    return {'status': 'completed', 'steps': [{'type': 'model_output', 'content': [
        {'type': 'audio', 'data': base64.b64encode(data).decode(), 'mime_type': mime}]}]}


def test_modern_request_and_audio(monkeypatch, tmp_path):
    requests = []
    response(monkeypatch, modern(wav_bytes()), requests)
    provider = GeminiTTS(style='Natural Vietnamese', speaker_voices={'LOIS': 'Kore'})
    target = tmp_path/'sample.wav'
    provider.synthesize('Xin chào', target, speaker='LOIS')
    assert target.read_bytes() == wav_bytes()
    request = requests[0]
    body = json.loads(request.data)
    assert body['input'][0]['content'][0] == {'type': 'text', 'text': 'Xin chào', 'annotations': [{'type': 'speech_metadata', 'style': 'Natural Vietnamese'}]}
    assert body['generation_config']['speech_config'] == [{'voice': 'Kore'}]
    assert body['store'] is False
    assert 'test-secret' not in request.full_url + request.data.decode() + provider.cache_id()
    assert request.get_header('X-goog-api-key') == 'test-secret'


def test_legacy_pcm(monkeypatch, tmp_path):
    pcm = b'\x10\x00'*2400
    response(monkeypatch, {'candidates': [{'finishReason': 'STOP', 'content': {'parts': [
        {'inlineData': {'data': base64.b64encode(pcm).decode(), 'mimeType': 'audio/L16;rate=24000'}}]}}]})
    target = tmp_path/'legacy.wav'
    GeminiTTS(model='gemini-2.5-flash-preview-tts').synthesize('Xin chào', target)
    with wave.open(str(target)) as audio:
        assert audio.getframerate() == 24000
        assert audio.readframes(2400) == pcm


@pytest.mark.parametrize('payload', [{}, modern(b'bad'), modern(b'odd', 'audio/L16'), modern(b'ab', 'audio/mp3'), modern(wav_bytes()[:-2])])
def test_invalid_response_preserves_target(monkeypatch, tmp_path, payload):
    response(monkeypatch, payload)
    target = tmp_path/'existing.wav'
    target.write_bytes(b'existing')
    with pytest.raises(TTSError, match='invalid audio'):
        GeminiTTS().synthesize('Xin chào', target)
    assert target.read_bytes() == b'existing'
    assert list(tmp_path.iterdir()) == [target]


@pytest.mark.parametrize('status,transient', [(400, False), (401, False), (403, False), (429, True), (503, True)])
def test_http_errors_are_sanitized(monkeypatch, tmp_path, status, transient):
    monkeypatch.setenv('GEMINI_API_KEY', 'test-secret')
    def fail(*args, **kwargs):
        raise urllib.error.HTTPError('https://example.invalid/test-secret', status, 'test-secret', {'Retry-After': '3'}, io.BytesIO(b'test-secret'))
    monkeypatch.setattr('jellydub.tts.gemini._open', fail)
    with pytest.raises(TTSError) as caught:
        GeminiTTS().synthesize('private transcript', tmp_path/'sample.wav')
    assert caught.value.transient is transient
    assert 'test-secret' not in str(caught.value)
    assert 'private transcript' not in str(caught.value)
    if transient:
        assert caught.value.retry_after_seconds == 3


def test_missing_key(monkeypatch, tmp_path):
    monkeypatch.delenv('GEMINI_API_KEY', raising=False)
    monkeypatch.delenv('GOOGLE_API_KEY', raising=False)
    with pytest.raises(TTSError, match='Missing Gemini API key'):
        GeminiTTS().synthesize('Xin chào', tmp_path/'sample.wav')


def test_cache_is_scoped_by_speaker():
    old = GeminiTTS(speaker_voices={'PETER': 'Charon', 'LOIS': 'Kore'})
    new = GeminiTTS(speaker_voices={'PETER': 'Charon', 'LOIS': 'Aoede'})
    assert old.cache_id_for('PETER') == new.cache_id_for('PETER')
    assert old.cache_id_for('LOIS') != new.cache_id_for('LOIS')
    assert old.cache_id_for(None) != GeminiTTS(style='calm').cache_id_for(None)


def test_config_factory_and_profile():
    settings = load_config(__import__('pathlib').Path('jellydub.gemini.toml'))
    options = tts_options(settings)
    assert isinstance(get_provider(settings.tts, **options), GeminiTTS)
    assert 'device' not in options and 'cache_dir' not in options
    with pytest.raises(JellyDubError, match='reference cloning'):
        Config(tts='gemini', reference=['voice.wav']).validate()


def test_no_redirect():
    assert _NoRedirect().redirect_request(None, None, 302, '', {}, 'https://elsewhere.invalid') is None


def test_common_pipeline_normalizes_and_reuses_cache(monkeypatch, tmp_path):
    from jellydub.models import Cue
    from jellydub.tts.synthesize import synthesize_cues
    requests = []
    response(monkeypatch, modern(wav_bytes()), requests)
    monkeypatch.setattr('jellydub.tts.synthesize.time.sleep', lambda _: None)
    provider = GeminiTTS()
    cues = [Cue(1, 0, 1, 'Xin chào')]
    results, failures = synthesize_cues(cues, provider, tmp_path)
    assert not failures and len(results) == 1
    with wave.open(str(results[0].path)) as audio:
        assert audio.getframerate() == 48000
        assert audio.getnchannels() == 1
    monkeypatch.delenv('GEMINI_API_KEY')
    cached, failures = synthesize_cues(cues, provider, tmp_path)
    assert cached == results and not failures
    assert len(requests) == 1


def test_retry_after_is_respected(monkeypatch, tmp_path):
    from jellydub.models import Cue
    from jellydub.tts.synthesize import synthesize_cues
    delays = []
    monkeypatch.setattr('jellydub.tts.synthesize.time.sleep', delays.append)
    provider = GeminiTTS()
    calls = []
    def synthesize(text, path, speaker=None):
        calls.append(text)
        if len(calls) == 1:
            error = TTSError('rate limited', transient=True)
            error.retry_after_seconds = 7
            raise error
        path.write_bytes(wav_bytes())
    monkeypatch.setattr(provider, 'synthesize', synthesize)
    results, failures = synthesize_cues([Cue(1, 0, 1, 'Xin chào')], provider, tmp_path)
    assert results and not failures and len(calls) == 2
    assert any(7 <= delay <= 8 for delay in delays)


def test_direction_preview_uses_separate_styles_and_reuses_cache(tmp_path, monkeypatch):
    from jellydub.models import Cue
    from jellydub.preview import create_preview
    from jellydub.subtitles import write_srt
    requests=[]
    response(monkeypatch,modern(wav_bytes()),requests)
    source=tmp_path/'source.srt'
    write_srt([Cue(1,0,1,'Không!'),Cue(2,1,2,'Không!')],source)
    plan=tmp_path/'speech.json'
    plan.write_text(json.dumps([{'index':1,'direction':{'emotion':'angry'}},
                                {'index':2,'direction':{'emotion':'sad'}}]))
    config=Config(tts='gemini',speech_plan=str(plan),cache_dir=tmp_path/'cache')
    output=create_preview(source,tmp_path/'preview',config,seconds=2,report=lambda message:None)
    assert output.is_file() and len(requests)==2
    parts=[json.loads(r.data)['input'][0]['content'][0] for r in requests]
    assert {p['text'] for p in parts}=={'Không!'}
    assert {p['annotations'][0]['style'] for p in parts}=={'Emotion: angry','Emotion: sad'}
    review=json.loads((tmp_path/'preview/review.json').read_text())
    assert review['speech_directions']['1']['requested']=={'emotion':'angry'}
    assert review['speech_warnings']==[]
    create_preview(source,tmp_path/'again',config,seconds=2,report=lambda message:None)
    assert len(requests)==2
