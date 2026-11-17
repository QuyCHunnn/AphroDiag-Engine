from pathlib import Path
import pytest
from jellydub import asr_openvino as ov
from jellydub import transcribe as t
from jellydub.models import Cue,JellyDubError


@pytest.fixture
def devices(monkeypatch):
    monkeypatch.setattr(t, '_cuda_available', lambda: False)
    monkeypatch.setattr(ov, 'gpu_available', lambda: True)
    monkeypatch.setattr(ov, 'load_audio', lambda path: [0.0])
    calls=[]
    def transcribe(audio,**kwargs):
        calls.append((kwargs['device'],kwargs['beam_size']))
        return [Cue(1,0,1,'Hello')], 'en'
    monkeypatch.setattr(ov,'transcribe',transcribe)
    return calls


def run(**kwargs):
    return t.transcribe_video(Path('video.mp4'),model_name='small',cache_dir=Path('cache'),
                              device=kwargs.pop('device','auto'),report=kwargs.pop('report',lambda text:None),**kwargs)


def test_auto_prefers_intel_gpu_without_changing_beam(devices):
    details={}
    cues,language=run(details=details)
    assert devices==[('GPU',5)] and language=='en' and cues
    assert details=={'backend':'openvino','device':'gpu','beam_size':5,'fallback':False}


def test_gpu_failure_retries_cpu_same_model_and_beam(devices,monkeypatch):
    calls=[];messages=[];details={}
    def fail_gpu(audio,**kwargs):
        calls.append((kwargs['device'],kwargs['model_name'],kwargs['beam_size']))
        if kwargs['device']=='GPU':raise RuntimeError('driver failure')
        return [Cue(1,0,1,'Hello')],'en'
    monkeypatch.setattr(ov,'transcribe',fail_gpu)
    assert run(details=details,report=messages.append)[1]=='en'
    assert calls==[('GPU','small',5),('CPU','small',5)]
    assert details['fallback'] and details['device']=='cpu'
    assert 'driver failure' in details['fallback_reason']
    assert any('same beam=5' in message for message in messages)


def test_explicit_gpu_reports_failure_without_silent_device_switch(devices,monkeypatch):
    calls=[]
    def fail(audio,**kwargs):
        calls.append(kwargs['device']);raise RuntimeError('unsupported')
    monkeypatch.setattr(ov,'transcribe',fail)
    with pytest.raises(JellyDubError,match='failed'):
        run(device='gpu')
    assert calls==['GPU']


def test_nvidia_preferred_over_intel_when_cuda_available(devices,monkeypatch):
    monkeypatch.setattr(t,'_cuda_available',lambda:True)
    calls=[]
    monkeypatch.setattr(t,'_faster_whisper',lambda v,**kw: calls.append(kw) or ([Cue(1,0,1,'Hello')],'en'))
    run()
    assert calls[0]['device']=='cuda' and calls[0]['beam_size']==5
    assert not devices


def test_force_cpu_does_not_try_gpu(devices,monkeypatch):
    calls=[]
    monkeypatch.setattr(t,'_faster_whisper',lambda v,**kw: calls.append(kw['device']) or ([Cue(1,0,1,'Hello')],'en'))
    run(device='cpu')
    assert calls==['cpu'] and not devices


def test_unsupported_language_is_not_a_gpu_failure(devices,monkeypatch):
    calls=[]
    monkeypatch.setattr(ov,'transcribe',lambda a,**kw: calls.append(kw['device']) or ([], 'fr'))
    with pytest.raises(JellyDubError,match='unsupported language'):
        run()
    assert calls==['GPU']


@pytest.mark.parametrize('kwargs',[{'backend':'bad'},{'backend':'openvino','device':'cuda'},
                                  {'backend':'faster-whisper','device':'gpu'},{'beam_size':0}])
def test_invalid_device_backend_combination(devices,kwargs):
    with pytest.raises(JellyDubError):run(**kwargs)


def test_short_speech_context_preserves_speech_and_avoids_overlap():
    speech = [{'start': 0, 'end': 9500}, {'start': 60000, 'end': 100000},
              {'start': 120000, 'end': 126000}]
    windows = ov._context_regions(speech, 128000)
    for original, window in zip(speech, windows):
        assert 0 <= window['start'] <= original['start']
        assert original['end'] <= window['end'] <= 128000
    assert all(a['end'] <= b['start'] for a, b in zip(windows, windows[1:]))
    assert windows[0]['end'] > speech[0]['end']
    assert windows[1] == speech[1]
    assert ov._context_regions([{'start': 0, 'end': 1000}], 1000) == [{'start': 0, 'end': 1000}]
