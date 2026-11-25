import io
import json
import math
from pathlib import Path
import struct
from types import SimpleNamespace
import subprocess
import wave

import pytest

from jellydub.config import Config, load_config
from jellydub.models import Cue, JellyDubError
from jellydub.pipeline import tts_options
from jellydub.speech import prepare_speech
from jellydub.tts import get_provider
from jellydub.tts.base import TTSError
from jellydub.tts.voxcpm2 import VoxCPM2TTS

GPU_LOG = b'VoxCPM2Runtime: custom component backend=OpenCL\nload_tensors: offloaded 29/29 layers to GPU\nElapsed: 26.133s\n'


def wav_bytes(seconds=0.2):
    buffer = io.BytesIO()
    with wave.open(buffer, 'wb') as audio:
        audio.setnchannels(1); audio.setsampwidth(2); audio.setframerate(48000)
        audio.writeframes(b''.join(struct.pack('<h', int(5000*math.sin(i*0.1))) for i in range(int(48000*seconds))))
    return buffer.getvalue()


@pytest.fixture
def assets(tmp_path):
    binary = tmp_path/'voxcpm2-cli'
    binary.write_text('test binary'); binary.chmod(0o755)
    model = tmp_path/'models'; model.mkdir()
    for part in ('BaseLM-Q8_0', 'Acoustic-F16'):
        (model/f'VoxCPM2-{part}.gguf').write_bytes(part.encode())
    return dict(model=str(model), binary=str(binary))


def fake_process(monkeypatch, *, log=GPU_LOG, payload=None, code=0, error=None):
    calls = []
    def run(command, stdout, stderr, timeout):
        calls.append(command)
        path=Path(command[command.index('-o')+1])
        path.write_bytes(wav_bytes() if payload is None else payload)
        stdout.write(log)
        if error: raise error
        return SimpleNamespace(returncode=code)
    monkeypatch.setattr('jellydub.tts.voxcpm2.subprocess',
                        SimpleNamespace(run=run, TimeoutExpired=subprocess.TimeoutExpired, STDOUT=subprocess.STDOUT))
    return calls


def test_factory_and_reference_options(assets, tmp_path):
    cfg=Config(tts='voxcpm2', tts_model=assets['model'], voxcpm2_binary=assets['binary'],
               tts_style='natural', reference=['default.wav', 'PETER=peter.wav'], tts_device='gpu')
    cfg.validate()
    provider=get_provider(cfg.tts, **tts_options(cfg))
    assert isinstance(provider,VoxCPM2TTS)
    assert provider.device=='gpu' and provider.reference==Path('default.wav')
    assert provider.speaker_refs=={'PETER':Path('peter.wav')}
    assert provider.max_concurrency==1
    profile=tmp_path/'config.toml'
    profile.write_text('[jellydub]\ntts="voxcpm2"\nvoxcpm2_timeout=123\n')
    assert load_config(profile).voxcpm2_timeout==123


def test_gpu_command_merges_direction_and_publishes_atomic(assets,tmp_path,monkeypatch):
    calls=fake_process(monkeypatch)
    p=VoxCPM2TTS(**assets,style='Vietnamese',device='auto')
    out=tmp_path/'speech.wav';p.synthesize('(angry)Không!',out)
    command=calls[0]
    assert command[command.index('-t')+1]=='(Vietnamese; angry)Không!'
    assert '--cpu' not in command and command[command.index('--n-gpu-layers')+1]=='99'
    assert out.is_file() and not out.with_name('speech.part.wav').exists()
    assert p.execution['backend']=='OpenCL' and p.execution['gpu_layers']==29
    assert p.execution['inference_seconds']==26.133


@pytest.mark.parametrize('log',[b'',b'custom component backend=CPU\noffloaded 29/29 layers to GPU',
                                  b'custom component backend=OpenCL\noffloaded 10/29 layers to GPU',
                                  b'custom component backend=OpenCL\noffloaded 0/0 layers to GPU'])
def test_gpu_refuses_cpu_fallback(assets,tmp_path,monkeypatch,log):
    fake_process(monkeypatch,log=log)
    out=tmp_path/'out.wav';out.write_bytes(b'preserve existing output')
    with pytest.raises(TTSError,match='refusing CPU fallback'):
        VoxCPM2TTS(**assets).synthesize('Xin chào',out)
    assert out.read_bytes()==b'preserve existing output'
    assert not out.with_name('out.part.wav').exists()


def test_explicit_cpu_disables_both_gpu_paths(assets,tmp_path,monkeypatch):
    calls=fake_process(monkeypatch,log=b'custom component backend=CPU\n')
    VoxCPM2TTS(**assets,device='cpu').synthesize('Xin chào',tmp_path/'out.wav')
    assert '--cpu' in calls[0] and calls[0][calls[0].index('--n-gpu-layers')+1]=='0'


@pytest.mark.parametrize('payload',[b'broken wav',wav_bytes()[:-2]])
def test_invalid_wav_is_not_published(assets,tmp_path,monkeypatch,payload):
    fake_process(monkeypatch,payload=payload)
    out=tmp_path/'out.wav'
    with pytest.raises(TTSError):VoxCPM2TTS(**assets).synthesize('Xin chào',out)
    assert not out.exists() and not out.with_name('out.part.wav').exists()


def test_timeout_and_process_errors_cleanup(assets,tmp_path,monkeypatch):
    for kwargs in ({'code':1},{'error':subprocess.TimeoutExpired('voxcpm2',1)}):
        fake_process(monkeypatch,**kwargs)
        out=tmp_path/'out.wav'
        with pytest.raises(TTSError):VoxCPM2TTS(**assets).synthesize('Xin chào',out)
        assert not out.exists() and not out.with_name('out.part.wav').exists()


def test_cache_tracks_model_runtime_reference_and_style(assets,tmp_path):
    ref=tmp_path/'ref.wav';ref.write_bytes(b'first reference')
    options={**assets,'reference':ref}
    first=VoxCPM2TTS(**options).cache_id_for('PETER')
    assert first!=VoxCPM2TTS(**options,style='angry').cache_id_for('PETER')
    ref.write_bytes(b'other reference')
    second=VoxCPM2TTS(**options).cache_id_for('PETER');assert first!=second
    lib=tmp_path/'libggml-opencl.so';lib.write_bytes(b'first runtime')
    third=VoxCPM2TTS(**options).cache_id_for('PETER');assert second!=third
    lib.write_bytes(b'other runtime')
    assert third!=VoxCPM2TTS(**options).cache_id_for('PETER')
    model=Path(assets['model'])/'VoxCPM2-BaseLM-Q8_0.gguf';model.write_bytes(b'other model')
    assert third!=VoxCPM2TTS(**options).cache_id_for('PETER')


def test_speech_plan_emotion_reference_keeps_subtitle(assets,tmp_path):
    ref=tmp_path/'ref.wav';ref.write_bytes(wav_bytes(3))
    plan=tmp_path/'speech.json'
    plan.write_text(json.dumps([{'index':1,'text_sub':'Không!','text_tts':'Không…',
                                'emotion':'angry, forceful','ref_audio':'ref.wav'}]))
    p=VoxCPM2TTS(**assets)
    cue=Cue(1,0,3,'Không!')
    text,speaker=prepare_speech([cue],p,str(plan))[1]
    assert text=='(angry, forceful)Không…' and cue.text=='Không!'
    assert p.speaker_refs[speaker]==ref and p.cache_id_for(speaker)
    with pytest.raises(TTSError,match='No VoxCPM2 reference'):
        p.cache_id_for('unmapped speaker')


@pytest.mark.parametrize('options',[{'voice':'Charon'},{'speaker_voices':{'PETER':'Charon'}},
                                     {'voxcpm2_timeout':True},{'voxcpm2_binary':''}])
def test_invalid_voxcpm2_settings(options):
    with pytest.raises(JellyDubError):Config(tts='voxcpm2',**options).validate()


def test_missing_and_ambiguous_models(assets,tmp_path):
    with pytest.raises(TTSError):VoxCPM2TTS(**{**assets,'binary':str(tmp_path/'missing')})
    with pytest.raises(TTSError):VoxCPM2TTS(**{**assets,'model':None})
    (Path(assets['model'])/'VoxCPM2-BaseLM-F16.gguf').touch()
    with pytest.raises(TTSError,match='exactly one BaseLM'):VoxCPM2TTS(**assets)
