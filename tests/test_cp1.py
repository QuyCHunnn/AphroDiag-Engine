"""CP1 regressions: scoped native tuning, content caches and legacy integrity."""
from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace
import json,sys
import pytest
from jellydub import pipeline
from jellydub.cache import cache_key
from jellydub.config import Config
from jellydub.ffmpeg import run_ffmpeg
from jellydub.models import Cue,JellyDubError
from jellydub.subtitles import write_srt,parse_srt
from jellydub.tts.base import TTSProvider,TTSError
from jellydub.tts.edge import EdgeTTS
from jellydub.tts.vieneu import VieNeuTTS
from jellydub.tts.synthesize import synthesize_cues,_speech_metadata
from jellydub.ffmpeg import duration


def tone(path):
    run_ffmpeg(['-f','lavfi','-i','sine=duration=0.15','-ar','48000','-ac','1',str(path)])


@pytest.mark.parametrize('provider_class',[EdgeTTS,VieNeuTTS])
def test_only_changed_speaker_regenerates(provider_class,tmp_path,monkeypatch):
    calls=[]
    def synth(self,text,path,speaker=None):
        calls.append((text,speaker));tone(path)
    monkeypatch.setattr(provider_class,'synthesize',synth)
    monkeypatch.setattr('jellydub.tts.synthesize.time.sleep',lambda _:None)
    voices={'PETER':'male','LOIS':'female'}
    first=provider_class(voice='male',speaker_voices=voices)
    cues=[Cue(1,0,1,'Hello.','PETER'),Cue(2,2,3,'Bye.','LOIS')]
    a,failed=synthesize_cues(cues,first,tmp_path/'tts');assert not failed
    changed=provider_class(voice='male',speaker_voices={**voices,'LOIS':'other female'})
    b,failed=synthesize_cues(cues,changed,tmp_path/'tts');assert not failed
    assert calls==[('Hello.','PETER'),('Bye.','LOIS'),('Bye.','LOIS')]
    assert a[0].path==b[0].path and a[1].path!=b[1].path


@pytest.mark.parametrize('corrupt',[False,True])
def test_exact_legacy_voice_cache_import_checks_hash(tmp_path,monkeypatch,corrupt):
    provider=VieNeuTTS(speaker_voices={'PETER':'male','LOIS':'female'})
    cue=Cue(1,0,1,'Hello.','PETER');cache=tmp_path/'tts';cache.mkdir()
    old=cache/(cache_key(provider.cache_id(),cue.text,cue.speaker)+'.wav');tone(old)
    _speech_metadata(old,duration(old))
    if corrupt:
        data=bytearray(old.read_bytes());data[-1]^=1;old.write_bytes(data)
    calls=[]
    monkeypatch.setattr(provider,'synthesize',lambda text,path,speaker=None:(calls.append(text),tone(path)))
    result,failed=synthesize_cues([cue],provider,cache)
    assert not failed and len(result)==1
    assert calls==(['Hello.'] if corrupt else [])
    assert result[0].path!=old and provider._model is None


class Speech(TTSProvider):
    def cache_id(self):return 'test-local'
    def synthesize(self,text,path,speaker=None):tone(path)


def test_jitter_only_remote_provider(tmp_path,monkeypatch):
    sleeps=[];monkeypatch.setattr('jellydub.tts.synthesize.time.sleep',sleeps.append)
    local=Speech();synthesize_cues([Cue(1,0,1,'Local')],local,tmp_path/'local')
    assert sleeps==[]
    remote=Speech();remote.is_remote=True
    synthesize_cues([Cue(1,0,1,'Remote')],remote,tmp_path/'remote')
    assert len(sleeps)==1 and .1<=sleeps[0]<=.3


@pytest.mark.parametrize('fail_at',[None,'infer','save'])
def test_vieneu_restores_thread_limits_on_all_exits(tmp_path,monkeypatch,fail_at):
    state={'threads':8};seen=[]
    class Limit:
        def __init__(self,limits,user_api):
            assert user_api=='blas';self.previous=state['threads'];state['threads']=limits
        def __enter__(self):return self
        def __exit__(self,*exc):state['threads']=self.previous
    class Model:
        def __init__(self,**kw):seen.append(kw)
        def infer(self,text,voice):
            assert state['threads']==2
            if fail_at=='infer':raise RuntimeError('infer failure')
            return 'audio'
        def save(self,audio,path):
            assert state['threads']==2
            if fail_at=='save':raise RuntimeError('save failure')
            Path(path).write_bytes(b'audio')
        def close(self):pass
    monkeypatch.setitem(sys.modules,'threadpoolctl',SimpleNamespace(threadpool_limits=Limit))
    monkeypatch.setitem(sys.modules,'vieneu',SimpleNamespace(Vieneu=Model))
    provider=VieNeuTTS(cpu_threads=3,blas_threads=2)
    if fail_at:
        with pytest.raises(TTSError):provider.synthesize('Hello',tmp_path/'out.wav')
    else:provider.synthesize('Hello',tmp_path/'out.wav')
    assert state['threads']==8 and seen[0]['threads']==3
    assert provider.cache_id_for(None)==VieNeuTTS(cpu_threads=1,blas_threads=1).cache_id_for(None)
    provider.close()


def test_cache_directory_alias_and_tuning_share_job(tmp_path,monkeypatch):
    monkeypatch.chdir(tmp_path);video=tmp_path/'video.mp4';video.write_bytes(b'video')
    srt=tmp_path/'video.vi.srt';write_srt([Cue(1,0,1,'Xin chào.')],srt)
    config=Config(cache_dir=Path('cache'));config.validate()
    a=pipeline._work_dir(video,srt,config)
    b=pipeline._work_dir(video,srt,replace(config,cache_dir=(tmp_path/'cache')))
    c=pipeline._work_dir(video,srt,replace(config,tts_blas_threads=1,tts_cpu_threads=3))
    assert a.resolve()==b.resolve()==c.resolve()


class Translator:
    def __init__(self,calls,identity='test'):self.calls,self.identity=calls,identity
    def cache_id(self):return self.identity
    def translate_batch(self,texts):self.calls.append(texts);return ['Vi '+s for s in texts]
    def close(self):pass


def test_edit_one_cue_reuses_other_job_translations(tmp_path,monkeypatch):
    calls=[];monkeypatch.setattr(pipeline,'get_translator',lambda **kw:Translator(calls,kw['model'] or 'test'))
    video=tmp_path/'video.mp4';video.write_bytes(b'video');srt=tmp_path/'video.en.srt'
    cues=[Cue(1,0,1,'One.'),Cue(2,2,3,'Two.'),Cue(3,4,5,'Three.')];write_srt(cues,srt)
    config=Config(source_lang='en',cache_dir=tmp_path/'cache')
    a=pipeline._work_dir(video,srt,config);pipeline._translated(srt,a,config,lambda _:None)
    cues[1].text='Changed.';write_srt(cues,srt)
    b=pipeline._work_dir(video,srt,config);out=pipeline._translated(srt,b,config,lambda _:None)
    assert a!=b and calls==[['One.','Two.','Three.'],['Changed.']]
    assert [c.text for c in parse_srt(out)]==['Vi One.','Vi Changed.','Vi Three.']
    different=replace(config,translator_model='other');pipeline._translated(srt,pipeline._work_dir(video,srt,different),different,lambda _:None)
    assert calls[-1]==['One.','Changed.','Three.']


def test_translation_imports_exact_legacy_context_key(tmp_path):
    from jellydub.translate import translate_cues
    calls=[];cues=[Cue(1,0,1,'Hello.')]
    old=tmp_path/'cache/old-job/translations';new=tmp_path/'cache/translations'
    first=translate_cues(cues,Translator(calls),old)
    second=translate_cues(cues,Translator(calls),new)
    assert first==second and calls==[['Hello.']] and list(new.glob('*.json'))


@pytest.mark.parametrize('field',['tts_blas_threads','tts_cpu_threads'])
@pytest.mark.parametrize('value',[-1,True,'2',1.5])
def test_thread_config_validation(field,value):
    with pytest.raises(JellyDubError,match=field):Config(**{field:value}).validate()


def test_contextual_translation_cache_tracks_future_lines(tmp_path):
    from jellydub.translate import translate_cues
    calls=[]
    class Contextual:
        batch_size=2
        def cache_id(self):return 'contextual-test'
        def translate_cues_batch(self,cues):
            calls.append([c.text for c in cues])
            return [c.text+' referring to '+cues[-1].text for c in cues]
    cues=[Cue(1,0,1,'A.'),Cue(2,2,3,'B.')]
    original=translate_cues(cues,Contextual(),tmp_path/'translations',strict_context_cache=True)
    assert translate_cues(cues,Contextual(),tmp_path/'translations',strict_context_cache=True)==original
    changed=translate_cues([cues[0],replace(cues[1],text='C.')],Contextual(),tmp_path/'translations',strict_context_cache=True)
    assert changed[0].text=='A. referring to C.'
    assert calls==[['A.','B.'],['A.','C.']]
