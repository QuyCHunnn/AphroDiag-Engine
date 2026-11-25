import json
import warnings
from pathlib import Path
from types import SimpleNamespace

import pytest

from jellydub.models import Cue, JellyDubError
from jellydub.speech import prepare_speech
from jellydub.subtitles import write_srt, parse_srt
from jellydub.tts.synthesize import synthesize_cues
from test_tts import Fake


def test_spoken_override_keeps_subtitles_and_separates_cache(tmp_path):
    cues = [Cue(1, 0, 1, 'Thôi được rồi.'), Cue(2, 1, 2, 'Thôi được rồi.')]
    plan = tmp_path / 'speech.json'
    plan.write_text(json.dumps([{'index': 1, 'text_sub': cues[0].text,
                                 'text_tts': 'Thôi… được rồi.'}]))
    class Recorder(Fake):
        def synthesize(self, text, out_path, speaker=None):
            texts.append(text)
            super().synthesize(text, out_path, speaker)
    texts = []
    provider = Recorder()
    results, failures = synthesize_cues(cues, provider, tmp_path / 'tts', speech_plan=str(plan))
    assert not failures and len(results) == 2
    assert set(texts) == {'Thôi… được rồi.', 'Thôi được rồi.'}
    assert results[0].path != results[1].path
    synthesize_cues(cues, provider, tmp_path / 'tts', speech_plan=str(plan))
    assert provider.calls == 2
    write_srt(cues, tmp_path / 'out.srt')
    assert [c.text for c in parse_srt(tmp_path / 'out.srt')] == [c.text for c in cues]


def test_vieneu_tag_and_reference_routing(tmp_path):
    from jellydub.ffmpeg import run_ffmpeg
    ref = tmp_path / 'sad.wav'
    run_ffmpeg(['-f', 'lavfi', '-i', 'sine=frequency=440:duration=3', str(ref)])
    plan = tmp_path / 'speech.json'
    plan.write_text(json.dumps([{'index': 1, 'text_tts': '[thở dài] Thôi được rồi.',
                                 'emotion': 'buồn', 'ref_audio': 'sad.wav'}]))
    provider = SimpleNamespace(name='vieneu', speaker_refs={}, speaker_voices={}, voice='Hải Đăng')
    text, speaker = prepare_speech([Cue(1, 0, 4, 'Thôi được rồi.')], provider, str(plan))[1]
    assert text.startswith('[thở dài]')
    assert provider.speaker_refs[speaker] == ref
    assert provider.speaker_voices[speaker] == 'Hải Đăng'


@pytest.mark.parametrize('row', [
    {'index': 1, 'text_sub': 'stale'},
    {'index': 1, 'text_tts': '[buồn] Xin chào'},
    {'index': 1, 'emotion': 'buồn'},
    {'index': 1, 'text_tts': 10},
    {'index': 1, 'ref_audio': 'missing.wav'},
])
def test_invalid_plan_fails_before_synthesis(tmp_path, row):
    plan = tmp_path / 'speech.json'
    plan.write_text(json.dumps([row]))
    with pytest.raises(JellyDubError):
        prepare_speech([Cue(1, 0, 1, 'Xin chào')], Fake(), str(plan))


@pytest.mark.parametrize('direction', [None, [], 'sad', {'unknown': 'x'}, {'emotion': 1},
                                      {'pace': ''}, {'emphasis': 'word'}, {'emphasis': [1]},
                                      {'emphasis': ['']}, {'ending': None}])
def test_invalid_direction_rejected_before_synthesis(tmp_path, direction):
    plan=tmp_path/'speech.json'
    plan.write_text(json.dumps([{'index':1,'direction':direction}]))
    with pytest.raises(JellyDubError, match='direction'):
        prepare_speech([Cue(1,0,2,'Thôi được rồi.')],Fake(),str(plan))


def test_same_plan_compiles_to_three_backends_without_subtitle_changes(tmp_path):
    from jellydub.tts.gemini import GeminiTTS
    from jellydub.ffmpeg import run_ffmpeg
    ref=tmp_path/'ref.wav'
    run_ffmpeg(['-f','lavfi','-i','sine=duration=3','-ar','16000','-ac','1',str(ref)])
    original='Thôi được rồi, anh nói đúng.'
    direction={'emotion':'buồn, miễn cưỡng','pace':'chậm','emphasis':['được rồi'],
               'ending':'hạ giọng','nonverbal':'thở dài trước câu'}
    plan=tmp_path/'speech.json'
    plan.write_text(json.dumps([{'index':1,'text_sub':original,'direction':direction,'ref_audio':'ref.wav'}]))
    cue=Cue(1,0,4,original,speaker='PETER')
    vieneu=SimpleNamespace(name='vieneu',speaker_refs={},speaker_voices={'PETER':'Quốc Tuấn'},voice='Hải Đăng')
    vox=SimpleNamespace(name='voxcpm2',speaker_refs={})
    gemini=GeminiTTS(style='Vietnamese',speaker_voices={'PETER':'Charon'})
    with pytest.warns(UserWarning):
        vi_text,vi_speaker=prepare_speech([cue],vieneu,str(plan))[1]
    assert vi_text=='[thở dài] Thôi được rồi… anh nói đúng.'
    assert vieneu.speaker_refs[vi_speaker]==ref
    assert vieneu.speaker_voices[vi_speaker]=='Quốc Tuấn'
    assert {x['field'] for x in vieneu.speech_warnings}=={'emotion','pace','emphasis','ending'}
    vox_text,vox_speaker=prepare_speech([cue],vox,str(plan))[1]
    assert vox_text.startswith('(Emotion: buồn, miễn cưỡng; Pace: chậm;')
    assert vox_text.endswith(')'+original) and vox.speaker_refs[vox_speaker]==ref
    with pytest.warns(UserWarning,match='reference audio is ignored'):
        gem_text,gem_speaker=prepare_speech([cue],gemini,str(plan))[1]
    assert gem_text==original and gemini.speaker_voices[gem_speaker]=='Charon'
    _,body=gemini._request(gem_text,gemini.speaker_voices[gem_speaker],gemini.speaker_styles[gem_speaker])
    part=body['input'][0]['content'][0]
    assert part['text']==original and 'Emphasize these exact words: được rồi' in part['annotations'][0]['style']
    assert part['annotations'][0]['style'].startswith('Vietnamese; Emotion:')
    assert cue.text==original and cue.text_tts is None and cue.direction is None
    assert (cue.start,cue.end,cue.speaker)==(0,4,'PETER')
    write_srt([cue],tmp_path/'out.srt')
    assert parse_srt(tmp_path/'out.srt')[0].text==original


def test_gemini_direction_routes_voice_and_cache_by_cue(tmp_path):
    from jellydub.tts.gemini import GeminiTTS
    p=GeminiTTS(speaker_voices={'PETER':'Charon'})
    cues=[Cue(1,0,1,'Không!',speaker='PETER',direction={'emotion':'angry'}),
          Cue(2,1,2,'Không!',speaker='PETER',direction={'emotion':'sad'}),
          Cue(3,2,3,'Không!',speaker='PETER')]
    spoken=prepare_speech(cues,p)
    assert len({p.cache_id_for(speaker) for _,speaker in spoken.values()})==3
    first=p.cache_id_for(spoken[1][1])
    prepare_speech(cues,p)
    assert p.cache_id_for(spoken[1][1])==first
    assert p.style=='' and spoken[3][1]=='PETER'
    legacy=GeminiTTS(model='gemini-2.5-flash-preview-tts')
    _,body=legacy._request('Không!','Charon',p.speaker_styles[spoken[1][1]])
    assert body['contents'][0]['parts'][0]['text']=='Emotion: angry\nRead only the following transcript:\nKhông!'


def test_direction_preserves_silent_overrides_and_merges_vox_prefix():
    p=SimpleNamespace(name='voxcpm2',speaker_refs={})
    cues=[Cue(1,0,1,'Skip me',text_tts='',direction={'emotion':'sad','nonverbal':'sigh'}),
          Cue(2,1,2,'Không!',text_tts='(tense)Không!',direction={'emotion':'sad (restrained)'})]
    spoken=prepare_speech(cues,p)
    assert spoken[1][0]==''
    assert spoken[2][0]=='(Emotion: sad （restrained）; tense)Không!'


def test_conflicting_legacy_and_new_emotion_rejected():
    with pytest.raises(JellyDubError,match='conflicting emotion'):
        prepare_speech([Cue(1,0,1,'Hi',emotion='sad',direction={'emotion':'angry'})],Fake())


def test_vieneu_unsupported_action_is_reported_once():
    p=SimpleNamespace(name='vieneu')
    cues=[Cue(1,0,1,'Hi',direction={'nonverbal':'cry after speaking'})]
    with pytest.warns(UserWarning,match='Unsupported VieNeu action'):
        assert prepare_speech(cues,p)[1][0]=='Hi'
    with warnings.catch_warnings(record=True) as notices:
        prepare_speech(cues,p)
    assert notices==[] and len(p.speech_warnings)==1


def test_vieneu_preview_records_unapplied_direction(tmp_path,monkeypatch):
    from jellydub.config import Config
    from jellydub.preview import create_preview
    from jellydub.tts.vieneu import VieNeuTTS
    from jellydub.ffmpeg import run_ffmpeg
    monkeypatch.setattr(VieNeuTTS,'synthesize',lambda self,text,path,speaker=None:
                        run_ffmpeg(['-f','lavfi','-i','sine=duration=0.2','-ar','48000','-ac','1',str(path)]))
    source=tmp_path/'source.srt'
    write_srt([Cue(1,0,1,'Xin chào.')],source)
    plan=tmp_path/'speech.json'
    plan.write_text(json.dumps([{'index':1,'direction':{'ending':'hạ giọng'}}]))
    config=Config(tts='vieneu',tts_device='cpu',speech_plan=str(plan),cache_dir=tmp_path/'cache')
    with pytest.warns(UserWarning,match='No direct VieNeu control'):
        create_preview(source,tmp_path/'preview',config,seconds=1,report=lambda message:None)
    review=json.loads((tmp_path/'preview/review.json').read_text())
    assert review['speech_warnings'][0]['field']=='ending'
    assert review['speech_directions']['1']['instructions'] is None
    assert review['cues'][0]['text_tts']=='Xin chào.'
