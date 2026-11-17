"""Regression coverage for listening previews, stage devices and ASR resume."""
import json
from pathlib import Path

import numpy as np
import pytest

from jellydub import devices, preview, transcribe
from jellydub.asr_cache import ASRTranscriptError, RegionCache, model_signature
from jellydub.config import Config
from jellydub.ffmpeg import run_ffmpeg
from jellydub.models import Cue, JellyDubError
from jellydub.subtitles import write_srt
from jellydub.tts.base import TTSProvider


def test_device_selection_is_backend_specific(monkeypatch):
    monkeypatch.setattr(devices, "torch_cuda_available", lambda: False)
    assert devices.resolve_device("TTS", "auto")[0] == "cpu"
    with pytest.raises(JellyDubError, match="no Intel"):
        devices.resolve_device("TTS", "gpu")
    with pytest.raises(JellyDubError, match="no CUDA"):
        devices.resolve_device("TTS", "cuda")
    monkeypatch.setattr(devices, "torch_cuda_available", lambda: True)
    assert devices.resolve_device("TTS", "auto") == ("cuda", None)
    assert devices.resolve_device("TTS", "cpu") == ("cpu", None)


def test_int8_voice_cache_never_imports_fp32(tmp_path, monkeypatch):
    from jellydub.tts.vieneu import VieNeuTTS
    from jellydub.tts.synthesize import synthesize_cues
    calls = []
    def synthesize(self, text, path, speaker=None):
        calls.append(self.precision)
        run_ffmpeg(["-f", "lavfi", "-i", "sine=duration=0.2", "-ac", "1", "-ar", "48000", str(path)])
    monkeypatch.setattr(VieNeuTTS, "synthesize", synthesize)
    cue = [Cue(1, 0, 1, "Xin chào", "PETER")]
    fp32 = VieNeuTTS(device="cpu")
    int8 = VieNeuTTS(device="cpu", precision="int8")
    a, failures = synthesize_cues(cue, fp32, tmp_path)
    assert not failures
    b, failures = synthesize_cues(cue, int8, tmp_path)
    assert not failures and a[0].path != b[0].path
    synthesize_cues(cue, int8, tmp_path)
    assert calls == ["fp32", "int8"]


def test_auto_cli_keeps_stage_devices_independent(monkeypatch):
    from typer.testing import CliRunner
    from jellydub.cli import app
    seen = []
    monkeypatch.setattr("jellydub.pipeline.auto", lambda v, s, c, **kw: seen.append(c) or {})
    result = CliRunner().invoke(app, ["auto", "movie.mp4", "--asr-device", "gpu",
                                    "--tts-device", "cpu", "--translate-device", "cuda"])
    assert result.exit_code == 0, result.output
    assert (seen[0].asr_device, seen[0].tts_device, seen[0].translate_device) == ("gpu", "cpu", "cuda")


def test_genai_rejects_unknown_source_language_before_audio_loading(tmp_path, monkeypatch):
    from jellydub import asr_openvino
    monkeypatch.setattr(asr_openvino, "load_audio", lambda p: pytest.fail("must not load audio"))
    with pytest.raises(JellyDubError, match="explicit source language"):
        transcribe.transcribe_video(Path("video"), model_name="small", device="gpu", backend="genai",
                                   cache_dir=tmp_path)


def test_native_worker_reuses_model_and_checkpoints(tmp_path, monkeypatch):
    import sys
    from types import SimpleNamespace
    from jellydub.asr_native_worker import main
    audio = np.concatenate([np.ones(32000, np.float32), np.full(32000, 2, np.float32)])
    np.save(tmp_path/"audio.npy", audio)
    loads, calls = [], []
    class NativeModel:
        def __init__(self, model, device, **properties):
            loads.append(device)
        def generate(self, audio, **settings):
            calls.append(settings)
            return SimpleNamespace(texts=["Hello"], chunks=[SimpleNamespace(text="Hello", start_ts=0., end_ts=1.)])
    monkeypatch.setitem(sys.modules, "openvino_genai", SimpleNamespace(__version__="test", WhisperPipeline=NativeModel))
    monkeypatch.setitem(sys.modules, "openvino", SimpleNamespace(__version__="test"))
    request = {"audio": str(tmp_path/"audio.npy"), "model": str(tmp_path/"model"),
               "cache": str(tmp_path/"cache"), "device": "GPU", "language": "en", "beam": 5,
               "regions": [{"start": 0, "end": 32000}, {"start": 32000, "end": 64000}]}
    first = main(request); second = main(request)
    assert loads == ["GPU"] and len(calls) == 2
    assert first["cues"] == second["cues"] and second["details"]["region_cache_hits"] == 2
    assert all(c["num_beams"] == 5 and c["no_repeat_ngram_size"] == 4 and c["max_new_tokens"] == 48 for c in calls)


@pytest.mark.parametrize("failure", [False, True])
def test_native_worker_process_protocol_and_cleanup(tmp_path, monkeypatch, failure):
    import sys
    from types import SimpleNamespace
    from jellydub import asr_native, asr_openvino
    monkeypatch.setitem(sys.modules, "faster_whisper.vad", SimpleNamespace(
        VadOptions=lambda: None, get_speech_timestamps=lambda *a: [{"start": 0, "end": 32000}]))
    monkeypatch.setattr(asr_openvino, "_model_path", lambda *a: tmp_path)
    interpreter = tmp_path/"isolated-python"
    payload = ({"error": "invalid timestamps", "error_type": "ASRTranscriptError"} if failure else
               {"cues": [{"index": 1, "start": 0., "end": 1., "text": "Hello", "speaker": None}],
                "language": "en", "details": {"compile_device": "GPU"}})
    interpreter.write_text(f"#!{sys.executable}\nimport json,sys,pathlib\nr=json.load(sys.stdin)\n"
                           f"pathlib.Path(r['result']).write_text({json.dumps(json.dumps(payload))})\n"
                           f"sys.exit({1 if failure else 0})\n")
    interpreter.chmod(0o755)
    options = dict(model_name="small", device="GPU", cache_dir=tmp_path/"cache", language="en",
                   beam_size=5, report=lambda s: None, details={}, python_executable=str(interpreter))
    if failure:
        with pytest.raises(ASRTranscriptError, match="invalid timestamps"):
            asr_native.transcribe(np.ones(32000, np.float32), **options)
    else:
        cues, language = asr_native.transcribe(np.ones(32000, np.float32), **options)
        assert language == "en" and cues[0].text == "Hello" and options["details"]["compile_device"] == "GPU"
    assert not list((tmp_path/"cache").glob("native-job-*"))


def test_region_checkpoint_invalidates_data_model_and_settings(tmp_path):
    audio = np.zeros(16, dtype=np.float32)
    model = tmp_path / "model"; model.mkdir()
    weights = model / "weights.bin"; weights.write_bytes(b"first")
    def cache(**kw):
        return RegionCache(tmp_path, model_signature(model), device="GPU", language="en", **kw)
    first = cache(beam_size=5); path = first.path_for(audio)
    result = {"chunks": [{"text": "Hello", "timestamp": [0, 1]}]}
    first.write(path, result)
    assert first.read(path) == result
    changed = audio.copy(); changed[0] = 1
    assert first.path_for(changed) != path
    assert cache(beam_size=4).path_for(audio) != path
    assert RegionCache(tmp_path, model_signature(model), device="GPU", language="en",
                       beam_size=5, precision="f16").path_for(audio) != path
    weights.write_bytes(b"other")
    assert cache(beam_size=5).path_for(audio) != path
    record = json.loads(path.read_text()); record["payload"] += " "
    path.write_text(json.dumps(record))
    assert first.read(path) is None


def test_transcript_error_does_not_rerun_entire_clip_on_cpu(tmp_path, monkeypatch):
    from jellydub import asr_openvino
    monkeypatch.setattr(transcribe, "_cuda_available", lambda: False)
    monkeypatch.setattr(asr_openvino, "gpu_available", lambda: True)
    monkeypatch.setattr(asr_openvino, "load_audio", lambda p: [])
    calls = []
    def invalid(audio, **kw):
        calls.append(kw["device"])
        raise ASRTranscriptError("invalid interval")
    monkeypatch.setattr(asr_openvino, "transcribe", invalid)
    with pytest.raises(ASRTranscriptError):
        transcribe.transcribe_video(Path("video"), model_name="small", device="auto", cache_dir=tmp_path)
    assert calls == ["GPU"]


def test_interrupted_asr_reuses_completed_region(tmp_path, monkeypatch):
    import sys
    from types import SimpleNamespace, ModuleType
    from jellydub import asr_openvino as ov
    def module(name, **attrs):
        value = ModuleType(name); value.__dict__.update(attrs)
        monkeypatch.setitem(sys.modules, name, value)
    config = SimpleNamespace(num_beams=5)
    request = SimpleNamespace(get_property=lambda name: ["GPU.0"])
    model = SimpleNamespace(generation_config=config, encoder=SimpleNamespace(request=request))
    module("optimum.intel.openvino", OVModelForSpeechSeq2Seq=SimpleNamespace(from_pretrained=lambda *a, **kw: model))
    processor = SimpleNamespace(tokenizer=None, feature_extractor=None)
    calls = []
    def recognize(audio, **kw):
        calls.append(float(audio[0]))
        if len(calls) == 2:
            raise RuntimeError("interrupted region")
        return {"chunks": [{"text": "Hello", "timestamp": [0, 1]}]}
    recognize.generation_config = config
    module("transformers", AutoProcessor=SimpleNamespace(from_pretrained=lambda *a, **kw: processor),
           pipeline=lambda *a, **kw: recognize)
    module("faster_whisper.vad", VadOptions=lambda: None,
           get_speech_timestamps=lambda *a: [{"start": 0, "end": 32000}, {"start": 32000, "end": 64000}])
    monkeypatch.setattr(ov, "_model_path", lambda *a: tmp_path / "model")
    audio = np.concatenate([np.ones(32000, dtype=np.float32), np.full(32000, 2, dtype=np.float32)])
    options = dict(model_name="small", device="GPU", cache_dir=tmp_path,
                   language="en", beam_size=5, report=lambda s: None)
    with pytest.raises(RuntimeError, match="interrupted"):
        ov.transcribe(audio, details={}, **options)
    details = {}
    cues, language = ov.transcribe(audio, details=details, **options)
    assert calls == [1, 2, 2]
    assert len(cues) == 2 and language == "en" and details["region_cache_hits"] == 1


class Tone(TTSProvider):
    def cache_id(self):
        return "preview-test"

    def synthesize(self, text, out_path, speaker=None):
        run_ffmpeg(["-f", "lavfi", "-i", "sine=duration=0.2", "-ac", "1", "-ar", "48000", str(out_path)])


def test_preview_keeps_complete_speech_and_reuses_full_dub_cache(tmp_path, monkeypatch):
    source = tmp_path / "reviewed.srt"
    write_srt([Cue(1, 10, 11, "Xin chào", "A"), Cue(2, 11, 14, "Chào bạn", "B"),
               Cue(3, 16, 17, "Ngoài mẫu", "C")], source)
    calls = []
    class ObservedTone(Tone):
        def synthesize(self, text, out_path, speaker=None):
            calls.append(speaker)
            super().synthesize(text, out_path, speaker)
    monkeypatch.setattr(preview, "get_provider", lambda *a, **kw: ObservedTone())
    config = Config(cache_dir=tmp_path / "cache")
    output = preview.create_preview(source, tmp_path / "out", config, start=10, seconds=2)
    record = json.loads((output.parent / "review.json").read_text())
    assert record["status"] == "pending_listening_review"
    assert record["duration"] == 4 and record["speakers"] == ["A", "B"]
    preview.create_preview(source, tmp_path / "second", config, start=10, seconds=2)
    assert calls == ["A", "B"]


def test_preview_rejects_source_collision(tmp_path):
    source = tmp_path / "preview.vi.srt"
    write_srt([Cue(1, 0, 1, "Xin chào")], source)
    original = source.read_bytes()
    with pytest.raises(JellyDubError, match="overwrite"):
        preview.create_preview(source, tmp_path, Config())
    assert source.read_bytes() == original
