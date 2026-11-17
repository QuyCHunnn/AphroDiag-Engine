"""Optional Intel/OpenVINO ASR with Whisper beam search and language detection.

Optimum's generation path is used rather than the GenAI remote-tensor beam path.
Imports and model downloads are lazy; CPU-only installations do not need this extra.
"""
from __future__ import annotations

import gc
import math
import logging
import warnings
from contextlib import contextmanager
import subprocess
import time
from pathlib import Path
from typing import Callable

from .models import Cue, JellyDubError
from .asr_cache import ASRTranscriptError, RegionCache, model_signature, append_region_cues
from .cache import cache_lock


def gpu_available() -> bool:
    try:
        import importlib.util
        if importlib.util.find_spec("optimum.intel.openvino") is None:
            return False
        import openvino as ov
        return any(device.startswith("GPU") for device in ov.Core().available_devices)
    except Exception:
        return False


def _model_path(name: str, cache: Path) -> Path:
    local = Path(name).expanduser()
    if local.is_dir():
        return local
    from huggingface_hub import snapshot_download
    return Path(snapshot_download(
        f"OpenVINO/whisper-{name}-int8-ov", cache_dir=str(cache),
        allow_patterns=["*.json", "*.xml", "*.bin", "*.txt"],
    ))


def load_audio(video: Path):
    try:
        import numpy as np
    except ImportError as exc:
        raise JellyDubError("Intel ASR requires: uv sync --extra asr --extra intel") from exc
    result = subprocess.run([
        "ffmpeg", "-v", "error", "-nostdin", "-i", str(video), "-map", "0:a:0",
        "-ar", "16000", "-ac", "1", "-f", "f32le", "-",
    ], capture_output=True)
    if result.returncode:
        raise JellyDubError(f"Cannot read ASR input audio: {result.stderr.decode(errors='replace').strip()}")
    if not result.stdout:
        raise JellyDubError("ASR input has no audio samples")
    return np.frombuffer(result.stdout, dtype=np.float32)


def _context_regions(speech, sample_count, sample_rate=16000):
    """Give subsecond utterances context without overlapping ASR windows.

    Whisper can hallucinate out-of-window timestamp tokens for tightly cropped
    subsecond VAD detections. Keep all detected speech and include surrounding
    audio (at least three seconds where the recording/neighbours allow it).
    """
    result = []
    for index, region in enumerate(speech):
        # Do not change already usable crops: extra context can change the
        # transcript of short game voice lines and introduce hallucinations.
        if region["end"] - region["start"] >= sample_rate:
            result.append(dict(region))
            continue
        low = 0 if index == 0 else (speech[index - 1]["end"] + region["start"]) // 2
        high = sample_count if index + 1 == len(speech) else (region["end"] + speech[index + 1]["start"]) // 2
        missing = max(0, 3 * sample_rate - (region["end"] - region["start"]))
        left = max(low, region["start"] - missing // 2)
        right = min(high, max(region["end"], left + 3 * sample_rate))
        left = max(low, min(left, right - 3 * sample_rate))
        result.append({"start": left, "end": right})
    return result


@contextmanager
def _quiet_backend_logs():
    loggers = [logging.getLogger(name) for name in ("transformers", "optimum", "nncf")]
    levels = [logger.level for logger in loggers]
    try:
        for logger in loggers:
            logger.setLevel(logging.ERROR)
        with warnings.catch_warnings():
            warnings.filterwarnings("ignore", category=SyntaxWarning, module=r"optimum\..*")
            yield
    finally:
        for logger, level in zip(loggers, levels):
            logger.setLevel(level)


@_quiet_backend_logs()

def transcribe(audio, *, model_name: str, device: str, cache_dir: Path,
               language: str | None, beam_size: int, report: Callable[[str], None],
               details: dict, on_region=None) -> tuple[list[Cue], str]:
    try:
        from optimum.intel.openvino import OVModelForSpeechSeq2Seq
        from transformers import AutoProcessor, pipeline
        from faster_whisper.vad import VadOptions, get_speech_timestamps
    except ImportError as exc:
        raise JellyDubError(f"Intel ASR import failed: {exc}; check runtime dependencies or reinstall extra intel") from exc

    speech = get_speech_timestamps(audio, VadOptions())
    if not speech:
        return [], language or "und"
    path = _model_path(model_name, cache_dir)
    model = recognizer = None
    try:
        model = OVModelForSpeechSeq2Seq.from_pretrained(
            str(path), device=device, export=False, local_files_only=True,
            ov_config={"INFERENCE_PRECISION_HINT": "f32"}, trust_remote_code=False,
        )
        processor = AutoProcessor.from_pretrained(str(path), local_files_only=True, trust_remote_code=False)
        if language is None:
            import numpy as np
            pieces, remaining = [], 30 * 16000
            for region in speech:
                piece = audio[region['start']:min(region['end'], region['start'] + remaining)]
                pieces.append(piece)
                remaining -= len(piece)
                if remaining <= 0:
                    break
            features = processor(np.concatenate(pieces),
                                 sampling_rate=16000, return_tensors="pt").input_features
            lang_id = int(model.detect_language(input_features=features,
                                               generation_config=model.generation_config)[0])
            language = next((token[2:-2] for token, value in model.generation_config.lang_to_id.items()
                             if value == lang_id), "und")
        if language not in {"en", "vi"}:
            return [], language
        model.generation_config.num_beams = beam_size
        model.generation_config.task = "transcribe"
        model.generation_config.language = language
        recognizer = pipeline("automatic-speech-recognition", model=model,
                              tokenizer=processor.tokenizer, feature_extractor=processor.feature_extractor)
        if recognizer.generation_config.num_beams != beam_size:
            raise JellyDubError("ASR pipeline changed the requested beam size")
        details["generation"] = {"num_beams": beam_size, "language": language, "task": "transcribe"}
        region_cache = RegionCache(cache_dir, model_signature(path), device=device,
                                   language=language, beam_size=beam_size)
        details["region_cache_hits"] = 0
        cues = []
        regions = _context_regions(speech, len(audio))
        for region_index, region in enumerate(regions, 1):
            first_cue = len(cues)
            start, end = region["start"], region["end"]
            region_started = time.perf_counter()
            report(f"ASR region {region_index}/{len(regions)}: {start / 16000:.2f}–{end / 16000:.2f}s")
            # Screams / effects can otherwise consume the decoder's entire
            # token budget repeating a single sound. Keep ample room for real
            # dialogue, but bound generation by the region's actual duration.
            token_budget = min(400, max(48, math.ceil((end - start) / 16000 * 12)))
            checkpoint = region_cache.path_for(audio[start:end])
            with cache_lock(checkpoint.with_suffix(".lock")):
                result = region_cache.read(checkpoint)
                if result is None:
                    result = recognizer(audio[start:end], return_timestamps=True,
                                        generate_kwargs={"no_repeat_ngram_size": 4,
                                                         "max_new_tokens": token_budget})
                else:
                    details["region_cache_hits"] += 1
                    report(f"ASR region {region_index}: reused completed checkpoint")
                report(f"ASR region {region_index}/{len(regions)} decoded in {time.perf_counter() - region_started:.2f}s")
                if not isinstance(result, dict):
                    raise ASRTranscriptError("ASR returned malformed chunks")
                append_region_cues(cues, result.get("chunks"), start/16000, end/16000, report)
                region_cache.write(checkpoint, result)
            if on_region:
                on_region(cues[first_cue:])
        execution = {}
        for name in ("encoder", "decoder", "decoder_with_past"):
            part = getattr(model, name, None)
            if part is not None and part.request is not None:
                compiled = part.request.get_compiled_model() if hasattr(part.request, "get_compiled_model") else part.request
                execution[name] = list(compiled.get_property("EXECUTION_DEVICES"))
        if device == "GPU" and (not execution or any(not d.startswith("GPU") for values in execution.values() for d in values)):
            raise RuntimeError(f"Unexpected ASR execution devices: {execution}")
        details["execution_devices"] = execution
        report(f"ASR execution devices: {execution}")
        return cues, language
    finally:
        recognizer = model = None
        gc.collect()
