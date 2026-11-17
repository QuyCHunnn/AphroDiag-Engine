"""Whisper ASR with GPU preference, explicit backend selection and CPU fallback."""
from __future__ import annotations

from pathlib import Path
from typing import Callable

from .models import Cue, JellyDubError
from .asr_cache import ASRTranscriptError


def _cuda_available() -> bool:
    try:
        import ctranslate2
        return ctranslate2.get_cuda_device_count() > 0
    except Exception:
        return False


def _faster_whisper(video, *, model_name, device, cache_dir, language, beam_size, on_region=None):
    try:
        from faster_whisper import WhisperModel
    except ImportError as exc:
        raise JellyDubError("ASR requires faster-whisper; install it with: uv sync --extra asr") from exc
    model = WhisperModel(model_name, device=device,
                         compute_type="float16" if device == "cuda" else "int8",
                         download_root=str(cache_dir))
    try:
        segments, info = model.transcribe(str(video), language=language, beam_size=beam_size, vad_filter=True)
        cues = []
        for segment in segments:
            text = segment.text.strip()
            start, end = float(segment.start), float(segment.end)
            if text and start >= 0 and end > start:
                cues.append(Cue(len(cues) + 1, start, end, text))
                if on_region:
                    on_region([cues[-1]])
        return cues, str(info.language).lower()
    finally:
        del model


def transcribe_video(video: Path, *, model_name: str, device: str, cache_dir: Path,
                     language: str | None = None, backend: str = "auto", beam_size: int = 5,
                     report: Callable[[str], None] = print, details: dict | None = None,
                     python_executable: str | None = None, on_region=None) -> tuple[list[Cue], str]:
    from . import asr_openvino
    if backend not in {"auto", "faster-whisper", "openvino", "genai"}:
        raise JellyDubError(f"Unknown ASR backend: {backend}")
    if device not in {"auto", "cpu", "cuda", "gpu"}:
        raise JellyDubError(f"Unknown ASR device: {device}")
    if type(beam_size) is not int or beam_size < 1:
        raise JellyDubError("ASR beam size must be a positive integer")
    requested_device = device
    if backend == "auto":
        if device == "gpu" or (device == "auto" and not _cuda_available() and asr_openvino.gpu_available()):
            backend = "openvino"
        else:
            backend = "faster-whisper"
    if backend == "genai" and language not in {"en", "vi"}:
        raise JellyDubError("Experimental GenAI ASR requires explicit source language en or vi")
    if (backend in {"openvino", "genai"} and device == "cuda") or (backend == "faster-whisper" and device == "gpu"):
        raise JellyDubError("Use openvino/gpu for Intel or faster-whisper/cuda for NVIDIA")
    if device == "auto":
        device = ("gpu" if asr_openvino.gpu_available() else "cpu") if backend in {"openvino", "genai"} else (
            "cuda" if _cuda_available() else "cpu")
    execution = details if details is not None else {}
    execution.update(backend=backend, device=device, beam_size=beam_size, fallback=False)
    audio = asr_openvino.load_audio(video) if backend in {"openvino", "genai"} else None

    def run(target):
        report(f"ASR backend={backend} device={target} beam={beam_size}")
        if backend == "genai":
            from . import asr_native
            return asr_native.transcribe(audio, model_name=model_name, device=target.upper(),
                                         cache_dir=cache_dir, language=language, beam_size=beam_size,
                                         report=report, details=execution, python_executable=python_executable,
                                         on_region=on_region)
        if backend == "openvino":
            return asr_openvino.transcribe(audio, model_name=model_name, device=target.upper(),
                                          cache_dir=cache_dir, language=language, beam_size=beam_size,
                                          report=report, details=execution,
                                          **({"on_region": on_region} if on_region is not None else {}))
        return _faster_whisper(video, model_name=model_name, device=target, cache_dir=cache_dir,
                               language=language, beam_size=beam_size,
                               **({"on_region": on_region} if on_region is not None else {}))
    failure = None
    try:
        cues, detected = run(device)
    except Exception as exc:
        if isinstance(exc, ASRTranscriptError):
            raise
        if requested_device != "auto" or device == "cpu":
            if isinstance(exc, JellyDubError):
                raise
            raise JellyDubError(f"ASR {backend}/{device} failed: {exc}") from exc
        failure = str(exc)
        execution.update(device="cpu", fallback=True, fallback_reason=failure)
    # Leave the exception scope before loading another model so GPU traceback
    # references cannot retain the failed model during the CPU retry.
    if failure is not None:
        reason = failure.strip().splitlines()[-1] if failure.strip() else "runtime error"
        report(f"Warning: ASR {backend}/{device} failed ({reason}); retrying CPU with the same beam={beam_size}")
        try:
            cues, detected = run("cpu")
        except Exception as cpu_error:
            raise JellyDubError(f"ASR failed on GPU and CPU: {cpu_error}") from cpu_error
    if not cues and detected == "und":
        raise JellyDubError("ASR did not detect any speech in the video")
    if detected not in {"en", "vi"}:
        raise JellyDubError(f"ASR detected unsupported language {detected!r}; JellyDub auto supports English and Vietnamese")
    if not cues:
        raise JellyDubError("ASR did not detect any speech in the video")
    return cues, detected
