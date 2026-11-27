"""Resumable stage orchestration; model lifetimes never overlap."""
from __future__ import annotations

from dataclasses import asdict, replace
import json
import logging
from pathlib import Path
from typing import Any, Callable

from .cache import atomic_write_bytes, atomic_write_text, cache_key, cache_lock, cached_file_matches, record_cached_file, file_digest, tmp_path_for
from .config import Config
from .export import export_mp3, export_mp4, mux_mkv, output_paths
from .ffmpeg import duration, probe, require_ffmpeg
from .mix import extract_audio, mix
from .models import JellyDubError, MixMode
from .separate import separate_background
from .subtitles import parse_srt, write_srt
from .timing import align, render_timeline
from .tts import get_provider
from .tts.synthesize import synthesize_cues
from .validate import check_audio

Reporter = Callable[[str], None]


class _WarningReporter(logging.Handler):
    def __init__(self, report: Reporter):
        super().__init__(level=logging.WARNING)
        self.report = report

    def emit(self, record: logging.LogRecord) -> None:
        self.report(f"Warning: {record.getMessage()}")


class _NoSubtitleError(JellyDubError):
    """No usable sidecar, embedded or online subtitle was found."""


def _fingerprint(path: Path) -> tuple:
    stat = path.stat()
    return str(path.resolve()), stat.st_size, stat.st_mtime_ns


def _work_dir(video: Path, srt: Path, config: Config) -> Path:
    refs = []
    for reference in config.reference or []:
        path = Path(reference.split("=", 1)[-1]).expanduser()
        refs.append(_fingerprint(path) if path.is_file() else str(path))
    # Scheduling does not change speech content. Preserve pre-workers cache keys
    # so existing jobs can resume when concurrency is adjusted.
    cache_settings = asdict(config)
    for option in ("tts_device", "translate_device", "asr_python", "tts_model"):
        if cache_settings[option] is None:
            cache_settings.pop(option)
    cache_settings.pop("voxcpm2_timeout")
    if config.tts != "voxcpm2" or cache_settings["voxcpm2_binary"] is None:
        cache_settings.pop("voxcpm2_binary")
    if not cache_settings["tts_style"]:
        cache_settings.pop("tts_style")
    if cache_settings["tts_precision"] == "fp32":
        cache_settings.pop("tts_precision")
    for option in ("tts_workers", "tts_blas_threads", "tts_cpu_threads", "pipeline_mode", "cache_dir", "subs", "download", "search_online", "providers",
                   "subtitle_credentials_env", "asr_model", "asr_device", "audio", "out_dir", "mkv", "mp4", "subtitle_font", "asr_backend", "asr_beam_size"):
        cache_settings.pop(option, None)
    key = cache_key(_fingerprint(video), srt.read_bytes().hex(), cache_settings, refs)
    work = config.cache_dir / f"{video.stem}-{key}"
    work.mkdir(parents=True, exist_ok=True)
    return work


def _inputs(video: Path, srt: Path, config: Config) -> None:
    config.validate()
    for path in (video, srt):
        if not path.is_file():
            raise JellyDubError(f"Input file does not exist: {path}")
    require_ffmpeg()


def get_translator(**options):
    # Import only after language detection has selected English.
    from .translate import get_translator as factory
    return factory(**options)


def translate_cues(*args, **options):
    from .translate import translate_cues as stage
    return stage(*args, **options)


def detect_source_language(srt: Path) -> tuple[str, str]:
    """Deterministic filename/diacritics heuristic; ambiguous input defaults to English."""
    import unicodedata
    tags = srt.name.lower().split(".")
    if tags[-1] == "srt" and any(tag in {"vi", "vie"} for tag in tags[1:-1]):
        return "vi", "Vietnamese filename tag"
    marked = set("ăâđêôơưàáảãạằắẳẵặầấẩẫậèéẻẽẹềếểễệìíỉĩịòóỏõọồốổỗộờớởỡợùúủũụừứửữựỳýỷỹỵ")
    texts = [unicodedata.normalize("NFC", cue.text).lower() for cue in parse_srt(srt) if cue.text.strip()]
    if texts and sum(any(char in marked for char in text) for text in texts) / len(texts) > 0.30:
        return "vi", "Vietnamese diacritics in more than 30% of cues"
    return "en", "no clear Vietnamese filename tag or diacritics"


def _translated(srt: Path, work: Path, config: Config, report: Reporter) -> Path:
    language, reason = ((config.source_lang, "explicit --source-lang") if config.source_lang != "auto"
                        else detect_source_language(srt))
    report(f"Source language: {language} ({reason})")
    if language == "vi":
        report("Vietnamese subtitles: skipping translation")
        return srt
    target = work / "translated.vi.srt"
    if not target.is_file() or not target.stat().st_size:
        report(f"Translating subtitles ({config.translator})…")
        translator = get_translator(**translator_options(config, report=report))
        try:
            # OPUS caches independent dialogue groups. Contextual LLM backends
            # deliberately invalidate the affected prompt batch and dependent
            # translated history when source text changes.
            shared = config.cache_dir / "translations"
            lock = config.cache_dir / "translation-locks" / f"{cache_key(translator.cache_id())}.lock"
            with cache_lock(lock):
                cues = translate_cues(parse_srt(srt), translator, shared,
                                      on_progress=lambda n, total: report(f"Translation {n}/{total}"),
                                      strict_context_cache=True)
            write_srt(cues, target)
        finally:
            translator.close()
    return target


def _publish_srt(source: Path, target: Path, inputs: tuple[Path, ...]) -> None:
    if source.resolve() == target.resolve():
        return
    if any(target.resolve() == p.resolve() for p in inputs):
        raise JellyDubError(f"Refusing to overwrite source file: {target}")
    atomic_write_bytes(target, source.read_bytes())


def translate(video: Path, srt: Path, config: Config, report: Reporter = print) -> Path:
    _inputs(video, srt, config)
    work = _work_dir(video, srt, config)
    translated = _translated(srt, work, config, report)
    target = output_paths(video, config.out_dir)["srt"]
    _publish_srt(translated, target, (video, srt))
    return target


def _stage(path: Path, action: Callable[[Path], None], *, inputs: tuple[Path, ...] = (), settings=None) -> None:
    signature = cache_key("stage-v2", [file_digest(source) for source in inputs], settings)
    with cache_lock(path.with_name(path.name + ".lock")):
        if cached_file_matches(path, signature):
            return
        path.parent.mkdir(parents=True, exist_ok=True)
        temporary = tmp_path_for(path)
        try:
            action(temporary)
            if not temporary.is_file() or not temporary.stat().st_size:
                raise JellyDubError(f"Stage produced no output: {path.name}")
            temporary.replace(path)
            record_cached_file(path, signature)
        finally:
            temporary.unlink(missing_ok=True)


def _checked_audio(path: Path, expected: float, work: Path) -> list[str]:
    """Reuse successful full-decode QC only when content and criteria match."""
    manifest = work / "qc" / f"{path.name}.json"
    with cache_lock(manifest.with_name(manifest.name + ".lock")):
        signature = {"version": 1, "sha256": file_digest(path), "expected_duration": expected}
        try:
            saved = json.loads(manifest.read_text())
            if saved == signature:
                return []
        except (OSError, ValueError):
            pass
        problems = check_audio(path, expected)
        if not problems:
            atomic_write_text(manifest, json.dumps(signature))
        return problems



def translator_options(config: Config, report: Reporter | None = None) -> dict:
    device = config.translate_device or config.device
    if config.translator == "opus-mt":
        from .devices import resolve_device
        device, reason = resolve_device("Translation", device)
        if report:
            report(f"Translation device={device}" + (f" ({reason})" if reason else ""))
    return {"name": config.translator, "model": config.translator_model, "device": device}


def tts_options(config: Config, report: Reporter | None = None) -> dict:
    """Share voice/reference settings between preview and full generation."""
    if config.tts == "gemini":
        from .tts.gemini import DEFAULT_MODEL
        if config.reference:
            raise JellyDubError("Gemini uses named voices; reference cloning is not supported")
        options = {"model": config.tts_model or DEFAULT_MODEL, "style": config.tts_style}
        if config.voice is not None:
            options["voice"] = config.voice
        if config.speaker_voices:
            options["speaker_voices"] = config.speaker_voices
        if report:
            report(f"TTS Gemini API model={options['model']} voice={config.voice or 'Charon'}")
        return options
    if config.tts == "voxcpm2":
        refs, unlabelled = {}, []
        for value in config.reference or []:
            if "=" in value:
                label, filename = value.split("=", 1)
                if not label.strip() or not filename.strip() or label in refs:
                    raise JellyDubError("VoxCPM2 references require unique nonempty LABEL=FILE entries")
                refs[label] = Path(filename).expanduser()
            else:
                unlabelled.append(Path(value).expanduser())
        if len(unlabelled) > 1:
            raise JellyDubError("Use LABEL=FILE for multiple reference voices")
        device = config.tts_device or config.device
        if report:
            report(f"TTS VoxCPM2 GGUF device={device} (auto requires GPU; no CPU fallback)")
        return {"model": config.tts_model, "binary": config.voxcpm2_binary,
                "style": config.tts_style, "device": device,
                "timeout": config.voxcpm2_timeout, "speaker_refs": refs,
                "reference": unlabelled[0] if unlabelled else None}
    device = config.tts_device or config.device
    if config.tts == "vieneu":
        from .devices import resolve_device
        requested = device
        device, reason = resolve_device("VieNeu TTS", requested)
        if report:
            report(f"TTS requested={requested} resolved={device}" + (f" ({reason})" if reason else ""))
    options = {"rate": config.rate, "pitch": config.pitch} if config.tts == "edge" else {"device": device,
                                                                                    "cache_dir": config.cache_dir / "voices"}
    if config.voice is not None:
        options["voice"] = config.voice
    if config.speaker_voices:
        options["speaker_voices"] = config.speaker_voices
    if config.tts == "vieneu":
        options.update(blas_threads=config.tts_blas_threads, cpu_threads=config.tts_cpu_threads)
        if config.tts_precision != "fp32":
            if device != "cpu":
                raise JellyDubError("VieNeu int8 currently requires tts_device=cpu")
            options["precision"] = config.tts_precision
        refs = {}
        unlabelled = []
        for value in config.reference or []:
            if "=" in value:
                label, filename = value.split("=", 1)
                refs[label] = Path(filename).expanduser()
            else:
                unlabelled.append(Path(value).expanduser())
        if len(unlabelled) > 1:
            raise JellyDubError("Use LABEL=FILE for multiple reference voices")
        if refs:
            options["speaker_refs"] = refs
        if unlabelled:
            options["reference"] = unlabelled[0]
    return options


def dub(video: Path, srt: Path | None, config: Config, report: Reporter = print, *,
        skip_timing_check: bool = False) -> dict[str, Path]:
    config.validate()
    if not video.is_file():
        raise JellyDubError(f"Input file does not exist: {video}")
    require_ffmpeg()
    original_srt = srt
    srt, language = _resolve_subtitle(video, srt, config, report,
                                      skip_explicit_sync=skip_timing_check)
    if language is not None:
        config = replace(config, source_lang=language)
    _inputs(video, srt, config)
    work = _work_dir(video, srt, config)
    outputs = output_paths(video, config.out_dir)
    for key in ("audio", "mkv", "mp4"):
        if any(outputs[key].resolve() == p.resolve() for p in (video, srt)):
            raise JellyDubError(f"Refusing to overwrite source: {outputs[key]}")
    translated = _translated(srt, work, config, report)
    cues = parse_srt(translated)
    _publish_srt(translated, outputs["srt"], (video, original_srt or srt))
    total = duration(video)
    if any(c.start < 0 or c.end > total + 0.05 or c.end < c.start for c in cues):
        raise JellyDubError("Subtitle timestamps fall outside the video duration")
    options = tts_options(config, report=report)
    report(f"Synthesizing {len(cues)} subtitle segments ({config.tts})…")
    provider = get_provider(config.tts, **options)
    try:
        results, failures = synthesize_cues(cues, provider, config.cache_dir / "tts", speech_plan=config.speech_plan, workers=config.tts_workers,
                                          on_progress=lambda n, count: report(f"TTS {n}/{count}"))
    finally:
        for notice in getattr(provider, "speech_warnings", []):
            report(f"Warning: TTS cue {notice['cue_index']} {notice['field']}: {notice['detail']}")
        if getattr(provider, "execution", {}).get("loaded"):
            report(f"TTS execution: {provider.execution}")
        provider.close()
    if failures:
        for cue, error in failures:
            report(f"Warning: segment {cue.index} failed: {error}")
        raise JellyDubError(f"{len(failures)} TTS segments failed. Re-run the same command to retry; successful segments are cached.")
    report("Aligning speech…")
    clips = align(cues, results, config.cache_dir / "aligned", max_tempo=config.max_tempo)
    warning_count = sum(bool(clip.warning) for clip in clips)
    if warning_count:
        report(f"Timing warnings: {warning_count} segment(s)")
    for clip in clips:
        if clip.warning:
            report(f"Warning: segment {clip.cue_index}: {clip.warning}")
    if any(c.start + c.duration > total + 0.02 for c in clips):
        raise JellyDubError("Aligned speech extends beyond the video; shorten the final subtitles and re-run to avoid truncation")
    timeline = work / "timeline.wav"
    _stage(timeline, lambda p: render_timeline(clips, total, p),
           inputs=tuple(clip.path for clip in clips),
           settings=(total, [(clip.start, clip.duration, clip.tempo) for clip in clips]))
    mode = MixMode(config.mix)
    original = background = None
    if mode != MixMode.DUB_ONLY:
        original = work / "original.wav"
        _stage(original, lambda p: extract_audio(video, p, stream_index=config.audio_stream),
               inputs=(video,), settings=config.audio_stream)
    if mode == MixMode.REPLACE:
        report("Separating original dialogue locally…")
        try:
            background = separate_background(original, work / "separation", device=config.device)
        except JellyDubError as exc:
            if not config.fallback_voiceover:
                raise JellyDubError(f"Separation failed: {exc}. Try --mix voiceover or --fallback-voiceover.") from exc
            report(f"Warning: separation failed; using voiceover: {exc}")
            mode = MixMode.VOICEOVER
    mixed = work / f"mixed-{mode.value}.wav"
    _stage(mixed, lambda p: mix(mode, timeline, p, original_wav=original, background_wav=background),
           inputs=tuple(path for path in (timeline, original, background) if path is not None),
           settings=mode.value)
    problems = _checked_audio(mixed, total, work)
    if problems:
        raise JellyDubError("Mixed audio validation failed: " + "; ".join(problems))
    # Encode external MP3 only on explicit request. Mux from the lossless mix.
    if config.audio:
        encoded = work / "export.mp3"
        _stage(encoded, lambda p: export_mp3(mixed, p), inputs=(mixed,))
        problems = _checked_audio(encoded, total, work)
        if problems:
            raise JellyDubError("Export validation failed: " + "; ".join(problems))
        _copy_output(encoded, outputs["audio"])
    if config.mkv:
        muxed = work / "export-direct.mkv"
        _stage(muxed, lambda p: mux_mkv(video, mixed, outputs["srt"], p),
               inputs=(video, mixed, outputs["srt"]), settings="mkv-v3-vietnamese-defaults")
        _copy_output(muxed, outputs["mkv"])
    if config.mp4:
        rendered = work / f"export-direct-{cache_key(config.subtitle_font)}.mp4"
        _stage(rendered, lambda p: export_mp4(
            video, mixed, outputs["srt"], p, subtitle_font=config.subtitle_font),
               inputs=(video, mixed, outputs["srt"]), settings=config.subtitle_font)
        _copy_output(rendered, outputs["mp4"])
    selected_outputs = {k: v for k, v in outputs.items()
                        if k == "srt" or (k == "audio" and config.audio)
                        or (k == "mkv" and config.mkv) or (k == "mp4" and config.mp4)}
    primary = "mkv" if config.mkv else "mp4" if config.mp4 else "audio" if config.audio else "srt"
    report(f"Done: {outputs[primary]}")
    return selected_outputs


def auto(video: Path, srt: Path | None, config: Config, report: Reporter = print) -> dict[str, Path]:
    """Transcribe English video and export Vietnamese MKV/SRT without review gates."""
    if srt is not None:
        raise JellyDubError("auto accepts only an English video; use dub for an external SRT")
    config = replace(config, source_lang="auto", subs="auto", mix="dub-only",
                     pipeline_mode="sequential", download=False, search_online=False,
                     fallback_voiceover=False)
    config.validate()
    if config.asr_backend == "genai":
        raise JellyDubError("auto requires language detection; use the auto, faster-whisper or openvino ASR backend")
    if not video.is_file():
        raise JellyDubError(f"Input file does not exist: {video}")
    require_ffmpeg()
    report_path = config.cache_dir / "reports" / f"{video.stem}-{cache_key(_fingerprint(video), asdict(config))}.json"
    if report_path.resolve() == video.resolve():
        raise JellyDubError(f"Refusing to overwrite source file: {report_path}")

    messages = []

    def emit(message: str) -> None:
        messages.append(message)
        report(message)

    selected = None
    source_language = None
    transcribed = False
    try:
        input_streams = probe(video).get("streams", [])
        if not any(stream.get("codec_type") == "video" for stream in input_streams):
            raise JellyDubError("auto requires a video file with a video stream")
        if not any(stream.get("codec_type") == "audio" for stream in input_streams):
            raise JellyDubError("auto requires a video with an audio track")
        emit(f"Transcribing video ({config.asr_model}, backend={config.asr_backend}, "
             f"device={config.asr_device}, beam={config.asr_beam_size})…")
        selected, source_language = _asr_subtitle(video, config, emit)
        transcribed = True
        if source_language != "en":
            raise JellyDubError(f"auto accepts English video only; ASR detected {source_language!r}")
        run_config = replace(config, source_lang="en", mkv=True)
        warning_handler = _WarningReporter(emit)
        logger = logging.getLogger()
        logger.addHandler(warning_handler)
        try:
            try:
                outputs = dub(video, selected, run_config, report=emit, skip_timing_check=True)
            except JellyDubError as exc:
                if "TTS segments failed." not in str(exc):
                    raise
                emit("Retrying failed TTS segments once; successful cached segments will be reused…")
                outputs = dub(video, selected, run_config, report=emit, skip_timing_check=True)
        finally:
            logger.removeHandler(warning_handler)
        video_duration = duration(video)
        subtitle_qc = _subtitle_quality(outputs["srt"], video_duration)
        warnings = _pipeline_warnings(messages, asr=transcribed)
        media = {}
        for key, path in outputs.items():
            item = {"path": str(path), "bytes": path.stat().st_size}
            if path.suffix.lower() in {".mp3", ".mkv", ".mp4"}:
                details = probe(path)
                item["duration_seconds"] = float(details["format"]["duration"])
                item["streams"] = [
                    {
                        "type": stream.get("codec_type"),
                        "codec": stream.get("codec_name"),
                        "language": stream.get("tags", {}).get("language"),
                        "title": stream.get("tags", {}).get("title"),
                    }
                    for stream in details.get("streams", [])
                ]
            media[key] = item
        _validate_auto_exports(media, video_duration)
        quality_status = ("needs_review" if subtitle_qc["issues"] or warnings or subtitle_qc["score"] < 0.75
                          else "automated_checks_passed")
        report_data = {
            "status": "completed",
            "quality_status": quality_status,
            "video": str(video),
            "subtitle": str(selected),
            "subtitle_source": "asr",
            "asr_execution": json.loads(selected.with_suffix(".json").read_text()).get("execution", {}) if transcribed else None,
            "subtitle_timing": "asr",
            "source_language": source_language,
            "translation": "performed",
            "semantic_translation_verified": False,
            "subtitle_qc": subtitle_qc,
            "pipeline_warnings": warnings,
            "outputs": media,
            "note": "Automated checks cannot verify translation meaning or perceived voice quality.",
        }
    except Exception as exc:
        report_data = {
            "status": "failed",
            "video": str(video),
            "subtitle": str(selected) if selected else None,
            "subtitle_source": "asr",
            "source_language": source_language,
            "error": str(exc),
            "pipeline_warnings": _pipeline_warnings(messages, asr=transcribed),
        }
        atomic_write_text(report_path, json.dumps(report_data, ensure_ascii=False, indent=2) + "\n")
        raise

    atomic_write_text(report_path, json.dumps(report_data, ensure_ascii=False, indent=2) + "\n")
    emit(f"Automated QC report: {report_path}")
    return {**outputs, "report": report_path}


def _asr_subtitle(video: Path, config: Config, report: Reporter, *, on_region=None) -> tuple[Path, str]:
    native_signature = []
    if config.asr_backend == "genai":
        from .asr_native import runtime_signature
        native_signature = [runtime_signature(config.asr_python)]
    key = cache_key("asr-v4-bounded", _fingerprint(video), config.asr_model, config.source_lang,
                    config.asr_backend, config.asr_device, config.asr_beam_size, *native_signature)
    folder = config.cache_dir / "transcriptions"
    subtitle = folder / f"{video.stem}-{key}.srt"
    metadata = folder / f"{video.stem}-{key}.json"
    if subtitle.is_file() and metadata.is_file():
        try:
            language = json.loads(metadata.read_text(encoding="utf-8"))["language"]
            if language in {"en", "vi"}:
                cached = parse_srt(subtitle)
                if on_region:
                    on_region(cached)
                report(f"Reusing cached ASR subtitles ({language})")
                return subtitle, language
        except (OSError, ValueError, KeyError, TypeError, JellyDubError):
            pass

    execution = {}
    cues, language = transcribe_video(
        video,
        model_name=config.asr_model,
        device=config.asr_device,
        backend=config.asr_backend,
        beam_size=config.asr_beam_size,
        report=report, details=execution,
        cache_dir=config.cache_dir / "asr-models",
        language=None if config.source_lang == "auto" else config.source_lang,
        **({"python_executable": config.asr_python} if config.asr_backend == "genai" else {}),
        **({"on_region": on_region} if on_region is not None else {}),
    )
    write_srt(cues, subtitle)
    atomic_write_text(metadata, json.dumps({"language": language, "execution": execution}, ensure_ascii=False))
    report(f"ASR created {len(cues)} subtitle cues ({language})")
    return subtitle, language


def _pipeline_warnings(messages: list[str], *, asr: bool = False) -> list[str]:
    warnings = []
    for message in messages:
        normalized = message.lower()
        relevant = (
            normalized.startswith(("warning:", "timing warnings:", "rejected "))
            or "offset unverified" in normalized
            or normalized.startswith("no sufficiently complete english reference")
            or normalized.startswith("no english timing reference")
        )
        if relevant and not (asr and "english timing reference" in normalized):
            warnings.append(message)
    return warnings


def _validate_auto_exports(media: dict[str, dict[str, Any]], video_duration: float) -> None:
    required = {"srt", "mkv"}
    missing = required - media.keys()
    if missing:
        raise JellyDubError(f"Automated export is missing: {', '.join(sorted(missing))}")
    mkv_streams = media["mkv"]["streams"]
    if not any(stream["type"] == "audio" and stream["language"] == "vie" for stream in mkv_streams):
        raise JellyDubError("MKV export is missing its Vietnamese audio track")
    if not any(stream["type"] == "subtitle" and stream["language"] == "vie" for stream in mkv_streams):
        raise JellyDubError("MKV export is missing its Vietnamese subtitle track")
    if "mp4" in media:
        mp4_streams = media["mp4"]["streams"]
        if not any(stream["type"] == "video" and stream["codec"] == "h264" for stream in mp4_streams):
            raise JellyDubError("MP4 export is missing its H.264 video stream")
        if not any(stream["type"] == "audio" and stream["codec"] == "aac" and
                   stream["language"] == "vie" for stream in mp4_streams):
            raise JellyDubError("MP4 export is missing its Vietnamese AAC audio stream")
    for key in ("audio", "mkv", "mp4"):
        if key not in media:
            continue
        delta = abs(media[key]["duration_seconds"] - video_duration)
        if delta > max(1.0, video_duration * 0.01):
            raise JellyDubError(f"{key.upper()} duration differs from the source by {delta:.2f} seconds")


def _subtitle_quality(path: Path, video_duration: float) -> dict:
    import math

    cues = parse_srt(path)
    issues = []
    assessment = assess_subtitle(cues, lang="vi")
    issues.extend({"type": "content_quality", "detail": item} for item in assessment.issues)
    score = float(assessment.score)
    if not math.isfinite(score):
        issues.append({"type": "invalid_quality_score"})
        score = 0.0
    previous = None
    for cue in cues:
        if not cue.text.strip():
            issues.append({"cue": cue.index, "type": "empty_text"})
        if cue.end > video_duration + 0.05:
            issues.append({"cue": cue.index, "type": "beyond_video"})
        if len(cue.text) > 60:
            issues.append({"cue": cue.index, "type": "long_text", "characters": len(cue.text)})
        if cue.duration > 0 and len(cue.text) / cue.duration > 24:
            issues.append({"cue": cue.index, "type": "fast_reading_rate",
                           "characters_per_second": round(len(cue.text) / cue.duration, 1)})
        if previous is not None and cue.start < previous.end - 0.1:
            issues.append({"cue": cue.index, "type": "overlap", "with_cue": previous.index})
        previous = cue
    return {"cue_count": len(cues), "score": round(score, 3), "issues": issues}


def _copy_output(source: Path, target: Path) -> None:
    """Publish potentially large media without reading it all into memory."""
    import shutil
    target.parent.mkdir(parents=True, exist_ok=True)
    temporary = tmp_path_for(target)
    try:
        shutil.copyfile(source, temporary)
        temporary.replace(target)
    finally:
        temporary.unlink(missing_ok=True)


# Lazy adapters preserve importability while subtitle-source modules/optional extras
# are unavailable. Tests patch these contracts, never the real network providers.
def find_local_subtitles(*args, **kwargs):
    from .subsource import find_local_subtitles as stage
    return stage(*args, **kwargs)


def extract_embedded_subtitles(*args, **kwargs):
    from .subsource import extract_embedded_subtitles as stage
    return stage(*args, **kwargs)


def download_subtitles(*args, **kwargs):
    from .subsource import download_subtitles as stage
    return stage(*args, **kwargs)


def transcribe_video(*args, **kwargs):
    from .transcribe import transcribe_video as stage
    return stage(*args, **kwargs)


def estimate_sync(*args, **kwargs):
    from .subsync import estimate_sync as stage
    return stage(*args, **kwargs)


def apply_sync(*args, **kwargs):
    from .subsync import apply_sync as stage
    return stage(*args, **kwargs)


def _source_dir(video: Path, config: Config) -> Path:
    folder = config.cache_dir / "subtitles" / f"{video.stem}-{cache_key(_fingerprint(video))}"
    folder.mkdir(parents=True, exist_ok=True)
    return folder


def _local_candidates(video: Path, language: str, work: Path, report: Reporter,
                      search_dirs: list[Path] | None = None):
    candidates = (list(find_local_subtitles(video, language, search_dirs=search_dirs))
                  if search_dirs else list(find_local_subtitles(video, language)))
    try:
        candidates.extend(extract_embedded_subtitles(video, language, work / "embedded"))
    except JellyDubError:
        report(f"Warning: could not extract embedded {language} subtitles")
    return sorted(candidates, key=lambda c: c.score, reverse=True)


class _ReferenceCues(list):
    """A cue list retaining alternative full-track timing references."""

    def __init__(self, alternatives):
        super().__init__(alternatives[0][1])
        self.alternatives = alternatives


def _english_reference(video: Path, work: Path, report: Reporter, *,
                       search_dirs: list[Path] | None = None, paired_subtitle: Path | None = None):
    references = []
    candidates = (_local_candidates(video, "en", work, report, search_dirs=search_dirs)
                  if search_dirs else _local_candidates(video, "en", work, report))
    for candidate in candidates:
        if (getattr(candidate, "metadata", {}) or {}).get("forced", False):
            report(f"Skipping forced English sync reference: {candidate.path}")
            continue
        if (paired_subtitle is not None
                and candidate.path.parent.resolve() == paired_subtitle.parent.resolve()
                and _subtitle_base(candidate.path) == _subtitle_base(paired_subtitle)):
            candidate.score = 1.0
            candidate.note = "matching language-paired subtitle"
        if candidate.score < 0.5:
            continue
        try:
            cues = parse_srt(candidate.path)
            if cues:
                references.append((candidate, cues))
        except JellyDubError:
            continue
    # Actual parsed cue counts beat metadata counts that may be missing/stale.
    references.sort(key=lambda item: (len(item[1]), item[0].origin == "embedded", item[0].score), reverse=True)
    if references:
        candidate, cues = references[0]
        report(f"Sync reference: {candidate.path} ({candidate.origin}, {len(cues)} cues)")
        return _ReferenceCues([(candidate.path, cues) for candidate, cues in references])
    report("No English timing reference; subtitle synchronization is unverified")
    return None


def _subtitle_base(path: Path) -> str:
    stem = path.stem.casefold()
    for suffix in (".english", ".eng", ".en", ".vietnamese", ".vie", ".vi"):
        if stem.endswith(suffix):
            return stem[:-len(suffix)]
    return stem


def _reference_for_candidate(reference, cues, report: Reporter):
    alternatives = getattr(reference, "alternatives", [("selected reference", reference)])
    for path, candidate_reference in alternatives:
        if candidate_reference is None:
            continue
        if len(candidate_reference) < 0.3 * len(cues):
            report(f"Skipping sparse English sync reference {path}: {len(candidate_reference)} cues for {len(cues)} candidate cues (below 30%)")
            continue
        return candidate_reference
    if reference is not None:
        report("No sufficiently complete English reference; subtitle synchronization is unverified")
    return None


def strip_junk(*args, **kwargs):
    from .subquality import strip_junk as stage
    return stage(*args, **kwargs)


def assess_subtitle(*args, **kwargs):
    from .subquality import assess_subtitle as stage
    return stage(*args, **kwargs)


def _best_candidate(candidates, reference, work: Path, report: Reporter):
    import math
    ranked = []
    for candidate in candidates:
        report(f"Candidate: {candidate.path} origin={candidate.origin} score={candidate.score:.3f} {candidate.note}")
        try:
            cues = parse_srt(candidate.path)
            if not cues:
                raise JellyDubError("empty subtitles")
            candidate_reference = _reference_for_candidate(reference, cues, report)
            quality = assess_subtitle(cues, lang=candidate.language, reference=candidate_reference)
            candidate.quality = quality.score
            candidate.issues = list(dict.fromkeys([*getattr(candidate, "issues", []), *quality.issues]))
            result = estimate_sync(cues, candidate_reference) if candidate_reference is not None else None
            reason = ""
            if not math.isfinite(quality.score) or quality.score < 0.5:
                reason = "subtitle quality below 0.5"
            if result is not None:
                values = (result.offset, result.scale, result.matched_ratio, result.residual)
                if (not all(math.isfinite(v) for v in values) or result.scale <= 0 or
                        result.matched_ratio < 0.5 or result.residual > 0.3):
                    reason = f"matched_ratio={result.matched_ratio:.3f}, residual={result.residual:.3f}s"
                sync_quality = max(0., min(1., result.matched_ratio)) * max(0., 1. - result.residual / 0.3)
            else:
                sync_quality = 0.5  # neutral, explicitly unverified
            local = candidate.origin in {"local", "sidecar", "embedded", "explicit"}
            prescore = 1.0 if local else max(0., min(1., getattr(candidate, "prescore", 0.0)))
            final = 0.4 * prescore + 0.4 * quality.score + 0.2 * sync_quality
            ranked.append((final, candidate, cues, result, reason))
        except JellyDubError:
            ranked.append((-1., candidate, [], None, "unreadable subtitle or synchronization failed"))
    ranked.sort(key=lambda row: (row[0], row[1].score), reverse=True)
    report("Rank | Origin | Uploader/trusted | Downloads | Rating | Quality | Sync offset | Final | Issues")
    for number, (final, candidate, cues, result, reason) in enumerate(ranked, 1):
        metadata = getattr(candidate, "metadata", {}) or {}
        uploader = metadata.get("uploader", {})
        if isinstance(uploader, dict):
            uploader = uploader.get("name") or uploader.get("rank") or "unknown"
        trusted = "trusted" if metadata.get("from_trusted") else "unverified"
        downloads = metadata.get("download_count", metadata.get("new_download_count", "?"))
        rating = metadata.get("ratings", metadata.get("rating", "?"))
        quality_value = getattr(candidate, "quality", None)
        quality_text = f"{quality_value:.3f}" if quality_value is not None else "?"
        offset = f"{result.offset:+.3f}s" if result is not None else "unverified"
        issues = "; ".join([*getattr(candidate, "issues", []), *([reason] if reason else [])]) or "none"
        report(f"{number} | {candidate.origin} | {uploader}/{trusted} | {downloads} | {rating} | {quality_text} | {offset} | {max(0., final):.3f} | {issues}")
        if reason:
            report(f"Rejected {candidate.path}: {reason}")
    for final, candidate, cues, result, reason in ranked:
        if reason:
            continue
        corrected = apply_sync(cues, result) if result is not None else cues
        cleaned = strip_junk(corrected)
        if not cleaned:
            report(f"Rejected {candidate.path}: no dialogue remains after removing ads/credits")
            continue
        sync_note = f"offset {result.offset:+.3f}s, scale {result.scale:.6f}" if result else "offset unverified"
        report(f"Using {candidate.origin} {candidate.language.upper()} subtitle {candidate.path}; {sync_note}; final={final:.3f}")
        if cleaned == cues:
            return candidate.path
        values = (result.offset, result.scale) if result else None
        target = work / f"selected-{cache_key(candidate.path.read_bytes().hex(), values, cleaned)}.{candidate.language}.srt"
        if not target.is_file() or not target.stat().st_size:
            write_srt(cleaned, target)
        return target
    return None


def _online_candidates(video, language, work, config, report, local_candidates, reference):
    import math
    if language == "vi" and config.download and not config.search_online:
        for candidate in local_candidates:
            if candidate.origin not in {"local", "sidecar", "embedded"} or candidate.language != "vi":
                continue
            try:
                cues = parse_srt(candidate.path)
                timing = _reference_for_candidate(reference, cues, report)
                if not cues or timing is None:
                    continue
                quality = assess_subtitle(cues, lang="vi", reference=timing)
                if not math.isfinite(quality.score) or quality.score < 0.9:
                    continue
                result = estimate_sync(cues, timing)
                if (all(math.isfinite(v) for v in (result.offset, result.scale, result.matched_ratio, result.residual))
                        and result.scale > 0 and result.matched_ratio >= 0.9 and result.residual <= 0.1
                        and strip_junk(apply_sync(cues, result))):
                    report("skipping online search: good local subtitle")
                    return []
            except JellyDubError:
                continue
    return _download_candidates(video, language, work, config, report)


def _download_candidates(video: Path, language: str, work: Path, config: Config, report: Reporter):
    import importlib.util
    import os
    if not config.download:
        report("Subtitle downloads disabled")
        return []
    if importlib.util.find_spec("subliminal") is None:
        report("Subtitle downloads skipped: install the opensubtitles extra")
        return []
    credentials = {key: os.environ[name] for key, name in config.subtitle_credentials_env.items()
                   if os.environ.get(name)}
    report(f"Searching online for {language.upper()} subtitles")
    try:
        return download_subtitles(video, language, work / "downloaded", providers=config.providers,
                                  credentials=credentials or None, max_downloads=2)
    except (JellyDubError, OSError):
        # Never interpolate provider exceptions: they may contain credentials.
        report("Warning: subtitle download unavailable; continuing with local sources")
        return []


def _resolve_subtitle(video: Path, explicit: Path | None, config: Config, report: Reporter, *,
                      skip_explicit_sync: bool = False):
    if explicit is not None:
        if not explicit.is_file():
            raise JellyDubError(f"Input file does not exist: {explicit}")
        report(f"Using explicit subtitle: {explicit}")
        language = config.source_lang if config.source_lang != "auto" else detect_source_language(explicit)[0]
        if language != "vi":
            return explicit, None
        if skip_explicit_sync:
            report("Using preselected subtitle timing; skipping redundant synchronization checks")
            return explicit, "vi"
        from types import SimpleNamespace
        work = _source_dir(video, config)
        sibling_dirs = ([explicit.parent] if explicit.parent.resolve() != video.parent.resolve() else None)
        reference = _english_reference(video, work, report, search_dirs=sibling_dirs,
                                       paired_subtitle=explicit)
        chosen = _best_candidate([SimpleNamespace(path=explicit, origin="explicit", language="vi", score=1.0,
                                                  note="user supplied")], reference, work, report)
        if chosen is None:
            raise JellyDubError("Explicit Vietnamese subtitle failed timing checks; supply subtitles matching this video")
        return chosen, "vi"
    work = _source_dir(video, config)
    reference = _english_reference(video, work, report)
    if config.subs != "translate":
        candidates = []
        if config.subs != "download":
            candidates.extend(_local_candidates(video, "vi", work, report))
        if config.subs in {"auto", "download"}:
            candidates.extend(_online_candidates(video, "vi", work, config, report, candidates, reference))
        chosen = _best_candidate(candidates, reference, work, report)
        if chosen is not None:
            return chosen, "vi"
        if config.subs in {"local", "download"}:
            raise JellyDubError(f"No acceptable Vietnamese subtitles found in --subs {config.subs} mode; try --subs auto or provide --srt")
    for candidate in _local_candidates(video, "en", work, report):
        try:
            if candidate.score >= 0.5 and parse_srt(candidate.path):
                report(f"No accepted Vietnamese subtitle; translating English subtitle {candidate.path}")
                return candidate.path, "en"
        except JellyDubError:
            continue
    raise _NoSubtitleError("No suitable subtitles found. Supply --srt EN.srt or VI.srt, or enable subtitle downloads")


def subs(video: Path, config: Config, *, lang: str = "vi", force: bool = False,
         report: Reporter = print) -> Path:
    """Find and sync a subtitle; atomically publish only when replacement is allowed."""
    config.validate()
    if lang not in {"vi", "en"}:
        raise JellyDubError("Subtitle language must be vi or en")
    if not video.is_file():
        raise JellyDubError(f"Input file does not exist: {video}")
    require_ffmpeg()
    work = _source_dir(video, config)
    reference = _english_reference(video, work, report) if lang == "vi" else None
    candidates = _local_candidates(video, lang, work, report)
    candidates.extend(_online_candidates(video, lang, work, config, report, candidates, reference))
    selected = _best_candidate(candidates, reference, work, report)
    if selected is None:
        raise JellyDubError(f"No acceptable {lang.upper()} subtitles found")
    target = (config.out_dir or video.parent) / f"{video.stem}.{lang}.srt"
    if target.resolve() == video.resolve():
        raise JellyDubError("Refusing to overwrite source video")
    if selected.resolve() == target.resolve():
        report(f"Subtitle already at {target}; unchanged")
        return target
    content = selected.read_bytes()
    if target.exists():
        if target.read_bytes() == content:
            report(f"Subtitle already at {target}; unchanged")
            return target
        if not force:
            raise JellyDubError(f"Subtitle output already exists: {target}; use --force to replace it")
    atomic_write_bytes(target, content)
    report(f"Saved subtitle: {target}")
    return target
