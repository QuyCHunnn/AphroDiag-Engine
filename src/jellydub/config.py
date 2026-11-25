"""TOML configuration with explicit CLI overrides."""
from __future__ import annotations

from dataclasses import dataclass, field, fields
from pathlib import Path
import tomllib

from .models import JellyDubError


@dataclass
class Config:
    source_lang: str = "auto"
    translator: str = "opus-mt"
    translator_model: str | None = None
    asr_model: str = "small"
    asr_device: str = "auto"
    asr_backend: str = "auto"
    asr_python: str | None = None
    pipeline_mode: str = "sequential"
    asr_beam_size: int = 5
    tts: str = "vieneu"
    tts_model: str | None = None
    voxcpm2_binary: str | None = None
    voxcpm2_timeout: int = 600
    speech_plan: str | None = None
    tts_style: str = ""
    tts_workers: int = 4
    tts_blas_threads: int = 2
    tts_cpu_threads: int = 0
    tts_precision: str = "fp32"
    voice: str | None = None
    speaker_voices: dict[str, str] = field(default_factory=dict)
    rate: str = "+0%"
    pitch: str = "+0Hz"
    reference: list[str] | None = None
    mix: str = "dub-only"
    mkv: bool = True
    mp4: bool = False
    audio: bool = False
    subtitle_font: str = "DejaVu Sans"
    out_dir: Path | None = None
    cache_dir: Path = Path("~/.cache/jellydub")
    audio_stream: int = 0
    fallback_voiceover: bool = False
    device: str = "auto"
    tts_device: str | None = None
    translate_device: str | None = None
    max_tempo: float = 1.25
    subs: str = "auto"
    download: bool = True
    search_online: bool = False
    providers: list[str] | None = None
    subtitle_credentials_env: dict[str, str] = field(default_factory=lambda: {
        "username": "OPENSUBTITLES_USERNAME", "password": "OPENSUBTITLES_PASSWORD",
        "apikey": "OPENSUBTITLES_APIKEY",
    })

    def validate(self) -> None:
        for key, choices in (("source_lang", {"auto", "en", "vi"}), ("tts", {"edge", "vieneu", "gemini", "voxcpm2"}),
                             ("pipeline_mode", {"sequential", "overlap"}),
                             ("tts_precision", {"fp32", "int8"}),
                             ("mix", {"dub-only", "voiceover", "replace"}),
                             ("translator", {"opus-mt", "agy", "agy-stream", "gemini"}),
                             ("subs", {"auto", "local", "download", "translate"})):
            if getattr(self, key) not in choices:
                raise JellyDubError(f"Invalid {key}: {getattr(self, key)!r}; choose {', '.join(sorted(choices))}")
        if self.pipeline_mode == "overlap" and (self.source_lang not in {"en", "vi"}
                or (self.source_lang == "en" and self.translator != "opus-mt")):
            raise JellyDubError("overlap requires explicit en/vi; English overlap currently supports opus-mt only")
        if type(self.tts_workers) is not int or self.tts_workers < 1:
            raise JellyDubError("tts_workers must be a positive integer")
        for key in ("tts_blas_threads", "tts_cpu_threads"):
            if type(getattr(self, key)) is not int or getattr(self, key) < 0:
                raise JellyDubError(f"{key} must be a non-negative integer")
        if type(self.audio_stream) is not int or self.audio_stream < 0:
            raise JellyDubError("audio_stream must be a non-negative integer")
        if not isinstance(self.max_tempo, (float, int)) or not 1 <= self.max_tempo <= 2:
            raise JellyDubError("max_tempo must be between 1 and 2")
        for key in ("mkv", "mp4", "audio", "fallback_voiceover", "download", "search_online"):
            if type(getattr(self, key)) is not bool:
                raise JellyDubError(f"{key} must be a boolean")
        for key in ("rate", "pitch", "device", "subtitle_font", "asr_model", "asr_device"):
            if not isinstance(getattr(self, key), str):
                raise JellyDubError(f"{key} must be a string")
        if not self.subtitle_font.strip():
            raise JellyDubError("subtitle_font must be a non-empty string")
        if not self.asr_model.strip():
            raise JellyDubError("asr_model must be a non-empty string")
        if self.asr_device not in {"auto", "cpu", "cuda", "gpu"}:
            raise JellyDubError("asr_device must be auto, cpu, cuda or gpu")
        for key in ("tts_device", "translate_device"):
            if getattr(self, key) not in {None, "auto", "cpu", "cuda", "gpu"}:
                raise JellyDubError(f"{key} must be auto, cpu, cuda or gpu")
        if self.asr_backend not in {"auto", "faster-whisper", "openvino", "genai"}:
            raise JellyDubError("asr_backend must be auto, faster-whisper, openvino or genai")
        if self.asr_python is not None and (not isinstance(self.asr_python, str) or not self.asr_python.strip()):
            raise JellyDubError("asr_python must be a nonempty interpreter path")
        if type(self.asr_beam_size) is not int or self.asr_beam_size < 1:
            raise JellyDubError("asr_beam_size must be a positive integer")
        if self.translator_model is not None and (
                not isinstance(self.translator_model, str) or not self.translator_model.strip()):
            raise JellyDubError("translator_model must be a nonempty string")
        if self.tts_model is not None and (not isinstance(self.tts_model, str) or not self.tts_model.strip()):
            raise JellyDubError("tts_model must be a nonempty string")
        if self.speech_plan is not None and (not isinstance(self.speech_plan, str) or not self.speech_plan.strip()):
            raise JellyDubError("speech_plan must be a nonempty JSON path")
        if self.voxcpm2_binary is not None and (not isinstance(self.voxcpm2_binary, str) or not self.voxcpm2_binary.strip()):
            raise JellyDubError("voxcpm2_binary must be a nonempty executable path")
        if type(self.voxcpm2_timeout) is not int or self.voxcpm2_timeout < 1:
            raise JellyDubError("voxcpm2_timeout must be a positive integer")
        if self.tts == "voxcpm2" and (self.voice is not None or self.speaker_voices):
            raise JellyDubError("VoxCPM2 uses reference audio or voice design instructions, not named voice presets")
        if not isinstance(self.tts_style, str):
            raise JellyDubError("tts_style must be a string")
        if self.tts == "gemini" and self.reference:
            raise JellyDubError("Gemini uses named voices; reference cloning is not supported")
        if self.voice is not None and not isinstance(self.voice, str):
            raise JellyDubError("voice must be a string")
        if (not isinstance(self.speaker_voices, dict) or
                any(not isinstance(k, str) or not k.strip() or not isinstance(v, str) or not v.strip()
                    for k, v in self.speaker_voices.items())):
            raise JellyDubError("speaker_voices must map speaker labels to nonempty voice names")
        if self.reference is not None and (not isinstance(self.reference, list) or
                                          any(not isinstance(v, str) for v in self.reference)):
            raise JellyDubError("reference must be an array of paths or LABEL=PATH strings")
        if self.providers is not None and (not isinstance(self.providers, list) or not self.providers or
                any(not isinstance(v, str) or not v.strip() for v in self.providers)):
            raise JellyDubError("providers must be a nonempty array of provider names")
        import re
        if (not isinstance(self.subtitle_credentials_env, dict) or
                any(k not in {"username", "password", "apikey"} or not isinstance(v, str) or
                    not re.fullmatch(r"[A-Za-z_][A-Za-z0-9_]*", v)
                    for k, v in self.subtitle_credentials_env.items())):
            raise JellyDubError("subtitle_credentials_env must map username/password/apikey to environment variable names")
        self.cache_dir = Path(self.cache_dir).expanduser()
        if self.out_dir is not None:
            self.out_dir = Path(self.out_dir).expanduser()


def load_config(path: Path | None = None, **overrides) -> Config:
    """Load [jellydub] TOML settings; non-None CLI values take precedence."""
    values = {}
    if path is not None:
        try:
            with path.open("rb") as handle:
                document = tomllib.load(handle)
            values = document.get("jellydub", document)
        except (OSError, tomllib.TOMLDecodeError) as exc:
            raise JellyDubError(f"Cannot read config {path}: {exc}") from exc
        if not isinstance(values, dict):
            raise JellyDubError("[jellydub] must be a TOML table")
    known = {f.name for f in fields(Config)}
    unknown = (set(values) | set(overrides)) - known
    if unknown:
        raise JellyDubError(f"Unknown configuration options: {', '.join(sorted(unknown))}")
    values = {**values, **{k: v for k, v in overrides.items() if v is not None}}
    try:
        config = Config(**values)
        config.validate()
    except (TypeError, ValueError) as exc:
        raise JellyDubError(f"Invalid configuration: {exc}") from exc
    return config
