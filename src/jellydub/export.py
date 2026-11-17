"""Atomic Jellyfin-compatible audio and Matroska exports."""
from __future__ import annotations

import os
from pathlib import Path

from .cache import tmp_path_for
from .ffmpeg import probe, run_ffmpeg
from .models import JellyDubError


def output_paths(video: Path, out_dir: Path | None = None, label: str = "AI-Dub") -> dict[str, Path]:
    """Return Vietnamese subtitle, external MP3 and optional video destinations."""
    if not label or any(c in label for c in '/\\\x00') or label in {'.', '..'}:
        raise JellyDubError("Output label must be a non-empty filename component")
    folder = out_dir if out_dir is not None else video.parent
    paths = {
        "srt": folder / f"{video.stem}.vi.srt",
        "audio": folder / f"{video.stem}.{label}.vi.mp3",
        "mkv": folder / f"{video.stem}.{label}.vi.mkv",
        "mp4": folder / f"{video.stem}.{label}.vi.mp4",
    }
    if any(p.resolve() == video.resolve() for p in paths.values()):
        raise JellyDubError("Export path would overwrite the source")
    return paths


def _export(args: list[str], output: Path, inputs: list[Path]) -> None:
    if any(output.resolve() == p.resolve() for p in inputs):
        raise JellyDubError("Export must not overwrite an input")
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = tmp_path_for(output)
    try:
        run_ffmpeg([*args, str(temporary)])
        os.replace(temporary, output)
    finally:
        temporary.unlink(missing_ok=True)


def export_mp3(mixed_wav: Path, out_mp3: Path, bitrate: str = "192k") -> None:
    """Encode stereo MP3 atomically."""
    _export(["-i", str(mixed_wav), "-map", "0:a:0", "-vn", "-ar", "48000", "-ac", "2", "-c:a", "libmp3lame", "-b:a", bitrate], out_mp3, [mixed_wav])


def mux_mkv(
    video: Path,
    dub_audio: Path,
    subtitle: Path,
    out_mkv: Path,
    title: str = "Tiếng Việt (AI Dub)",
    subtitle_title: str = "Phụ đề tiếng Việt",
) -> None:
    """Copy original streams and append selectable Vietnamese audio and subtitle tracks."""
    streams = probe(video)["streams"]
    audio_count = sum(s.get("codec_type") == "audio" for s in streams)
    subtitle_count = sum(s.get("codec_type") == "subtitle" for s in streams)
    defaults = []
    for kind, count in (("a", audio_count), ("s", subtitle_count)):
        for index in range(count):
            defaults.extend([f"-disposition:{kind}:{index}", "-default"])
    _export([
        "-i", str(video), "-i", str(dub_audio), "-i", str(subtitle),
        "-map", "0", "-map", "1:a:0", "-map", "2:0",
        "-map_metadata", "0", "-map_chapters", "0", "-c", "copy", *defaults,
        f"-c:a:{audio_count}", "aac", f"-b:a:{audio_count}", "192k",
        f"-metadata:s:a:{audio_count}", "language=vie",
        f"-metadata:s:a:{audio_count}", f"title={title}",
        f"-disposition:a:{audio_count}", "default",
        f"-metadata:s:s:{subtitle_count}", "language=vie",
        f"-metadata:s:s:{subtitle_count}", f"title={subtitle_title}",
        f"-disposition:s:{subtitle_count}", "default",
    ], out_mkv, [video, dub_audio, subtitle])


def _filter_escape(value: str) -> str:
    return value.replace("\\", "\\\\").replace("'", "\\'")


def export_mp4(
    video: Path,
    dub_audio: Path,
    subtitle: Path,
    out_mp4: Path,
    subtitle_font: str = "DejaVu Sans",
) -> None:
    """Encode an H.264/AAC MP4 with the Vietnamese SRT rendered into the picture."""
    if not subtitle_font.strip():
        raise JellyDubError("Subtitle font must be a non-empty string")
    subtitle_filter = (
        f"subtitles=filename='{_filter_escape(subtitle.resolve().as_posix())}':"
        f"force_style='FontName={_filter_escape(subtitle_font)},FontSize=16,"
        "Outline=2,Shadow=0,MarginV=48,MarginL=24,MarginR=24,Alignment=2'"
    )
    _export([
        "-i", str(video), "-i", str(dub_audio),
        "-map", "0:v:0", "-map", "1:a:0",
        "-vf", subtitle_filter,
        "-c:v", "libx264", "-preset", "medium", "-crf", "20", "-pix_fmt", "yuv420p",
        "-c:a", "aac", "-b:a", "192k", "-movflags", "+faststart",
        "-metadata:s:a:0", "language=vie",
        "-metadata:s:a:0", "title=Tiếng Việt (AI Dub)",
    ], out_mp4, [video, dub_audio, subtitle])
