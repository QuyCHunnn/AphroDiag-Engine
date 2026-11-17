"""Small, reproducible listening samples using the full pipeline's voice cache."""
from dataclasses import replace
import json
import math
from pathlib import Path

from .cache import atomic_write_text, file_digest
from .config import Config
from .models import JellyDubError
from .pipeline import tts_options
from .subtitles import parse_srt, write_srt
from .timing import align, render_timeline
from .tts import get_provider
from .tts.synthesize import _has_speakable_content, synthesize_cues
from .validate import check_audio


def create_preview(srt: Path, out_dir: Path, config: Config, *, start=0.0,
                   seconds=20.0, report=print) -> Path:
    """Render complete cues starting in a window; never truncate spoken audio.

    Input is reviewed Vietnamese text. The manifest records pending human
    acceptance; technical QC cannot approve pronunciation or acting.
    """
    config.validate()
    if not math.isfinite(start) or start < 0 or not math.isfinite(seconds) or not 1 <= seconds <= 30:
        raise JellyDubError("Preview requires start >= 0 and seconds between 1 and 30")
    cues = [replace(c, start=c.start-start, end=c.end-start)
            for c in parse_srt(srt) if start <= c.start < start+seconds]
    if not any(_has_speakable_content(c.text) for c in cues):
        raise JellyDubError("No speech in the selected preview window")
    out_dir.mkdir(parents=True, exist_ok=True)
    outputs = [out_dir / name for name in ("preview.wav", "preview.vi.srt", "review.json")]
    if srt.resolve() in [p.resolve() for p in outputs]:
        raise JellyDubError("Preview output would overwrite the input subtitle")
    # Invalidate an older review before starting potentially failing synthesis.
    manifest = {"status": "generating", "input_sha256": file_digest(srt),
                "start": start, "requested_seconds": seconds}
    atomic_write_text(outputs[2], json.dumps(manifest, indent=2))
    provider = get_provider(config.tts, **tts_options(config, report=report))
    try:
        identities = {str(c.index): provider.cache_id_for(c.speaker)
                      if hasattr(provider, "cache_id_for") else provider.cache_id() for c in cues}
        results, failures = synthesize_cues(cues, provider, config.cache_dir / "tts",
                                          workers=config.tts_workers)
    finally:
        manifest["execution"] = getattr(provider, "execution", None)
        provider.close()
    if failures:
        raise JellyDubError(f"{len(failures)} preview segments failed; successful speech is cached")
    clips = align(cues, results, config.cache_dir / "aligned", max_tempo=config.max_tempo)
    total = max(seconds, max(c.end for c in cues), max(c.start+c.duration for c in clips))
    render_timeline(clips, total, outputs[0])
    problems = check_audio(outputs[0], total)
    if problems:
        raise JellyDubError(f"Preview audio QC failed: {problems}")
    write_srt(cues, outputs[1])
    manifest.update(status="pending_listening_review", duration=total,
                    audio_sha256=file_digest(outputs[0]), provider=config.tts,
                    voice_identities=identities,
                    speakers=sorted({c.speaker or "default" for c in cues}),
                    cues=[{"index": c.index, "speaker": c.speaker, "text": c.text} for c in cues],
                    timing_warnings=[c.warning for c in clips if c.warning],
                    criteria=["pronunciation", "meaning", "speaker identity", "emotion", "rhythm"])
    atomic_write_text(outputs[2], json.dumps(manifest, ensure_ascii=False, indent=2))
    report(f"Listening review pending: {outputs[0]} ({len(manifest['speakers'])} speakers)")
    return outputs[0]
