"""Resumable per-cue speech synthesis and normalization."""
from __future__ import annotations

import os
import json
import shutil
import random
import re
import threading
from concurrent.futures import ThreadPoolExecutor, as_completed
import time
from pathlib import Path
from typing import Callable

from ..cache import atomic_write_text, cache_key, cache_lock, file_digest, tmp_path_for
from ..ffmpeg import duration, run_ffmpeg
from ..models import Cue, SynthResult, SAMPLE_RATE
from .base import TTSError, TTSProvider
from ..performance_trace import span


def _valid_duration(path):
    if not path.is_file() or path.stat().st_size == 0:
        return None
    try:
        value = duration(path)
        return value if value > 0 else None
    except Exception:
        return None


def _speech_metadata(path, measured):
    atomic_write_text(path.with_suffix(".json"), json.dumps({
        "version": 1, "sha256": file_digest(path), "duration": measured,
    }))


def _cached_duration(path, *, adopt_legacy=True):
    measured = _valid_duration(path)
    if measured is None:
        return None
    manifest = path.with_suffix(".json")
    if manifest.exists():
        try:
            saved = json.loads(manifest.read_text())
            if (saved.get("version") != 1 or saved.get("sha256") != file_digest(path)
                    or saved.get("duration") != measured):
                return None
        except (OSError, ValueError, AttributeError):
            return None
    elif adopt_legacy:
        # Legacy normalized WAV: import once, then protect future reuse by hash.
        _speech_metadata(path, measured)
    return measured


def _has_speakable_content(text: str) -> bool:
    """Ignore symbol-only cues and lines consisting entirely of annotations.

    This is only a classification check: spoken text and its cache key retain
    their original spelling, whitespace, and markers.
    """
    for line in text.splitlines():
        line = line.strip(" \t*♪♫#~−–—-")
        if re.fullmatch(r"(?:(?:\[[^\[\]]*\]|\([^()]*\))\s*)+", line):
            continue
        if any(char.isalpha() or char.isdigit() for char in line):
            return True
    return False


class _AdaptiveLimiter:
    """Limit new requests; already running calls finish when the limit shrinks."""

    def __init__(self, cap):
        self.cap = self.allowed = cap
        self.active = self.successes = 0
        self.condition = threading.Condition()

    def acquire(self):
        with self.condition:
            self.condition.wait_for(lambda: self.active < self.allowed)
            self.active += 1

    def release(self, *, success=False, transient=False):
        with self.condition:
            self.active -= 1
            if transient:
                self.allowed = max(1, self.allowed // 2)
                self.successes = 0
            elif success:
                self.successes += 1
                if self.successes >= 20:
                    self.allowed = min(self.cap, self.allowed + 1)
                    self.successes = 0
            else:
                self.successes = 0
            self.condition.notify_all()


def synthesize_cues(
    cues: list[Cue], provider: TTSProvider, cache_dir: Path, retries: int = 4,
    on_progress: Callable[[int, int], None] | None = None, *, workers: int = 1,
    speech_plan: str | None = None,
) -> tuple[list[SynthResult], list[tuple[Cue, str]]]:
    """Synthesize unique lines concurrently, returning results/failures in cue order.

    Progress runs only on the calling thread, once per completed cue (including
    duplicates and non-speech cues). Provider concurrency defaults to one for safety.
    """
    from ..speech import prepare_speech
    spoken = prepare_speech(cues, provider, speech_plan)
    cache_dir = Path(cache_dir)
    cache_dir.mkdir(parents=True, exist_ok=True)
    cap = max(1, min(workers, getattr(provider, "max_concurrency", 1)))
    limiter = _AdaptiveLimiter(cap)
    groups = {}
    outcomes = {}
    completed = 0

    def progress(amount):
        nonlocal completed
        for _ in range(amount):
            completed += 1
            if on_progress:
                on_progress(completed, len(cues))

    for cue in cues:
        if _has_speakable_content(spoken[cue.index][0]):
            groups.setdefault(spoken[cue.index], []).append(cue)
        else:
            progress(1)

    identity_for = getattr(provider, "cache_id_for", lambda speaker: provider.cache_id())
    legacy_for = getattr(provider, "legacy_cache_ids_for", lambda speaker: [])
    identities = {speaker: identity_for(speaker) for _, speaker in groups}
    legacy_ids = {speaker: legacy_for(speaker) for _, speaker in groups}

    def generate_locked(key, target):
        text, speaker = key
        raw = target.with_suffix(".raw.wav")
        temporary = tmp_path_for(target)
        measured = _cached_duration(target)
        if measured is not None:
            return target, measured, None
        for attempt in range(max(0, retries) + 1):
            success = transient = False
            error = None
            retry_after = 0
            limiter.acquire()
            try:
                if getattr(provider, "is_remote", False):
                    time.sleep(random.uniform(0.1, 0.3))
                with span("tts.synthesize", provider=provider.name):
                    provider.synthesize(text, raw, speaker=speaker)
                with span("tts.normalize"):
                    run_ffmpeg(["-i", str(raw), "-af",
                            "silenceremove=start_periods=1:start_duration=0.01:start_threshold=-50dB,"
                            "areverse,silenceremove=start_periods=1:start_duration=0.01:start_threshold=-50dB,areverse",
                            "-ar", str(SAMPLE_RATE), "-ac", "1", "-c:a", "pcm_s16le", str(temporary)])
                measured = _valid_duration(temporary)
                if measured is None:
                    raise TTSError("TTS produced missing or silent audio", transient=True)
                os.replace(temporary, target)
                _speech_metadata(target, measured)
                success = True
            except Exception as exc:
                transient = isinstance(exc, TTSError) and exc.transient
                error = str(exc)
                retry_after = getattr(exc, "retry_after_seconds", 0)
            finally:
                limiter.release(success=success, transient=transient)
                raw.unlink(missing_ok=True)
                temporary.unlink(missing_ok=True)
            if success:
                return target, measured, None
            if not transient or attempt >= max(0, retries):
                return target, None, error
            time.sleep(max(retry_after, min(2 ** attempt, 8)) + random.uniform(0, 1))

    legacy_dirs = list(cache_dir.parent.glob("*/tts")) if cache_dir.name == "tts" else []

    def generate(key):
        text, speaker = key
        filename = cache_key(identities[speaker], text, speaker) + ".wav"
        target = cache_dir / filename
        with cache_lock(target.with_suffix(".lock")):
            if not target.exists():
                filenames = dict.fromkeys([filename] + [cache_key(identity, text, speaker) + ".wav"
                                                       for identity in legacy_ids[speaker]])
                candidates = (folder / name for folder in [cache_dir, *legacy_dirs] for name in filenames)
                for candidate in candidates:
                    if candidate != target and _cached_duration(candidate, adopt_legacy=False) is not None:
                        temporary = tmp_path_for(target)
                        try:
                            shutil.copy2(candidate, temporary)
                            os.replace(temporary, target)
                        finally:
                            temporary.unlink(missing_ok=True)
                        break
            return generate_locked(key, target)

    with ThreadPoolExecutor(max_workers=cap) as executor:
        pending = {executor.submit(generate, key): key for key in groups}
        for future in as_completed(pending):
            key = pending[future]
            try:
                outcomes[key] = future.result()
            except Exception as exc:
                outcomes[key] = (None, None, str(exc))
            progress(len(groups[key]))

    results, failures = [], []
    for cue in cues:
        if not _has_speakable_content(spoken[cue.index][0]):
            continue
        target, measured, error = outcomes[spoken[cue.index]]
        if error is None:
            results.append(SynthResult(cue.index, target, measured))
        else:
            failures.append((cue, error))
    return results, failures
