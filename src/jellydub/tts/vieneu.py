"""Local VieNeu v3 Turbo SDK adapter with reusable manual voice profiles."""
from __future__ import annotations

import hashlib
import json
import os
import warnings
import threading
from contextlib import nullcontext
from pathlib import Path

from ..cache import cache_key, tmp_path_for
from ..ffmpeg import duration
from .base import TTSError, TTSProvider
from ..performance_trace import span


class VieNeuTTS(TTSProvider):
    name = "vieneu"
    max_concurrency = 1
    _thread_limit_lock = threading.RLock()

    def __init__(self, voice="Hải Đăng", reference=None, speaker_refs=None,
                 cache_dir=None, device="cpu", speaker_voices=None, blas_threads=2, cpu_threads=0,
                 precision="fp32"):
        self.voice = voice
        self.speaker_voices = dict(speaker_voices or {})
        self.reference = Path(reference) if reference else None
        self.speaker_refs = {key: Path(value) for key, value in (speaker_refs or {}).items()}
        self.cache_dir = Path(cache_dir) if cache_dir else Path.home() / ".cache/jellydub/voices"
        self.device = device
        if precision not in {"fp32", "int8"} or (precision == "int8" and device != "cpu"):
            raise ValueError("VieNeu precision must be fp32, or int8 with device=cpu")
        self.precision = precision
        self.blas_threads, self.cpu_threads = blas_threads, cpu_threads
        for name, value in (("blas_threads", blas_threads), ("cpu_threads", cpu_threads)):
            if type(value) is not int or value < 0:
                raise ValueError(f"{name} must be a non-negative integer")
        self._model = None
        self._profiles = {}
        self.execution = {"requested_device": device, "loaded": False}

    @staticmethod
    def _fingerprint(path):
        if path is None:
            return None
        try:
            return hashlib.sha256(path.read_bytes()).hexdigest()
        except OSError:
            return f"missing:{path}"

    def cache_id(self):
        suffix = ":int8" if self.precision == "int8" else ""
        if self.speaker_voices:
            return "vieneu-v3turbo:speakers:" + cache_key(
                self.voice, self.device, self.speaker_voices, self._fingerprint(self.reference),
                {k: self._fingerprint(v) for k, v in self.speaker_refs.items()}) + suffix
        return "vieneu-v3turbo:" + cache_key(
            self.voice, self.device, self._fingerprint(self.reference),
            {k: self._fingerprint(v) for k, v in self.speaker_refs.items()}) + suffix

    def cache_id_for(self, speaker):
        ref = self.speaker_refs.get(speaker) if speaker else self.reference
        return "vieneu-v3turbo:cue-v1:" + cache_key(
            self.speaker_voices.get(speaker, self.voice), self.device, self._fingerprint(ref)) + (
                ":int8" if self.precision == "int8" else "")

    def legacy_cache_ids_for(self, speaker):
        # Only the exact old configuration is safe to import. Do not guess
        # which voice another job's mapping resolved to.
        return [self.cache_id()]

    def _load(self):
        if self._model is None:
            from vieneu import Vieneu
            options = {"mode": "v3turbo", "device": self.device}
            if self.cpu_threads:
                options["threads"] = self.cpu_threads
            if self.precision != "fp32":
                options["precision"] = self.precision
            self._model = Vieneu(**options)
            backend = getattr(self._model, "backend", "unknown")
            engine = getattr(self._model, "engine", None)
            session = getattr(engine, "sess_pre", None)
            self.execution.update(loaded=True, backend=backend,
                                  device=str(getattr(engine, "device", self.device)),
                                  providers=session.get_providers() if session else [],
                                  precision=self.precision,
                                  graph_path=str(getattr(session, "_model_path", "unknown")),
                                  blas_threads=self.blas_threads, cpu_threads=self.cpu_threads)
        return self._model

    def _voice(self, model, speaker):
        preset = self.speaker_voices.get(speaker, self.voice)
        ref = self.speaker_refs.get(speaker) if speaker else self.reference
        if speaker and ref is None:
            if speaker in self.speaker_voices:
                return preset
            warnings.warn(f"No reference for speaker {speaker!r}; using preset {self.voice}.", stacklevel=2)
            return self.voice
        if ref is None:
            return preset
        try:
            valid = ref.is_file() and 3 <= duration(ref) <= 8
        except Exception:
            valid = False
        if not valid:
            warnings.warn(f"Reference {ref} must be a readable 3–8 second clip; using preset {preset}.", stacklevel=2)
            return preset
        key = cache_key("v3turbo", self._fingerprint(ref))
        if key in self._profiles:
            return self._profiles[key]
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        profile_path = self.cache_dir / f"{key}.json"
        profile = None
        if profile_path.exists():
            try:
                import numpy as np  # provided by the optional SDK, never imported for Edge
                data = json.loads(profile_path.read_text(encoding="utf-8"))["presets"][key]
                data["speaker_emb"] = np.asarray(data["speaker_emb"], dtype=np.float32)
                if data.get("codes") is not None:
                    data["codes"] = np.asarray(data["codes"], dtype=np.int64)
                profile = data
            except (ValueError, KeyError, TypeError):
                pass
        if profile is None:
            model.add_voice(key, str(ref))
            temporary = tmp_path_for(profile_path)
            try:
                model.save_voices(str(temporary))
                os.replace(temporary, profile_path)
            finally:
                temporary.unlink(missing_ok=True)
            profile = model.get_preset_voice(key)
        self._profiles[key] = profile
        return profile

    def synthesize(self, text: str, out_path: Path, speaker=None):
        try:
            # Native BLAS limits are process-wide. Serialize our scoped limits
            # and restore them on success and exceptions, including save errors.
            with self._thread_limit_lock:
                with span("tts.load", precision=self.precision):
                    model = self._load()
                limits = nullcontext()
                if self.blas_threads and (self.device == "cpu" or getattr(model, "backend", None) == "onnx"):
                    from threadpoolctl import threadpool_limits
                    limits = threadpool_limits(limits=self.blas_threads, user_api="blas")
                with limits:
                    with span("tts.infer", precision=self.precision):
                        audio = model.infer(text, voice=self._voice(model, speaker))
                    model.save(audio, str(out_path))
        except Exception as exc:
            raise TTSError(f"VieNeu-TTS failed (install the vieneu extra for local synthesis): {exc}") from exc

    def close(self):
        model, self._model = self._model, None
        self._profiles.clear()
        if model is not None:
            model.close()
