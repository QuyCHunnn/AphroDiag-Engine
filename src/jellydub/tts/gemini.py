"""Gemini speech generation over the official REST API; secrets stay in headers."""
import base64
import binascii
import io
import json
import os
from pathlib import Path
import re
import urllib.error
import urllib.request
import wave

from ..cache import cache_key, tmp_path_for
from .base import TTSError, TTSProvider

DEFAULT_MODEL = "gemini-3.8-flash-lite-tts"


class _NoRedirect(urllib.request.HTTPRedirectHandler):
    def redirect_request(self, req, fp, code, msg, headers, newurl):
        return None


def _open(request, timeout):
    return urllib.request.build_opener(_NoRedirect()).open(request, timeout=timeout)


class GeminiTTS(TTSProvider):
    name = "gemini"
    max_concurrency = 2
    is_remote = True

    def __init__(self, voice="Charon", model=DEFAULT_MODEL, style="", speaker_voices=None):
        if not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9._-]*", model):
            raise ValueError("Invalid Gemini TTS model id")
        self.voice, self.model, self.style = voice, model, style
        self.speaker_voices = dict(speaker_voices or {})
        for value in [voice, *self.speaker_voices.values()]:
            if not isinstance(value, str) or not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_-]*", value):
                raise ValueError("Invalid Gemini voice name")
        if not isinstance(style, str):
            raise ValueError("Gemini TTS style must be a string")

    def cache_id(self):
        return "gemini-tts:v1:"+cache_key(self.model, self.voice, self.style, self.speaker_voices)

    def cache_id_for(self, speaker):
        return "gemini-tts:cue-v1:"+cache_key(self.model, self.speaker_voices.get(speaker, self.voice), self.style)

    def _request(self, text, voice):
        if self.model.startswith("gemini-3.8-"):
            part = {"type": "text", "text": text}
            if self.style:
                part["annotations"] = [{"type": "speech_metadata", "style": self.style}]
            return "https://generativelanguage.googleapis.com/v1beta/interactions", {
                "model": self.model, "input": [{"type": "user_input", "content": [part]}],
                "response_format": {"type": "audio", "mime_type": "audio/wav"},
                "generation_config": {"speech_config": [{"voice": voice}]}, "store": False,
            }
        # Earlier TTS models use GenerateContent and return headerless PCM.
        prompt = (self.style+"\nRead only the following transcript:\n"+text) if self.style else text
        return f"https://generativelanguage.googleapis.com/v1beta/models/{self.model}:generateContent", {
            "contents": [{"parts": [{"text": prompt}]}],
            "generationConfig": {"responseModalities": ["AUDIO"],
                                 "speechConfig": {"voiceConfig": {"prebuiltVoiceConfig": {"voiceName": voice}}}},
        }

    @staticmethod
    def _audio(payload, modern):
        if modern:
            if payload.get("status") not in (None, "completed"):
                raise ValueError("Incomplete interaction")
            parts = [p for step in payload.get("steps", []) if step.get("type") == "model_output"
                     for p in step.get("content", []) if p.get("type") == "audio" and p.get("data")]
            if len(parts) != 1:
                raise ValueError("Missing or ambiguous audio")
            return base64.b64decode(parts[0]["data"], validate=True), parts[0].get("mime_type", "audio/wav")
        candidates = payload.get("candidates", [])
        if not candidates or candidates[0].get("finishReason") not in (None, "STOP"):
            raise ValueError("Missing audio candidate")
        parts = [p["inlineData"] for p in candidates[0]["content"]["parts"] if p.get("inlineData")]
        if len(parts) != 1:
            raise ValueError("Missing or ambiguous audio")
        return base64.b64decode(parts[0]["data"], validate=True), parts[0].get("mimeType", "")

    @staticmethod
    def _write_audio(path, data, mime):
        if mime.lower().split(";")[0] in {"audio/wav", "audio/x-wav", "audio/wave"}:
            with wave.open(io.BytesIO(data), "rb") as audio:
                expected = audio.getnframes()*audio.getnchannels()*audio.getsampwidth()
                if (audio.getcomptype() != "NONE" or audio.getnchannels() not in {1, 2}
                        or audio.getsampwidth() != 2 or not 8000 <= audio.getframerate() <= 96000
                        or expected <= 0 or len(audio.readframes(audio.getnframes())) != expected):
                    raise ValueError("Invalid WAV")
            path.write_bytes(data)
        elif mime.lower().startswith("audio/l16"):
            rate = re.search(r"(?:^|;)\s*rate=(\d+)", mime, re.I)
            rate = int(rate.group(1)) if rate else 24000
            if not 8000 <= rate <= 96000 or not data or len(data) % 2:
                raise ValueError("Invalid PCM")
            with wave.open(str(path), "wb") as audio:
                audio.setnchannels(1); audio.setsampwidth(2); audio.setframerate(rate)
                audio.writeframes(data)
        else:
            raise ValueError("Unsupported audio MIME type")

    def synthesize(self, text, out_path: Path, speaker=None):
        key = os.environ.get("GEMINI_API_KEY") or os.environ.get("GOOGLE_API_KEY")
        if not key:
            raise TTSError("Missing Gemini API key; set GEMINI_API_KEY")
        voice = self.speaker_voices.get(speaker, self.voice)
        url, body = self._request(text, voice)
        request = urllib.request.Request(url, data=json.dumps(body, ensure_ascii=False).encode(),
                                         headers={"Content-Type": "application/json", "x-goog-api-key": key}, method="POST")
        try:
            with _open(request, timeout=120) as response:
                raw = response.read(32*1024*1024+1)
            if len(raw) > 32*1024*1024:
                raise ValueError("Oversized response")
            payload = json.loads(raw)
        except urllib.error.HTTPError as exc:
            status = exc.code
            retry_after = exc.headers.get("Retry-After") if exc.headers else None
            exc.close()
            error = TTSError(f"Gemini TTS request failed (HTTP {status}); check API key, quota and model access",
                             transient=status in {408, 429, 500, 502, 503, 504})
            if error.transient and retry_after:
                from ..translate import _retry_after_seconds
                delay = _retry_after_seconds(retry_after)
                if delay <= 120:
                    error.retry_after_seconds = delay
                else:
                    error.transient = False
            raise error from None
        except (OSError, ValueError):
            raise TTSError("Gemini TTS connection or response failed", transient=True) from None
        temporary = tmp_path_for(out_path)
        try:
            data, mime = self._audio(payload, self.model.startswith("gemini-3.8-"))
            self._write_audio(temporary, data, mime)
            os.replace(temporary, out_path)
        except (ValueError, TypeError, KeyError, AttributeError, IndexError, EOFError, wave.Error, binascii.Error):
            raise TTSError("Gemini TTS returned missing, blocked or invalid audio", transient=True) from None
        finally:
            temporary.unlink(missing_ok=True)
