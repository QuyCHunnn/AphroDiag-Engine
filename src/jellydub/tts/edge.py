"""Microsoft Edge's online TTS service (no voice cloning)."""
from __future__ import annotations

import asyncio
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

from ..cache import cache_key
from .base import TTSError, TTSProvider


class EdgeTTS(TTSProvider):
    name = "edge"
    max_concurrency = 8
    is_remote = True

    def __init__(self, voice="vi-VN-HoaiMyNeural", rate="+0%", pitch="+0Hz", volume="+0%", speaker_voices=None):
        self.voice, self.rate, self.pitch, self.volume = voice, rate, pitch, volume
        self.speaker_voices = dict(speaker_voices or {})

    def cache_id(self):
        if self.speaker_voices:
            return "edge:speakers:" + cache_key(self.voice, self.rate, self.pitch, self.volume, self.speaker_voices)
        return "edge:" + cache_key(self.voice, self.rate, self.pitch, self.volume)

    def cache_id_for(self, speaker):
        return "edge:" + cache_key(self.speaker_voices.get(speaker, self.voice),
                                    self.rate, self.pitch, self.volume)

    def legacy_cache_ids_for(self, speaker):
        return [self.cache_id()]

    def synthesize(self, text: str, out_path: Path, speaker=None):
        async def generate():
            import edge_tts
            voice = self.speaker_voices.get(speaker, self.voice)
            await edge_tts.Communicate(text, voice, rate=self.rate,
                                       pitch=self.pitch, volume=self.volume).save(str(out_path))
        try:
            try:
                asyncio.get_running_loop()
            except RuntimeError:
                asyncio.run(generate())
            else:
                # The provider is synchronous, including when called from an async CLI host.
                with ThreadPoolExecutor(max_workers=1) as executor:
                    executor.submit(lambda: asyncio.run(generate())).result()
        except Exception as exc:
            status = getattr(exc, "status", None)
            transient = (isinstance(exc, (OSError, TimeoutError)) or
                         type(exc).__name__ in {"NoAudioReceived", "ClientConnectionError", "ServerDisconnectedError", "ClientPayloadError", "ServerTimeoutError"} or
                         (isinstance(status, int) and (status >= 500 or status in {408, 429})))
            raise TTSError(f"Edge-TTS failed: {exc}", transient=transient) from exc
