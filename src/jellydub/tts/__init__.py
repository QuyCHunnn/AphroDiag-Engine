"""Lazy TTS provider selection."""
from .base import TTSError, TTSProvider
from ..models import JellyDubError

__all__ = ["TTSError", "TTSProvider", "get_provider"]


def get_provider(name: str, **options) -> TTSProvider:
    if name == "edge":
        from .edge import EdgeTTS
        return EdgeTTS(**options)
    if name == "vieneu":
        from .vieneu import VieNeuTTS
        return VieNeuTTS(**options)
    if name == "voxcpm2":
        from .voxcpm2 import VoxCPM2TTS
        return VoxCPM2TTS(**options)
    if name == "gemini":
        from .gemini import GeminiTTS
        return GeminiTTS(**options)
    raise JellyDubError(f"Unknown TTS provider: {name}")
