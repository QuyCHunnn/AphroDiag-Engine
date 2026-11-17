# Gemini TTS

VieNeu is the default TTS provider. Select cloud Gemini explicitly instead of local VieNeu:

```bash
read -rsp 'Gemini API key: ' GEMINI_API_KEY
export GEMINI_API_KEY
printf '\n'
.venv/bin/jellydub auto video.mp4 --config jellydub.gemini.toml
unset GEMINI_API_KEY
```

The key is read from `GEMINI_API_KEY` (or `GOOGLE_API_KEY`) at request time. It is never a configuration field or cache identifier. CLI overrides include `--tts gemini`, `--tts-model`, `--tts-style`, and `--voice`; the same configuration works with `preview`, `dub`, `auto`, `serve`, and `stream`.

The supplied profile selects `gemini-3.8-flash-lite-tts`, Charon, two concurrent requests, voiceover mixing, and MKV/SRT output. Translation remains OPUS; ASR remains local. To preserve a machine-specific GPU ASR setup, add these TTS options to that setup or pass its ASR options explicitly. Cloud speech speed depends on API latency and quota, rather than the local GPU.

Optional manual character mapping:

```toml
[jellydub.speaker_voices]
PETER = "Charon"
LOIS = "Kore"
```

These are named Gemini voices, not cloned character voices. VieNeu reference WAV files are rejected for this provider. Cache identities include model, style and resolved voice per speaker, so changing one character preserves other characters' speech. Existing VieNeu audio cannot be reused as Gemini audio.

Modern models use the official Interactions API with transcript and style metadata separated. Earlier 2.5/3.1 model IDs use GenerateContent. HTTP failures are sanitized; transient errors retry with bounded backoff and Retry-After support. Redirects are disabled to avoid forwarding credentials. Responses are validated and written atomically, then the common pipeline normalizes the audio and applies cue timing.

## Verified sample, 2026-11-17

`artifacts/gemini-tts/sample.vi.wav`: live Flash-Lite/Charon Vietnamese request succeeded in **3.419 s** for **7.840 s** of audio (24 kHz PCM WAV). Full FFmpeg decode passed. This is a single request, not a complete-video benchmark; listening acceptance remains pending. Timing excludes ASR, translation, normalization, cue fitting and export.

Reproduce a small live request without saving the key:

```bash
.venv/bin/python scripts/preview_gemini_tts.py
```

Official API reference: https://aistudio.google.com/docs/speech-generation

Validation: the full suite passed **516 tests**, with 5 network/model tests deselected by the repository's existing marker policy. After adding two integration checks, all **18 Gemini tests** passed separately, including common-pipeline 48 kHz normalization, cache reuse without credentials, and Retry-After handling. `git diff --check` and CLI `auto --help` passed. Human listening and whole-video latency are still pending.

To return to the default local engine, omit the Gemini profile and run `.venv/bin/jellydub auto video.mp4 --tts vieneu`. No automatic Gemini/VieNeu fallback is enabled; backend selection remains explicit for consistent voices.
