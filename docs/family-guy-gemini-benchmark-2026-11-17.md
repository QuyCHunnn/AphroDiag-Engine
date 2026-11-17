# Family Guy: fresh Gemini pipeline run, 2026-11-17

Result: **failed due to Gemini HTTP 429; no completed video**. The user subsequently confirmed that API requests were exhausted. No further API calls were made after that confirmation.

Source: `videos/Family Guy - Peter falls down the stairs.mp4`, duration **156.665034 s**. A byte copy was placed in a new input directory containing only that video, with no SRT or embedded subtitle track. Existing historical SRT files outside the input directory were not used or deleted. Processing cache started absent; downloaded model weights remained available. A fresh CLI process loaded all models. Subtitle downloads were disabled. ASR reported **0 region cache hits** and no fallback.

Configuration: native OpenVINO GenAI Whisper Small, Iris Xe GPU, beam=5, f32 execution; OPUS translation CPU; Gemini `gemini-3.8-flash-lite-tts`, Charon, two workers; overlap mode; voiceover/MKV output requested. No manually prepared ledger or transcript.

Measured elapsed CLI process time: **452.038 s (7 min 32.04 s)**, from before process launch through exit. There were no code fixes, manual restarts or debugging pauses during this interval. Automatic retries and pipeline fallback are included. This is the time to failure, **not a successful whole-pipeline processing time**. No output rendering/decode time can be claimed because TTS did not complete.

| Measurement | Seconds | Interpretation |
|---|---:|---|
| ASR completed from process start | 100.15 | New 41-cue English SRT |
| ASR load/compile | 13.99 | Included in ASR elapsed |
| ASR generation | 74.98 | Nine regions, no transcript cache hits |
| Translation load | 58.76 | Worker + fallback process loads |
| Translation generation | 35.39 | Across eight calls |
| Actual TTS request intervals, wall-time union | 53.72 | Excludes waiting between retries; incomplete speech |
| TTS normalization | 1.73 | 13 successful clips |
| Full CLI elapsed to failure | 452.04 | Includes quota waits, automatic fallback/retry and shutdown |

Stage measurements overlap and must not be added to reconstruct elapsed time. The trace summarizer's `asr_tts_infer_overlap_seconds=0` refers to the local `tts.infer` span, which cloud Gemini does not emit; it does not establish that cloud API requests never overlapped ASR.

The progressive worker failed at **257.86 s** after HTTP 429 for three cues, triggering normal fallback. Final translation completed at **283.68 s**. The first full TTS pass ended at **387.21 s** and retried missing segments. The last error was logged at **449.84 s**, followed by process shutdown. Overall, **13 segments succeeded, 28 remained failed**, with **83 API attempts / 70 failed attempts** across progressive and fallback attempts. The translated SRT exists; MKV does not.

Artifacts:

- `artifacts/gemini-family-guy/run01/report.json`
- `artifacts/gemini-family-guy/run01/messages.jsonl`
- `artifacts/gemini-family-guy/run01/spans.jsonl`
- `artifacts/gemini-family-guy/run01/trace-summary.json`
- `artifacts/gemini-family-guy/run01/output/Family Guy - Peter falls down the stairs.vi.srt`

No estimate of successful whole-video latency is inferred from the partial TTS results. A valid retest requires available quota and a new, empty processing cache again. Known quota exhaustion should stop further provider calls rather than repeat the same request budget through multiple pipeline layers; this run exposes that missing protection, but no runtime change was made during this measurement.
