"""Real Intel GPU/Small/beam=5 cold-region-cache versus checkpoint reuse.

Run after other benchmarks, with predownloaded models; no network or model change.
"""
from dataclasses import asdict
import json
import os
from pathlib import Path
import time

os.environ["HF_HUB_OFFLINE"] = "1"
root = Path(__file__).resolve().parents[1]
base = root / "artifacts/checkpoints-2026-11-17/asr-region-cache"
if base.exists():
    raise SystemExit("Use a fresh output directory; existing evidence is preserved")
base.mkdir(parents=True)
started = time.perf_counter()
from optimum.intel.openvino.modeling_seq2seq import OVDecoder
from jellydub.transcribe import transcribe_video

record = {"import_seconds": time.perf_counter()-started,
          "scope": "20-second sample, predownloaded Whisper Small, Intel GPU, beam=5", "runs": []}
decoder_calls = 0
original = OVDecoder.forward


def observed(self, *args, **kwargs):
    global decoder_calls
    decoder_calls += 1
    return original(self, *args, **kwargs)


OVDecoder.forward = observed
transcripts = []
try:
    for run in (1, 2):
        decoder_calls = 0
        details = {}
        started = time.perf_counter()
        cues, language = transcribe_video(
            root / "artifacts/performance-audit/asr/sample-20s.wav",
            model_name=(root / "artifacts/igpu-setup/whisper-small-model-path.txt").read_text().strip(),
            device="gpu", backend="openvino", cache_dir=base / "cache",
            language="en", beam_size=5, details=details,
            report=lambda s: print(s, flush=True))
        transcripts.append([asdict(c) for c in cues])
        record["runs"].append({"run": run, "seconds": time.perf_counter()-started,
                               "decoder_calls": decoder_calls, "details": details,
                               "language": language, "cues": transcripts[-1]})
        (base / "report.json").write_text(json.dumps(record, ensure_ascii=False, indent=2))
    record["same_transcript"] = transcripts[0] == transcripts[1]
    record["passed"] = (record["same_transcript"] and record["runs"][0]["decoder_calls"] > 0
                        and record["runs"][1]["decoder_calls"] == 0
                        and record["runs"][1]["details"]["region_cache_hits"] > 0)
    (base / "report.json").write_text(json.dumps(record, ensure_ascii=False, indent=2))
    if not record["passed"]:
        raise SystemExit("ASR region cache verification failed")
    print(json.dumps(record, indent=2))
finally:
    OVDecoder.forward = original
