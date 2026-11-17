"""Balanced CPU ONNX thread sweep, fixed BLAS=2 and identical per-cue seeds."""
import hashlib
import json
import os
from pathlib import Path
import time

os.environ["HF_HUB_OFFLINE"] = "1"
import numpy as np
from threadpoolctl import threadpool_limits
from jellydub.subtitles import parse_srt
from jellydub.tts.vieneu import VieNeuTTS

root = Path(__file__).resolve().parents[1]
base = root / "artifacts/checkpoints-2026-11-17/onnx-thread-sweep"
if base.exists():
    raise SystemExit("Existing benchmark evidence is preserved")
base.mkdir(parents=True)
voices = json.loads((root / "artifacts/family-guy-vieneu/speaker-ledger.json").read_text())["voices"]
cues = parse_srt(root / "artifacts/family-guy-vieneu/reviewed.vi.srt")
selected = [next(c for c in cues if c.speaker == speaker) for speaker in ("PETER", "LOIS", "STEWIE")]
record = {"status": "running", "blas_threads": 2, "sessions": [], "runs": []}


def save():
    (base / "report.json").write_text(json.dumps(record, indent=2))


try:
    for threads in (1, 2, 4, 4, 2, 1):
        provider = VieNeuTTS(device="cpu", blas_threads=2, cpu_threads=threads)
        try:
            started = time.perf_counter(); model = provider._load()
            record["sessions"].append({"threads": threads, "load_seconds": time.perf_counter()-started,
                                       "actual_threads": model.engine.ort_intra_op_threads})
            with threadpool_limits(limits=2, user_api="blas"):
                np.random.seed(123); model.infer("Xin chào.", voice=voices["PETER"])
                for cue in selected:
                    seed = int.from_bytes(hashlib.sha256((str(cue.speaker)+"|"+cue.text).encode()).digest()[:4], "little")
                    np.random.seed(seed); started = time.perf_counter()
                    audio = model.infer(cue.text, voice=voices[cue.speaker])
                    elapsed = time.perf_counter()-started
                    record["runs"].append({"threads": threads, "cue": cue.index, "seconds": elapsed,
                                           "sha256": hashlib.sha256(np.asarray(audio, dtype=np.float32).tobytes()).hexdigest()})
                    save(); print(threads, cue.speaker, round(elapsed, 3), flush=True)
        finally:
            provider.close()
    record["same_waveforms"] = all(len({r["sha256"] for r in record["runs"] if r["cue"] == c.index}) == 1
                                    for c in selected)
    record["total_inference_seconds"] = {str(t): sum(r["seconds"] for r in record["runs"] if r["threads"] == t)
                                          for t in (1, 2, 4)}
    record["status"] = "measured"; save()
    print(json.dumps({k: v for k, v in record.items() if k != "runs"}, indent=2))
except BaseException as exc:
    record.update(status="failed", error=str(exc)); save(); raise
