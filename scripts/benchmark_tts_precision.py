"""Compare upstream FP32/INT8 exports of the same VieNeu revision and voices.

Prepare weights separately; benchmark starts offline and excludes model downloads.
"""
import argparse
import hashlib
import json
import os
from pathlib import Path
import time

root = Path(__file__).resolve().parents[1]
base = root/"artifacts/checkpoints-2026-11-17/tts-precision"
parser = argparse.ArgumentParser(); parser.add_argument("--prepare-only", action="store_true")
args = parser.parse_args()
if args.prepare_only:
    from huggingface_hub import snapshot_download
    revision = (Path.home()/".cache/huggingface/hub/models--pnnbao-ump--VieNeu-TTS-v3-Turbo/refs/main").read_text().strip()
    path = snapshot_download("pnnbao-ump/VieNeu-TTS-v3-Turbo", revision=revision, allow_patterns=["onnx_int8/*"])
    print("Prepared INT8 at same cached FP32 revision:", path)
    raise SystemExit(0)
if base.exists():
    raise SystemExit("Preserve prior results; use a fresh output directory")
base.mkdir(parents=True)
os.environ["HF_HUB_OFFLINE"] = "1"
import numpy as np
from threadpoolctl import threadpool_limits
from vieneu import Vieneu
from jellydub.subtitles import parse_srt

voices = json.loads((root/"artifacts/family-guy-vieneu/speaker-ledger.json").read_text())["voices"]
cues = parse_srt(root/"artifacts/family-guy-vieneu/reviewed.vi.srt")
selected = [next(c for c in cues if c.speaker == speaker) for speaker in ("PETER", "LOIS", "STEWIE")]
models = {}
record = {"status": "running", "sessions": [], "runs": [], "blas_threads": 2,
          "quality_status": "pending listening review if waveform changes"}
def save():
    (base/"report.json").write_text(json.dumps(record, ensure_ascii=False, indent=2))
try:
    for precision in ("fp32", "int8"):
        started = time.perf_counter()
        models[precision] = Vieneu(mode="v3turbo", device="cpu", precision=precision)
        engine = models[precision].engine
        record["sessions"].append({"precision": precision, "seconds": time.perf_counter()-started,
                                    "graph": engine.sess_pre._model_path,
                                    "providers": engine.sess_pre.get_providers()})
        save()
    with threadpool_limits(limits=2, user_api="blas"):
        for model in models.values():
            np.random.seed(123); model.infer("Xin chào.", voice=voices["PETER"])
        for precision in ("fp32", "int8", "int8", "fp32"):
            for cue in selected:
                np.random.seed(int.from_bytes(hashlib.sha256((str(cue.speaker)+"|"+cue.text).encode()).digest()[:4], "little"))
                started = time.perf_counter()
                audio = models[precision].infer(cue.text, voice=voices[cue.speaker])
                elapsed = time.perf_counter()-started
                audio = np.asarray(audio, dtype=np.float32)
                if not audio.size or not np.isfinite(audio).all():
                    raise RuntimeError("Invalid audio")
                models[precision].save(audio, str(base/f"{precision}-{cue.speaker}.wav"))
                record["runs"].append({"precision": precision, "cue": cue.index, "seconds": elapsed,
                                       "samples": audio.size, "sha256": hashlib.sha256(audio.tobytes()).hexdigest()})
                save(); print(precision, cue.speaker, round(elapsed, 3), flush=True)
    totals = {p: sum(r["seconds"] for r in record["runs"] if r["precision"] == p) for p in models}
    record["speedup"] = totals["fp32"]/totals["int8"]
    record["same_waveforms"] = all(len({r["sha256"] for r in record["runs"] if r["cue"] == c.index}) == 1 for c in selected)
    record["status"] = "measured"; save()
    print(json.dumps({k: v for k, v in record.items() if k != "runs"}, indent=2))
except BaseException as exc:
    record.update(status="failed", error=str(exc)); save(); raise
finally:
    for model in models.values():
        model.close()
