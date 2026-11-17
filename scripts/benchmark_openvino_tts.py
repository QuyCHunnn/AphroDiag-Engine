"""Probe real Intel GPU TTS using the same VieNeu graph weights and presets."""
import hashlib
import json
import os
from pathlib import Path
import time
import argparse

os.environ["HF_HUB_OFFLINE"] = "1"
import numpy as np
from threadpoolctl import threadpool_limits
from jellydub.subtitles import parse_srt
from jellydub.tts.vieneu import VieNeuTTS
from jellydub.tts.openvino_session import OpenVINOSession

root = Path(__file__).resolve().parents[1]
parser = argparse.ArgumentParser(); parser.add_argument("--run", default="openvino-tts")
args = parser.parse_args()
base = root / "artifacts/checkpoints-2026-11-17" / args.run
if base.exists():
    raise SystemExit("Existing evidence is preserved; use a new benchmark directory")
base.mkdir(parents=True)
voices = json.loads((root / "artifacts/family-guy-vieneu/speaker-ledger.json").read_text())["voices"]
cues = parse_srt(root / "artifacts/family-guy-vieneu/reviewed.vi.srt")
selected = [next(c for c in cues if c.speaker == speaker) for speaker in ("PETER", "LOIS", "STEWIE")]
record = {"status": "running", "scope": "same FP32 graphs, CPU codec/heads/sampling, Intel GPU transformer sessions",
          "compiles": [], "runs": []}
provider = VieNeuTTS(device="cpu", blas_threads=2)


def save():
    (base / "report.json").write_text(json.dumps(record, ensure_ascii=False, indent=2))


try:
    model = provider._load(); engine = model.engine
    original = {name: getattr(engine, name) for name in ("sess_pre", "sess_dec", "sess_ac")}
    import openvino as ov
    core = ov.Core()
    gpu = {}
    for name, session in original.items():
        started = time.perf_counter()
        gpu[name] = OpenVINOSession(session._model_path, "GPU", base / "compiled", core=core)
        record["compiles"].append({"session": name, "seconds": time.perf_counter()-started,
                                    "devices": gpu[name].execution_devices})
        save(); print("compiled", name, record["compiles"][-1], flush=True)
    with threadpool_limits(limits=2, user_api="blas"):
        for mode in ("sdk", "gpu-acoustic", "gpu-all"):
            for name in original:
                setattr(engine, name, gpu[name] if mode == "gpu-all" or
                        (mode == "gpu-acoustic" and name == "sess_ac") else original[name])
            for cue in selected:
                seed = int.from_bytes(hashlib.sha256((str(cue.speaker)+"|"+cue.text).encode()).digest()[:4], "little")
                np.random.seed(seed); started = time.perf_counter()
                audio = model.infer(cue.text, voice=voices[cue.speaker])
                elapsed = time.perf_counter()-started
                audio = np.asarray(audio, dtype=np.float32)
                path = base / f"{mode}-{cue.speaker}.wav"; model.save(audio, str(path))
                record["runs"].append({"mode": mode, "cue": cue.index, "seconds": elapsed,
                                       "samples": audio.size,
                                       "sha256": hashlib.sha256(audio.tobytes()).hexdigest()})
                save(); print(mode, cue.speaker, round(elapsed, 3), flush=True)
    record["same_waveforms"] = all(len({r["sha256"] for r in record["runs"] if r["cue"] == c.index}) == 1
                                    for c in selected)
    record["sessions"] = {name: {"calls": s.calls, "seconds": s.seconds,
                                  "devices": s.execution_devices} for name, s in gpu.items()}
    record["status"] = "measured"; save()
    print(json.dumps(record, indent=2))
except BaseException as exc:
    record.update(status="failed", error=str(exc)); save(); raise
finally:
    provider.close()
