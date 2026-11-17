"""ABBA prototype: fuse acoustic heads without changing model, voices or sampling."""
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
from jellydub.tts.fused import install_fused_heads

root = Path(__file__).resolve().parents[1]
parser = argparse.ArgumentParser()
parser.add_argument("--engine", choices=["heads", "frame-cpu", "frame-gpu"], default="heads")
parser.add_argument("--run", default="fused-heads")
parser.add_argument("--precision", choices=["f32", "f16"], default="f32")
args = parser.parse_args()
base = root / "artifacts/checkpoints-2026-11-17" / args.run
if base.exists():
    raise SystemExit("Preserve previous results; choose a fresh benchmark directory")
base.mkdir(parents=True)
voices = json.loads((root / "artifacts/family-guy-vieneu/speaker-ledger.json").read_text())["voices"]
cues = parse_srt(root / "artifacts/family-guy-vieneu/reviewed.vi.srt")
selected = [next(c for c in cues if c.speaker == speaker) for speaker in ("PETER", "LOIS", "STEWIE")]
provider = VieNeuTTS(voice=voices["PETER"], speaker_voices=voices, device="cpu", blas_threads=2)
record = {"status": "running", "engine": args.engine, "precision": args.precision, "runs": [], "blas_threads": 2}


def save():
    (base / "report.json").write_text(json.dumps(record, ensure_ascii=False, indent=2))


try:
    started = time.perf_counter()
    model = provider._load()
    record["load_seconds"] = time.perf_counter()-started
    with threadpool_limits(limits=2, user_api="blas"):
        started = time.perf_counter()
        if args.engine == "heads":
            fused = install_fused_heads(model.engine)
        else:
            from jellydub.tts.frame_graph import install_frame_graph
            fused = install_frame_graph(model.engine, device="CPU" if args.engine == "frame-cpu" else "GPU", precision=args.precision)
        record["fusion_setup_seconds"] = time.perf_counter()-started
        record["providers"] = fused.get("providers", fused.get("devices"))
        record["onnx_threads"] = model.engine.ort_intra_op_threads
        optimized = model.engine._acoustic_frame
        original = fused["original"]
        # Warm both paths with the same input; exclude setup and warmup from throughput.
        for function in (original, optimized):
            model.engine._acoustic_frame = function
            np.random.seed(123)
            model.infer("Xin chào.", voice=voices["PETER"])
        for mode in ("sdk", "fused", "fused", "sdk"):
            model.engine._acoustic_frame = original if mode == "sdk" else optimized
            for cue in selected:
                seed = int.from_bytes(hashlib.sha256((str(cue.speaker)+"|"+cue.text).encode()).digest()[:4], "little")
                np.random.seed(seed)
                started = time.perf_counter()
                audio = model.infer(cue.text, voice=voices[cue.speaker])
                elapsed = time.perf_counter()-started
                audio = np.asarray(audio, dtype=np.float32)
                record["runs"].append({"mode": mode, "cue": cue.index, "speaker": cue.speaker,
                                       "seconds": elapsed, "samples": audio.size,
                                       "sha256": hashlib.sha256(audio.tobytes()).hexdigest()})
                save()
                print(mode, cue.speaker, round(elapsed, 3), flush=True)
    groups = {c.index: [r for r in record["runs"] if r["cue"] == c.index] for c in selected}
    record["same_waveforms"] = all(len({r["sha256"] for r in rows}) == 1 for rows in groups.values())
    totals = {mode: sum(r["seconds"] for r in record["runs"] if r["mode"] == mode)
              for mode in ("sdk", "fused")}
    record["speedup"] = totals["sdk"] / totals["fused"]
    record["status"] = "measured"
    record["adopt_candidate"] = record["same_waveforms"] and record["speedup"] > 1.05
    save()
    print(json.dumps({k: v for k, v in record.items() if k != "runs"}, indent=2))
except BaseException as exc:
    record.update(status="failed", error=str(exc)); save()
    raise
finally:
    provider.close()
