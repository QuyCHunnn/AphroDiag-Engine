"""Exercise the CLI's native ASR adapter on the actual 20s and Family Guy sources."""
from dataclasses import asdict
import json
from pathlib import Path
import time
from jellydub.cache import file_digest
from jellydub.subtitles import write_srt
from jellydub.transcribe import transcribe_video

root = Path(__file__).resolve().parents[1]
base = root / "artifacts/checkpoints-2026-11-17/native-cli"
if base.exists():
    raise SystemExit("Preserve previous evidence; use a fresh benchmark directory")
base.mkdir(parents=True)
model = (root/"artifacts/igpu-setup/whisper-small-model-path.txt").read_text().strip()
record = {"scope": "native adapter, region VAD/guard/checkpoint, Small/beam=5/f32 on Iris",
          "model_downloads_included": False, "runs": []}
for label, source in [("sample20", root/"artifacts/performance-audit/asr/sample-20s.wav"),
                      ("family-guy", root/"videos/Family Guy - Peter falls down the stairs.mp4")]:
    before = file_digest(source)
    previous = None
    for number in (1, 2):
        details = {}; started = time.perf_counter()
        cues, language = transcribe_video(
            source, model_name=model, device="gpu", cache_dir=base/label, language="en",
            backend="genai", beam_size=5, python_executable=str(root/".venv-native-asr-next/bin/python"),
            details=details, report=lambda s: print(s, flush=True))
        elapsed = time.perf_counter()-started
        serial = [asdict(c) for c in cues]
        if previous is not None and serial != previous:
            raise RuntimeError("Checkpoint reuse changed the transcript")
        previous = serial
        write_srt(cues, base/f"{label}-{number}.en.srt")
        record["runs"].append({"source": label, "run": number, "seconds": elapsed,
                               "language": language, "details": details, "cues": serial,
                               "source_unchanged": file_digest(source) == before})
        (base/"report.json").write_text(json.dumps(record, ensure_ascii=False, indent=2))
        print(label, number, round(elapsed, 3), flush=True)
record["status"] = "passed"
(base/"report.json").write_text(json.dumps(record, ensure_ascii=False, indent=2))
