"""Compare completed, fresh-cache CP1 TTS reports by cue identity and waveform."""
import argparse
import json
from pathlib import Path

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument("directory", type=Path)
args = parser.parse_args()
a, b = [json.loads((args.directory / variant / "report.json").read_text())
        for variant in ("baseline8", "optimized2")]
if any(r.get("status") != "passed" for r in (a, b)):
    raise SystemExit("Both fresh-cache runs must finish successfully")
identities = lambda r: [(c["speaker"], c["text"], c["seed"]) for c in r["calls"]]
if identities(a) != identities(b):
    raise SystemExit("Input/seed mismatch")
first = {c["cue_index"]: c for c in a["clips"]}
second = {c["cue_index"]: c for c in b["clips"]}
if len(first) != 44 or first.keys() != second.keys():
    raise SystemExit("Expected the same complete 44 speech cues")
different = [i for i in first if first[i]["sha256"] != second[i]["sha256"]]
report = {"scope": "44 reviewed speech cues; TTS load/inference/normalization, no ASR/translation/export",
          "baseline_seconds": a["wall_seconds"], "optimized_seconds": b["wall_seconds"],
          "speedup": a["wall_seconds"] / b["wall_seconds"],
          "same_waveform_count": 44-len(different), "different_waveform_cues": different,
          "baseline_model_load_seconds": a.get("model_load_seconds"),
          "optimized_model_load_seconds": b.get("model_load_seconds")}
(args.directory / "ab-result.json").write_text(json.dumps(report, indent=2))
print(json.dumps(report, indent=2))
if different:
    raise SystemExit("Waveform equality gate failed")
