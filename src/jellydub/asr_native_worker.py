"""Isolated native GenAI worker; run with its own compatible OpenVINO environment."""
from dataclasses import asdict
import json
import math
from pathlib import Path
import sys
import time

# The isolated environment needs only GenAI + NumPy, not the main Torch stack.
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from jellydub.asr_cache import RegionCache, append_region_cues, model_signature
from jellydub.cache import atomic_write_text, cache_lock
from jellydub.performance_trace import span


def main(request):
    import numpy as np
    import openvino as ov
    import openvino_genai as genai
    audio = np.load(request["audio"], allow_pickle=False, mmap_mode="r")
    root, model_path = Path(request["cache"]), Path(request["model"])
    device, language, beam = request["device"], request["language"], request["beam"]
    precision = request.get("precision", "f32")
    if precision not in {"f32", "f16"}:
        raise ValueError("Native ASR precision must be f32 or f16")
    cache = RegionCache(root, model_signature(model_path), device=device, language=language,
                        beam_size=beam, backend="genai", precision=precision)
    details = {"genai_version": genai.__version__, "openvino_version": ov.__version__,
               "compile_device": device, "model_loaded": False, "region_cache_hits": 0, "regions": [],
               "precision": precision,
               "execution_verification": "explicit device compilation; per-component devices not exposed by GenAI"}
    pipe = None
    cues = []
    def report(message):
        print(json.dumps({"message": message}), flush=True)
    for number, region in enumerate(request["regions"], 1):
        first_cue = len(cues)
        start, end = region["start"], region["end"]
        path = cache.path_for(audio[start:end])
        started = time.perf_counter()
        with cache_lock(path.with_suffix(".lock")):
            result = cache.read(path)
            if result is None:
                if pipe is None:
                    (root/"compiled").mkdir(parents=True, exist_ok=True)
                    properties = {"INFERENCE_PRECISION_HINT": precision, "CACHE_DIR": str((root/"compiled").resolve())}
                    if device == "GPU":
                        properties["GPU_ENABLE_SDPA_OPTIMIZATION"] = False
                    load_started = time.perf_counter()
                    with span("asr.load_compile", device=device):
                        pipe = genai.WhisperPipeline(str(model_path), device, **properties)
                    details["model_loaded"] = True
                    details["model_load_compile_seconds"] = time.perf_counter()-load_started
                budget = min(400, max(48, math.ceil((end-start)/16000*12)))
                with span("asr.generate", device=device, region=number, audio_seconds=(end-start)/16000):
                    generated = pipe.generate(audio[start:end], language=f"<|{language}|>", task="transcribe",
                                          return_timestamps=True, num_beams=beam,
                                          no_repeat_ngram_size=4, max_new_tokens=budget)
                chunks = generated.chunks or []
                if not chunks and any(t.strip() for t in generated.texts):
                    from jellydub.asr_cache import ASRTranscriptError
                    raise ASRTranscriptError("Native ASR returned speech without timestamp chunks")
                result = {"chunks": [{"text": c.text, "timestamp": [c.start_ts, c.end_ts]} for c in chunks]}
            else:
                details["region_cache_hits"] += 1
            append_region_cues(cues, result["chunks"], start/16000, end/16000, report)
            cache.write(path, result)
        elapsed = time.perf_counter()-started
        details["regions"].append({"start": start/16000, "end": end/16000, "seconds": elapsed})
        report(f"Native ASR region {number}/{len(request['regions'])}: {elapsed:.2f}s")
        print(json.dumps({"region_cues": [asdict(c) for c in cues[first_cue:]]}, ensure_ascii=False), flush=True)
    return {"cues": [asdict(c) for c in cues], "language": language, "details": details}


if __name__ == "__main__":
    request = json.loads(sys.stdin.read())
    try:
        result = main(request)
    except Exception as exc:
        result = {"error": str(exc), "error_type": type(exc).__name__}
        atomic_write_text(Path(request["result"]), json.dumps(result))
        raise
    atomic_write_text(Path(request["result"]), json.dumps(result, ensure_ascii=False))
