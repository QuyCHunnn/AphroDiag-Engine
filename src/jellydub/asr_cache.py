"""Content-validated checkpoints for completed OpenVINO speech regions."""
import hashlib
import json
import math
from importlib.metadata import PackageNotFoundError, version

from .cache import atomic_write_text, cache_key, file_digest
from .models import Cue, JellyDubError


class ASRTranscriptError(JellyDubError):
    """Invalid transcript data; repeating the whole clip on CPU cannot repair it."""


def append_region_cues(cues, chunks, start, end, report):
    """Validate relative timestamps before publishing/checkpointing a region."""
    if not isinstance(chunks, list):
        raise ASRTranscriptError("ASR returned malformed chunks")
    for chunk in chunks:
        if (not isinstance(chunk, dict) or not isinstance(chunk.get("text"), str)
                or not isinstance(chunk.get("timestamp"), (list, tuple))
                or len(chunk["timestamp"]) != 2
                or any(t is not None and (not isinstance(t, (int, float)) or not math.isfinite(t))
                       for t in chunk["timestamp"])):
            raise ASRTranscriptError("ASR returned malformed chunk timestamps")
    for index, chunk in enumerate(chunks):
        text = chunk["text"].strip()
        left, right = chunk["timestamp"]
        if left is None:
            left = cues[-1].end-start if cues else 0.0
        if right is None:
            following = chunks[index+1]["timestamp"][0] if index+1 < len(chunks) else None
            right = following if following is not None else end-start
            report("Warning: ASR estimated a missing end timestamp; review this subtitle")
        left, right = max(start, start+float(left)), min(end, start+float(right))
        if text:
            if right <= left:
                raise ASRTranscriptError(f"ASR returned an invalid subtitle interval {left:.3f}–{right:.3f} "
                                         f"in audio region {start:.3f}–{end:.3f} for {text!r}; refusing to drop speech")
            cues.append(Cue(len(cues)+1, left, right, text))


def model_signature(path):
    # Snapshot paths alone miss edits to local models and processor settings.
    return [(str(p.relative_to(path)), file_digest(p)) for p in sorted(path.rglob("*"))
            if p.is_file() and p.suffix in {".bin", ".xml", ".json", ".txt", ".model"}]


class RegionCache:
    def __init__(self, root, model, *, device, language, beam_size, backend="optimum", precision="f32"):
        self.root = root / "regions-v1"
        self.settings = {"model": model, "device": device, "language": language,
                         "beam": beam_size, "precision": precision, "policy": 1}
        self.settings["runtime"] = {}
        packages = ["optimum-intel", "transformers", "openvino", "faster-whisper"]
        if backend != "optimum":
            self.settings["backend"] = backend
            packages.append("openvino-genai")
        for package in packages:
            try:
                self.settings["runtime"][package] = version(package)
            except PackageNotFoundError:
                self.settings["runtime"][package] = None

    def path_for(self, audio):
        digest = hashlib.sha256(memoryview(audio).cast("B")).hexdigest()
        return self.root / (cache_key(self.settings, digest) + ".json")

    def read(self, path):
        try:
            record = json.loads(path.read_text())
            payload = record["payload"]
            expected = hashlib.sha256(payload.encode()).hexdigest()
            if expected != record["sha256"]:
                return None
            result = json.loads(payload)
            if not isinstance(result, dict) or not isinstance(result.get("chunks"), list):
                return None
            return result
        except (OSError, ValueError, TypeError, KeyError, AttributeError):
            return None

    def write(self, path, result):
        # Only JSON primitives are retained; no executable/pickled model objects.
        payload = json.dumps({"chunks": result.get("chunks", [])}, ensure_ascii=False,
                             allow_nan=False)
        atomic_write_text(path, json.dumps({"payload": payload,
                          "sha256": hashlib.sha256(payload.encode()).hexdigest()}))
