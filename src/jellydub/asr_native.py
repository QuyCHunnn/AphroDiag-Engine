"""Opt-in GenAI ASR through an isolated interpreter, with bounded region resume."""
import json
from pathlib import Path
import subprocess
import sys
import tempfile

from .asr_cache import ASRTranscriptError
from .models import Cue, JellyDubError


def runtime_signature(python_executable=None):
    from .cache import file_digest
    interpreter = Path(python_executable or sys.executable).expanduser().absolute()
    root = interpreter.parent.parent
    files = sorted(root.glob("lib/python*/site-packages/openvino*dist-info/METADATA"))
    worker = Path(__file__).with_name("asr_native_worker.py")
    return {"python": str(interpreter), "worker": file_digest(worker),
            "distributions": [(str(p.relative_to(root)), file_digest(p)) for p in files]}


def transcribe(audio, *, model_name, device, cache_dir, language, beam_size,
               report, details, python_executable=None, on_region=None, precision="f32"):
    import numpy as np
    from faster_whisper.vad import VadOptions, get_speech_timestamps
    from .asr_openvino import _context_regions, _model_path
    if language not in {"en", "vi"}:
        raise JellyDubError("Experimental GenAI ASR requires explicit source language en or vi")
    if precision not in {"f32", "f16"}:
        raise JellyDubError("Native ASR precision must be f32 or experimental f16")
    speech = get_speech_timestamps(audio, VadOptions())
    if not speech:
        return [], language
    model_path = _model_path(model_name, cache_dir)
    cache_dir.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix="native-job-", dir=cache_dir) as directory:
        folder = Path(directory)
        np.save(folder/"audio.npy", audio, allow_pickle=False)
        request = {"audio": str((folder/"audio.npy").resolve()), "model": str(model_path.resolve()),
                   "cache": str((cache_dir/"genai").resolve()), "device": device,
                   "language": language, "beam": beam_size,
                   "precision": precision,
                   "regions": _context_regions(speech, len(audio)),
                   "result": str((folder/"result.json").resolve())}
        # Preserve the venv symlink path: resolving it to /usr/bin/python loses
        # the isolated environment's pyvenv.cfg and installed GenAI packages.
        interpreter = str(Path(python_executable).expanduser().absolute()) if python_executable else sys.executable
        with (folder/"stderr.log").open("w+") as errors:
            process = subprocess.Popen([interpreter, str(Path(__file__).with_name("asr_native_worker.py"))],
                                       stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=errors, text=True)
            try:
                process.stdin.write(json.dumps(request)); process.stdin.close()
                for line in process.stdout:
                    try:
                        event = json.loads(line)
                    except ValueError:
                        report(line.rstrip()[:500]); continue
                    if isinstance(event, dict) and "message" in event:
                        report(str(event["message"]))
                    if isinstance(event, dict) and "region_cues" in event and on_region:
                        on_region([Cue(**cue) for cue in event["region_cues"]])
                code = process.wait()
            finally:
                if process.poll() is None:
                    process.terminate()
                    try:
                        process.wait(timeout=10)
                    except subprocess.TimeoutExpired:
                        process.kill(); process.wait()
                process.stdout.close()
            errors.seek(0)
            stderr = errors.read()
        try:
            result = json.loads((folder/"result.json").read_text())
        except (OSError, ValueError) as exc:
            raise JellyDubError(f"Native ASR worker failed ({code}); check --asr-python and GenAI installation: {stderr[-2000:]}") from exc
        if "error" in result or code:
            error = ASRTranscriptError if result.get("error_type") == "ASRTranscriptError" else JellyDubError
            raise error(f"Native ASR worker failed ({code}): {result.get('error', stderr[-2000:])}")
        details.update(result["details"])
        return [Cue(**cue) for cue in result["cues"]], result["language"]
