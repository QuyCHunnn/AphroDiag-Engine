"""Local VoxCPM2 GGUF CLI adapter; GPU requests never silently fall back."""
from pathlib import Path
import os
import re
import shutil
import subprocess
import tempfile
import wave

from ..cache import cache_key, file_digest, tmp_path_for
from ..ffmpeg import duration
from .base import TTSError, TTSProvider


class VoxCPM2TTS(TTSProvider):
    name = "voxcpm2"
    max_concurrency = 1

    def __init__(self, model=None, binary=None, style="", reference=None,
                 speaker_refs=None, device="gpu", timeout=600):
        self.device = "gpu" if device == "auto" else device
        if self.device not in {"cpu", "gpu", "cuda"}:
            raise ValueError("VoxCPM2 device must be auto, gpu, cuda or cpu")
        if not isinstance(style, str) or type(timeout) is not int or timeout < 1:
            raise ValueError("VoxCPM2 requires a string style and positive integer timeout")
        self.style, self.timeout = style, timeout
        self.reference = Path(reference).expanduser() if reference else None
        self.speaker_refs = {k: Path(v).expanduser() for k, v in (speaker_refs or {}).items()}
        executable = binary or shutil.which("voxcpm2-cli")
        if not executable or not Path(executable).expanduser().is_file():
            raise TTSError("Set voxcpm2_binary to a built voxcpm2-cli executable (see docs/voxcpm2.md)")
        self.binary = Path(executable).expanduser().resolve()
        if not os.access(self.binary, os.X_OK):
            raise TTSError(f"VoxCPM2 executable is not executable: {self.binary}")
        if not model or not Path(model).expanduser().is_dir():
            raise TTSError("VoxCPM2 tts_model must be a directory containing BaseLM and Acoustic GGUF files")
        folder = Path(model).expanduser()
        self.models = []
        for component in ("BaseLM", "Acoustic"):
            files = sorted(folder.glob(f"VoxCPM2-{component}-*.gguf"))
            if len(files) != 1:
                raise TTSError(f"VoxCPM2 needs exactly one {component} GGUF in {folder}")
            self.models.append(files[0].resolve())
        self.runtime_assets = [self.binary, *sorted({p.resolve() for pattern in ("libggml*.so*", "libllama*.so*")
                                                    for p in self.binary.parent.glob(pattern) if p.is_file()})]
        self._digests = {}
        self.execution = {"requested_device": device, "loaded": False}

    def _digest(self, path):
        if path is None:
            return None
        path = Path(path).resolve()
        if path not in self._digests:
            try:
                self._digests[path] = file_digest(path)
            except OSError as exc:
                raise TTSError(f"Cannot read VoxCPM2 asset {path}: {exc}") from exc
        return self._digests[path]

    def _reference_for(self, speaker):
        ref = self.speaker_refs.get(speaker, self.reference)
        if ref is None and self.speaker_refs:
            raise TTSError(f"No VoxCPM2 reference for speaker {speaker!r}; add LABEL=FILE or a default reference")
        return ref

    def cache_id_for(self, speaker):
        return "voxcpm2-gguf:cue-v1:" + cache_key(
            [self._digest(p) for p in self.runtime_assets], [self._digest(p) for p in self.models],
            self.device, self.style, self._digest(self._reference_for(speaker)),
            {"seed": 42, "steps": 200, "timesteps": 10, "cfg": 2})

    def cache_id(self):
        return "voxcpm2-gguf:v1:" + cache_key(
            self.cache_id_for(None) if self.reference or not self.speaker_refs else None,
            {k: self.cache_id_for(k) for k in self.speaker_refs})

    def synthesize(self, text, out_path: Path, speaker=None):
        ref = self._reference_for(speaker)
        if ref is not None:
            try:
                if not ref.is_file() or not 3 <= duration(ref) <= 8:
                    raise ValueError("expected a readable 3–8 second reference")
            except Exception as exc:
                raise TTSError(f"Invalid VoxCPM2 reference {ref}: {exc}") from exc
        if self.style:
            # Merge a per-cue direction into the single model-supported prefix.
            direction = re.match(r"^\(([^()]*)\)", text)
            style = self.style
            if direction:
                style += "; " + direction[1]
                text = text[direction.end():]
            text = f"({style}){text}"
        out_path = Path(out_path)
        temporary = tmp_path_for(out_path)
        command = [str(self.binary), "--seed", "42", "--steps", "200",
                   "--timesteps", "10", "--cfg", "2", "-t", text,
                   "-o", str(temporary)]
        # --cpu alone only selects CPU for acoustic components in this runtime.
        # Zero GPU layers is also required to make explicit CPU mode honest.
        command += ["--cpu", "--n-gpu-layers", "0"] if self.device == "cpu" else ["--n-gpu-layers", "99"]
        if ref is not None:
            command += ["-r", str(ref.resolve())]
        command += [str(p) for p in self.models]
        try:
            with tempfile.TemporaryFile(mode="w+b") as log:
                result = subprocess.run(command, stdout=log, stderr=subprocess.STDOUT,
                                        timeout=self.timeout)
                log.seek(0, os.SEEK_END)
                size = log.tell()
                log.seek(max(0, size - 1024 * 1024))
                output = log.read().decode("utf-8", errors="replace")
            if result.returncode:
                raise TTSError(f"VoxCPM2 exited {result.returncode}: {output[-2000:]}")
            backend = re.search(r"custom component backend=(\S+)", output)
            layers = re.search(r"offloaded (\d+)/(\d+) layers to GPU", output)
            if self.device != "cpu" and (not backend or backend[1].upper().startswith("CPU")
                    or not layers or int(layers[1]) != int(layers[2]) or int(layers[1]) == 0
                    or (self.device == "cuda" and "CUDA" not in backend[1].upper())):
                raise TTSError("VoxCPM2 did not confirm all BaseLM layers and acoustic components on GPU; refusing CPU fallback")
            with wave.open(str(temporary), "rb") as audio:
                frames = audio.getnframes()
                expected = frames * audio.getnchannels() * audio.getsampwidth()
                if (audio.getcomptype() != "NONE" or audio.getnchannels() != 1
                        or audio.getframerate() != 48000 or frames <= 0
                        or len(audio.readframes(frames)) != expected):
                    raise TTSError("VoxCPM2 returned invalid or truncated WAV audio")
                seconds = frames / audio.getframerate()
            # 200 patches * 160 ms for VoxCPM2. Reject output at the generation cap.
            if seconds >= 32:
                raise TTSError("VoxCPM2 reached the 200-frame speech cap; split this cue into shorter text")
            elapsed = re.search(r"Elapsed: ([\d.]+)s", output)
            self.execution.update(loaded=True, backend=backend[1] if backend else "CPU",
                                  gpu_layers=int(layers[1]) if layers else 0,
                                  audio_seconds=seconds,
                                  inference_seconds=float(elapsed[1]) if elapsed else None)
            os.replace(temporary, out_path)
        except (OSError, subprocess.TimeoutExpired, wave.Error, EOFError) as exc:
            raise TTSError(f"VoxCPM2 synthesis failed: {exc}") from exc
        finally:
            temporary.unlink(missing_ok=True)
