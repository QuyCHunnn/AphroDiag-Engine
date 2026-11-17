"""Experimental OPUS adapter sharing the exact existing generation/guard policy."""
from pathlib import Path
import json
import time

from .cache import cache_key, file_digest
from .models import JellyDubError
from .translate import OpusMTTranslator


class OpenVINOOpusTranslator(OpusMTTranslator):
    def __init__(self, model, device="cpu"):
        super().__init__(model, device)
        self.execution = {"requested_device": device, "loaded": False}

    def cache_id(self):
        root = Path(self.model_name).expanduser()
        manifest = root / "export-manifest.json"
        if not manifest.is_file():
            raise JellyDubError("OpenVINO OPUS requires a local export with export-manifest.json")
        return "opus-mt-openvino:v1-guarded:" + cache_key(file_digest(manifest), self.device, "f32")

    def _load(self):
        if self._model is not None:
            return
        if self.device not in {"cpu", "gpu"}:
            raise JellyDubError("OpenVINO OPUS requires explicit cpu or gpu")
        self.cache_id()
        started = time.perf_counter()
        try:
            from optimum.intel.openvino import OVModelForSeq2SeqLM
            from transformers import MarianTokenizer
            root = Path(self.model_name).expanduser()
            self._tokenizer = MarianTokenizer.from_pretrained(str(root), local_files_only=True)
            self._model = OVModelForSeq2SeqLM.from_pretrained(
                str(root), device=self.device.upper(), export=False, local_files_only=True,
                ov_config={"INFERENCE_PRECISION_HINT": "f32"}, trust_remote_code=False)
            devices = {}
            for name in ("encoder", "decoder", "decoder_with_past"):
                part = getattr(self._model, name, None)
                request = getattr(part, "request", None)
                if request is not None:
                    compiled = request.get_compiled_model() if hasattr(request, "get_compiled_model") else request
                    devices[name] = list(compiled.get_property("EXECUTION_DEVICES"))
            expected = self.device.upper()
            if not devices or any(not value.startswith(expected) for values in devices.values() for value in values):
                raise JellyDubError(f"Unexpected OPUS execution devices: {devices}")
            self.execution.update(loaded=True, load_compile_seconds=time.perf_counter()-started,
                                  execution_devices=devices, precision="f32")
        except Exception as exc:
            self._model = self._tokenizer = None
            raise JellyDubError(f"OpenVINO OPUS load failed: {exc}") from exc
