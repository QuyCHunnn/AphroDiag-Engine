"""Experimental OpenVINO session bridge for VieNeu's existing ONNX graphs.

This keeps the SDK prompt, voices, sampling and codec unchanged. GPU throughput
and waveform equivalence must be measured before exposing it as a normal backend.
"""
import time
from pathlib import Path
import shutil


class OpenVINOSession:
    def __init__(self, source, device, cache_dir, *, core=None):
        import openvino as ov
        import onnx
        cache_dir.mkdir(parents=True, exist_ok=True)
        core = core if core is not None else ov.Core()
        # Both frontends reject symlinked external weights in HF snapshots.
        # Stage regular files inside this experiment, leaving HF files untouched.
        source = Path(source)
        stage = cache_dir.parent / "onnx-source"
        stage.mkdir(parents=True, exist_ok=True)
        graph = onnx.load(str(source), load_external_data=False)
        for tensor in graph.graph.initializer:
            for item in tensor.external_data:
                if item.key != "location":
                    continue
                relative = Path(item.value)
                if relative.is_absolute() or ".." in relative.parts:
                    raise ValueError("External weights must use a relative path inside the model directory")
                target = stage / relative
                if not target.exists():
                    target.parent.mkdir(parents=True, exist_ok=True)
                    shutil.copyfile(source.parent / relative, target)
        target_graph = stage / source.name
        onnx.save_model(graph, str(target_graph))
        model = core.read_model(str(target_graph))
        self.compiled = core.compile_model(model, device, {
            "INFERENCE_PRECISION_HINT": "f32", "PERFORMANCE_HINT": "LATENCY",
            "CACHE_DIR": str(cache_dir.resolve()),
        })
        self.request = self.compiled.create_infer_request()
        self.execution_devices = list(self.compiled.get_property("EXECUTION_DEVICES"))
        if device.upper().startswith("GPU") and any(not d.startswith("GPU") for d in self.execution_devices):
            raise RuntimeError(f"Unexpected TTS execution devices: {self.execution_devices}")
        self.calls = 0
        self.seconds = 0.0

    def run(self, output_names, feed):
        if output_names is not None:
            raise ValueError("This prototype supports all graph outputs only")
        started = time.perf_counter()
        try:
            self.request.infer(feed)
            # A persistent InferRequest reuses output buffers. The SDK retains
            # K/V arrays for the next step; detach them before the next inference.
            return [self.request.get_output_tensor(i).data.copy()
                    for i in range(len(self.compiled.outputs))]
        finally:
            self.calls += 1
            self.seconds += time.perf_counter()-started
