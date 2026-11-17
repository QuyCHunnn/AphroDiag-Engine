"""Backend-specific device selection; OpenVINO GPU is not a Torch GPU."""
from .models import JellyDubError


def hardware_capabilities():
    """Probe native GPU libraries outside a potentially threaded CLI host."""
    import json
    import subprocess
    import sys
    code = (
        "import json; from jellydub.transcribe import _cuda_available; "
        "from jellydub.asr_openvino import gpu_available; "
        "from jellydub.devices import torch_cuda_available; "
        "print(json.dumps(dict(asr_cuda=_cuda_available(), "
        "asr_openvino=gpu_available(), torch_cuda=torch_cuda_available())))"
    )
    try:
        result = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True, timeout=60)
        if result.returncode:
            return {"error": f"hardware probe exited {result.returncode}"}
        return json.loads(result.stdout.splitlines()[-1])
    except (OSError, ValueError, IndexError, subprocess.TimeoutExpired) as exc:
        return {"error": str(exc)}


def torch_cuda_available():
    try:
        from importlib.metadata import version
        if version("torch").endswith("+cpu"):
            return False
        import torch
        return torch.cuda.is_available()
    except ImportError:
        return False


def resolve_device(stage, requested):
    if requested not in {"auto", "cpu", "cuda", "gpu"}:
        raise JellyDubError(f"Unknown {stage} device: {requested}")
    cuda = torch_cuda_available() if requested in {"auto", "cuda"} else False
    if requested == "cuda" and not cuda:
        raise JellyDubError(f"{stage} requested CUDA but the installed Torch runtime has no CUDA device")
    if requested == "gpu":
        raise JellyDubError(f"{stage} has no Intel/OpenVINO backend; use cpu or auto")
    actual = ("cuda" if cuda else "cpu") if requested == "auto" else requested
    reason = "Torch CUDA unavailable; using CPU" if requested == "auto" and not cuda else None
    return actual, reason
