"""Verify Intel GPU inference in WSL using isolated OpenVINO (no AUTO fallback).
Run outside restricted sandboxes that hide /dev/dxg:
  .venv-igpu/bin/python scripts/check_igpu.py
"""
import json
from pathlib import Path
import platform
import time

import numpy as np
import openvino as ov
from openvino import opset13 as ops

root = Path(__file__).resolve().parents[1]
output = root / 'artifacts/igpu-setup'
output.mkdir(parents=True, exist_ok=True)
core = ov.Core()
report = {'platform': platform.platform(), 'openvino_version': ov.__version__,
          'dxg_visible': Path('/dev/dxg').exists(), 'available_devices': core.available_devices,
          'requested_device': 'GPU', 'benchmarks': []}
if not any(device.startswith('GPU') for device in core.available_devices):
    (output/'verification.json').write_text(json.dumps(report, indent=2))
    raise SystemExit('GPU unavailable. Check clinfo and run outside the restricted sandbox.')
report['gpu_name'] = core.get_property('GPU', 'FULL_DEVICE_NAME')
rng = np.random.default_rng(42)
for size in (256, 512, 1024):
    a = rng.normal(0, .1, (size,size)).astype(np.float32)
    b = rng.normal(0, .1, (size,size)).astype(np.float32)
    left = ops.parameter([size,size], np.float32, name='left')
    right = ops.parameter([size,size], np.float32, name='right')
    model = ov.Model([ops.matmul(left,right,False,False)], [left,right], f'matmul-{size}')
    start = time.perf_counter()
    compiled = core.compile_model(model, 'GPU', {'INFERENCE_PRECISION_HINT': 'f32'})
    compile_seconds = time.perf_counter()-start
    actual = compiled([a,b])[0].copy()
    reference = a @ b
    np.testing.assert_allclose(actual, reference, rtol=1e-3, atol=1e-4)
    times = []
    for _ in range(10):
        start = time.perf_counter(); compiled([a,b]); times.append(time.perf_counter()-start)
    devices = list(compiled.get_property('EXECUTION_DEVICES'))
    if not devices or not all(d.startswith('GPU') for d in devices):
        raise RuntimeError(f'Unexpected inference devices: {devices}')
    item = {'matrix_size':size,'execution_devices':devices,'compile_seconds':compile_seconds,
            'median_inference_ms':float(np.median(times)*1000),
            'max_absolute_error':float(np.max(np.abs(actual-reference))), 'numerical_check':'passed'}
    report['benchmarks'].append(item)
    print(json.dumps(item),flush=True)
report['status'] = 'gpu_inference_verified'
(output/'verification.json').write_text(json.dumps(report,indent=2))
print('Verified GPU:', report['gpu_name'])
