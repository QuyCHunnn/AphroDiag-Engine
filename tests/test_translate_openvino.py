"""The experimental backend shares generation and isolates precision/device caches."""
import json
from pathlib import Path
from types import SimpleNamespace
import pytest
from jellydub.translate_openvino import OpenVINOOpusTranslator
from jellydub.translate import OpusMTTranslator
from jellydub.models import JellyDubError


def test_same_guarded_generation_implementation():
    assert OpenVINOOpusTranslator.translate_batch is OpusMTTranslator.translate_batch


def test_cache_isolated_from_torch_export_revision_and_device(tmp_path):
    path=tmp_path/'export-manifest.json';path.write_text(json.dumps({'source_revision':'a','quantized':False}))
    a=OpenVINOOpusTranslator(str(tmp_path),'cpu').cache_id()
    b=OpenVINOOpusTranslator(str(tmp_path),'gpu').cache_id()
    assert a!=b and a!=OpusMTTranslator().cache_id()
    path.write_text(json.dumps({'source_revision':'b','quantized':False}))
    assert OpenVINOOpusTranslator(str(tmp_path),'cpu').cache_id()!=a


def test_requires_explicit_device_and_local_export(tmp_path):
    with pytest.raises(JellyDubError,match='local export'):
        OpenVINOOpusTranslator(str(tmp_path)).cache_id()
    with pytest.raises(JellyDubError,match='explicit cpu or gpu'):
        OpenVINOOpusTranslator(str(tmp_path),'auto')._load()
