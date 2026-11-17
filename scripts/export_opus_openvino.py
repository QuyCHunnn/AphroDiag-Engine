"""Export the cached OPUS revision once; no quantization or new weights."""
from pathlib import Path
import json,os,time
os.environ['HF_HUB_OFFLINE']='1'
from jellydub.cache import file_digest
root=Path(__file__).resolve().parents[1]
base=root/'artifacts/next-optimization/opus-openvino-fp32'
if base.exists():raise SystemExit('Existing export is preserved')
from huggingface_hub import snapshot_download
source=Path(snapshot_download('Helsinki-NLP/opus-mt-en-vi',local_files_only=True))
from optimum.intel.openvino import OVModelForSeq2SeqLM
from transformers import MarianTokenizer
started=time.perf_counter()
model=OVModelForSeq2SeqLM.from_pretrained(str(source),export=True,compile=False,load_in_8bit=False,ov_config={'INFERENCE_PRECISION_HINT':'f32'},trust_remote_code=False)
model.save_pretrained(base)
MarianTokenizer.from_pretrained(str(source),local_files_only=True).save_pretrained(base)
record={'source':str(source),'source_revision':source.name,'export_seconds':time.perf_counter()-started,'quantized':False,'files':{p.name:file_digest(p) for p in base.iterdir() if p.is_file()}}
(base/'export-manifest.json').write_text(json.dumps(record,indent=2));print(json.dumps(record,indent=2))
