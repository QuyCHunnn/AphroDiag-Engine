"""Verify alternative OpenVINO Whisper backend, preserving beam and language detection."""
import argparse,json,time,wave
from pathlib import Path
import numpy as np
from optimum.intel.openvino import OVModelForSpeechSeq2Seq
from transformers import AutoProcessor,pipeline
p=argparse.ArgumentParser();p.add_argument('--model',default='tiny');p.add_argument('--device',default='GPU');args=p.parse_args()
root=Path(__file__).resolve().parents[1];folder=root/'artifacts/optimization-checkpoints/cp3';folder.mkdir(parents=True,exist_ok=True)
model_dir=Path((root/f'artifacts/igpu-setup/whisper-{args.model}-model-path.txt').read_text().strip())
with wave.open(str(root/'artifacts/performance-audit/asr/sample-20s.wav')) as f:audio=np.frombuffer(f.readframes(f.getnframes()),dtype=np.int16).astype(np.float32)/32768
start=time.perf_counter()
model=OVModelForSpeechSeq2Seq.from_pretrained(str(model_dir),device=args.device,local_files_only=True,ov_config={'INFERENCE_PRECISION_HINT':'f32'})
processor=AutoProcessor.from_pretrained(str(model_dir),local_files_only=True)
print('Model loaded',round(time.perf_counter()-start,3),flush=True)
recognizer=pipeline('automatic-speech-recognition',model=model,tokenizer=processor.tokenizer,feature_extractor=processor.feature_extractor)
features=processor(audio,sampling_rate=16000,return_tensors='pt').input_features
lang_id=int(model.detect_language(input_features=features,generation_config=model.generation_config)[0])
language=next(key[2:-2] for key,value in model.generation_config.lang_to_id.items() if value==lang_id)
print('Detected language',language,flush=True)
start=time.perf_counter()
result=recognizer(audio,return_timestamps=True,return_language=True,generate_kwargs={'num_beams':5,'task':'transcribe','language':language})
execution={}
for part in ('encoder','decoder','decoder_with_past'):
 child=getattr(model,part,None)
 if child is not None and child.request is not None:
  request=child.request.get_compiled_model() if hasattr(child.request,'get_compiled_model') else child.request
  execution[part]=list(request.get_property('EXECUTION_DEVICES'))
report={'device':args.device,'num_beams':5,'model':args.model,'seconds':time.perf_counter()-start,'result':result,'detected_language':language,'execution_devices':execution}
(folder/f'optimum-{args.model}-{args.device.lower()}.json').write_text(json.dumps(report,indent=2));print(json.dumps(report,indent=2))
