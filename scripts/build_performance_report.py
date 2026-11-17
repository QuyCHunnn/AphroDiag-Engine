"""Build the measured audit report from retained JSON evidence."""
from collections import Counter
from pathlib import Path
from statistics import median
import json

root=Path(__file__).resolve().parents[1]
base=root/'artifacts/performance-audit/run-20261116'
load=lambda p: json.loads(p.read_text())
profile=load(base/'profile.json')
audio=load(base/'audio-ab/results.json')
first,warm,mkv=profile['runs']
mp4=next(s for s in first['stages'] if s['stage']=='export_mp4')
old=median(x['wall_seconds'] for x in audio if x['mode']=='mp3_then_aac')
new=median(x['wall_seconds'] for x in audio if x['mode']=='direct_wav_to_aac')
lines=['# Kiểm tra hiệu năng JellyDub — 2026-11-16','',
'## Phạm vi và cách đo','',
'- Video công khai n0LY3XcZaCs, dài 211 giây; SRT Việt 38 cue; Edge-TTS, voiceover, giọng NamMinh.',
'- Seed đúng 38 file giọng có khóa provider/text/speaker khớp. Cấm gọi TTS online trong harness. Đây là cold **media**, không phải cold toàn bộ dịch/ASR/TTS.',
'- Chạy tuần tự các benchmark tính toán; không chạy đồng thời ASR với render/audio A/B. Lượt media baseline có tác vụ chuẩn bị mẫu/tải model và kiểm tra metadata nhỏ trong thời gian render, nên đây là số đo vận hành thực tế, không phải benchmark phòng lab hoàn toàn cô lập.',
'- Wall time đo bằng perf_counter; CPU của tiến trình điều phối và các tiến trình con đo bằng getrusage. CPU-giây cộng dồn trên nhiều core có thể lớn hơn wall time. Không phải phần trăm GPU hoặc công suất điện.',
'- Baseline engine không sửa thuật toán. Hash source lưu tại `baseline-source-hashes.json`. Số đo dưới đây là trên máy hiện tại; không đại diện mọi video/phần cứng.','',
'## 1. Đường xử lý media hiện tại','',
'| Lượt chạy | Wall giây | CPU tiến trình điều phối giây | CPU tiến trình con giây | ffprobe | ffmpeg |',
'|---|---:|---:|---:|---:|---:|']
for run in profile['runs']:
 c=Counter()
 for x in run['commands']:c[x['tool']]+=x['count']
 lines.append(f"| {run['label']} | {run['wall_seconds']:.3f} | {run['python_cpu_seconds']:.3f} | {run['child_cpu_seconds']:.3f} | {c['ffprobe']} | {c['ffmpeg']} |")
lines += ['',f"Bỏ MP4 giảm thời gian từ {first['wall_seconds']:.3f}s xuống {mkv['wall_seconds']:.3f}s trong lượt cold-media tương ứng: nhanh hơn {first['wall_seconds']/mkv['wall_seconds']:.2f} lần. MKV-only của engine hiện tại vẫn xuất MP3; chưa phải bản sửa chỉ MKV+SRT.",'',
'| Stage ở lượt MKV+MP4 | Wall giây | CPU tiến trình con giây |','|---|---:|---:|']
for s in sorted(first['stages'],key=lambda s:s['wall_seconds'],reverse=True):
 if s['wall_seconds']>.1:lines.append(f"| {s['stage']} | {s['wall_seconds']:.3f} | {s['child_cpu_seconds']:.3f} |")
lines += ['',f"Riêng export_mp4 chiếm **{100*mp4['wall_seconds']/first['wall_seconds']:.2f}% wall time** và **{100*mp4['child_cpu_seconds']/first['child_cpu_seconds']:.2f}% CPU-giây tiến trình con**. Source gọi FFmpeg libx264/preset medium và render phụ đề. Đây là nút thắt đã đo của workload này. Tiến trình Python điều phối chỉ dùng {first['python_cpu_seconds']:.3f} CPU-giây; không có bằng chứng cần viết lại engine bằng ngôn ngữ khác để xử lý nút thắt này.",'',
'## 2. MP3 trung gian — A/B ba cặp','',
'| Đường xuất audio/MKV | Trung vị wall giây |','|---|---:|',
f'| WAV → MP3 → AAC/MKV | {old:.3f} |',f'| WAV → AAC/MKV trực tiếp | {new:.3f} |','',
f"Đo ba cặp, đảo thứ tự giữa các lượt. Bỏ MP3 tiết kiệm trung vị **{old-new:.3f}s**, khoảng **{100*(old-new)/old:.1f}%** của đoạn xuất audio/MKV; không được hiểu là giảm từng ấy phần trăm của toàn pipeline. Hai đường giữ video stream-copy; chưa nghiệm thu nghe sự khác biệt chất lượng AAC.",'',
'## 3. Cache và chi phí kiểm tra lại','',
'- Chỉ thay out_dir, mp4 hoặc subtitle_font đã làm đổi work_dir. Cache giọng cũ có 38 WAV; các work_dir mới không có WAV. Không cần suy đoán: đường dẫn và số file được ghi trong `cache-key-audit.json`.',
'- Lượt warm còn 80 lần ffprobe: 38 cho giọng đã cache, 38 cho clip đã căn thời gian và các probe phụ. Hai lần check_audio còn decode/quét âm lượng toàn audio dù file stage đã có cache.',
f"- Riêng align ở lượt warm mất {next(s['wall_seconds'] for s in warm['stages'] if s['stage']=='align'):.3f}s. Hai check_audio tổng {sum(s['wall_seconds'] for s in warm['stages'] if s['stage']=='check_audio'):.3f}s. Đây là overhead thực khi chạy lại, không phải chi phí model AI."]
meta=base/'wav-metadata-ab.json'
if meta.exists():
 data=load(meta);a=median(x['ffprobe_seconds'] for x in data);b=median(x['wav_header_seconds'] for x in data)
 lines += [f"- A/B metadata 38 WAV, ba lượt: ffprobe trung vị {a:.4f}s; đọc header PCM bằng wave trung vị {b:.4f}s; chênh thời lượng lớn nhất {max(x['max_duration_difference_seconds'] for x in data):.8f}s. Đây chỉ là đọc metadata, không thay được kiểm tra corruption/full decode."]
lines += ['','Dung lượng cache của video mẫu:']
for item in load(base/'cache-storage.json'):
 lines.append(f"- `{Path(item['work_dir']).name.rsplit('-',1)[-1]}`: {item['file_count']} file, {item['total_bytes']/1e6:.2f} MB. Có nhiều bản WAV trung gian và bản export trong cache; cần chính sách giữ/xóa theo job, không tự xóa cache đang dùng.")
lines += ['','## 4. Intel GPU: thực tế và ranh giới','',
'- Máy có Intel Iris Xe 8086:9A49. Ngoài sandbox WSL có /dev/dxg; OpenCL và OpenVINO nhận GPU. Ba smoke test số học chạy trên GPU.0 và đạt so sánh số học.',
'- Nhận định cũ “WSL chưa có GPU” từ kiểm tra trong sandbox là sai; xem [đính chính môi trường](../artifacts/performance-audit/ENVIRONMENT-CORRECTION.md). CUDA=False không phải cách phát hiện Intel GPU.',
'- Lượt engine với SRT Việt + Edge-TTS + voiceover không chạy ASR, không dịch và không tách nền bằng model cục bộ. Không có công đoạn model CUDA/Intel GPU cần chạy trong workload này; GPU thấp không tự chứng minh lỗi.',
'- Engine hiện dùng faster-whisper/CTranslate2 cho ASR và PyTorch CPU/CUDA cho dịch; chưa tích hợp OpenVINO. Các benchmark ASR sau là **backend thử nghiệm riêng**, không phải số đo tốc độ ASR hiện tại của engine.',
'- Cùng model OpenVINO INT8, cùng WAV 16 kHz và cấu hình giải mã, chạy CPU và GPU tuần tự, không dùng AUTO fallback. Load/compile ghi riêng; mỗi sample chạy hai lượt. Có ba đoạn 20/40/60 giây từ cùng video, chưa phải ba miền dữ liệu độc lập.']
for model in ('tiny','small','small-beam1'):
 directory=root/'artifacts/performance-audit/asr'
 if model!='tiny':directory=directory/model
 cp=directory/'cpu.json';gp=directory/'gpu.json'
 if not (cp.exists() and gp.exists()):
  lines += ['',f'### Whisper {model}: chưa có đủ hai backend để kết luận'];continue
 cpu,gpu=load(cp),load(gp)
 if model=='small' and not gpu['samples']:
  lines += ['', '### Whisper Small, beam=5: GPU thất bại', '',
            '| Audio giây | CPU lượt 1/2 giây | GPU |','|---:|---:|---|']
  for a in cpu['samples']:
   cv='/'.join(f"{x['wall_seconds']:.3f}" for x in a['trials'])
   lines.append(f"| {a['duration_seconds']} | {cv} | Lỗi ở mẫu 20s đầu tiên; không có thời gian inference thành công |")
  lines += ['', 'GPU load/compile thành công nhưng generate với beam=5 lỗi `IRemoteTensor: Not Implemented`. Không coi lượt lỗi là GPU chậm hoặc GPU không tồn tại. CPU với cùng beam=5 chạy được. Lượt beam=1 được kiểm tra riêng bên dưới; không so CPU beam=5 với GPU beam=1.',
            'Cùng triệu chứng được báo cáo trong [OpenVINO GenAI issue #2746](https://github.com/openvinotoolkit/openvino.genai/issues/2746). Đây là bằng chứng hỗ trợ giới hạn đường beam search ở runtime đang thử; chưa chứng minh nội bộ lỗi nằm ở driver hay phần nào của thư viện. Bằng chứng local: `asr/small/gpu-failure.json`.']
  continue
 lines += ['',f'### Whisper {model}, num_beams={cpu["settings"]["num_beams"]}','',
 f"Load/compile CPU {cpu['model_load_compile_seconds']:.3f}s; GPU {gpu['model_load_compile_seconds']:.3f}s.",'',
 '| Audio giây | CPU lượt 1/2 giây | GPU lượt 1/2 giây |','|---:|---:|---:|']
 for a,b in zip(cpu['samples'],gpu['samples']):
  cv='/'.join(f"{x['wall_seconds']:.3f}" for x in a['trials']);gv='/'.join(f"{x['wall_seconds']:.3f}" for x in b['trials'])
  lines.append(f"| {a['duration_seconds']} | {cv} | {gv} |")
 lines += ['',f"Model revision: `{Path(cpu['model_path']).name}`. Transcript/chunk đầy đủ lưu trong JSON CPU/GPU. Văn bản giữa hai device có thể khác, chưa đo WER/CER với bản chép lời đã nghe duyệt; không lấy tốc độ thay cho nghiệm thu chất lượng."]
lines += ['', '### Diễn giải giới hạn kết quả GPU', '',
'Với Tiny beam=1, GPU không nhanh hơn một cách nhất quán trong các lượt này. Với Small beam=1, mẫu 20s CPU nhanh hơn; 40/60s GPU nhanh hơn khoảng 1.4–1.5 lần khi chỉ tính inference. Load/compile GPU tách riêng và có thể xóa lợi ích ở job ngắn. Không so CPU Small beam=5 với GPU beam=1 để tuyên bố tăng tốc; chất lượng transcript giữa các cấu hình chưa được nghiệm thu.',
'Lỗi GPU Small beam=5 đã tái hiện hai lần; lần thứ hai có exception trong asr/small/gpu.json. Không tự động hạ beam rồi gọi đó là cùng cấu hình/chất lượng. Chưa nâng runtime để kiểm chứng bản khác có sửa lỗi hay không.',
'', '## 5. Danh sách vấn đề và ưu tiên','',
'| Ưu tiên | Vấn đề đã xác nhận | Bằng chứng | Hướng sửa |',
'|---|---|---|---|',
'| P1 | Auto ép xuất MP4 dù mục tiêu chỉ MKV/SRT | pipeline.auto force mp4=True; stage export_mp4 chiếm phần lớn thời gian/CPU | Mặc định chỉ MKV+SRT, giữ video stream-copy, MP4 opt-in |',
'| P1 | Encode audio qua MP3 rồi encode AAC tiếp | Source export_mp3 rồi mux_mkv; A/B ba cặp | Dùng WAV đã phối để encode AAC trực tiếp; MP3 chỉ opt-in |',
'| P1 | Cache TTS phụ thuộc cấu hình xuất không liên quan | Đổi folder/font/MP4 làm đổi work_dir | Tách cache giọng theo provider/text/voice; cache export theo dependency riêng |',
'| P2 | Quá nhiều probe cho WAV đã cache | 80 ffprobe ở lượt warm | Đọc header PCM, tái sử dụng metadata khi file không đổi |',
'| P2 | Hai lần quét full audio lặp ở warm run | check_audio tổng khoảng 3 giây | Cache kết quả QC theo hash/version, kiểm tra lại khi audio đổi |',
'| P2 | Clip timing tạo/đọc nhiều file và subprocess | 38 FFmpeg khi align cold; 38 ffprobe khi warm | Tối ưu metadata trước; chỉ gộp filter/timeline sau khi profile lại |',
'| P2 | Intel GPU chưa có backend trong engine, beam=5 còn lỗi ở stack thử nghiệm | Tiny/Small beam=1 chạy; Small beam=5 fail hai lần | Không flip GPU mặc định; kiểm chứng runtime/beam/quality trước tích hợp |',
'| P2 | Cache giữ nhiều bản trung gian | Số file/byte tại cache-storage.json | Chính sách lưu theo job/quota; giữ khả năng retry, không xóa ngầm |','',
'## 6. Những điều chưa được chứng minh','',
'- Chưa đo cold-start toàn bộ dịch + ASR + TTS online của engine. Không có kết luận về latency dịch vụ Edge hoặc model dịch OPUS từ phép đo SRT Việt này.',
'- Chưa có phép đo peak GPU utilization, nhiệt độ, điện năng hay bộ nhớ đỉnh. Không khẳng định nguyên nhân GPU nhanh/chậm là transfer/kernel overhead nếu chưa profiling mức kernel.',
'- Chưa sửa thuật toán/mặc định xuất file trong src/jellydub ở đợt audit này. Runtime Intel và môi trường thử riêng đã cấu hình; bản vá tối ưu production là bước tiếp theo theo các ưu tiên trên.',
'- Các tệp video đầu ra audit là dữ liệu thử, không thay bộ bàn giao đã duyệt. Chưa nghiệm thu nghe/VLC.','',
'## Kiểm chứng artifact', '', 'Hai đường audio/MKV được decode và so hash bitstream video với nguồn tại `run-20261116/verification.json`. Source engine được đối chiếu hash trước/sau. Không coi kiểm tra này là nghiệm thu nghe.', '', '## Tái lập','',
'```bash',
'# Chọn output mới vì harness yêu cầu cold media cache.',
'.venv/bin/python scripts/profile_engine.py artifacts/performance-audit/run-new',
'.venv/bin/python scripts/benchmark_audio_path.py',
'.venv/bin/python scripts/benchmark_wav_metadata.py',
'.venv-igpu/bin/python scripts/benchmark_igpu_asr.py --device CPU',
'.venv-igpu/bin/python scripts/benchmark_igpu_asr.py --device GPU',
'.venv-igpu/bin/python scripts/benchmark_igpu_asr.py --device CPU --model small --beams 5',
'# Lệnh beam=5 trên GPU hiện dự kiến fail; dùng để tái hiện lỗi.',
'.venv-igpu/bin/python scripts/benchmark_igpu_asr.py --device GPU --model small --beams 5',
'.venv-igpu/bin/python scripts/benchmark_igpu_asr.py --device CPU --model small --beams 1',
'.venv-igpu/bin/python scripts/benchmark_igpu_asr.py --device GPU --model small --beams 1',
'```','',
'Các script audio/header tham chiếu lượt run-20261116. GPU phải chạy ngoài sandbox che /dev/dxg. Môi trường/dependency: docs/igpu-wsl.md và artifacts/igpu-setup/requirements.lock.txt. Model download bằng scripts/download_igpu_whisper.py [tiny|small], chỉ tải weights/config, không chạy remote code.','',
'Nguồn API: [OpenVINO WhisperPipeline](https://docs.openvino.ai/2025/api/genai_api/_autosummary/openvino_genai.WhisperPipeline.html), [Intel GPU/WSL](https://docs.openvino.ai/2024/get-started/configurations/configurations-intel-gpu.html).']
path=root/'docs/performance-audit-2026-11-16.md';path.write_text('\n'.join(lines)+'\n');print(path)
