# Kiểm tra hiệu năng JellyDub — 2026-11-16

## Phạm vi và cách đo

- Video công khai n0LY3XcZaCs, dài 211 giây; SRT Việt 38 cue; Edge-TTS, voiceover, giọng NamMinh.
- Seed đúng 38 file giọng có khóa provider/text/speaker khớp. Cấm gọi TTS online trong harness. Đây là cold **media**, không phải cold toàn bộ dịch/ASR/TTS.
- Chạy tuần tự các benchmark tính toán; không chạy đồng thời ASR với render/audio A/B. Lượt media baseline có tác vụ chuẩn bị mẫu/tải model và kiểm tra metadata nhỏ trong thời gian render, nên đây là số đo vận hành thực tế, không phải benchmark phòng lab hoàn toàn cô lập.
- Wall time đo bằng perf_counter; CPU của tiến trình điều phối và các tiến trình con đo bằng getrusage. CPU-giây cộng dồn trên nhiều core có thể lớn hơn wall time. Không phải phần trăm GPU hoặc công suất điện.
- Baseline engine không sửa thuật toán. Hash source lưu tại `baseline-source-hashes.json`. Số đo dưới đây là trên máy hiện tại; không đại diện mọi video/phần cứng.

## 1. Đường xử lý media hiện tại

| Lượt chạy | Wall giây | CPU tiến trình điều phối giây | CPU tiến trình con giây | ffprobe | ffmpeg |
|---|---:|---:|---:|---:|---:|
| both-cold-media | 285.447 | 0.232 | 1003.810 | 83 | 45 |
| both-warm | 7.624 | 0.739 | 8.797 | 80 | 2 |
| mkv-only-cold-media | 27.110 | 0.200 | 27.440 | 83 | 44 |

Bỏ MP4 giảm thời gian từ 285.447s xuống 27.110s trong lượt cold-media tương ứng: nhanh hơn 10.53 lần. MKV-only của engine hiện tại vẫn xuất MP3; chưa phải bản sửa chỉ MKV+SRT.

| Stage ở lượt MKV+MP4 | Wall giây | CPU tiến trình con giây |
|---|---:|---:|
| export_mp4 | 258.388 | 977.819 |
| mux_mkv | 9.882 | 9.011 |
| align | 5.506 | 4.893 |
| export_mp3 | 4.141 | 3.607 |
| mix | 2.288 | 2.099 |
| extract_audio | 1.627 | 1.349 |
| check_audio | 1.389 | 1.120 |
| check_audio | 1.179 | 1.132 |
| synthesize_cues | 0.765 | 2.616 |
| _resolve_subtitle | 0.144 | 0.085 |

Riêng export_mp4 chiếm **90.52% wall time** và **97.41% CPU-giây tiến trình con**. Source gọi FFmpeg libx264/preset medium và render phụ đề. Đây là nút thắt đã đo của workload này. Tiến trình Python điều phối chỉ dùng 0.232 CPU-giây; không có bằng chứng cần viết lại engine bằng ngôn ngữ khác để xử lý nút thắt này.

## 2. MP3 trung gian — A/B ba cặp

| Đường xuất audio/MKV | Trung vị wall giây |
|---|---:|
| WAV → MP3 → AAC/MKV | 13.607 |
| WAV → AAC/MKV trực tiếp | 9.302 |

Đo ba cặp, đảo thứ tự giữa các lượt. Bỏ MP3 tiết kiệm trung vị **4.305s**, khoảng **31.6%** của đoạn xuất audio/MKV; không được hiểu là giảm từng ấy phần trăm của toàn pipeline. Hai đường giữ video stream-copy; chưa nghiệm thu nghe sự khác biệt chất lượng AAC.

## 3. Cache và chi phí kiểm tra lại

- Chỉ thay out_dir, mp4 hoặc subtitle_font đã làm đổi work_dir. Cache giọng cũ có 38 WAV; các work_dir mới không có WAV. Không cần suy đoán: đường dẫn và số file được ghi trong `cache-key-audit.json`.
- Lượt warm còn 80 lần ffprobe: 38 cho giọng đã cache, 38 cho clip đã căn thời gian và các probe phụ. Hai lần check_audio còn decode/quét âm lượng toàn audio dù file stage đã có cache.
- Riêng align ở lượt warm mất 2.943s. Hai check_audio tổng 2.956s. Đây là overhead thực khi chạy lại, không phải chi phí model AI.
- A/B metadata 38 WAV, ba lượt: ffprobe trung vị 2.2488s; đọc header PCM bằng wave trung vị 0.0008s; chênh thời lượng lớn nhất 0.00000033s. Đây chỉ là đọc metadata, không thay được kiểm tra corruption/full decode.

Dung lượng cache của video mẫu:
- `4b4bfd9c11116ab2`: 82 file, 163.66 MB. Có nhiều bản WAV trung gian và bản export trong cache; cần chính sách giữ/xóa theo job, không tự xóa cache đang dùng.
- `57ac87416b79ac4f`: 81 file, 147.27 MB. Có nhiều bản WAV trung gian và bản export trong cache; cần chính sách giữ/xóa theo job, không tự xóa cache đang dùng.

## 4. Intel GPU: thực tế và ranh giới

- Máy có Intel Iris Xe 8086:9A49. Ngoài sandbox WSL có /dev/dxg; OpenCL và OpenVINO nhận GPU. Ba smoke test số học chạy trên GPU.0 và đạt so sánh số học.
- Nhận định cũ “WSL chưa có GPU” từ kiểm tra trong sandbox là sai; xem [đính chính môi trường](../artifacts/performance-audit/ENVIRONMENT-CORRECTION.md). CUDA=False không phải cách phát hiện Intel GPU.
- Lượt engine với SRT Việt + Edge-TTS + voiceover không chạy ASR, không dịch và không tách nền bằng model cục bộ. Không có công đoạn model CUDA/Intel GPU cần chạy trong workload này; GPU thấp không tự chứng minh lỗi.
- Engine hiện dùng faster-whisper/CTranslate2 cho ASR và PyTorch CPU/CUDA cho dịch; chưa tích hợp OpenVINO. Các benchmark ASR sau là **backend thử nghiệm riêng**, không phải số đo tốc độ ASR hiện tại của engine.
- Cùng model OpenVINO INT8, cùng WAV 16 kHz và cấu hình giải mã, chạy CPU và GPU tuần tự, không dùng AUTO fallback. Load/compile ghi riêng; mỗi sample chạy hai lượt. Có ba đoạn 20/40/60 giây từ cùng video, chưa phải ba miền dữ liệu độc lập.

### Whisper tiny, num_beams=1

Load/compile CPU 1.137s; GPU 12.146s.

| Audio giây | CPU lượt 1/2 giây | GPU lượt 1/2 giây |
|---:|---:|---:|
| 20 | 1.048/0.671 | 1.762/3.417 |
| 40 | 1.614/1.468 | 2.242/1.907 |
| 60 | 2.610/2.929 | 3.186/2.837 |

Model revision: `a850762d97243dee30f46ca309720541af619ab0`. Transcript/chunk đầy đủ lưu trong JSON CPU/GPU. Văn bản giữa hai device có thể khác, chưa đo WER/CER với bản chép lời đã nghe duyệt; không lấy tốc độ thay cho nghiệm thu chất lượng.

### Whisper Small, beam=5: GPU thất bại

| Audio giây | CPU lượt 1/2 giây | GPU |
|---:|---:|---|
| 20 | 34.760/35.477 | Lỗi ở mẫu 20s đầu tiên; không có thời gian inference thành công |
| 40 | 38.507/47.251 | Lỗi ở mẫu 20s đầu tiên; không có thời gian inference thành công |
| 60 | 54.430/53.083 | Lỗi ở mẫu 20s đầu tiên; không có thời gian inference thành công |

GPU load/compile thành công nhưng generate với beam=5 lỗi `IRemoteTensor: Not Implemented`. Không coi lượt lỗi là GPU chậm hoặc GPU không tồn tại. CPU với cùng beam=5 chạy được. Lượt beam=1 được kiểm tra riêng bên dưới; không so CPU beam=5 với GPU beam=1.
Cùng triệu chứng được báo cáo trong [OpenVINO GenAI issue #2746](https://github.com/openvinotoolkit/openvino.genai/issues/2746). Đây là bằng chứng hỗ trợ giới hạn đường beam search ở runtime đang thử; chưa chứng minh nội bộ lỗi nằm ở driver hay phần nào của thư viện. Bằng chứng local: `asr/small/gpu-failure.json`.

### Whisper small-beam1, num_beams=1

Load/compile CPU 1.536s; GPU 2.963s.

| Audio giây | CPU lượt 1/2 giây | GPU lượt 1/2 giây |
|---:|---:|---:|
| 20 | 3.621/2.993 | 3.772/4.258 |
| 40 | 5.672/5.414 | 3.862/3.610 |
| 60 | 9.477/9.345 | 6.491/6.703 |

Model revision: `5b831719e093f86e1970be663e524fe001489f9b`. Transcript/chunk đầy đủ lưu trong JSON CPU/GPU. Văn bản giữa hai device có thể khác, chưa đo WER/CER với bản chép lời đã nghe duyệt; không lấy tốc độ thay cho nghiệm thu chất lượng.

### Diễn giải giới hạn kết quả GPU

Với Tiny beam=1, GPU không nhanh hơn một cách nhất quán trong các lượt này. Với Small beam=1, mẫu 20s CPU nhanh hơn; 40/60s GPU nhanh hơn khoảng 1.4–1.5 lần khi chỉ tính inference. Load/compile GPU tách riêng và có thể xóa lợi ích ở job ngắn. Không so CPU Small beam=5 với GPU beam=1 để tuyên bố tăng tốc; chất lượng transcript giữa các cấu hình chưa được nghiệm thu.
Lỗi GPU Small beam=5 đã tái hiện hai lần; lần thứ hai có exception trong asr/small/gpu.json. Không tự động hạ beam rồi gọi đó là cùng cấu hình/chất lượng. Chưa nâng runtime để kiểm chứng bản khác có sửa lỗi hay không.

## 5. Danh sách vấn đề và ưu tiên

| Ưu tiên | Vấn đề đã xác nhận | Bằng chứng | Hướng sửa |
|---|---|---|---|
| P1 | Auto ép xuất MP4 dù mục tiêu chỉ MKV/SRT | pipeline.auto force mp4=True; stage export_mp4 chiếm phần lớn thời gian/CPU | Mặc định chỉ MKV+SRT, giữ video stream-copy, MP4 opt-in |
| P1 | Encode audio qua MP3 rồi encode AAC tiếp | Source export_mp3 rồi mux_mkv; A/B ba cặp | Dùng WAV đã phối để encode AAC trực tiếp; MP3 chỉ opt-in |
| P1 | Cache TTS phụ thuộc cấu hình xuất không liên quan | Đổi folder/font/MP4 làm đổi work_dir | Tách cache giọng theo provider/text/voice; cache export theo dependency riêng |
| P2 | Quá nhiều probe cho WAV đã cache | 80 ffprobe ở lượt warm | Đọc header PCM, tái sử dụng metadata khi file không đổi |
| P2 | Hai lần quét full audio lặp ở warm run | check_audio tổng khoảng 3 giây | Cache kết quả QC theo hash/version, kiểm tra lại khi audio đổi |
| P2 | Clip timing tạo/đọc nhiều file và subprocess | 38 FFmpeg khi align cold; 38 ffprobe khi warm | Tối ưu metadata trước; chỉ gộp filter/timeline sau khi profile lại |
| P2 | Intel GPU chưa có backend trong engine, beam=5 còn lỗi ở stack thử nghiệm | Tiny/Small beam=1 chạy; Small beam=5 fail hai lần | Không flip GPU mặc định; kiểm chứng runtime/beam/quality trước tích hợp |
| P2 | Cache giữ nhiều bản trung gian | Số file/byte tại cache-storage.json | Chính sách lưu theo job/quota; giữ khả năng retry, không xóa ngầm |

## 6. Những điều chưa được chứng minh

- Chưa đo cold-start toàn bộ dịch + ASR + TTS online của engine. Không có kết luận về latency dịch vụ Edge hoặc model dịch OPUS từ phép đo SRT Việt này.
- Chưa có phép đo peak GPU utilization, nhiệt độ, điện năng hay bộ nhớ đỉnh. Không khẳng định nguyên nhân GPU nhanh/chậm là transfer/kernel overhead nếu chưa profiling mức kernel.
- Chưa sửa thuật toán/mặc định xuất file trong src/jellydub ở đợt audit này. Runtime Intel và môi trường thử riêng đã cấu hình; bản vá tối ưu production là bước tiếp theo theo các ưu tiên trên.
- Các tệp video đầu ra audit là dữ liệu thử, không thay bộ bàn giao đã duyệt. Chưa nghiệm thu nghe/VLC.

## Kiểm chứng artifact

Hai đường audio/MKV được decode và so hash bitstream video với nguồn tại `run-20261116/verification.json`. Source engine được đối chiếu hash trước/sau. Không coi kiểm tra này là nghiệm thu nghe.

## Tái lập

```bash
# Chọn output mới vì harness yêu cầu cold media cache.
.venv/bin/python scripts/profile_engine.py artifacts/performance-audit/run-new
.venv/bin/python scripts/benchmark_audio_path.py
.venv/bin/python scripts/benchmark_wav_metadata.py
.venv-igpu/bin/python scripts/benchmark_igpu_asr.py --device CPU
.venv-igpu/bin/python scripts/benchmark_igpu_asr.py --device GPU
.venv-igpu/bin/python scripts/benchmark_igpu_asr.py --device CPU --model small --beams 5
# Lệnh beam=5 trên GPU hiện dự kiến fail; dùng để tái hiện lỗi.
.venv-igpu/bin/python scripts/benchmark_igpu_asr.py --device GPU --model small --beams 5
.venv-igpu/bin/python scripts/benchmark_igpu_asr.py --device CPU --model small --beams 1
.venv-igpu/bin/python scripts/benchmark_igpu_asr.py --device GPU --model small --beams 1
```

Các script audio/header tham chiếu lượt run-20261116. GPU phải chạy ngoài sandbox che /dev/dxg. Môi trường/dependency: docs/igpu-wsl.md và artifacts/igpu-setup/requirements.lock.txt. Model download bằng scripts/download_igpu_whisper.py [tiny|small], chỉ tải weights/config, không chạy remote code.

Nguồn API: [OpenVINO WhisperPipeline](https://docs.openvino.ai/2025/api/genai_api/_autosummary/openvino_genai.WhisperPipeline.html), [Intel GPU/WSL](https://docs.openvino.ai/2024/get-started/configurations/configurations-intel-gpu.html).
