# Audit điểm nghẽn pipeline — 2026-11-16

Phạm vi: bản repo/runtime đang chạy, gồm phần tích hợp đã thêm tại workspace. Không quy mọi điểm nghẽn cho upstream JellyDub. Lượt audit này chỉ thêm script đo và báo cáo; chưa áp dụng tuning hoặc đổi backend/giọng vào pipeline chính.

## Kết luận có số đo

- Family Guy 156.665 giây, lượt đầy đủ 615.50 giây: khối TTS+normalize 396.64s (64.4%), ASR 145.72s (23.7%), dịch 33.56s (5.5%). Ba khối chiếm 93.6%; dựng timeline PCM bằng Python chỉ 0.07s.
- BLAS đối chứng không cProfile, thứ tự 2–8–2–8, giữ ORT 4 luồng: BLAS8 trung bình 6.686s, BLAS2 2.870s, nhanh 2.33×. Cả bốn waveform có SHA-256 giống nhau. Chỉ là mẫu 2 giây, chưa tuyên bố cả video nhanh tương ứng.
- Profile có instrumentation hai mẫu: Peter 8.993→2.637s, Lois 13.097→4.427s khi BLAS8→2; waveform giống nhau ở mỗi text giữa các cấu hình. Đối chứng trên giúp giảm ảnh hưởng thứ tự/warmup.
- Mẫu Peter: 25 frame, 16 codebook/frame → 400 acoustic ONNX runs, thêm 24 decoder runs, 1 prefill, 1 codec. Mẫu Lois 44 frame → 704 acoustic runs. Heads và sampling còn được tính bằng NumPy ở host.
- ASR mẫu 20 giây: SDK import 27.30s; transcribe 26.98s sau import, encoder/decoder GPU.0 nhưng logits/device tương thích Torch là CPU, 61 decoder forwards. Không cộng thời gian forward và compile wrapper vì các timer có thể lồng nhau.
- Runtime: Intel Iris Xe iGPU, Torch 2.14.1+cpu, CUDA/XPU unavailable; ONNX providers Azure/CPU. OpenVINO GPU khả dụng không làm ONNX/Torch tự có GPU backend.

## Python và native thực sự nằm ở đâu

ONNX Runtime, OpenVINO, BLAS, Torch kernels và FFmpeg có phần thực thi native. Process mang tên python hoặc CPU self time của process không chứng minh Python bytecode chiếm thời gian đó. cProfile self time trong hàm `samp` chứa phép `@` gọi native BLAS; không được diễn giải toàn bộ thành pure Python.

Tuy nhiên VieNeu lite có vòng autoregressive theo frame/codebook bằng Python, nhiều lượt gọi graph nhỏ và NumPy output heads/sampling ngoài graph. ASR Optimum có generation/beam trên CPU, `start_async()` rồi `wait()` từng bước và clone logits thành CPU tensor. Việc chuyển thêm phần này vào graph/native có cơ sở khảo sát, nhưng chưa có phép đo so sánh Rust/C++/fused graph để cam kết mức tăng tốc.

Intel iGPU chia sẻ RAM; audit không gọi mọi lần materialize logits là truyền qua PCIe, và không suy GPU utilization từ CPU usage.

## Danh sách đầy đủ trong phạm vi audit

### F01 — BLAS 8 luồng chậm hơn 2 luồng, trong khi ORT có pool 4 luồng riêng (P0)

**Bằng chứng — Đã đo A/B:** Cùng audio 2 giây, thứ tự 2–8–2–8: 6.686 → 2.870s (2.33×), hash cả 4 lần giống nhau.

**Next step:** Giới hạn BLAS trong phạm vi TTS; expose riêng ORT/BLAS threads, không đổi thread toàn process vĩnh viễn.

**Kiểm chứng:** A/B 1/2/4/8 trên bộ câu cố định; fixed seed, hash audio và thời gian full video.

### F02 — TTS không có đường GPU trong cấu hình hiện tại (P0)

**Bằng chứng — Mã + runtime:** pipeline.py:188 auto→cpu; VieNeu ONNX chọn CPUExecutionProvider; runtime ORT chỉ có Azure/CPU.

**Next step:** Chọn backend có khả năng chạy GPU đúng phần cứng, giữ CPU fallback có lý do; không chỉ sửa chuỗi cpu thành gpu.

**Kiểm chứng:** Báo requested/actual device từng stage; provider thực thi GPU được xác nhận bằng runtime.

### F03 — Một phần phép tính và sampling TTS nằm ngoài ONNX graph (P1)

**Bằng chứng — Mã + profile:** 16 codebook/frame; mẫu Peter: 25 frame, 400 acoustic ONNX calls + NumPy output heads/sampling; 3.52s audio: 704 acoustic calls.

**Next step:** Prototype đưa heads/sampling và acoustic frame vào graph/native, giảm số lần đi qua ranh giới Python/runtime.

**Kiểm chứng:** Giữ model/seed/voice; so output/độ dài/chất lượng; đếm runtime calls và tốc độ, không chỉ % GPU.

### F04 — TTS xử lý một cue/lần, chưa batch các câu (P1)

**Bằng chứng — Mã nguồn:** VieNeuTTS.max_concurrency=1, adapter chỉ gọi infer; SDK CPU giữ RLock và chạy tuần tự.

**Next step:** Bổ sung batch interface khi backend hỗ trợ; gom theo độ dài; không tăng Python workers mù quáng.

**Kiểm chứng:** So throughput và VRAM/RAM; không chồng/mất câu; CPU thread contention không tăng.

### F05 — Chính sách chọn thiết bị chưa thống nhất theo từng công đoạn (P1)

**Bằng chứng — Mã + runtime:** ASR thấy OpenVINO GPU, translator chỉ auto CUDA/CPU; PyTorch là +cpu, CUDA/XPU false.

**Next step:** Tách asr/tts/translate backend và device; doctor in bảng capability thực; auto chọn đường khả dụng.

**Kiểm chứng:** CPU-only/Intel/NVIDIA đều có test lựa chọn và fallback; phần cứng chưa test phải ghi rõ.

### F06 — ASR GPU vẫn có vòng generation/beam trên CPU, đồng bộ từng bước (P1)

**Bằng chứng — Mã + profile:** Optimum start_async→wait→logits CPU clone; mẫu 20s: 61 decoder forwards, logits cpu, encoder/decoder GPU.0.

**Next step:** Đo/fuse generation/native backend có hỗ trợ đúng beam=5; giảm materialize và sync nếu API hỗ trợ.

**Kiểm chứng:** So transcript/timestamp và runtime count; chưa gán toàn bộ 145.7s cho Python hay copy.

### F07 — Precision chưa được lựa chọn qua A/B theo phần cứng (P1)

**Bằng chứng — Cấu hình xác nhận, lợi ích chưa đo:** ASR INFERENCE_PRECISION_HINT=f32 trên weights INT8; TTS ONNX FP32; adapter chưa expose lựa chọn precision.

**Next step:** A/B FP32/FP16/INT8 đúng backend/device; không suy rằng đổi dtype chắc chắn nhanh/chất lượng tương đương.

**Kiểm chứng:** Nghe/reference transcript, không NaN/mất câu; ghi cả load, inference, chất lượng.

### F08 — ASR từng sinh lặp dài ở tiếng hét; guard hiện tại chưa có regression nội dung đầy đủ (P0)

**Bằng chứng — Đã đo + đã giảm lỗi:** Family Guy 353.44→145.72s sau no-repeat + token budget; raw SRT cũ có hàng nghìn chữ A.

**Next step:** Giữ guard; kiểm tra cap có chạm trần/truncated, phân biệt phi ngôn ngữ; thêm bộ EN/VI nói nhanh/tiếng hét.

**Kiểm chứng:** Không lặp vô hạn; không cắt câu thật vì token cap; preserve âm gốc cho hiệu ứng.

### F09 — Fallback ASR bắt Exception rộng và chạy lại cả CPU; chưa checkpoint từng vùng (P1)

**Bằng chứng — Mã + ca lỗi trước:** Data/timestamp error cũng có thể kích hoạt cùng-model CPU retry; cues chỉ xuất sau khi xong toàn clip.

**Next step:** Phân loại lỗi backend với lỗi transcript; retry/repair vùng lỗi, checkpoint theo vùng có version/model/hash.

**Kiểm chứng:** Lỗi dữ liệu không gây rerun CPU vô ích; resume không lặp toàn clip hoặc mất vùng đã xong.

### F10 — Import/load/compile lặp khi mở process/job mới (P1)

**Bằng chứng — Đã đo mẫu, chưa suy ra toàn video:** Mẫu ASR: import SDK 27.30s, transcribe 26.98s; compile wrappers ~5.07s; probe TTS init 34.18s (khác process/context).

**Next step:** Tái sử dụng model/session trong batch CLI; khảo sát compile cache; tách first-install/cold/warm.

**Kiểm chứng:** Batch nhiều video không load lại model mỗi cue/job; theo dõi RAM; không bắt file đầu ra phụ thuộc server.

### F11 — Dịch OPUS hiện chạy CPU, không có Intel backend (P1)

**Bằng chứng — Mã + số đo:** Family Guy 33.56s CPU; auto kiểm tra torch.cuda, không OpenVINO/XPU.

**Next step:** Prototype backend dịch phù hợp Intel/NVIDIA; batch theo độ dài, giữ cache nội dung và context.

**Kiểm chứng:** Đo nghĩa/thuật ngữ và tốc độ; không đổi model nhỏ hơn chỉ để lấy benchmark đẹp.

### F12 — Chờ ngẫu nhiên và subprocess FFmpeg được áp dụng cả TTS local (P2)

**Bằng chứng — Mã + CPU child time:** sleep 0.1–0.3s mỗi miss: 43 cue tương đương 4.3–12.9s chờ, chưa kể retry; FFmpeg normalize 1 process/cue; TTS child CPU 11.54s.

**Next step:** Chỉ rate-limit provider mạng; đo rồi batch/in-process normalize nếu giữ được âm thanh.

**Kiểm chứng:** Hash/timing/loudness không đổi; đo spawn/normalize riêng; đây không phải toàn bộ 396.6s.

### F13 — Đổi một giọng làm cache các giọng khác đổi theo (P1)

**Bằng chứng — Đã tái hiện:** Chỉ đổi LOIS, key PETER vẫn đổi vì cache_id chứa toàn speaker_voices.

**Next step:** Khóa cache theo resolved voice/ref của chính cue, model và tham số thực ảnh hưởng output.

**Kiểm chứng:** Đổi Lois chỉ tạo lại câu Lois; Peter/Brian reuse; sửa ref chỉ ảnh hưởng đúng người.

### F14 — Đường dẫn cache tương đối/tuyệt đối tạo job key khác dù cùng thư mục (P1)

**Bằng chứng — Đã tái hiện:** cache_dir còn nằm trong asdict dùng cho _work_dir; hai đường dẫn alias tạo hai job khác nhau.

**Next step:** Canonicalize paths và loại tham số vị trí lưu khỏi fingerprint nội dung.

**Kiểm chứng:** Hai alias cùng root dùng chung job/cache, không render/mix lại vô ích.

### F15 — Cache dịch bị đặt dưới workdir phụ thuộc toàn bộ SRT (P1)

**Bằng chứng — Mã nguồn:** _translated dùng work/translations; SRT đổi làm workdir đổi, mất reuse các câu dịch không đổi.

**Next step:** Dùng cache dịch chia sẻ theo model/text/nhóm context/glossary version.

**Kiểm chứng:** Sửa một câu chỉ dịch lại nhóm context bị ảnh hưởng; không dùng nhầm ngữ cảnh cũ.

### F16 — Các stage chạy nối tiếp, chưa có lịch CPU/GPU/batch thực sự (P2)

**Bằng chứng — Mã nguồn, lợi ích cần A/B:** Thời gian các stage cộng gần bằng tổng; GPU không có việc khi TTS CPU chạy 396.6s.

**Next step:** Sau khi backend ổn, thử pipeline bounded queue/batch; ưu tiên việc độc lập, không tranh CPU giữa translator và TTS.

**Kiểm chứng:** Đo throughput/peak RAM/latency/chất lượng; không giả định thêm thread luôn nhanh.

### F17 — Gán nhiều preset không tạo ra diễn xuất nhân vật (P0 chất lượng)

**Bằng chứng — Mã + phản hồi người dùng:** Adapter chỉ text+voice; style v3 Turbo deprecated/ignored; cue rời không có hướng dẫn hội thoại/cảm xúc.

**Next step:** Chọn backend/voice qua mẫu đối thoại có ngữ cảnh và delivery thực được hỗ trợ; không chỉ đổi tên preset.

**Kiểm chứng:** Người dùng chấp nhận mẫu Peter/Lois/Stewie trước khi render cả clip; nam/nữ khác với tuổi/cảm xúc.

### F18 — Nhận diện/tách người nói đang làm thủ công theo ledger (P1 chất lượng)

**Bằng chứng — Ca clip + mã:** Whisper gộp Peter/Lois và Brian/Stewie; mapping chuẩn bị sẵn, không phải diarization; gender QA chỉ kiểm preset.

**Next step:** Thiết kế speaker-turn/diarization và review boundary; gán voice ổn định; phân biệt đúng preset với đúng người thật.

**Kiểm chứng:** So bộ clip có nhãn, đo lỗi đổi người/nhầm giọng/overlap; uncertain phải flag thay vì tự coi đạt.

### F19 — Gate hiện tại thiên về file kỹ thuật, không chặn giọng/bản dịch kém (P0 chất lượng)

**Bằng chứng — Kết quả bàn giao + mã:** Decode/hash/track pass nhưng người dùng không chấp nhận giọng; OPUS được biên tập trước.

**Next step:** Gate mẫu nghe + nghĩa + delivery + timing; lưu nội dung đã duyệt và regenerate đúng câu.

**Kiểm chứng:** Không dùng pytest/ffprobe thay nghiệm thu nghe; dừng bulk generation khi sample không đạt.

### F20 — Chưa có đo hoàn chỉnh toàn bộ workflow không chuẩn bị thủ công (P1 vận hành)

**Bằng chứng — Báo cáo đo đã ghi rõ:** 615.50s và sửa 52.12s không gồm ledger/biên tập/tải model; áp review 0.002s không phải thời gian con người dịch/duyệt.

**Next step:** Đo tách install/model, cold process, cache rỗng/ấm, manual review và toàn workflow; ghi cache-hit, provider, precision, thread count.

**Kiểm chứng:** Một báo cáo cho từng mode, không gộp/gán nhầm cold với warm, tự động với thủ công.

## Checkpoint triển khai đề xuất

| CP | Công việc | Điều kiện qua |
|---|---|---|
| CP1 | Thread tuning theo scope TTS, bỏ jitter local; sửa cache giọng/path/dịch | Fixed seed/audio không đổi khi tuning; đổi một voice/cue chỉ tính lại dependency đúng; chạy lại toàn clip và so baseline |
| CP2 | Gate chất lượng 15–20s, nhiều vai trước bulk generation | Người dùng chấp nhận phát âm/cảm xúc/nhịp; không coi preset nam/nữ là đủ |
| CP3 | Chọn backend/device từng stage theo phần cứng và backend đã đạt chất lượng | Runtime xác nhận thiết bị thật; CPU fallback rõ lý do; Intel/NVIDIA/CPU có profile tương ứng đã kiểm chứng hoặc ghi chưa test |
| CP4 | Prototype fuse TTS/ASR generation, precision A/B, batch/session reuse, typed retry/checkpoint | Không mất/cắt câu, beam giữ nguyên, chất lượng không giảm; đo native call count và toàn pipeline |
| CP5 | Đưa speaker-turn/context và cache/resume vào flow CLI chung | Video mới không phụ thuộc ledger hardcode; uncertain cần review, sửa riêng câu/người nói |
| CP6 | Benchmark nghiệm thu cuối | Tách install/cold/warm/manual; đo latency, CPU/RAM/GPU thực, nghĩa/giọng/timing và chỉ MKV+SRT; không hứa mốc thời gian trước đo |

CP1 có thể cải thiện CPU mà không đổi giọng. CP2 phải hoàn thành trước khi đầu tư port một backend TTS mà người dùng vẫn không thích. P0/P1/P2 là ưu tiên công việc, không phải lỗi nào cũng có thể sửa chỉ trong wrapper Python.

## Phần không nên tối ưu trước

- Wave/PCM timeline 0.07s, điều phối/gán theo bảng vài ms: không phải nguyên nhân 10 phút.
- FFmpeg/AAC và QA chạy CPU là hợp lý cho MKV stream-copy. Export 18.60s có thể đo tối ưu sau, nhưng không phải 396.6s TTS.
- Chín preset chia sẻ một model; chưa có bằng chứng chậm vì tải chín model.
- Cache y hệt cấu hình đã chạy lại Delta Force 2.52s; vấn đề cache mới tìm là invalidation quá rộng/alias, không phải mọi cache đều hỏng.
- Không đổi beam/model, tắt QC hoặc giảm precision mù quáng để làm số đo đẹp. Không bật GPU provider rồi bỏ qua các heads còn nằm ở CPU.

## Source và bằng chứng

- `src/jellydub/pipeline.py:188`: TTS auto→CPU; `:51`: job key có cache_dir; `:105`: cache dịch dưới workdir.
- `src/jellydub/tts/vieneu.py:17,39,48`: concurrency 1, toàn mapping trong cache_id, lazy model per-provider.
- `src/jellydub/tts/synthesize.py:144`: sleep local; normalize bằng FFmpeg từng cue.
- `.venv/lib/python3.12/site-packages/vieneu/_v3_turbo_engine/onnx_runtime_lite.py:157,327,414`: CPU EP, NumPy heads, frame/codebook loop.
- `.venv/lib/python3.12/site-packages/vieneu/v3turbo.py:530,588`: CPU sequential chunks; style ignored; GPU batch path là backend khác.
- `.venv/lib/python3.12/site-packages/optimum/intel/openvino/modeling_seq2seq.py:1020`: async→wait→CPU logits; `modeling_base.py:328`: torch compatibility device cpu.
- `src/jellydub/asr_openvino.py:118,155`: precision hint và bounded generation.
- `src/jellydub/translate.py:51` và `pyproject.toml:43`: CPU/CUDA selection và CPU wheels.
- Các số đo: `artifacts/pipeline-bottlenecks-2026-11-16/{tts-profile.json,blas-abab.json,asr-profile.json,cache-findings.json,findings.json}`. Có `.prof`, script tái lập và WAV hash đối chiếu trong cùng thư mục.
- Baseline đầy đủ: `artifacts/family-guy-vieneu/measured-01/report.json`; các phép đo micro bổ sung không thay thế benchmark full-video.

Giới hạn: chưa port GPU TTS, chưa thử FP16/INT8 trong audit này, chưa đo GPU occupancy/kernel trace, chưa có benchmark C++/Rust so sánh. Không cài/gỡ dependency hoặc sửa pipeline chính trong lượt audit.
