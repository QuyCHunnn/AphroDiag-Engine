# Family Guy — đo pipeline và bản nhiều giọng

Nguồn `videos/Family Guy - Peter falls down the stairs.mp4`: **156.665 giây** (2 phút 36.7 giây), 624×352.

Lượt đầy đủ, cache xử lý mới: **615.50 giây** (10 phút 15.5 giây), gồm kiểm tra file. Thời gian đến khi có hai file, chưa kiểm tra: **607.89 giây**.

| Công việc | Giây | Thiết bị |
|---|---:|---|
| Đọc metadata đầu vào | 0.125 | CPU / ffprobe |
| Chép lời Anh (Whisper Small, beam=5) | 145.715 | Intel GPU / OpenVINO; audio decode + VAD on CPU |
| Gán/tách lượt thoại theo bảng đã duyệt | 0.004 | CPU / reviewed source-specific ledger |
| Dịch tự động OPUS Anh → Việt | 33.558 | CPU / PyTorch OPUS EN-VI |
| Áp dụng biên tập tiếng Việt đã chuẩn bị | 0.002 | CPU / prepared review overrides |
| Tạo giọng + chuẩn hóa WAV (43 lượt thoại) | 396.635 | CPU / VieNeu v3 Turbo ONNX FP32 |
| Căn thời gian | 6.294 | CPU / FFmpeg |
| Dựng timeline PCM | 0.070 | CPU / PCM |
| Trích âm thanh nguồn | 2.549 | CPU / FFmpeg |
| Trộn voiceover | 4.310 | CPU / FFmpeg |
| Xuất MKV/SRT (encode AAC, copy video) | 18.602 | CPU / FFmpeg; video stream copy |
| Kiểm tra metadata, hash hình, decode toàn bộ | 7.606 | CPU / ffprobe + FFmpeg full decode |

Phần TTS chiếm **64.4%** thời gian. ASR encoder/decoder xác nhận `GPU.0`; VAD và giải mã audio chạy CPU. TTS ghi nhận `CPUExecutionProvider`; OPUS ghi nhận `cpu`. FFmpeg encode âm thanh và stream-copy hình; không render lại hình, không sử dụng GPU video encode.

## Bản sửa cuối và tổng chi phí xử lý

Lượt đầu có 47 dòng, 43 clip giọng. Rà lại thấy một câu Peter/Lois còn bị gộp; tách thành **48 dòng, 44 clip, 9 giọng**. Chỉ tạo mới **2 clip**, các clip không đổi dùng cache. Lượt chỉnh sửa + xuất/kiểm tra file mất **52.12 giây** (tạo giọng 22.19 giây).

**Tổng hai lượt máy chạy có đo: 667.63 giây = 11 phút 07.6 giây.** Đây là lượt đầy đủ cộng lượt sửa. Không trình bày tổng này như một lượt tự động sạch, và không cộng thời gian hội thoại/chờ/chuẩn bị thủ công hoặc các lượt chẩn đoán trước đó.

## Giọng và người nói

| Nhân vật | Preset VieNeu | Giới tính preset |
|---|---|---|
| PETER | Quốc Tuấn | Nam |
| LOIS | Ngọc Huyền | Nữ |
| CHRIS | Phạm Tuyên | Nam |
| MEG | Đoan Trang | Nữ |
| BRIAN | Xuân Vĩnh | Nam |
| STEWIE | Quang Sơn | Nam |
| TV_NARRATOR | Minh Đức | Nam |
| VAMPIRE | Thái Sơn | Nam |
| WEREWOLF | Đức Trí | Nam |

Không dùng Hải Đăng/Trúc Ly. Các preset dùng chung một model VieNeu v3 Turbo ONNX FP32, không tải chín model. Giọng là preset, không clone/nhái chính xác diễn viên hoặc giọng trẻ em. Nhân vật được gán bằng bảng đã rà cho video này; chưa có diarization tự động cho mọi video. Bốn dòng tiếng hét/rên/cười là chú thích; giữ âm thanh gốc thay vì bắt TTS đọc ký tự la hét.

## Phạm vi phép đo và chuẩn bị

Từ file video, lượt đầy đủ thật sự gọi ASR GPU, OPUS CPU, VieNeu CPU với cache SRT/dịch/giọng/căn thời gian trống. Model đã tải sẵn; thời gian load/compile trong mỗi công đoạn vẫn được đo. Bảng người nói, sửa mốc đổi người và bản biên tập tiếng Việt được chuẩn bị thủ công trước lượt đo; thời gian chuẩn bị đó **không nằm trong tổng pipeline tự động**. Raw ASR và raw OPUS được giữ để đối chiếu; không tuyên bố nhận diện người nói/dịch đạt nghiệm thu hoàn toàn tự động. Chưa nghe nghiệm thu thủ công trong VLC.

Nguồn chính để rà là video người dùng cung cấp. Có đối chiếu thêm ngữ cảnh câu hỏi Peter/Lois và câu cuối tại [transcript tập The Blind Side](https://transcripts.foreverdreaming.org/viewtopic.php?t=21302); clip có cảnh TV khác bản phát sóng nên không dùng transcript ngoài thay toàn bộ clip.

## Lỗi ASR đã tìm và sửa

Lượt chẩn đoán cũ mất **353.44 giây**: tiếng hét bị lặp thành hàng nghìn chữ A. Giới hạn `no_repeat_ngram_size=4` và ngân sách token theo thời lượng vùng audio (48–400 token), giữ beam=5. Lượt đo sau sửa mất **145.72 giây**. Có log tiến độ/thời gian từng vùng VAD và tăng phiên bản cache ASR. Đây không phải đổi model nhỏ hơn hoặc giảm beam.

## Kiểm tra file cuối

- 48 dòng SRT, 44 clip giọng, 9 preset riêng; đúng giới tính preset cho hai nhân vật nữ và các nhân vật nam.
- Không cảnh báo chồng/lệch placement; không clip nào đọc vượt khoảng phụ đề.
- Audio Việt và SRT Việt `vie`, default=1; SRT gắn trong MKV khớp SRT rời.
- Video stream có hash giống nguồn, source SHA-256 không đổi; hai lượt decode toàn bộ không lỗi.
- Chỉ MKV + SRT trong thư mục bàn giao. Ảnh kiểm tra tiếng Việt/nhân vật: `qa-peter.jpg`, `qa-lois.jpg`, `qa-stewie.jpg`.
- 111 test liên quan pass sau sửa ASR/routing; không dùng mock test thay kiểm tra video thực.

## Tái lập và bằng chứng

```bash
# Lượt đầy đủ mới, tên run phải chưa tồn tại; GPU phải thấy được trong WSL.
.venv/bin/python scripts/benchmark_family_guy.py --run measured-02

# Dựng lại bản đã duyệt, dùng lại cache (không chép lời/dịch lại).
.venv/bin/python scripts/repair_family_guy.py
```

Bảng cho đúng nguồn: `artifacts/family-guy-vieneu/speaker-ledger.json`; bảng lượt đầu: `speaker-ledger-first-pass.json`. Script kiểm tra hash nguồn và ASR so với bảng để tránh gán nhầm video/câu.

Số liệu gốc: `measured-01/report.json`, `repair-report.json`, `summary.json`, `final-verification.json`; log `measured-01.log`, `repair.log`; toàn bộ nằm trong `artifacts/family-guy-vieneu/`. Tổng thời gian suy luận model và FFmpeg đã đo không bao gồm tải model lần đầu.
