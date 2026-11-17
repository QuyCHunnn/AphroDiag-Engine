# Delta Force — bản VieNeu

Giữ nguyên video, 16 câu phụ đề Việt đã biên tập và mapping người nói của bản trước. Chỉ thay TTS:

- Nam: Hải Đăng, 10 câu.
- Nữ: Trúc Ly, 6 câu.
- VieNeu 3.8.3, v3 Turbo, ONNX/CPU FP32; không dùng Edge-TTS, không clone giọng nhân vật gốc. Hai preset được xác nhận trong danh sách preset của SDK và tạo mẫu thực tế.
- Sau khi tải model lần đầu, lệnh dựng chạy với `HF_HUB_OFFLINE=1`. Không sử dụng GPU cho bước TTS trong môi trường Intel hiện tại.

Đầu ra riêng ở `delivery/delta-force-vieneu/` chỉ có MKV và SRT. Bản Edge ở `delivery/delta-force/` được giữ nguyên để đối chiếu.

Mẫu nghe thử: `artifacts/delta-force-vieneu/male.wav` và `female.wav`. Mẫu nam tạo 2.00 giây âm thanh trong 7.95 giây xử lý; nữ 1.68 giây âm thanh trong 5.61 giây xử lý, không gồm tải/load model. Đây là số đo mẫu, không ngoại suy tốc độ video khác.

## Dựng lại

```bash
.venv/bin/python scripts/rebuild_delta_force_vieneu.py
```

Cần model đã tải từ bước thử mẫu. Script giữ cache, kiểm tra hash nguồn rồi dùng SRT đã duyệt; không chạy lại ASR/dịch. Cấu hình tên giọng từng nhân vật: `artifacts/delta-force-vieneu/settings.toml`. Lệnh và thời gian thực tế: `run.json` cùng thư mục.

## Thay đổi engine

`[jellydub.speaker_voices]` dùng được với `tts="vieneu"`, chọn preset theo nhãn trong SRT. Reference cụ thể vẫn có ưu tiên cao hơn preset; reference lỗi fallback về preset của đúng người nói. Cache phân biệt mapping giọng để không dùng nhầm giọng Edge hoặc cấu hình cũ.

Kiểm thử liên quan: 165 passed. Đã chạy thử hai giọng và toàn bộ 16 câu bằng SDK thực; mock test chỉ kiểm tra thêm routing/cache/fallback. Bản dịch và xác định người nói vẫn theo [quy trình bản đầu](delta-force-run.md), chưa phải tự nhận diện nhân vật/giới tính cho mọi video. Chưa nghe nghiệm thu thủ công bằng VLC; hai mẫu và bản MKV dùng để người dùng đánh giá giọng.

## Kết quả file cuối

Tạo mới 16 câu giọng, căn thời gian, trộn voiceover và xuất MKV/SRT mất **122.16 giây** trên CPU, chạy offline. Không gồm tải model, ASR hoặc dịch (dùng lại SRT đã duyệt). Có đủ 16 clip, không cảnh báo timing và không câu nào đọc vượt thời gian phụ đề. Hình stream-copy có hash giống nguồn; toàn bộ file giải mã không lỗi. Audio Việt và phụ đề Việt là track mặc định; chỉ hai file MKV/SRT trong thư mục bàn giao. Số liệu: `artifacts/delta-force-vieneu/verification.json`.

## Đo lại cùng cấu hình

Chạy lại đúng cùng lệnh cho video Delta Force (189.8 giây) mất **2.52 giây**, không gọi `VieNeuTTS.synthesize` lần nào (new_speech_calls=0); lấy 16 câu từ cache mất 0.027 giây. Bằng chứng `artifacts/delta-force-vieneu/replay-profile.json` và `profile_replay.py`. Lượt 122.16 giây là tạo giọng VieNeu mới sau khi đổi khỏi Edge, không phải chạy lại cùng voice có cache. Không dùng số đo này để kết luận cho video 2:17 khác nếu chưa biết file/lệnh/config thực tế.
